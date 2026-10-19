"""OODD (code chính thức) và hai biến thể trên cùng host bộ nhớ với CURE: OODD-core và hàng đợi hết hạn theo tuổi."""
from __future__ import annotations

import json
from pathlib import Path
from queue import PriorityQueue

import numpy as np
import torch

from ...data.benchmark import data_root
from ...ops import DEVICE
from ..base import Method, Output, Session, knn_context, register
from ..memory import run_memory
from ._official import oodd_knn as official

T_OUT = Path(__file__).resolve().parents[4] / "configs" / "oodd_t_out.json"


def informative_gallery(bench):
    """Gallery informative của OODD: file 5 crop nếu có, ko thì xấp xỉ 1 view (ImageNet-200)."""
    path = data_root() / f"oodd_gallery_{bench.name}.npz"
    if path.exists():
        d = np.load(path)
        return d["activation_log"], d["idx"], "5-crop"
    msp = torch.softmax(torch.as_tensor(bench.train_logits, dtype=torch.float32), 1).max(1).values
    idx = official.informative_order(msp, torch.as_tensor(bench.train_labels))
    return official.normalizer(bench.train_feats).astype(np.float32), idx, "1-view"


@register("oodd")
class OODD(Method):
    """OODD chạy conf_postprocess gốc; t_out=True thêm 128 mẫu OOD phụ trợ theo danh sách chính thức."""

    def __init__(self, K1=10, K2=5, alpha=0.5, queue_size=512, batch=512, t_out=False):
        self.K1, self.K2, self.alpha, self.queue_size, self.batch, self.t_out = K1, K2, alpha, queue_size, batch, t_out

    def fit(self, bench):
        super().fit(bench)
        self.activation_log, self.idx, self.gallery_kind = informative_gallery(bench)
        self.t_out_index = json.loads(T_OUT.read_text(encoding="utf-8")).get(bench.name, {}) if self.t_out else {}
        return self

    def aux(self, domain):
        if not self.t_out:
            return None
        if domain not in self.t_out_index:
            raise KeyError(f"OODD T-out: ko có danh sách cho {self.bench.name}/{domain}")
        spec = self.t_out_index[domain]
        return official.normalizer(self.bench.array(f"{spec['set']}_feats")[spec["idx"]]).astype(np.float32)

    def score(self, stream):
        feats = official.normalizer(stream.feats).astype(np.float32)
        # Code gốc dùng self.K cho T-out nhưng ko gán ở cấu hình mặc định (sẽ lỗi); ta dùng K1.
        s, trace = official.conf_postprocess(self.activation_log, self.idx, feats, self.K1, self.K2, self.alpha,
                                             self.queue_size, self.batch, self.aux(stream.domain), K_prior=self.K1)
        return Output(-s, trace)

    def session(self, domain=None):
        return OODDSession(self, self.aux(domain))


class OODDSession(Session):
    """Vòng lặp conf_postprocess gốc tách theo batch: hàng ưu tiên, memory_bank T-out và điền trước bằng mẫu phụ trợ."""

    def __init__(self, m: OODD, aux):
        super().__init__()
        self.m = m
        # Bản gốc tải lại ftrain lên GPU mỗi lần gọi; khi triển khai ftrain nằm sẵn trên GPU.
        self.ftrain = torch.tensor(m.activation_log[m.idx[:int(m.alpha * m.activation_log.shape[0])]], device=DEVICE)
        self.queue, self.pool, self.memory_bank = PriorityQueue(), {}, None
        if aux is not None:
            # Mẫu phụ trợ mang chỉ số âm; ScoreData chỉ so điểm nên cấu trúc heap giống bản gốc.
            prior = self.train_scores(aux, m.K1)
            self.push(prior, aux, -len(aux))
            self.memory_bank = aux[:5]

    @torch.no_grad()
    def train_scores(self, data, K):
        """batched_matrix_multiply(ftrain, data, K) với ftrain thường trú: cùng phép nhân, cùng khối 128 cột."""
        d = torch.tensor(data, device=DEVICE)
        return np.concatenate([official.kth_largest_per_column(self.ftrain @ d[s:s + 128].T, K)
                               for s in range(0, len(d), 128)])

    def push(self, scores, feats, first):
        for j in range(len(scores)):
            self.pool[first + j] = feats[j]
            self.queue.put(official.ScoreData(scores[j], first + j))
            if self.queue.qsize() > self.m.queue_size:
                self.queue.get()
        ids = [item.data for item in list(self.queue.queue)]
        self.pool = {d: self.pool[d] for d in ids}
        return ids

    def step(self, feats, logits=None):
        batch = official.normalizer(feats).astype(np.float32)
        score = self.train_scores(batch, self.m.K1)
        ids = self.push(score, batch, self.t)
        food = np.stack([self.pool[d] for d in ids])
        if self.memory_bank is not None:
            food = np.concatenate([food, self.memory_bank], axis=0)
        # Bản gốc lỗi khi hàng đợi có ít hơn K2 mẫu (chỉ gặp khi batch < K2, ở đầu luồng): lấy láng giềng xa nhất có được.
        out = Output(-(score - official.batched_matrix_multiply(food, batch, min(self.m.K2, len(food)))),
                     [(self.t, np.array([d for d in ids if d >= 0], dtype=int))])
        self.t += len(feats)
        return out




@register("age_queue")
class AgeExpiryQueue(Method):
    """Đối chứng cùng host: hàng đợi OODD-core nhưng mẫu quá `ttl` mẫu tuổi bị xóa; ko hậu kiểm, ko lối lùi."""

    def __init__(self, ttl=512, batch=128, capacity=512, k_mem=5):
        self.ttl, self.batch, self.capacity, self.k_mem = ttl, batch, capacity, k_mem

    def score(self, stream):
        ctx = knn_context(stream)
        r = run_memory(stream.x, ctx.b, ctx.passes, ctx.shadow, batch=self.batch, capacity=self.capacity,
                       k_mem=self.k_mem, ttl=self.ttl)
        return Output(r.scores, r.trace)
