"""CURE: hàng chờ hậu kiểm, chứng nhận e-BH vào CDM, gộp Cauchy có hiệu chuẩn, lùi về điểm tĩnh (docs/method.md)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..ops import DEVICE, l2
from ..stats import (
    CERTIFIERS,
    RULES,
    betabinom_pvalues,
    combine,
    conformal_combine,
    excess_counts,
    loo_p,
    oracle_certifier,
    rank_p,
)
from .base import Method, Output, Session, knn_context, knn_reference, register
from .memory import run_memory


def static_scores(logits, base: str) -> np.ndarray:
    """Điểm MSP/Energy hướng OOD, tính bằng torch float32 như bản gốc."""
    t = torch.as_tensor(logits, dtype=torch.float32)
    return -(torch.softmax(t, 1).max(1).values if base == "msp" else torch.logsumexp(t, 1)).numpy()


def base_scores(stream, base: str):
    """Điểm nền hướng OOD cho luồng và cho C. kNN dùng s0."""
    if base == "knn":
        ctx = knn_context(stream)
        return -ctx.b, -ctx.cb
    return static_scores(stream.logits, base), static_scores(stream.bench.cal_logits, base)


@register("cure")
class CURE(Method):

    def __init__(self, base="knn", window=256, batch=128, capacity=512, k_mem=5, k_shadow=10, screen_q=0.2,
                 certifier="ebh", alpha=0.2, rule="cauchy", w_mem=1.0, wake=None, memory="cdm", fallback=True):
        """Mặc định là cấu hình chính thức; certifier: ebh | elond | all | oracle_labels (chẩn đoán, đọc nhãn)."""
        self.base, self.window, self.batch, self.capacity = base, window, batch, capacity
        self.k_mem, self.k_shadow, self.screen_q = k_mem, k_shadow, screen_q
        self.certifier, self.alpha, self.rule, self.w_mem = certifier, alpha, rule, w_mem
        self.wake, self.memory, self.fallback = wake, memory, fallback
        self.requires_logits = base != "knn"

    def memory_channel(self, stream):
        """Kênh bộ nhớ chỉ phụ thuộc s0 của kNN, nên dùng chung giữa các nền (kNN/MSP/Energy)."""
        key = ("cure_mem", self.window, self.batch, self.capacity, self.k_mem, self.k_shadow, self.screen_q,
               self.certifier, self.alpha, str(self.wake), self.memory)
        if key not in stream.cache:
            ctx = knn_context(stream, self.screen_q)
            cnt, npass = excess_counts(stream.x, ctx.passes, ctx.shadow, self.k_shadow, self.window)
            p_e = betabinom_pvalues(cnt, npass, self.k_shadow, len(ctx.shadow))
            # oracle_labels là ngoại lệ duy nhất đọc stream.y: cận trên của hậu kiểm, chỉ để ablation.
            cert = (oracle_certifier(stream.y) if self.certifier == "oracle_labels"
                    else CERTIFIERS[self.certifier](p_e, alpha=self.alpha))
            stream.cache[key] = run_memory(
                stream.x, ctx.b, ctx.passes, ctx.shadow, batch=self.batch, capacity=self.capacity, k_mem=self.k_mem,
                ttl=self.window, certifier=cert,
                queue_in_memory=self.memory != "cdm", wake=tuple(self.wake) if self.wake else None,
                k_shadow=self.k_shadow, calib=(stream.bench.C, ctx.cb))
        return stream.cache[key]

    def score(self, stream):
        mem = self.memory_channel(stream)
        s, s_cal = base_scores(stream, self.base)
        p, stat, covered = conformal_combine(s, s_cal, mem.pvals, mem.mem_cal, len(stream), self.rule, self.w_mem)
        return self.finish(s, p, stat, covered, rank_p(s, s_cal), mem.trace, mem.activity(self.batch))

    def finish(self, s, p, stat, covered, p_static, trace, stats) -> Output:
        """Điểm và p cuối: kênh gộp khi có bộ nhớ, ko thì lùi về điểm tĩnh (trừ khi fallback=False)."""
        if not self.fallback:
            return Output(-np.log(p) + 1e-4 * stat, trace, p, stats)
        # Phá hòa: 1e-4·T giữa các mẫu cùng p bão hòa; 1e-7·s giữ thứ tự của điểm nền.
        sc = np.where(covered, -np.log(p) + 1e-4 * stat, -np.log(p_static)) + 1e-7 * s
        return Output(sc, trace, np.where(covered, p, p_static), stats)

    def session(self, domain=None):
        return CURESession(self)


class CURESession(Session):
    """CURE từng batch: N_e của mẫu chờ được cộng dần khi các mẫu sau nó đến, nên hậu kiểm chỉ dùng mẫu đã đến."""

    def __init__(self, m: CURE):
        super().__init__()
        if m.certifier not in RULES or m.memory != "cdm" or m.wake:
            raise ValueError("Phiên trực tuyến của CURE cần memory=cdm, ko wake, certifier ebh | elond | all")
        self.m, B = m, m.bench
        self.cb, self.tau, shadow = knn_reference(B, m.screen_q)
        # Chuẩn hóa lại như excess_counts để bán kính và độ tương tự của hậu kiểm khớp score().
        self.shadow = F.normalize(torch.as_tensor(shadow, device=DEVICE), dim=1)
        self.n_shadow = len(shadow)
        self.cal_x = torch.as_tensor(B.C, device=DEVICE)
        self.s_cal = -self.cb if m.base == "knn" else static_scores(B.cal_logits, m.base)
        self.ps_cal = loo_p(self.s_cal)
        self.certify = RULES[m.certifier](alpha=m.alpha)
        d = B.dim
        # Hàng chờ: chỉ số, s0, N_e, n_pass, bán kính r_e, x (cho CDM) và x chuẩn hóa lại (cho hậu kiểm).
        self.q = {"idx": np.empty(0, np.int64), "s0": np.empty(0, np.float32), "cnt": np.empty(0, np.int64),
                  "npass": np.empty(0, np.int64), "rad": torch.empty(0, device=DEVICE),
                  "x": torch.empty(0, d, device=DEVICE), "xn": torch.empty(0, d, device=DEVICE)}
        self.cdm_idx, self.cdm_x = np.empty(0, np.int64), torch.empty(0, d, device=DEVICE)
        self.cdm_sizes, self.covered, self.certified, self.evicted = [], 0, 0, 0
        self.cdm_view = None  # (mem, mem_x, aff_c): chỉ đổi khi CDM đổi, tránh tính lại C x CDM mỗi bước

    @torch.inference_mode()
    def step(self, feats, logits=None):
        m, s, n = self.m, self.t, len(feats)
        x = l2(feats)
        xt = torch.as_tensor(x, device=DEVICE)
        s0 = m.bench.knn(x)
        xn = F.normalize(xt, dim=1)
        new = {"idx": np.arange(s, s + n), "s0": s0, "cnt": np.zeros(n, np.int64), "npass": np.zeros(n, np.int64),
               "rad": (xn @ self.shadow.T).topk(m.k_shadow, dim=1).values[:, -1], "x": xt, "xn": xn}
        # Mẫu đầu batch có thể nằm đúng cuối cửa sổ của mẫu sắp hết hạn: đếm batch trước rồi mới hậu kiểm.
        self.count(self.q, new)
        self.count(new, new)
        self.review(s)
        self.cdm_sizes.append(len(self.cdm_idx))
        self.enqueue(new)
        static = -s0 if m.base == "knn" else static_scores(logits, m.base)
        p_static = rank_p(static, self.s_cal)
        p, stat, covered, trace = np.ones(n), np.zeros(n), np.zeros(n, bool), []
        if len(self.cdm_idx):
            if self.cdm_view is None:
                order = np.argsort(self.cdm_idx)
                mem_x = self.cdm_x[torch.as_tensor(order, device=DEVICE)]
                k = min(m.k_mem, len(order))
                aff_c = (self.cal_x @ mem_x.T).topk(k, dim=1).values[:, -1].cpu().numpy()
                self.cdm_view = (self.cdm_idx[order], mem_x, aff_c)
            mem, mem_x, aff_c = self.cdm_view
            k = min(m.k_mem, len(mem))
            # CDM chỉ chứa mẫu đã hậu kiểm nên ko có mẫu của batch này: ko cần bỏ tự-khớp.
            aff = (xt @ mem_x.T).topk(k, dim=1).values[:, -1].cpu().numpy()
            t_test, t_cal = -(s0 - aff), -(self.cb - aff_c)
            p_mem = rank_p(t_test, t_cal)
            stat = combine([p_static, p_mem], [1.0, m.w_mem], m.rule)
            p = rank_p(stat, combine([self.ps_cal, loo_p(t_cal)], [1.0, m.w_mem], m.rule))
            covered[:], trace = True, [(s, mem)]
            self.covered += n
        self.t += n
        return m.finish(static, p, stat, covered, p_static, trace, None)

    def review(self, s):
        """Hậu kiểm các mẫu đã chờ đủ `window` mẫu: chứng nhận vào CDM (FIFO) hoặc trục xuất."""
        q, m = self.q, self.m
        exp = (s - q["idx"]) >= m.window
        if not exp.any():
            return
        ok = self.certify(betabinom_pvalues(q["cnt"][exp], q["npass"][exp], m.k_shadow, self.n_shadow))
        exp_t = torch.as_tensor(exp, device=DEVICE)
        self.cdm_idx = np.concatenate((self.cdm_idx, q["idx"][exp][ok]))[-m.capacity:]
        self.cdm_x = torch.cat((self.cdm_x, q["x"][exp_t][torch.as_tensor(ok, device=DEVICE)]))[-m.capacity:]
        self.certified += int(ok.sum())
        if ok.any():
            self.cdm_view = None
        self.evicted += int((~ok).sum())
        self.q = {key: v[~exp_t] if torch.is_tensor(v) else v[~exp] for key, v in q.items()}

    def count(self, q, new):
        """Cộng vào N_e, n_pass của các mẫu trong q những mẫu mới đến sau chúng trong cửa sổ và qua sàng lọc."""
        f = np.flatnonzero(new["s0"] <= self.tau)
        if len(f) == 0 or len(q["idx"]) == 0:
            return
        lag = torch.as_tensor(new["idx"][f][None, :] - q["idx"][:, None], device=DEVICE)
        inwin = (lag > 0) & (lag <= self.m.window)
        hit = inwin & ((q["xn"] @ new["xn"][torch.as_tensor(f, device=DEVICE)].T) >= q["rad"][:, None])
        q["cnt"] += hit.sum(1).cpu().numpy()
        q["npass"] += inwin.sum(1).cpu().numpy()

    def enqueue(self, new):
        """Thêm batch vào hàng chờ, giữ `capacity` mẫu có s0 thấp nhất theo đúng thứ tự argpartition của run_memory."""
        cand = {key: torch.cat((v, new[key])) if torch.is_tensor(v) else np.concatenate((v, new[key]))
                for key, v in self.q.items()}
        keep = min(self.m.capacity, len(cand["idx"]))
        sel = np.argpartition(cand["s0"], keep - 1)[:keep]
        sel_t = torch.as_tensor(sel, device=DEVICE)
        self.q = {key: v[sel_t] if torch.is_tensor(v) else v[sel] for key, v in cand.items()}

    def activity(self):
        return {"cdm_mean": float(np.mean(self.cdm_sizes)), "coverage": self.covered / self.t,
                "certified": self.certified, "evicted": self.evicted}
