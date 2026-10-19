"""AdaODD cài lại theo bài, đối chiếu ở docs/reimplementations/adaodd.md."""
from __future__ import annotations

import numpy as np
import torch

from ...ops import DEVICE, l2
from ..base import Method, Output, Session, register

# Sec. VI-A "Hyper-parameters details". k = None: quy tắc k = 1‰ cỡ tập train (Sec. VI-C "Effect of k").
# CIFAR-100: bài chỉ cho k = 200; γ, κ lấy theo CIFAR-10 (cùng nhóm benchmark nhỏ).
PRESETS = {
    "cifar10": {"k": 50, "gamma": 1.5, "kappa": 5.0},
    "cifar100": {"k": 200, "gamma": 1.5, "kappa": 5.0},
    "imagenet": {"k": None, "gamma": 1.0, "kappa": 10.0},
}
AGGREGATIONS = ("avg", "kth", "median")  # Sec. VI-B "How to combine the score", Fig. 4/9


def preset_for(name: str) -> dict:
    """Bộ siêu tham số của bài ứng với benchmark (theo tên bộ ID)."""
    if name.startswith("imagenet"):
        return PRESETS["imagenet"]
    if name.startswith("cifar100"):
        return PRESETS["cifar100"]
    return PRESETS["cifar10"]  # gồm cả CIFAR-10 SupCon/DINOv2 (Bảng II dùng chung cấu hình)


def knn_score(q: torch.Tensor, mem: torch.Tensor, wts: torch.Tensor, k: int, agg: str = "avg") -> torch.Tensor:
    """S(x) = (1/k) Σ -‖z - z_(i)‖₂ · s_(i) (Alg. 1 dòng 5-6); lớn = quen. Láng giềng xếp theo khoảng cách thuần."""
    k = min(k, len(mem))
    d, i = torch.cdist(q, mem).topk(k, dim=1, largest=False)
    if agg == "avg":
        return -(d * wts[i]).mean(1)
    if agg == "kth":
        return -(d[:, -1] * wts[i[:, -1]])
    if agg == "median":  # z_(⌊k/2⌋), chỉ số tính từ 1
        j = max(k // 2, 1) - 1
        return -(d[:, j] * wts[i[:, j]])
    raise ValueError(f"agg phải thuộc {AGGREGATIONS}")


@register("adaodd")
class AdaODD(Method):
    """AdaODD: kNN có trọng số trên bộ nhớ mở rộng theo luồng (Alg. 1)."""

    def __init__(self, k=None, gamma=None, kappa=None, tpr=0.95, batch=128, agg="avg", normalize=True):
        if agg not in AGGREGATIONS:
            raise ValueError(f"agg phải thuộc {AGGREGATIONS}")
        self.k, self.gamma, self.kappa = k, gamma, kappa
        self.tpr, self.batch, self.agg, self.normalize = tpr, batch, agg, normalize

    def _resolve(self, bench, n_train: int):
        p = preset_for(bench.name)
        k = self.k if self.k is not None else p["k"]
        if k is None:
            k = max(1, round(n_train / 1000))
        gamma = self.gamma if self.gamma is not None else p["gamma"]
        kappa = self.kappa if self.kappa is not None else p["kappa"]
        if gamma < 1 or kappa < 1:
            raise ValueError("Bài yêu cầu γ ≥ 1 và κ ≥ 1")
        return int(k), float(gamma), float(kappa)

    def _feats(self, a):
        return a if self.normalize else np.ascontiguousarray(a, dtype=np.float32)

    def fit(self, bench):
        super().fit(bench)
        self.mem0 = self._feats(bench.G if self.normalize else bench.train_feats)
        self.k_, self.gamma_, self.kappa_ = self._resolve(bench, len(self.mem0))
        cal = self._feats(bench.C if self.normalize else bench.cal_feats)
        with torch.inference_mode():
            mem = torch.as_tensor(self.mem0, device=DEVICE)
            sc = knn_score(torch.as_tensor(cal, device=DEVICE), mem, torch.ones(len(mem), device=DEVICE),
                           self.k_, self.agg)
        # round: 1 - 0.95 = 0.05000000000000004 làm lệch bit λ so với vòng 53.
        self.lam = np.quantile(sc.cpu().numpy(), round(1 - self.tpr, 12))
        return self

    def score(self, stream):
        sess = self.session()
        x = stream.x if self.normalize else self._feats(stream.feats)
        out = np.concatenate([sess.adapt(x[s:s + self.batch]) for s in range(0, len(x), self.batch)])
        self.memory_ = (sess.mem, sess.wts)
        return Output(-out)

    def session(self, domain=None):
        return AdaODDSession(self)

    @torch.inference_mode()
    def reevaluate(self, feats: np.ndarray) -> float:
        """ID reevaluation accuracy (Sec. VI-B, Fig. 5): tỉ lệ mẫu ID có S ≥ λ khi chấm bằng bộ nhớ sau thích nghi."""
        mem, wts = self.memory_
        x = l2(feats) if self.normalize else self._feats(feats)
        sc = knn_score(torch.as_tensor(x, device=DEVICE), mem, wts, self.k_, self.agg)
        return float((sc.cpu().numpy() >= self.lam).mean())


class AdaODDSession(Session):
    """Bộ nhớ và trọng số của AdaODD giữ giữa các batch."""

    def __init__(self, m: AdaODD):
        super().__init__()
        self.m = m
        self.mem = torch.as_tensor(m.mem0, device=DEVICE)
        self.wts = torch.ones(len(self.mem), device=DEVICE)

    @torch.inference_mode()
    def adapt(self, x: np.ndarray) -> np.ndarray:
        """Chấm batch rồi nạp mẫu chắc chắn OOD (trọng số κ) và chắc chắn ID (trọng số 1) vào bộ nhớ."""
        m = self.m
        q = torch.as_tensor(x, device=DEVICE)
        sc = knn_score(q, self.mem, self.wts, m.k_, m.agg)
        add_ood, add_id = sc < m.lam * m.gamma_, sc > m.lam / m.gamma_
        self.mem = torch.cat([self.mem, q[add_ood], q[add_id]])
        self.wts = torch.cat([self.wts, torch.full((int(add_ood.sum()),), m.kappa_, device=DEVICE),
                              torch.ones(int(add_id.sum()), device=DEVICE)])
        self.t += len(x)
        return sc.cpu().numpy()

    def step(self, feats, logits=None):
        return Output(-self.adapt(l2(feats) if self.m.normalize else self.m._feats(feats)))
