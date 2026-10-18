"""AdaODD cài lại theo bài, đối chiếu ở docs/reimplementations/adaodd.md."""
from __future__ import annotations

import numpy as np
import torch

from ...ops import DEVICE, l2
from ..base import Method, Output, register

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

    @torch.inference_mode()
    def score(self, stream):
        mem = torch.as_tensor(self.mem0, device=DEVICE)
        wts = torch.ones(len(mem), device=DEVICE)
        xt = torch.as_tensor(stream.x if self.normalize else self._feats(stream.feats), device=DEVICE)
        lo, hi = self.lam * self.gamma_, self.lam / self.gamma_
        out = np.empty(len(xt), np.float32)
        for s in range(0, len(xt), self.batch):
            q = xt[s:s + self.batch]
            sc = knn_score(q, mem, wts, self.k_, self.agg)
            out[s:s + self.batch] = sc.cpu().numpy()
            add_ood, add_id = sc < lo, sc > hi
            mem = torch.cat([mem, q[add_ood], q[add_id]])
            wts = torch.cat([wts, torch.full((int(add_ood.sum()),), self.kappa_, device=DEVICE),
                             torch.ones(int(add_id.sum()), device=DEVICE)])
        self.memory_ = (mem, wts)
        return Output(-out)

    @torch.inference_mode()
    def reevaluate(self, feats: np.ndarray) -> float:
        """ID reevaluation accuracy (Sec. VI-B, Fig. 5): tỉ lệ mẫu ID có S ≥ λ khi chấm bằng bộ nhớ sau thích nghi."""
        mem, wts = self.memory_
        x = l2(feats) if self.normalize else self._feats(feats)
        sc = knn_score(torch.as_tensor(x, device=DEVICE), mem, wts, self.k_, self.agg)
        return float((sc.cpu().numpy() >= self.lam).mean())
