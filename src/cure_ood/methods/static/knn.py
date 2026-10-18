"""kNN tĩnh: điểm nền của mọi phương pháp test-time."""
from __future__ import annotations

import itertools

import numpy as np
from sklearn.metrics import roc_auc_score

from ...ops import kth_sim, l2
from ..base import Method, Output, knn_context, register

K_GRID = [1, 2, 5, 10, 25, 50, 100, 200, 500, 750, 1000]  # lưới K của OpenOOD v1.5


@register("knn")
class KNN(Method):
    """OOD = -cosine tới láng giềng thứ K trong gallery (tương đương đơn điệu với kNN-OOD trên đặc trưng L2)."""

    def score(self, stream):
        return Output(-knn_context(stream).b)


def select_k(bench, grid=K_GRID) -> tuple[int, float]:
    """Chọn K bằng APS trên (id_val, val_ood); dùng để sinh trường `k` trong configs/benchmarks."""
    if not bench.val_ood:
        raise ValueError(f"{bench.name}: ko có val_ood để chọn K")
    p, auroc = aps_select(lambda f, z, K: -kth_sim(l2(f), bench.G, K), {"K": grid},
                          (bench.cal_feats, None), bench.val_ood_data())
    return p["K"], auroc


def aps_select(score_fn, grid: dict, val_id, val_ood, valid=None) -> tuple[dict, float]:
    """Chọn tổ hợp có AUROC (val ID vs val OOD) cao nhất; hòa thì giữ tổ hợp gặp trước, như OpenOOD."""
    best = None
    for vals in itertools.product(*grid.values()):
        p = dict(zip(grid, vals))
        if valid is not None and not valid(p):
            continue
        si, so = score_fn(*val_id, **p), score_fn(*val_ood, **p)
        a = roc_auc_score(np.r_[np.zeros(len(si)), np.ones(len(so))], np.r_[si, so])
        if best is None or a > best[1]:
            best = (p, a)
    return best
