"""Chỉ số phát hiện (OOD là lớp dương) và DCR (tỉ lệ mẫu ID trong bộ nhớ dùng để chấm)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def detection_metrics(y: np.ndarray, s: np.ndarray) -> dict:
    fpr, tpr, _ = roc_curve(y, s)
    return {"auroc": float(roc_auc_score(y, s)), "aupr_out": float(average_precision_score(y, s)),
            "aupr_in": float(average_precision_score(1 - y, -s)),
            "fpr95": float(fpr[np.searchsorted(tpr, 0.95)]), "tpr_fpr5": float(np.interp(0.05, fpr, tpr))}


def dcr_counts(memory, y: np.ndarray) -> list[int] | None:
    """[số lượt mẫu ID, tổng số lượt mẫu, số batch có bộ nhớ]; DCR = phần tử 0 / phần tử 1, cộng dồn qua batch."""
    if memory is None:
        return None
    n_id = sum(int((y[np.asarray(d, int)] == 0).sum()) for _, d in memory if len(d))
    return [n_id, sum(len(d) for _, d in memory), len(memory)]
