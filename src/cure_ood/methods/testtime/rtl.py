"""RTL++ (code chính thức) và bản rút gọn dùng ở vòng 21-57."""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from ..base import Method, Output, knn_context, register
from ._official.rtl import RTL_OOD


@register("rtl_plus")
class RTLPlus(Method):
    """RTL++ chạy RTL_OOD.reranking gốc theo batch (chấm cả luồng cần ma trận n×n, vượt RAM)."""

    def __init__(self, base="msp", batch=512, percent=0.6, alpha=1e-5):
        self.base, self.batch, self.percent, self.alpha = base, batch, percent, alpha
        self.requires_logits = base == "msp"

    def score(self, stream):
        if self.base == "msp":
            z = stream.logits - stream.logits.max(1, keepdims=True)
            y0 = -(np.exp(z) / np.exp(z).sum(1, keepdims=True)).max(1)
        else:
            y0 = -knn_context(stream).b
        X, n = stream.feats, len(stream)
        step = self.batch or n
        out = np.empty(n)
        with warnings.catch_warnings():  # code gốc ko đặt max_iter; alpha = 1e-5 thường ko hội tụ hẳn
            warnings.simplefilter("ignore", ConvergenceWarning)
            for s in range(0, n, step):
                out[s:s + step] = RTL_OOD().reranking(X[s:s + step], y0[s:s + step].reshape(-1, 1), self.percent,
                                                      self.alpha).ravel()
        return Output(out)


