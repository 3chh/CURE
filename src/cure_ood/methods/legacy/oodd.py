"""Bản dùng ở vòng 21-57, giữ để tái hiện bảng cũ."""
from __future__ import annotations

from ..base import Method, Output, knn_context, register
from ..memory import run_memory


@register("oodd_core")
class OODDCore(Method):
    """OODD rút gọn của vòng 21-57: chỉ hàng đợi ưu tiên."""

    def __init__(self, batch=128, capacity=512, k_mem=5):
        self.batch, self.capacity, self.k_mem = batch, capacity, k_mem

    def score(self, stream):
        ctx = knn_context(stream)
        r = run_memory(stream.x, ctx.b, ctx.passes, ctx.shadow, batch=self.batch, capacity=self.capacity, k_mem=self.k_mem)
        return Output(r.scores, r.trace)
