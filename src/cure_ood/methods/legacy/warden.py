"""Bản dùng ở vòng 21-57, giữ để tái hiện bảng cũ."""
from __future__ import annotations

import numpy as np
import torch

from ...ops import DEVICE
from ...stats import EPS, rank_p
from ..base import Method, Output, knn_context, register

# ---------------------------------------------------------------------------------------------------------------
# Bản tái lập cũ (vòng 53), giữ để tái hiện số đã công bố trong bảng; KHÔNG trung thành với bài:
# xếp hạng bằng min hai p (bài xếp hạng bằng điểm nền), nạp trước khi chấm cùng batch, ko làm trắng, ko CDC.
R53_GRID = ((0.5, 0.1), (0.1, 0.2), (0.05, 0.2))  # chọn theo AUROC test: cận trên lạc quan


def _ebh_r53(p, delta, kappa):
    """e-BH của bản cũ, giữ nguyên thứ tự sắp xếp để tái hiện từng bit."""
    e = kappa * np.power(np.maximum(p, EPS), kappa - 1)
    order, m = np.argsort(-e), len(p)
    ok = e[order] >= m / (delta * np.arange(1, m + 1))
    sel = np.zeros(m, bool)
    if ok.any():
        sel[order[: np.flatnonzero(ok)[-1] + 1]] = True
    return sel


@register("warden_r53")
class WARDENRound53(Method):
    """Bản cũ: điểm = -log min(1, 2·min(p_gốc, p_bank)) - 1e-6·s0; chỉ dùng để tái hiện kết quả vòng 53."""

    def __init__(self, kappa=0.5, delta=0.1, batch=64, cap=512):
        self.kappa, self.delta, self.batch, self.cap = kappa, delta, batch, cap

    @torch.inference_mode()
    def score(self, stream):
        ctx = knn_context(stream)
        b, n = ctx.b, len(stream)
        xt = torch.as_tensor(stream.x, device=DEVICE)
        rt = torch.as_tensor(self.bench.C, device=DEVICE)
        p_base, p_dict = rank_p(-b, -ctx.cb), np.ones(n)
        bank, trace = [], []
        for s in range(0, n, self.batch):
            idx = np.arange(s, min(n, s + self.batch))
            sel = idx[_ebh_r53(p_base[idx], self.delta, self.kappa)]
            if len(sel):
                bank.append(sel)
                while sum(len(v) for v in bank) > self.cap:
                    bank.pop(0)
            if not bank:
                continue
            ids = np.concatenate(bank)
            sims = xt[idx] @ xt[ids].T
            own = torch.as_tensor(ids[None, :] == idx[:, None], device=DEVICE)
            prox = torch.where(own, torch.tensor(-4.0, device=DEVICE), sims).max(1).values.cpu().numpy()
            prox_r = (rt @ xt[ids].T).max(1).values.cpu().numpy()
            p_dict[idx] = rank_p(prox, prox_r)
            trace.append((s, ids))
        return Output(-np.log(np.minimum(1, 2 * np.minimum(p_base, p_dict))) - 1e-6 * b, trace)
