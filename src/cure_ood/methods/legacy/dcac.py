"""Bản dùng ở vòng 21-57, giữ để tái hiện bảng cũ."""
from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from ...ops import DEVICE
from ..base import Method, Output, register


def normalized_entropy(z) -> np.ndarray:
    """Entropy chuẩn hóa như code gốc (ln chia log2 C), float64 để lấy phân vị."""
    t = torch.as_tensor(z, dtype=torch.float32)
    return (-(torch.softmax(t, 1) * torch.log_softmax(t, 1)).sum(1) / np.log2(t.shape[1])).numpy().astype(np.float64)


@register("dcac_msp_r53")
class DCAC(Method):
    """DCAC bản vòng 21-57 (viết lại dạng vector, batch 128)."""
    requires_logits = True
    score_type = "msp"

    def __init__(self, shot_capacity=20, lower_percentile=95.0, alpha=0.9, topk=20, batch=128):
        self.shot_capacity, self.lower_percentile, self.alpha = shot_capacity, lower_percentile, alpha
        self.topk, self.batch = topk, batch

    def fit(self, bench):
        super().fit(bench)
        self.threshold = float(np.percentile(normalized_entropy(bench.train_logits), self.lower_percentile))
        return self

    @torch.inference_mode()
    def run(self, stream):
        """Hai điểm MSP/Energy cùng một lần chạy cache, dùng chung giữa hai biến thể."""
        key = ("dcac", self.shot_capacity, self.lower_percentile, self.alpha, self.topk, self.batch, self.threshold)
        if key in stream.cache:
            return stream.cache[key]
        feats = F.normalize(torch.as_tensor(stream.x, device=DEVICE), dim=1)
        logits = torch.as_tensor(np.asarray(stream.logits, np.float32), device=DEVICE)
        n, n_cls = logits.shape
        prob = torch.softmax(logits, 1)
        entropy = -(prob * torch.log_softmax(logits, 1)).sum(1) / np.log2(n_cls)
        admit = (entropy > self.threshold).cpu().numpy()
        pred = logits.argmax(1).cpu().numpy()
        caches = [deque(maxlen=self.shot_capacity) for _ in range(n_cls)]
        msp, energy = torch.empty(n, device=DEVICE), torch.empty(n, device=DEVICE)
        trace = []
        k = min(self.topk, n_cls)
        for s in range(0, n, self.batch):
            e = min(s + self.batch, n)
            for t in np.flatnonzero(admit[s:e]) + s:
                caches[pred[t]].append(int(t))
            idx = [t for c in caches for t in c]
            trace.append((s, idx))
            z = logits[s:e]
            if idx:
                cp = prob[idx]
                tv, ti = cp.topk(k, dim=1)
                aff = feats[s:e] @ feats[idx].T
                z = z - float(self.alpha) * (aff @ torch.zeros_like(cp).scatter(1, ti, tv))
            msp[s:e] = -torch.softmax(z, 1).max(1).values
            energy[s:e] = -torch.logsumexp(z, 1)
        stream.cache[key] = ({"msp": msp.cpu().numpy(), "energy": energy.cpu().numpy()}, trace)
        return stream.cache[key]

    def score(self, stream):
        scores, trace = self.run(stream)
        return Output(scores[self.score_type], trace)


@register("dcac_energy_r53")
class DCACEnergy(DCAC):
    score_type = "energy"
