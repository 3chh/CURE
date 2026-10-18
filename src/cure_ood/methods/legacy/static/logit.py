"""Phương pháp tĩnh chỉ dùng logit: MSP, MaxLogit, Energy, GEN."""
from __future__ import annotations

import numpy as np

from ....ops import logsumexp, softmax
from ...base import register
from .base import StaticMethod

OPENOOD = "https://github.com/Jingkang50/OpenOOD"


@register("msp_legacy")
class MSP(StaticMethod):
    requires_logits = True

    def ood_score(self, feats, logits):
        return -softmax(logits).max(1)


@register("maxlogit_legacy")
class MaxLogit(StaticMethod):
    requires_logits = True

    def ood_score(self, feats, logits):
        return -logits.max(1)


@register("energy_legacy")
class Energy(StaticMethod):
    requires_logits = True
    grid = {"T": [0.1, 0.5, 1, 1.5, 2.0]}
    defaults = {"T": 1}

    def ood_score(self, feats, logits, T):
        return -(T * logsumexp(logits / T))


@register("gen_legacy")
class GEN(StaticMethod):
    requires_logits = True
    grid = {"gamma": [0.01, 0.1, 0.5, 1, 2, 5, 10], "M": [1, 2, 3, 4, 5, 6, 7, 50, 100, 200]}
    defaults = {"gamma": 0.1, "M": 10}

    def prepare(self, bench):
        self.n_classes = bench.fc[0].shape[0]

    def valid(self, p):
        return p["M"] <= self.n_classes

    def ood_score(self, feats, logits, gamma, M):
        p = np.sort(softmax(logits), 1)[:, ::-1][:, :M]
        return ((p ** gamma) * ((1 - p) ** gamma)).sum(1)
