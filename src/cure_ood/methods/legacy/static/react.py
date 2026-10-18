"""ReAct: cắt kích hoạt penultimate ở phân vị p của train rồi tính energy."""
from __future__ import annotations

import numpy as np

from ....ops import logsumexp
from ...base import register
from .base import StaticMethod, train_stats


@register("react_legacy")
class ReAct(StaticMethod):
    requires_logits = True
    grid = {"percentile": [85, 90, 95, 99]}
    defaults = {"percentile": 90}

    def prepare(self, bench):
        self.st, self.clip = train_stats(bench), {}

    def ood_score(self, feats, logits, percentile):
        if percentile not in self.clip:
            self.clip[percentile] = np.percentile(self.st.tr, percentile)
        W, b = self.st.fc
        return -logsumexp(np.minimum(feats, self.clip[percentile]) @ W.T + b)
