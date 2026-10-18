"""NCI: độ thẳng hàng giữa đặc trưng (đã trừ tâm toàn cục) và trọng số lớp dự đoán, cộng chuẩn L1."""
from __future__ import annotations

import numpy as np

from ...base import register
from .base import StaticMethod, train_stats


@register("nci_legacy")
class NCI(StaticMethod):
    requires_logits = True
    grid = {"alpha": [1e-4, 1e-3, 1e-2, 1e-1]}
    defaults = {"alpha": 1e-2}

    def prepare(self, bench):
        st = train_stats(bench)
        self.W, self.mu_g = st.fc[0], st.mu_g

    def ood_score(self, feats, logits, alpha):
        w, c = self.W[logits.argmax(1)], feats - self.mu_g
        cos = (w * c).sum(1) / (np.linalg.norm(w, axis=1) * np.linalg.norm(c, axis=1))
        return -(cos * np.linalg.norm(w, axis=1) + alpha * np.abs(feats).sum(1))
