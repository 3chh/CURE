"""CADRef: độ lệch tương đối so với tâm lớp dự đoán, tách theo dấu trọng số lớp."""
from __future__ import annotations

import numpy as np

from ....ops import logsumexp
from ...base import register
from .base import StaticMethod, train_stats


@register("cadref_legacy")
class CADRef(StaticMethod):
    requires_logits = True

    def prepare(self, bench):
        st = train_stats(bench)
        self.W = st.fc[0]
        pred = st.zt.argmax(1)
        # Tâm theo lớp dự đoán (ko phải nhãn thật), như mã chính thức.
        self.mu_pred = np.stack([st.tr[pred == c].mean(0) if (pred == c).any() else st.tr.mean(0)
                                 for c in range(len(self.W))])
        self.global_energy = logsumexp(st.zt).mean()

    def ood_score(self, feats, logits):
        yh = logits.argmax(1)
        dist, sg = feats - self.mu_pred[yh], np.sign(self.W[yh])
        l1 = np.abs(feats).sum(1)
        ep = np.clip(dist * sg, 0, None).sum(1) / l1
        en = np.clip(-dist * sg, 0, None).sum(1) / l1
        return ep / logsumexp(logits) + en / self.global_energy
