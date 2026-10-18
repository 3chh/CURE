"""ViM: energy trừ phần dư chiếu lên không gian con chính của đặc trưng train."""
from __future__ import annotations

import numpy as np

from ....ops import logsumexp
from ...base import register
from .base import StaticMethod, train_stats


@register("vim_legacy")
class ViM(StaticMethod):
    requires_logits = True
    grid = {"dim": [1, 16, 32, 64, 128, 256]}
    defaults = {"dim": 256}

    def default_params(self, bench):
        # Như OpenOOD: D = 256 với đặc trưng ≤ 512 chiều, 1000 với 2048 chiều.
        return {"dim": 256 if bench.train_feats.shape[1] <= 512 else 1000}

    def prepare(self, bench):
        st = train_stats(bench)
        W, b = st.fc
        self.u = -np.linalg.pinv(W) @ b  # gốc tọa độ: điểm có logit bằng 0
        ev, vec = np.linalg.eigh(np.cov((st.tr - self.u).T))
        self.vec, self.st, self.alpha = vec[:, np.argsort(ev)[::-1]], st, {}

    def _null(self, dim):
        return self.vec[:, dim:] if dim < self.vec.shape[1] else self.vec[:, -1:]

    def ood_score(self, feats, logits, dim):
        NS = self._null(dim)
        if dim not in self.alpha:  # α: cân thang virtual logit với max logit trung bình trên train
            res = np.linalg.norm((self.st.tr - self.u) @ NS, axis=1)
            self.alpha[dim] = self.st.zt.max(1).mean() / res.mean()
        return -(logsumexp(logits) - self.alpha[dim] * np.linalg.norm((feats - self.u) @ NS, axis=1))
