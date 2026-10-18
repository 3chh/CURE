"""fDBD: khoảng cách trung bình tới các biên quyết định, chuẩn hóa theo khoảng cách tới tâm train."""
from __future__ import annotations

import numpy as np

from ...base import register
from .base import StaticMethod, train_stats


@register("fdbd_legacy")
class FDBD(StaticMethod):
    requires_logits = True
    grid = {"norm": [True, False]}
    defaults = {"norm": True}

    def __init__(self, aps=True, chunk=256, **hparams):
        super().__init__(aps, **hparams)
        self.chunk = chunk  # tensor (n, lớp, chiều) quá lớn với ImageNet-200 nếu tính một lần

    def prepare(self, bench):
        st = train_stats(bench)
        (self.W, self.b), self.mu_g = st.fc, st.mu_g

    def _dist(self, f, z):
        yh, C = z.argmax(1), len(self.W)
        dw = self.W[yh][:, None, :] - self.W[None, :, :]
        db = self.b[yh][:, None] - self.b[None, :]
        num = np.abs(np.einsum("ncd,nd->nc", dw, f) + db)
        den = np.linalg.norm(dw, axis=2)
        mask = np.arange(C)[None, :] != yh[:, None]
        return (num / np.maximum(den, 1e-12) * mask).sum(1) / (C - 1)

    def ood_score(self, feats, logits, norm):
        d = np.concatenate([self._dist(feats[s:s + self.chunk], logits[s:s + self.chunk])
                            for s in range(0, len(feats), self.chunk)])
        r = np.linalg.norm(feats - self.mu_g, axis=1)
        # Bản gốc chia rồi nhân lại khi norm=False; giữ nguyên để khớp từng bit.
        s = d / r
        return -(s if norm else s * r)
