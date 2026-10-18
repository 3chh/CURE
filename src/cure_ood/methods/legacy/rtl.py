"""Bản dùng ở vòng 21-57, giữ để tái hiện bảng cũ."""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.preprocessing import StandardScaler

from ..base import Method, Output, knn_context, register


@register("rtl_plus_legacy")
class RTLPlusLegacy(Method):
    """RTL++ bản vòng 21-57: đầu vào kNN, PCA 128 chiều."""

    def __init__(self, batch=512, percent=0.6, alpha=1e-5, dim=128, pca_fit=20000):
        self.batch, self.percent, self.alpha, self.dim, self.pca_fit = batch, percent, alpha, dim, pca_fit

    def fit(self, bench):
        super().fit(bench)
        G = bench.G
        sub = G[np.random.default_rng(0).permutation(len(G))[: self.pca_fit]]
        self.pca = PCA(n_components=min(self.dim, G.shape[1]), random_state=0).fit(sub)
        return self

    def score(self, stream):
        x, y0 = stream.x, -knn_context(stream).b
        out = np.empty(len(x))
        for s in range(0, len(x), self.batch):
            X, y = self.pca.transform(x[s:s + self.batch]), y0[s:s + self.batch].reshape(-1, 1)
            if len(X) < 16:
                out[s:s + self.batch] = y.ravel()
                continue
            Xh = np.eye(len(X)) - X @ np.linalg.pinv(X.T @ X) @ X.T
            en = ElasticNet(alpha=self.alpha, l1_ratio=1.0, fit_intercept=True, warm_start=True, selection="cyclic",
                            max_iter=2000)
            with warnings.catch_warnings():  # alpha = 1e-5 hầu như ko hội tụ; chỉ cần thứ hạng của γ
                warnings.simplefilter("ignore", ConvergenceWarning)
                en.fit(StandardScaler().fit_transform(Xh), (Xh @ y).ravel())
            idx = np.argsort(np.abs(en.coef_))[: int(len(X) * self.percent)]  # mẫu có γ nhỏ = ít ngoại lai
            out[s:s + self.batch] = LinearRegression().fit(X[idx], y[idx]).predict(X).ravel()
        return Output(out)
