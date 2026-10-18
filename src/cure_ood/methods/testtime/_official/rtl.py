# Vendored from the official RTL code (Fan et al., "Test-Time Linear Out-of-Distribution Detection", CVPR 2024),
#   file RTL_plus.py, class RTL_OOD (copy kept at baselines_official/RTL/RTL_plus.py in the project archive).
# Changes, all marked "ADAPTED":
#   1. print() calls and the module-level set_seed(1) / script part (WideResNet, data loaders) are removed.
#   2. Only the code paths used by the defaults are kept: reduce="none", norm="l2", gamma_residual.
import numpy as np
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.preprocessing import StandardScaler, normalize


class RTL_OOD(object):

    def __init__(self, reduce="none", d=5, norm='l2', ood_score=None):
        self.ood_score = ood_score
        self.initial_embed(reduce, d)
        self.initial_norm(norm)
        self.gamma_in = self.gamma_residual
        self.scaler = StandardScaler()

    def linear_reranking(self, X, y=None, index=None):
        self.linear_reg = LinearRegression()
        X_in = X[index]
        y_in = y[index]
        self.linear_reg.fit(X_in, y_in)
        return self.linear_reg.predict(X)

    def reranking(self, X, y=None, percent=0.6, alpha=1e-5):
        if y is None:
            y = self.ood_score(X).reshape(-1, 1)
        gammas = self.compute_gamma(X, y, alpha)
        index_in = np.argsort(gammas)[:int(len(X) * percent)]
        return self.linear_reranking(X, y, index_in)

    def compute_gamma(self, X, y=None, alpha=1e-5):
        if y is None:
            y = self.ood_score(X).reshape(-1, 1)
        normed_X = self.norm(X)
        X = self.embed(normed_X)
        H = np.dot(np.dot(X, np.linalg.pinv(np.dot(X.T, X))), X.T)
        X_hat = np.eye(H.shape[0]) - H
        y_hat = np.dot(X_hat, y)
        return self.gamma_in(X_hat, y_hat, alpha)

    def gamma_residual(self, X_hat, y_hat, alpha):
        self.elasticnet = ElasticNet(alpha=alpha, l1_ratio=1.0, fit_intercept=True,
                                     warm_start=True, selection='cyclic')
        X_hat = self.scaler.fit_transform(X_hat)
        self.elasticnet.fit(X_hat, y_hat)
        if self.elasticnet.coef_.ndim == 2:
            coefs = self.elasticnet.coef_.T
            coefs = np.sum(np.abs(coefs), axis=1)
            return coefs
        elif self.elasticnet.coef_.ndim == 1:
            coefs = np.abs(self.elasticnet.coef_)
            return coefs
        else:
            raise RuntimeError

    def initial_embed(self, reduce, d):
        reduce = reduce.lower()
        assert reduce in ['pca', 'none']  # ADAPTED (2)
        if reduce == 'pca':
            from sklearn.decomposition import PCA
            embed = PCA(n_components=d)
        if reduce == 'none':
            self.embed = lambda x: x
        else:
            self.embed = lambda x: embed.fit_transform(x)

    def initial_norm(self, norm):
        norm = norm.lower()
        assert norm in ['l2', 'none']
        if norm == 'l2':
            self.norm = lambda x: normalize(x)
        else:
            self.norm = lambda x: x
