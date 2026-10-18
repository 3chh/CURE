"""DynProto cài lại theo bài, đối chiếu ở docs/reimplementations/dynproto.md."""
from __future__ import annotations

from collections import deque

import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import Birch

from ...ops import softmax
from ..base import Method, Output, register

# Lưới tìm α như code OWTTT (np.arange(0, 1, 0.01)); bài chỉ nói "search α over (0, 1)".
ALPHA_GRID = np.arange(0, 1, 0.01)


def log_msp_odds(z: np.ndarray) -> np.ndarray:
    """logit(MSP) = log p_max - log(1 - p_max), tính ổn định từ logit."""
    z = np.asarray(z, np.float64)
    top = z.argmax(1)
    rest = z.copy()
    rest[np.arange(len(z)), top] = -np.inf
    return z[np.arange(len(z)), top] - logsumexp(rest, axis=1)


def neglabel_log_odds(v: np.ndarray, p_id: np.ndarray, p_ood: np.ndarray, temp: float, k: float) -> np.ndarray:
    """logit(S_DynProto) của Eq. 4 = lse(cos(v,p_id)/τ) - log K - lse(cos(v,p_ood)/τ)."""
    lpos = logsumexp((v @ p_id.T).astype(np.float64) / temp, axis=1)
    lneg = logsumexp((v @ p_ood.T).astype(np.float64) / temp, axis=1)
    return lpos - np.log(k) - lneg


def sigmoid(a: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * a))


def within_variance(s: np.ndarray, th: float, weighted: bool) -> float:
    """Tiêu chí Eq. 5 tại ngưỡng th; nhóm rỗng thì loại ngưỡng (như OWTTT)."""
    hi, lo = s[s >= th], s[s < th]
    if len(hi) == 0 or len(lo) == 0:
        return np.inf
    if weighted:  # compute_os_variance của OWTTT: phương sai trong nhóm có trọng số (Otsu)
        return (len(lo) * lo.var() + len(hi) * hi.var()) / len(s)
    return lo.var() + hi.var()  # Eq. 5 đúng như in: tổng hai phương sai, ko trọng số


def adaptive_alpha(s: np.ndarray, rule: str = "owttt") -> float:
    """α của Eq. 5: argmin trên lưới; mọi ngưỡng bị loại thì trả 0 (ko nạp mẫu nào), như argmin của OWTTT."""
    crit = [within_variance(s, th, rule == "owttt") for th in ALPHA_GRID]
    return float(ALPHA_GRID[int(np.argmin(crit))])


def birch_prototypes(F: np.ndarray, threshold: float, branching_factor: int) -> np.ndarray:
    """Gom một cache bằng BIRCH (ko gom toàn cục), mỗi cụm lấy trung bình thành một nguyên mẫu (§III-D)."""
    if len(F) == 1:
        return F.copy()
    lab = Birch(threshold=threshold, branching_factor=branching_factor, n_clusters=None).fit_predict(F)
    return np.stack([F[lab == c].mean(0) for c in np.unique(lab)])


@register("dynproto")
class DynProto(Method):
    """DynProto: nguyên mẫu OOD tiến hóa từ cache theo lớp (Alg. 1)."""
    requires_logits = True

    def __init__(self, batch=128, cache_size=30, t_cold=5, beta=5.0, k=5.0, temp=0.01,
                 birch_threshold=0.5, birch_branching=50, alpha_rule="owttt"):
        if alpha_rule not in ("owttt", "eq5"):
            raise ValueError("alpha_rule phải là 'owttt' hoặc 'eq5'")
        self.batch, self.cache_size, self.t_cold, self.beta = batch, cache_size, t_cold, beta
        self.k, self.temp, self.alpha_rule = k, temp, alpha_rule
        self.birch_threshold, self.birch_branching = birch_threshold, birch_branching

    def fit(self, bench):
        super().fit(bench)
        G, y = bench.G, np.asarray(bench.train_labels)
        n_cls = bench.train_logits.shape[1]
        # Nguyên mẫu ID: trung bình đặc trưng train theo nhãn thật (dòng 12); lớp vắng thì vector 0 (cos = 0).
        P = np.stack([G[y == c].mean(0) if (y == c).any() else np.zeros(G.shape[1]) for c in range(n_cls)])
        self.p_id = (P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)).astype(np.float32)
        # θ = phân vị β của MSP trên train ID (§III-E).
        self.theta = float(np.percentile(softmax(bench.train_logits).max(1), self.beta))
        return self

    def score(self, stream):
        x, z = stream.x.astype(np.float32), stream.logits
        n_cls, d = self.p_id.shape
        yhat, msp, base = np.asarray(z).argmax(1), softmax(z).max(1), log_msp_odds(z)
        caches = [deque(maxlen=self.cache_size) for _ in range(n_cls)]
        proto_of = [np.zeros((0, d), np.float32) for _ in range(n_cls)]
        p_ood = np.zeros((0, d), np.float32)
        out, trace = np.zeros(len(x)), []
        self.alphas = []
        for t, s in enumerate(range(0, len(x), self.batch)):
            e = min(s + self.batch, len(x))
            v = x[s:e]
            if len(p_ood) and t >= self.t_cold:
                # Dòng 3-4: chấm bằng nguyên mẫu hiện có rồi tìm α trên điểm S của batch.
                s_dp = sigmoid(neglabel_log_odds(v, self.p_id, p_ood, self.temp, self.k))
                alpha = adaptive_alpha(s_dp, self.alpha_rule)
                self.alphas.append(alpha)
                admit = s_dp < alpha
            elif t < self.t_cold:
                admit = msp[s:e] < self.theta
            else:
                admit = np.zeros(e - s, bool)  # dòng 8: hết cold start mà cache rỗng thì ko còn đường nạp
            changed = set()
            for j in np.flatnonzero(admit):
                caches[yhat[s + j]].append(s + j)
                changed.add(int(yhat[s + j]))
            for c in changed:
                P = birch_prototypes(x[np.asarray(caches[c])], self.birch_threshold, self.birch_branching)
                proto_of[c] = (P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)).astype(np.float32)
            if changed:
                p_ood = np.concatenate(proto_of)
            # Dòng 11-17: cache ≠ ∅ thì chấm bằng nguyên mẫu vừa cập nhật, ngược lại dùng base detector.
            if len(p_ood):
                out[s:e] = -neglabel_log_odds(v, self.p_id, p_ood, self.temp, self.k)
                trace.append((s, [i for q in caches for i in q]))
            else:
                out[s:e] = -base[s:e]
                trace.append((s, []))
        return Output(out, trace)
