"""DynProto bản vòng 53 (giữ để tái hiện kết quả cũ; bản bám bài ở dynproto.py): nguyên mẫu OOD động gom bằng BIRCH từ bộ đệm theo lớp dự đoán, chấm kiểu NegLabel."""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import Birch

from ...ops import softmax
from ..base import Method, Output, register


def otsu(s, bins=64) -> float:
    """Ngưỡng cực đại phương sai giữa hai lớp trên điểm của một batch."""
    h, edges = np.histogram(s, bins=bins)
    c = (edges[:-1] + edges[1:]) / 2
    best, thr = -1.0, float(np.median(s))
    for i in range(1, bins):
        w0, w1 = h[:i].sum(), h[i:].sum()
        if w0 == 0 or w1 == 0:
            continue
        m0, m1 = (h[:i] * c[:i]).sum() / w0, (h[i:] * c[i:]).sum() / w1
        between = w0 * w1 * (m0 - m1) ** 2
        if between > best:
            best, thr = between, edges[i]
    return thr


@register("dynproto_r53")
class DynProtoR53(Method):
    """DynProto bản vòng 51-57."""
    requires_logits = True

    def __init__(self, batch=128, buffer=30, t_cold=5, beta=5.0, k_neg=5, temp=0.01, birch_threshold=0.5):
        self.batch, self.buffer, self.t_cold, self.beta = batch, buffer, t_cold, beta
        self.k_neg, self.temp, self.birch_threshold = k_neg, temp, birch_threshold

    def fit(self, bench):
        super().fit(bench)
        G, z = bench.G, bench.train_logits
        cls, n_cls = z.argmax(1), z.shape[1]
        P = np.stack([G[cls == c].mean(0) if (cls == c).any() else np.zeros(G.shape[1]) for c in range(n_cls)])
        self.P = (P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)).astype(np.float32)
        self.q = np.percentile(softmax(z).max(1), self.beta)
        return self

    def score(self, stream):
        x, z = stream.x, stream.logits
        n_cls, d = self.P.shape
        yhat, msp = z.argmax(1), softmax(z).max(1)
        buffers = [[] for _ in range(n_cls)]
        proto_of = [np.zeros((0, d), np.float32) for _ in range(n_cls)]
        protos = np.zeros((0, d), np.float32)
        out, trace = np.zeros(len(x)), []
        for t, s in enumerate(range(0, len(x), self.batch)):
            e = min(s + self.batch, len(x))
            v = x[s:e]
            # Tính trong không gian log: exp(cos/τ) với τ = 0,01 tràn float32.
            lpos = logsumexp((v @ self.P.T).astype(np.float64) / self.temp, axis=1)
            if len(protos):
                lneg = logsumexp((v @ protos.T).astype(np.float64) / self.temp, axis=1)
                score = 1.0 / (1.0 + self.k_neg * np.exp(np.clip(lneg - lpos, -700, 700)))  # lớn = giống ID
            else:
                score = np.ones(len(v))
            trace.append((s, [i for b in buffers for i in b]))
            out[s:e] = -score
            admit = msp[s:e] < self.q if t < self.t_cold else score < otsu(score)
            changed = set()
            for j in np.flatnonzero(admit):
                c = yhat[s + j]
                buffers[c].append(s + j)
                if len(buffers[c]) > self.buffer:
                    buffers[c].pop(0)
                changed.add(c)
            for c in changed:
                F = x[np.asarray(buffers[c])]
                if len(F) < 2:
                    proto_of[c] = F
                    continue
                lab = Birch(threshold=self.birch_threshold, n_clusters=None).fit_predict(F)
                Q = np.stack([F[lab == k].mean(0) for k in np.unique(lab)])
                proto_of[c] = Q / np.linalg.norm(Q, axis=1, keepdims=True)
            if any(len(p) for p in proto_of):
                protos = np.concatenate(proto_of)
        return Output(out, trace)
