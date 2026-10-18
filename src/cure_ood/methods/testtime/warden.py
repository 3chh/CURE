"""WARDEN cài lại theo bài, đối chiếu ở docs/reimplementations/warden.md."""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.stats import binom
from sklearn.covariance import LedoitWolf

from ...ops import DEVICE, l2
from ...stats import EPS, rank_p
from ..base import Method, Output, register

# Điểm vận hành của bài (Sec. 4.3 sau Cor. 10, Sec. 5, Sec. 7): a = 0.1, δ = α = 0.1, K = 64, η = 0.05, λ = 1/2.
PAPER = {"a": 0.1, "delta": 0.1, "alpha": 0.1, "batch": 64, "eta": 0.05, "lam": 0.5}


def ebh_admit(p: np.ndarray, delta: float, a: float) -> np.ndarray:
    """e-BH mức δ trên một batch (Lemma 8): e_i = a·p_i^(a-1), nhận k* e-value lớn nhất với e_(k) ≥ K/(δk)."""
    K = len(p)
    keep = np.zeros(K, bool)
    if K == 0:
        return keep
    e = a * np.power(np.maximum(p, EPS), a - 1)
    order = np.argsort(-e, kind="stable")
    ok = e[order] >= K / (delta * np.arange(1, K + 1))
    if ok.any():
        keep[order[: int(np.flatnonzero(ok)[-1]) + 1]] = True
    return keep


def admission_p_threshold(a: float, delta: float) -> float:
    """κ(δ) = (aδ)^(1/(1-a)): mọi mẫu được nạp đều có p ≤ κ(δ) (Lemma 8)."""
    return (a * delta) ** (1.0 / (1.0 - a))


def kappa_bar(m: int, kappa: float, eta: float) -> float:
    """κ̄ = max{u: Pr[Bin(m,u) < ⌈κ(m+1)⌉] ≥ η}, cận trên nhị thức của đuôi κ của reserve (Lemma 8)."""
    c = math.ceil(kappa * (m + 1) - 1e-9)
    lo, hi = 0.0, 1.0
    for _ in range(100):  # Pr giảm theo u nên chia đôi được
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if binom.cdf(c - 1, m, mid) >= eta else (lo, mid)
    return lo


def storey_upper(p: np.ndarray, lam: float = 0.5, eta: float = 0.05) -> float:
    """π̂_up = [π̂_λ + (1/(1-λ))·sqrt(log(2/η)/(2n))] cắt về [0, 1] (CDC bước 1)."""
    n = len(p)
    pi_lam = 1.0 - np.count_nonzero(p > lam) / (n * (1.0 - lam))
    return float(np.clip(pi_lam + math.sqrt(math.log(2 / eta) / (2 * n)) / (1 - lam), 0.0, 1.0))


def cdc_threshold(scores: np.ndarray, p: np.ndarray, alpha: float, lam: float = 0.5, eta: float = 0.05) -> float:
    """τ̂ = F̂_mix⁻¹(1 - α(1 - π̂_up) + ε_n), ε_n = sqrt(log(2/η)/(2n)) + 1/n; mức ≥ 1 thì τ̂ = +∞ (Thm. 11)."""
    n = len(scores)
    level = 1 - alpha * (1 - storey_upper(p, lam, eta)) + math.sqrt(math.log(2 / eta) / (2 * n)) + 1 / n
    if level >= 1:
        return math.inf
    return float(np.sort(scores)[math.ceil(level * n) - 1])  # phân vị trái của CDF thực nghiệm


def ledoit_wolf_whitener(feats: np.ndarray):
    """Trả (μ, W) để z = (f - μ)·W có hiệp phương sai Ledoit-Wolf bằng I; chỉ fit trên train-ID (Sec. 7)."""
    f = np.asarray(feats, np.float64)
    lw = LedoitWolf().fit(f)
    val, vec = np.linalg.eigh(lw.covariance_)
    return lw.location_, (vec / np.sqrt(np.maximum(val, EPS))) @ vec.T


def kth_dist(q: np.ndarray, g: np.ndarray, k: int, chunk: int = 2048) -> np.ndarray:
    """Khoảng cách Euclid tới láng giềng thứ k trong g (kNN-to-bank, Sun et al.)."""
    gt = torch.as_tensor(np.ascontiguousarray(g, np.float32), device=DEVICE)
    out = [torch.cdist(torch.as_tensor(np.ascontiguousarray(q[s:s + chunk], np.float32), device=DEVICE), gt)
           .topk(k, dim=1, largest=False).values[:, -1].cpu().numpy() for s in range(0, len(q), chunk)]
    return np.concatenate(out) if out else np.zeros(0, np.float32)


@register("warden")
class WARDEN(Method):
    """WARDEN: xếp hạng bằng kNN đã làm trắng, bộ nhớ chỉ dùng cho quyết định (Sec. 4-5)."""

    def __init__(self, a=0.1, delta=0.1, alpha=0.1, batch=128, cap=512, k=None, whiten=True,
                 cdc_window=1024, eta=0.05, lam=0.5):
        # batch = 128 theo giao thức của ta (bài: K = 64); cap, k, cdc_window bài ko nêu số.
        self.a, self.delta, self.alpha, self.batch, self.cap = a, delta, alpha, batch, cap
        self.k, self.whiten, self.cdc_window, self.eta, self.lam = k, whiten, cdc_window, eta, lam

    def _embed(self, feats):
        """L2 như Sun et al. rồi làm trắng; ko chuẩn hóa lại vì sau khi trừ tâm, L2 xóa mất độ lớn."""
        f = l2(feats)
        if not self.whiten:
            return f
        return ((f.astype(np.float64) - self._mu) @ self._W).astype(np.float32)

    def fit(self, bench):
        self.bench = bench
        self._k = self.k or bench.k
        if self.whiten:
            self._mu, self._W = ledoit_wolf_whitener(l2(bench.train_feats))
        self._G = self._embed(bench.train_feats)
        self._R = self._embed(bench.cal_feats)  # reserve đóng băng: ID giữ lại, ko nằm trong gallery
        self._res_score = kth_dist(self._R, self._G, self._k)
        return self

    def base_score(self, stream) -> tuple[np.ndarray, np.ndarray]:
        """(khoảng cách tới láng giềng thứ k trong bank, đặc trưng đã nhúng); lớn = lạ."""
        z = self._embed(stream.feats)
        return kth_dist(z, self._G, self._k), z

    @torch.inference_mode()
    def score(self, stream):
        s, z = self.base_score(stream)
        n, half = len(s), self.alpha / 2
        p_base = rank_p(s, self._res_score)
        p_dict = np.ones(n)
        tau_cdc = np.full(n, np.nan)
        flag_cdc, flag_cdc_half = np.zeros(n, bool), np.zeros(n, bool)
        zt, rt = torch.as_tensor(z, device=DEVICE), torch.as_tensor(self._R, device=DEVICE)
        dictionary = np.zeros(0, np.int64)  # chỉ số mẫu theo thứ tự nạp
        trace, admitted = [], []
        for st in range(0, n, self.batch):
            idx = np.arange(st, min(n, st + self.batch))
            # Kênh từ điển: D_{t-1} đo được trước batch nên p vẫn siêu đều so với reserve.
            if len(dictionary):
                dt = zt[dictionary]
                prox = -torch.cdist(zt[idx], dt).min(1).values.cpu().numpy()
                prox_r = -torch.cdist(rt, dt).min(1).values.cpu().numpy()
                p_dict[idx] = rank_p(prox, prox_r)
                trace.append((st, dictionary.copy()))
            # CDC: ngưỡng cho batch t chỉ dùng các batch < t (quyết định rồi mới cập nhật).
            lo = max(0, st - self.cdc_window)
            if st > lo:
                tau = cdc_threshold(s[lo:st], p_base[lo:st], self.alpha, self.lam, self.eta)
                tau_h = cdc_threshold(s[lo:st], p_base[lo:st], half, self.lam, self.eta)
                tau_cdc[idx] = tau
                flag_cdc[idx], flag_cdc_half[idx] = s[idx] > tau, s[idx] > tau_h
            else:  # chưa có cửa sổ: dùng ngưỡng tĩnh của reserve
                flag_cdc[idx], flag_cdc_half[idx] = p_base[idx] <= self.alpha, p_base[idx] <= half
            # Nạp: bằng chứng chỉ đọc reserve và scorer đóng băng, ko đọc từ điển.
            new = idx[ebh_admit(p_base[idx], self.delta, self.a)]
            if len(new):
                admitted.append(new)
                dictionary = np.r_[dictionary, new][-self.cap:]
        dict_ch = p_dict <= half
        self.last_ = {
            "p_base": p_base, "p_dict": p_dict, "tau_cdc": tau_cdc,
            "flags": (p_base <= half) | dict_ch,  # WARDEN như bài báo cáo
            "flags_cdc": flag_cdc,  # CDC đứng riêng ở mức α
            "flags_cdc_dict": flag_cdc_half | dict_ch,  # ghép của ta: kênh gốc dùng ngưỡng CDC ở α/2
            "admitted": np.concatenate(admitted) if admitted else np.zeros(0, np.int64),
        }
        return Output(s, trace)






