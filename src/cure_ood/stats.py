"""Công cụ thống kê của CURE: p-value conformal, gộp Cauchy/Fisher, beta-binomial, chứng nhận e-BH/e-LOND và các bản ablation."""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch
from scipy.stats import betabinom

from .ops import DEVICE

EPS = 1e-12
Certifier = Callable[[np.ndarray], np.ndarray]


def rank_p(stat, ref) -> np.ndarray:
    """p conformal (1 + #{ref ≥ stat}) / (n + 1); stat, ref hướng OOD (lớn = lạ)."""
    r = np.sort(np.asarray(ref))
    return (1 + len(r) - np.searchsorted(r, stat, side="left")) / (len(r) + 1)


def loo_p(ref) -> np.ndarray:
    """p bỏ-chính-nó cho các mẫu tham chiếu: #{j: s_j ≥ s_i} / n."""
    ref = np.asarray(ref)
    return (len(ref) - np.searchsorted(np.sort(ref), ref, side="left")) / len(ref)


def combine(ps: Sequence[np.ndarray], weights: Sequence[float], rule: str) -> np.ndarray:
    """Thống kê gộp các p-value (lớn = lạ)."""
    p = np.clip(np.stack(ps), EPS, 1 - EPS)
    w = np.asarray(weights, float)[:, None]
    if rule == "fisher":
        return (w * -np.log(p)).sum(0) / w.sum()
    if rule == "cauchy":
        return (w * np.tan((0.5 - p) * np.pi)).sum(0) / w.sum()
    raise ValueError(rule)


def conformal_combine(static_test, static_cal, mem_p, mem_cal, n, rule="cauchy", w_mem=1.0):
    """Gộp p tĩnh với p bộ nhớ, hiệu chuẩn lại trên C theo bộ nhớ từng batch; trả (p, thống kê, mặt nạ có bộ nhớ)."""
    ps_test, ps_cal = rank_p(static_test, static_cal), loo_p(static_cal)
    out, stat, covered = np.ones(n), np.zeros(n), np.zeros(n, bool)
    for s, e, c_mem in mem_cal:
        t_test = combine([ps_test[s:e], mem_p[s:e]], [1.0, w_mem], rule)
        t_cal = combine([ps_cal, loo_p(c_mem)], [1.0, w_mem], rule)
        out[s:e], stat[s:e], covered[s:e] = rank_p(t_test, t_cal), t_test, True
    return out, stat, covered


def id_threshold(cal_scores: np.ndarray, q: float) -> float:
    """Ngưỡng sàng lọc τ: phân vị q của điểm giống-ID trên C."""
    return float(np.quantile(np.asarray(cal_scores, dtype=np.float64), q))


@torch.inference_mode()
def excess_counts(x, passes, shadow, k, window, block=1024):
    """Hậu kiểm: đếm mẫu qua sàng lọc đến sau rơi vào quả cầu quanh mỗi mẫu; trả (N_e, n_pass), N_e = -1 khi chưa đủ cửa sổ."""
    n = len(x)
    xt = torch.nn.functional.normalize(torch.as_tensor(x, device=DEVICE), dim=1)
    sh = torch.nn.functional.normalize(torch.as_tensor(shadow, device=DEVICE), dim=1)
    radius = torch.cat([(xt[i:i + 4096] @ sh.T).topk(k, dim=1).values[:, -1] for i in range(0, n, 4096)])
    pt = torch.as_tensor(passes, device=DEVICE)
    counts, npass = np.full(n, -1, np.int64), np.zeros(n, np.int64)
    for s in range(0, n, block):
        e = torch.arange(s, min(n, s + block), device=DEVICE)
        f = torch.arange(s + 1, min(n, s + block + window), device=DEVICE)
        lag = f[None, :] - e[:, None]
        inwin = (lag > 0) & (lag <= window) & pt[f][None, :]
        hit = inwin & ((xt[e] @ xt[f].T) >= radius[e][:, None])
        c = torch.where(e + window < n, hit.sum(1), torch.tensor(-1, device=DEVICE))
        counts[s:s + len(e)] = c.cpu().numpy()
        npass[s:s + len(e)] = inwin.sum(1).cpu().numpy()
    return counts, npass


def betabinom_pvalues(counts, npass, k, m) -> np.ndarray:
    """p của hậu kiểm: dưới null "mẫu là đuôi ID", N_e ~ BetaBinomial(n_pass, k, m - k + 1)."""
    p = np.ones(len(counts))
    ok = counts >= 0
    p[ok] = betabinom.sf(counts[ok] - 1, npass[ok], k, m - k + 1)
    return p


def ebh_certifier(p: np.ndarray, alpha=0.2, kappa=0.5) -> Certifier:
    """e-BH trên các mẫu hết hạn cùng batch; FDR ≤ alpha dưới mọi phụ thuộc."""
    def certify(expired):
        keep = np.zeros(len(expired), bool)
        if len(expired) == 0:
            return keep
        ev = kappa * np.maximum(p[expired], EPS) ** (kappa - 1)
        order, m = np.argsort(-ev), len(expired)
        ok = ev[order] >= m / (alpha * np.arange(1, m + 1))
        if ok.any():
            keep[order[: int(np.flatnonzero(ok)[-1]) + 1]] = True
        return keep
    return certify


def elond_certifier(p: np.ndarray, alpha=0.2, kappa=0.5) -> Certifier:
    """e-LOND (online, xét từng mẫu); FDR ≤ alpha dưới mọi phụ thuộc. Có trạng thái: tạo mới cho mỗi luồng."""
    j = np.arange(1, 2 * len(p) + 10)
    gamma = 0.07720838 * np.log(np.maximum(j, 2)) / (j * np.exp(np.sqrt(np.log(j))))
    state = {"t": 0, "r": 0}

    def certify(expired):
        keep = np.zeros(len(expired), bool)
        for i, e in enumerate(expired):
            if kappa * max(p[e], EPS) ** (kappa - 1) >= 1.0 / (alpha * gamma[state["t"]] * (state["r"] + 1)):
                keep[i] = True
                state["r"] += 1
            state["t"] += 1
        return keep
    return certify


def all_certifier(p: np.ndarray, alpha=None) -> Certifier:
    """Bỏ hậu kiểm (ablation): mọi mẫu hết hạn đều vào CDM."""
    return lambda expired: np.ones(len(expired), bool)


def oracle_certifier(y: np.ndarray) -> Certifier:
    """Chẩn đoán, đọc nhãn: chứng nhận đúng các mẫu OOD thật, cận trên của hậu kiểm; ko bao giờ là mặc định."""
    return lambda expired: np.asarray(y)[expired] == 1


CERTIFIERS = {"ebh": ebh_certifier, "elond": elond_certifier, "all": all_certifier}
