"""Host bộ nhớ động dùng chung cho CURE, OODD-core và hàng đợi theo tuổi."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.stats import binom

from ..ops import DEVICE
from ..stats import Certifier


@dataclass
class MemoryRun:
    scores: np.ndarray  # điểm thô kiểu OODD: -(s0 - aff), lớn = lạ
    pvals: np.ndarray | None = None  # p kênh bộ nhớ, hiệu chuẩn trên C (khi có calib)
    mem_cal: list = field(default_factory=list)  # (s, e, thống kê bộ nhớ của C) cho batch có bộ nhớ
    trace: list = field(default_factory=list)  # (s, chỉ số mẫu trong bộ nhớ) cho batch có bộ nhớ, để đo DCR


@torch.inference_mode()
def run_memory(x, base, passes, shadow, *, batch=128, capacity=512, k_mem=5, ttl=None, certifier: Certifier | None = None,
               queue_in_memory=True, wake=None, k_shadow=10, calib=None) -> MemoryRun:
    """Chạy hàng chờ và bộ nhớ theo batch; trả điểm thô, p kênh bộ nhớ (khi có calib) và vết bộ nhớ."""
    n = len(x)
    xt = torch.as_tensor(x, device=DEVICE)
    queue, cdm = np.empty(0, np.int64), []
    run = MemoryRun(np.empty(n, np.float32))
    if calib is not None:
        cal_x, cal_b = torch.as_tensor(calib[0], device=DEVICE), np.asarray(calib[1], np.float32)
        cal_sorted = np.sort(-cal_b)
        run.pvals = np.empty(n, np.float64)
    if wake is not None:
        sh_t = torch.as_tensor(shadow, device=DEVICE)
    for s in range(0, n, batch):
        e = min(n, s + batch)
        if ttl is not None:
            expired = queue[(s - queue) >= ttl]
            if len(expired):
                if certifier is not None:
                    cdm.extend(int(j) for j in expired[certifier(expired)])
                    cdm = cdm[-capacity:]  # FIFO: bỏ mẫu cũ nhất
                queue = queue[(s - queue) < ttl]
        cand = np.concatenate((queue, np.arange(s, e)))
        keep = min(capacity, len(cand))
        # Thứ tự trong hàng chờ quyết định mẫu nào còn trong CDM sau khi cắt: giữ đúng argpartition.
        queue = cand[np.argpartition(base[cand], keep - 1)[:keep]]
        active = np.asarray(cdm, np.int64)
        if wake is not None and len(active):
            active = _awake(xt, sh_t, active, passes, s, e, wake, k_shadow, len(shadow))
        mem = np.unique(np.concatenate((queue, active))) if queue_in_memory else np.unique(active)
        if len(mem) == 0:
            run.scores[s:e] = -base[s:e]
            if calib is not None:
                run.pvals[s:e] = (1 + len(cal_sorted) - np.searchsorted(cal_sorted, -base[s:e], side="left")) / (len(cal_sorted) + 1)
            continue
        k = min(k_mem, len(mem))
        sims = xt[s:e] @ xt[mem].T
        aff = sims.topk(k, dim=1).values[:, -1].cpu().numpy()
        run.scores[s:e] = -(base[s:e] - aff)
        run.trace.append((s, mem.copy()))
        if calib is None:
            continue
        # Thống kê của mẫu luồng bỏ tự-khớp để hoán đổi được với C (C ko bao giờ nằm trong bộ nhớ).
        own = torch.as_tensor(mem[None, :] == np.arange(s, e)[:, None], device=DEVICE)
        aff_ex = torch.where(own, torch.tensor(-4.0, device=DEVICE), sims).topk(k, dim=1).values[:, -1].cpu().numpy()
        aff_c = (cal_x @ xt[mem].T).topk(k, dim=1).values[:, -1].cpu().numpy()
        t_test, t_cal = -(base[s:e] - aff_ex), -(cal_b - aff_c)
        run.mem_cal.append((s, e, t_cal))
        ref = np.sort(t_cal)
        run.pvals[s:e] = (1 + len(ref) - np.searchsorted(ref, t_test, side="left")) / (len(ref) + 1)
    return run


def _awake(xt, sh_t, active, passes, s, e, wake, k_shadow, m):
    """Trạng thái ngủ: mẫu CDM chỉ chấm điểm khi gần đây có đủ mẫu qua sàng lọc quanh nó (kiểm định nhị thức)."""
    window, level = wake
    rec = np.arange(max(0, s - window), e)
    rec = rec[passes[rec]]
    if len(rec) == 0:
        return active[:0]
    rad = (xt[active] @ sh_t.T).topk(k_shadow, dim=1).values[:, -1]
    hits = ((xt[active] @ xt[rec].T) >= rad[:, None]) & torch.as_tensor(rec[None, :] != active[:, None], device=DEVICE)
    h = hits.sum(1).cpu().numpy()
    return active[binom.sf(h - 1, len(rec), k_shadow / (m + 1)) <= level]
