"""CURE: hàng chờ hậu kiểm, chứng nhận e-BH vào CDM, gộp Cauchy có hiệu chuẩn, lùi về điểm tĩnh (docs/method.md)."""
from __future__ import annotations

import numpy as np
import torch

from ..stats import CERTIFIERS, betabinom_pvalues, conformal_combine, excess_counts, oracle_certifier, rank_p
from .base import Method, Output, knn_context, register
from .memory import run_memory


def base_scores(stream, base: str):
    """Điểm nền hướng OOD cho luồng và cho C. kNN dùng s0; MSP/Energy tính bằng torch float32 như bản gốc."""
    if base == "knn":
        ctx = knn_context(stream)
        return -ctx.b, -ctx.cb
    out = []
    for z in (stream.logits, stream.bench.cal_logits):
        t = torch.as_tensor(z, dtype=torch.float32)
        out.append(-(torch.softmax(t, 1).max(1).values if base == "msp" else torch.logsumexp(t, 1)).numpy())
    return tuple(out)


@register("cure")
class CURE(Method):

    def __init__(self, base="knn", window=256, batch=128, capacity=512, k_mem=5, k_shadow=10, screen_q=0.2,
                 certifier="ebh", alpha=0.2, rule="cauchy", w_mem=1.0, wake=None, memory="cdm", fallback=True):
        """Mặc định là cấu hình chính thức; certifier: ebh | elond | all | oracle_labels (chẩn đoán, đọc nhãn)."""
        self.base, self.window, self.batch, self.capacity = base, window, batch, capacity
        self.k_mem, self.k_shadow, self.screen_q = k_mem, k_shadow, screen_q
        self.certifier, self.alpha, self.rule, self.w_mem = certifier, alpha, rule, w_mem
        self.wake, self.memory, self.fallback = wake, memory, fallback
        self.requires_logits = base != "knn"

    def memory_channel(self, stream):
        """Kênh bộ nhớ chỉ phụ thuộc s0 của kNN, nên dùng chung giữa các nền (kNN/MSP/Energy)."""
        key = ("cure_mem", self.window, self.batch, self.capacity, self.k_mem, self.k_shadow, self.screen_q,
               self.certifier, self.alpha, str(self.wake), self.memory)
        if key not in stream.cache:
            ctx = knn_context(stream, self.screen_q)
            cnt, npass = excess_counts(stream.x, ctx.passes, ctx.shadow, self.k_shadow, self.window)
            p_e = betabinom_pvalues(cnt, npass, self.k_shadow, len(ctx.shadow))
            # oracle_labels là ngoại lệ duy nhất đọc stream.y: cận trên của hậu kiểm, chỉ để ablation.
            cert = (oracle_certifier(stream.y) if self.certifier == "oracle_labels"
                    else CERTIFIERS[self.certifier](p_e, alpha=self.alpha))
            stream.cache[key] = run_memory(
                stream.x, ctx.b, ctx.passes, ctx.shadow, batch=self.batch, capacity=self.capacity, k_mem=self.k_mem,
                ttl=self.window, certifier=cert,
                queue_in_memory=self.memory != "cdm", wake=tuple(self.wake) if self.wake else None,
                k_shadow=self.k_shadow, calib=(stream.bench.C, ctx.cb))
        return stream.cache[key]

    def score(self, stream):
        mem = self.memory_channel(stream)
        s, s_cal = base_scores(stream, self.base)
        p, stat, covered = conformal_combine(s, s_cal, mem.pvals, mem.mem_cal, len(stream), self.rule, self.w_mem)
        stats = mem.activity(self.batch)
        if not self.fallback:
            return Output(-np.log(p) + 1e-4 * stat, mem.trace, p, stats)
        p_static = rank_p(s, s_cal)
        # Phá hòa: 1e-4·T giữa các mẫu cùng p bão hòa; 1e-7·s giữ thứ tự của điểm nền.
        sc = np.where(covered, -np.log(p) + 1e-4 * stat, -np.log(p_static)) + 1e-7 * s
        return Output(sc, mem.trace, np.where(covered, p, p_static), stats)
