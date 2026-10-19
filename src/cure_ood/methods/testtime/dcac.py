"""DCAC: cache theo lớp dự đoán gồm mẫu entropy cao, trừ bớt logit của mẫu giống chúng."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ...ops import DEVICE
from ..base import Method, Output, Session, register
from ._official import dcac as official


@register("dcac_msp")
class DCACOfficial(Method):
    """DCAC chạy run_test gốc; topk = min(20, số lớp) vì CIFAR-10 chỉ có 10 lớp."""
    requires_logits = True
    score_type = "msp"

    def __init__(self, batch=512, **cfg):
        self.batch, self.cfg = batch, {**official.CONFIG, **cfg}

    def fit(self, bench):
        super().fit(bench)
        self.n_cls = bench.train_logits.shape[1]
        self.cfg["topk"] = min(self.cfg["topk"], self.n_cls)
        self.id_entropy = official.train_entropy(bench.train_logits, self.n_cls)
        return self

    def score(self, stream):
        key = ("dcac_official", self.batch, tuple(sorted(self.cfg.items())))
        if key not in stream.cache:  # MSP và Energy dùng chung một lần chạy cache
            stream.cache[key] = official.run_test(stream.logits, stream.feats, self.id_entropy, self.cfg, self.n_cls,
                                                  self.batch, DEVICE)
        msp, energy, trace = stream.cache[key]
        return Output(-(msp if self.score_type == "msp" else energy), trace)

    def session(self, domain=None):
        return DCACSession(self)


class DCACSession(Session):
    """Thân vòng lặp run_test gốc cho từng batch; cache theo lớp là trạng thái giữa các batch."""

    def __init__(self, m: DCACOfficial):
        super().__init__()
        self.m, self.cache = m, {}
        self.lower = np.percentile(m.id_entropy, m.cfg["lower"])

    @torch.no_grad()
    def step(self, feats, logits=None):
        cfg, n = self.m.cfg, len(feats)
        logits = torch.as_tensor(logits, dtype=torch.float32, device=DEVICE)
        features = torch.as_tensor(feats, dtype=torch.float32, device=DEVICE)
        features = features / features.norm(dim=1, keepdim=True)
        loss = official.softmax_entropy(logits)
        prob_map = logits.softmax(1)
        pred = torch.argmax(logits, dim=1)
        prop_entropy = official.get_entropy(loss, self.m.n_cls)
        final_logits, trace = logits.clone(), []
        for i in range(n):
            if cfg["enabled"] and self.lower < prop_entropy[i].item():
                official.update_cache(self.cache, int(pred[i].item()), [features[i].unsqueeze(0), loss[i].item(),
                                      prob_map[i].unsqueeze(0), self.t + i], cfg["shot_capacity"], True)
        if cfg["enabled"] and self.cache:
            final_logits -= official.compute_cache_logits(features, self.cache, cfg["alpha"], cfg["topk"])
            trace = [(self.t, np.array([item[3] for c in self.cache.values() for item in c], dtype=int))]
        self.t += n
        if self.m.score_type == "msp":
            return Output(-np.max(F.softmax(final_logits, dim=-1).cpu().numpy(), axis=1), trace)
        return Output(-torch.logsumexp(final_logits, dim=1).cpu().numpy(), trace)


@register("dcac_energy")
class DCACOfficialEnergy(DCACOfficial):
    score_type = "energy"






