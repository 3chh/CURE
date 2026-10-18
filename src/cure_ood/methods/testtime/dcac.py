"""DCAC: cache theo lớp dự đoán gồm mẫu entropy cao, trừ bớt logit của mẫu giống chúng."""
from __future__ import annotations

from ...ops import DEVICE
from ..base import Method, Output, register
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


@register("dcac_energy")
class DCACOfficialEnergy(DCACOfficial):
    score_type = "energy"






