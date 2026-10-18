"""H58a: bản vendor của DCAC cho đúng kết quả của run_test gốc (unimodal/ood_method/msp.py) trên cùng đầu vào."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cure_ood.methods.testtime._official import dcac as vendored

SRC = Path(os.environ.get("CURE_DCAC_REPO", Path(os.environ.get("CURE_LEGACY", ".")) / "baselines_official" / "DCAC"))
UNI = SRC / "unimodal"


def upstream():
    if not (UNI / "ood_method" / "msp.py").exists():
        pytest.skip(f"thiếu repo DCAC gốc ({UNI}); đặt CURE_DCAC_REPO")
    sys.path.insert(0, str(UNI))
    try:
        from ood_method import msp, utils
    finally:
        sys.path.remove(str(UNI))
    return msp, utils


class FakeModel(torch.nn.Module):
    """get_feature trả (logit, đặc trưng) đã tính sẵn; ảnh đầu vào chỉ là chỉ số hàng."""

    def __init__(self, logits, feats):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.logits, self.feats = torch.as_tensor(logits), torch.as_tensor(feats)

    def get_feature(self, idx):
        return self.logits[idx], self.feats[idx]


@pytest.mark.parametrize("score", ["msp", "energy"])
def test_vendored_matches_upstream(score):
    msp, utils = upstream()
    rng = np.random.default_rng(0)
    n, c, d = 1500, 10, 32
    logits = rng.normal(0, 3, (n, c)).astype(np.float32)
    feats = rng.normal(size=(n, d)).astype(np.float32)
    is_id = rng.random(n) < 0.7
    id_entropy = vendored.train_entropy(rng.normal(0, 4, (5000, c)).astype(np.float32), c)
    cfg = {**vendored.CONFIG, "topk": c}
    batches = [(torch.arange(s, min(s + 512, n)), ["ID" if is_id[i] else "OOD" for i in range(s, min(s + 512, n))])
               for s in range(0, n, 512)]
    up = msp.run_test(SimpleNamespace(num_classes=c, method=score), batches, FakeModel(logits, feats), cfg, id_entropy)
    s_msp, s_energy, _ = vendored.run_test(logits, feats, id_entropy, cfg, c, 512, "cpu")
    s = s_msp if score == "msp" else s_energy
    assert utils.get_measures(s[is_id], s[~is_id], 0.95) == up
