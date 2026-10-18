"""H58a cho phương pháp tĩnh: file OpenOOD vendor ko bị sửa, và FeatureNet tái tạo đúng logit của checkpoint."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cure_ood.data import Benchmark
from cure_ood.methods.static.openood import FeatureNet

from .conftest import need_features

VENDOR = Path(__file__).resolve().parents[1] / "src" / "cure_ood" / "methods" / "static" / "_openood"


def test_vendored_files_unchanged():
    man = json.loads((VENDOR / "MANIFEST.json").read_text(encoding="utf-8"))
    for rel, digest in man["sha256"].items():
        assert hashlib.sha256((VENDOR / rel).read_bytes()).hexdigest() == digest, rel


@pytest.mark.parametrize("bench", ["cifar10", "cifar100", "imagenet200"])
def test_feature_net_reproduces_logits(bench):
    B = Benchmark.load(bench)
    need_features(B)
    z = FeatureNet(*B.fc)(torch.as_tensor(B.cal_feats)).detach().numpy()
    assert np.allclose(z, B.cal_logits, atol=1e-4)
