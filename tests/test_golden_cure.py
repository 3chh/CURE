"""CURE, kNN, OODD, hàng đợi theo tuổi phải ra đúng từng bit kết quả vòng 57 (AUROC và DCR)."""
import pytest

from cure_ood.data import Benchmark
from cure_ood.eval.metrics import dcr_counts, detection_metrics
from cure_ood.methods import build

from .conftest import golden_row, need_features

CASES = [("imagenet200", "textures", 0.5, 0), ("cifar10", "svhn", 0.5, 0), ("cifar10_supcon", "tin", 0.5, 1),
         ("cifar100", "texture", 0.2, 2), ("cifar10_dinov2", "places365", 0.5, 0), ("cifar10", "cifar100", 0.05, 1)]
ARMS = {"cure": ("cure", {}), "cure_msp": ("cure", {"base": "msp"}), "cure_energy": ("cure", {"base": "energy"}),
        "knn": ("knn", {}), "oodd": ("oodd_core", {}), "ttl": ("age_queue", {})}


@pytest.mark.parametrize("bench,domain,pi,seed", CASES)
def test_round57_exact(bench, domain, pi, seed):
    B = Benchmark.load(bench)
    need_features(B)
    ref = golden_row("round57_parts", bench, domain, pi, seed)
    stream = B.stream(domain, pi, seed)
    for label, (name, params) in ARMS.items():
        m = build(name, **params)
        if m.requires_logits and not B.has_logits:
            continue
        out = m.fit(B).score(stream)
        assert detection_metrics(stream.y, out.scores)["auroc"] == ref["metrics"][label]["auroc"], label
        if label in ref["dcr"]:
            assert dcr_counts(out.memory, stream.y) == ref["dcr"][label], label
