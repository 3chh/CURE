"""Runner chỉ bỏ qua cặp (phương pháp, benchmark) khi có lý do rõ ràng, ko chạy sai im lặng."""
import pytest

from cure_ood.data import Benchmark
from cure_ood.methods import build

from .conftest import need_features


@pytest.mark.parametrize("bench,t_out_ok", [("cifar10", True), ("cifar10_supcon", True), ("cifar10_dinov2", False)])
def test_oodd_t_out_support(bench, t_out_ok):
    B = Benchmark.load(bench)
    need_features(B)
    assert (build("oodd", t_out=True).supports(B) is None) == t_out_ok


def test_logit_methods_skip_feature_only_benchmark():
    B = Benchmark.load("cifar10_dinov2")
    need_features(B)
    assert build("dcac_msp").supports(B) and build("cure", base="msp").supports(B)
    assert build("cure").supports(B) is None and build("knn").supports(B) is None
