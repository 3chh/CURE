"""Chỉ mục gallery: mặc định trùng kth_sim, các biến thể đúng cỡ và gần bản chính xác."""
import numpy as np
import pytest
import torch

from cure_ood.data.benchmark import Benchmark
from cure_ood.data.gallery import ExactIndex, build_index
from cure_ood.ops import DEVICE, kth_sim, l2

RNG = np.random.default_rng(0)
G = l2(RNG.normal(size=(2000, 32)).astype(np.float32))
Q = l2(RNG.normal(size=(300, 32)).astype(np.float32))
LABELS = np.repeat(np.arange(10), 200)


def test_default_is_bit_exact():
    assert np.array_equal(build_index(G, None).kth_sim(Q, 5), kth_sim(Q, G, 5))
    assert np.array_equal(ExactIndex(G, g=torch.as_tensor(G, device=DEVICE)).kth_sim(Q, 5), kth_sim(Q, G, 5))


@pytest.mark.parametrize("frac,n", [(0.1, 200), (0.01, 20), (0.0001, 1)])
def test_subsample_size(frac, n):
    assert build_index(G, {"subsample": frac}).size == n


def test_prototypes_per_class():
    idx = build_index(G, {"prototypes": 5}, LABELS)
    assert idx.size == 50
    assert np.allclose(np.linalg.norm(idx.g.cpu().numpy(), axis=1), 1, atol=1e-5)


def test_faiss_flat_matches_exact_and_approx_is_close():
    pytest.importorskip("faiss")
    exact = kth_sim(Q, G, 5)
    assert np.allclose(build_index(G, {"faiss": "Flat"}).kth_sim(Q, 5), exact, atol=1e-5)
    hnsw = build_index(G, {"faiss": "HNSW32", "ef_search": 256}).kth_sim(Q, 5)
    assert np.mean(np.abs(hnsw - exact) < 1e-5) > 0.9  # efSearch lớn gần như tìm đúng
    ivf1 = build_index(G, {"faiss": "IVF16,Flat", "nprobe": 1}).kth_sim(Q, 5)
    assert np.all(ivf1 <= exact + 1e-5)  # tìm xấp xỉ chỉ có thể bỏ sót láng giềng gần, ko tìm ra gần hơn


def test_bf16_close():
    assert np.abs(build_index(G, {"precision": "bf16"}).kth_sim(Q, 5) - kth_sim(Q, G, 5)).max() < 2e-2


def test_unknown_spec_rejected():
    with pytest.raises(ValueError):
        build_index(G, {"subsample": 0.5, "typo": 1})


def test_variant_inherits_base():
    b = Benchmark.load({"name": "c10_sub", "base": "cifar10", "index": {"subsample": 0.1}, "k": "aps"})
    base = Benchmark.load("cifar10")
    assert (b.name, b.index, b.k) == ("c10_sub", {"subsample": 0.1}, "aps")
    assert (b.features, b.near, b.far, b.stream_salt) == (base.features, base.near, base.far, base.stream_salt)
