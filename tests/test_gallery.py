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


@pytest.fixture
def syn(tmp_path, monkeypatch):
    """Benchmark tổng hợp nhỏ để kiểm chỉ mục được dùng nhất quán."""
    import yaml
    rng = np.random.default_rng(0)
    np.savez(tmp_path / "syn.npz", id_train_feats=rng.normal(size=(600, 16)).astype(np.float32),
             id_train_labels=np.repeat(np.arange(6), 100),
             id_val_feats=rng.normal(size=(120, 16)).astype(np.float32),
             id_test_feats=rng.normal(size=(200, 16)).astype(np.float32),
             id_test_labels=np.zeros(200), ood_feats=rng.normal(size=(150, 16)).astype(np.float32),
             ood_labels=np.zeros(150))
    cfg = tmp_path / "syn.yaml"
    cfg.write_text(yaml.safe_dump({"name": "syn", "features": "syn.npz", "k": 5, "near": [], "far": ["ood"]}), encoding="utf-8")
    monkeypatch.setenv("CURE_FEATURES", str(tmp_path))
    return str(cfg)


@pytest.mark.parametrize("index", [{"subsample": 0.1}, {"prototypes": 5}, {"precision": "bf16"}])
def test_calibration_scored_with_same_index(syn, index):
    """C và tập đối chứng S phải được chấm bằng đúng chỉ mục của biến thể (điều kiện để p conformal hợp lệ)."""
    from cure_ood.methods.base import knn_reference
    b = Benchmark.load({"name": "v", "base": syn, "index": index})
    cb, tau, shadow = knn_reference(b, 0.2)
    assert np.array_equal(cb, b.gallery_index.kth_sim(b.C, b.k_knn))
    assert not np.array_equal(cb, Benchmark.load(syn).knn(b.C))  # khác chỉ mục gốc, tức đã dùng chỉ mục mới
    assert len(shadow) == int((cb <= tau).sum()) > 0


def test_session_and_score_share_the_index(syn):
    """Phiên trực tuyến và score() cùng đi qua Benchmark.knn, nên đổi chỉ mục là đổi cả hai."""
    from cure_ood.methods import build
    b = Benchmark.load({"name": "v", "base": syn, "index": {"subsample": 0.1}})
    st = b.stream("ood", 0.5, 0)
    m = build("cure", certifier="elond", batch=16).fit(b)
    ref = m.score(st)
    st.cache.clear()
    sess = m.session()
    out = np.concatenate([sess.step(st.feats[i:i + 16]).scores for i in range(0, len(st.feats), 16)])
    assert np.allclose(ref.scores, out, atol=1e-5)


@pytest.fixture
def sharded(tmp_path, monkeypatch):
    """Cùng dữ liệu, một bản nạp cả mảng và một bản chia 3 mảnh fp16, để so kết quả."""
    import yaml
    rng = np.random.default_rng(1)
    feats = rng.normal(size=(900, 16)).astype(np.float16)
    labels = np.repeat(np.arange(9), 100).astype(np.int16)
    for i in range(3):
        np.save(tmp_path / f"shard_feats_{i}.npy", feats[i * 300:(i + 1) * 300])
        np.save(tmp_path / f"shard_labels_{i}.npy", labels[i * 300:(i + 1) * 300])
    np.savez(tmp_path / "syn.npz", id_train_feats=feats.astype(np.float32), id_train_labels=labels,
             id_val_feats=rng.normal(size=(120, 16)).astype(np.float32),
             id_test_feats=rng.normal(size=(200, 16)).astype(np.float32),
             id_test_labels=np.zeros(200), ood_feats=rng.normal(size=(150, 16)).astype(np.float32),
             ood_labels=np.zeros(150))
    base = {"name": "syn", "features": "syn.npz", "k": 5, "near": [], "far": ["ood"]}
    (tmp_path / "syn.yaml").write_text(yaml.safe_dump(base), encoding="utf-8")
    (tmp_path / "syn_sh.yaml").write_text(yaml.safe_dump({**base, "name": "syn_sh", "gallery_shards": "shard_feats_*.npy"}),
                                          encoding="utf-8")
    monkeypatch.setenv("CURE_FEATURES", str(tmp_path))
    return str(tmp_path / "syn.yaml"), str(tmp_path / "syn_sh.yaml")


@pytest.mark.parametrize("index", [None, {"subsample": 0.2}, {"prototypes": 3}, {"precision": "bf16"}])
def test_sharded_gallery_matches_whole(sharded, index):
    whole, shard = (Benchmark.load({"name": "v", "base": p, **({"index": index} if index else {})}) for p in sharded)
    assert shard.shards.size == 900 and shard.dim == 16
    q = whole.C
    assert np.allclose(whole.knn(q, 5), shard.knn(q, 5), atol=1e-6)


def test_sharded_refuses_to_materialise(sharded):
    b = Benchmark.load(sharded[1])
    with pytest.raises(RuntimeError, match="theo mảnh"):
        _ = b.train_feats
    assert np.array_equal(b.train_labels, np.repeat(np.arange(9), 100))


@pytest.mark.parametrize("n,item,want", [(50_000, 4, 2048), (3_520, 4, 2048), (1_281_167, 4, 209), (1_281_167, 2, 419)])
def test_budget_chunk(n, item, want):
    """Gallery nhỏ giữ nguyên khối 2048 (ko đổi kết quả cũ); gallery lớn bị chia nhỏ để ko tràn bộ nhớ."""
    idx = ExactIndex.__new__(ExactIndex)
    idx.size, idx.g = n, torch.empty(0, dtype=torch.float32 if item == 4 else torch.bfloat16)
    assert idx.budget_chunk(2048) == want


def test_big_gallery_chunking_matches_small_chunk():
    """Chia khối nhỏ cho cùng kết quả: tính đúng đắn ko phụ thuộc kích thước khối (trên CPU)."""
    idx = build_index(G, None)
    assert np.allclose(idx.kth_sim(Q, 5, chunk=7), kth_sim(Q, G, 5), atol=1e-6)
