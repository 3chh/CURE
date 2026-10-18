"""AdaODD: từng thành phần so với giá trị tính tay theo bài, cộng một luồng thật làm smoke test."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cure_ood.methods import build
from cure_ood.methods.testtime.adaodd import AdaODD, knn_score, preset_for

S2 = np.sqrt(2.0)


def t(a):
    return torch.as_tensor(np.asarray(a, np.float32))


def fake_bench(G, C, name="cifar10"):
    G, C = np.asarray(G, np.float32), np.asarray(C, np.float32)
    return SimpleNamespace(name=name, G=G, C=C, train_feats=G, cal_feats=C)


def fake_stream(X):
    X = np.asarray(X, np.float32)
    return SimpleNamespace(x=X, feats=X)


def test_footnote2_example():
    # Chú thích 2 của bài: x = [2,2], M = {([1,1],1), ([0,0],1)}, k = 1 → S = -√2 ≈ -1.4.
    mem, w = t([[1, 1], [0, 0]]), t([1, 1])
    assert knn_score(t([[2, 2]]), mem, w, 1).item() == pytest.approx(-S2, abs=1e-6)
    # Thêm láng giềng OOD ([2.5,2.5], κ): S = -0.707 κ.
    mem2, w2 = torch.cat([mem, t([[2.5, 2.5]])]), torch.cat([w, t([3.0])])
    assert knn_score(t([[2, 2]]), mem2, w2, 1).item() == pytest.approx(-S2 / 2 * 3.0, abs=1e-6)


def test_score_aggregations():
    # Khoảng cách tới 4 láng giềng: 1, 2, 3, 4 (trên trục x); trọng số 1, 2, 1, 3.
    mem, w = t([[1, 0], [2, 0], [3, 0], [4, 0]]), t([1, 2, 1, 3])
    q = t([[0, 0]])
    assert knn_score(q, mem, w, 4, "avg").item() == pytest.approx(-(1 + 4 + 3 + 12) / 4)
    assert knn_score(q, mem, w, 4, "kth").item() == pytest.approx(-12)
    assert knn_score(q, mem, w, 4, "median").item() == pytest.approx(-4)  # z_(⌊4/2⌋) = z_(2)
    # Láng giềng chọn theo khoảng cách thuần, ko theo khoảng cách có trọng số.
    assert knn_score(q, mem, t([5, 1, 1, 1]), 1).item() == pytest.approx(-5)


def test_threshold_is_id_quantile():
    G = [[0, 0]]
    C = [[float(i), 0] for i in range(1, 21)]  # S = -1..-20
    m = AdaODD(k=1, gamma=1.5, kappa=5.0, normalize=False).fit(fake_bench(G, C))
    assert m.lam == pytest.approx(np.quantile(-np.arange(1, 21), 0.05))
    assert m.lam == pytest.approx(-19.05)


def test_memory_update_rule_hand_computed():
    # λ = -1 (C chỉ có 1 mẫu ở khoảng cách 1), γ = 2: OOD nếu S < -2, ID nếu S > -0.5.
    G, C = [[0, 0]], [[1, 0]]
    m = AdaODD(k=1, gamma=2.0, kappa=4.0, tpr=0.95, batch=1, normalize=False).fit(fake_bench(G, C))
    assert m.lam == pytest.approx(-1.0)
    X = [[3, 0],     # S = -3 → nạp (z, κ)
         [3.5, 0],   # láng giềng gần nhất [3,0] (0.5 · 4 = 2) → S = -2: ko nạp (bất đẳng thức chặt)
         [0.2, 0],   # S = -0.2 → nạp (z, 1)
         [1, 0]]     # S = -0.8: vùng giữa, ko nạp
    s = -m.score(fake_stream(X)).scores
    assert s == pytest.approx([-3, -2, -0.2, -0.8], abs=1e-5)
    mem, w = m.memory_
    assert np.allclose(mem.cpu().numpy(), [[0, 0], [3, 0], [0.2, 0]])
    assert w.cpu().numpy().tolist() == [1, 4, 1]


def test_batch_defers_update():
    # batch = 2: mẫu thứ 2 chưa thấy mẫu thứ 1 trong bộ nhớ; batch = 1 thì thấy (Alg. 1).
    G, C = [[0, 0]], [[1, 0]]
    X = [[3, 0], [3.1, 0]]
    s1 = -AdaODD(k=1, gamma=2.0, kappa=4.0, batch=1, normalize=False).fit(fake_bench(G, C)).score(fake_stream(X)).scores
    s2 = -AdaODD(k=1, gamma=2.0, kappa=4.0, batch=2, normalize=False).fit(fake_bench(G, C)).score(fake_stream(X)).scores
    assert s1[1] == pytest.approx(-0.4, abs=1e-5)  # 0.1 · κ
    assert s2[1] == pytest.approx(-3.1, abs=1e-5)


def test_batch1_matches_numpy_reference():
    rng = np.random.default_rng(0)
    G, C, X = rng.normal(size=(200, 8)), rng.normal(size=(50, 8)), rng.normal(size=(60, 8)) * 1.3
    l2 = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True)
    G, C, X = l2(G), l2(C), l2(X)
    k, gamma, kappa = 5, 1.2, 3.0
    m = AdaODD(k=k, gamma=gamma, kappa=kappa, batch=1).fit(fake_bench(G, C))
    got = -m.score(fake_stream(X)).scores

    def S(x, M, W):
        d = np.linalg.norm(M - x, axis=1)
        i = np.argsort(d)[:k]
        return -(d[i] * W[i]).mean()
    lam = np.quantile([S(c, G, np.ones(len(G))) for c in C], 0.05)
    M, W, ref = G.copy(), np.ones(len(G)), []
    for x in X:
        s = S(x, M, W)
        ref.append(s)
        if s < lam * gamma:
            M, W = np.vstack([M, x]), np.r_[W, kappa]
        if s > lam / gamma:
            M, W = np.vstack([M, x]), np.r_[W, 1.0]
    assert m.lam == pytest.approx(lam, abs=1e-5)
    assert got == pytest.approx(np.array(ref), abs=1e-4)


def test_presets_and_k_rule():
    assert preset_for("cifar10_supcon") == {"k": 50, "gamma": 1.5, "kappa": 5.0}
    assert preset_for("cifar100")["k"] == 200
    G = np.eye(4, dtype=np.float32).repeat(1000, 0)[:3520]
    m = AdaODD().fit(fake_bench(G, G[:10], name="imagenet200"))
    assert (m.k_, m.gamma_, m.kappa_) == (4, 1.0, 10.0)  # k = 1‰ · 3520
    with pytest.raises(ValueError):
        AdaODD(gamma=0.5).fit(fake_bench(G, G[:10]))


def test_reevaluate():
    G, C = [[0, 0]], [[1, 0]]
    m = AdaODD(k=1, gamma=2.0, kappa=4.0, batch=1, normalize=False).fit(fake_bench(G, C))
    m.score(fake_stream([[3, 0]]))
    # Mẫu ID [0.5,0] → S = -0.5 ≥ λ = -1; mẫu [2.5,0] gần ([3,0],κ) → S = -2 < λ.
    assert m.reevaluate(np.array([[0.5, 0], [2.5, 0]], np.float32)) == pytest.approx(0.5)


def test_smoke_imagenet200_textures():
    from cure_ood.data import Benchmark
    from cure_ood.eval.metrics import detection_metrics

    from .conftest import need_features
    B = Benchmark.load("imagenet200")
    need_features(B)
    stream = B.stream("textures", 0.05, 0)
    auroc = detection_metrics(stream.y, build("adaodd").fit(B).score(stream).scores)["auroc"]
    assert 0.5 < auroc <= 1.0
