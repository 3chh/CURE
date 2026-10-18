"""WARDEN: kiểm từng thành phần trên dữ liệu tổng hợp (số tính tay) và một lần chạy thật trên ImageNet-200."""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cure_ood.eval.metrics import detection_metrics
from cure_ood.methods import build
from cure_ood.methods.testtime.warden import (
    WARDEN,
    admission_p_threshold,
    cdc_threshold,
    ebh_admit,
    kappa_bar,
    kth_dist,
    ledoit_wolf_whitener,
    storey_upper,
)
from cure_ood.ops import l2


def test_ebh_admit_hand_computed():
    # a = 0.5: e = 0.5·p^-0.5 = [50, 50, 0.71, 0.53]; ngưỡng K/(δk) = [40, 20, 13.3, 10] → k* = 2.
    assert ebh_admit(np.array([1e-4, 1e-4, 0.5, 0.9]), 0.1, 0.5).tolist() == [True, True, False, False]
    # e = 5 < 40 ở hạng 1, và hạng k cần e ≥ 40/k nên ko nhận ai.
    assert not ebh_admit(np.full(4, 0.01), 0.1, 0.5).any()
    assert ebh_admit(np.zeros(0), 0.1, 0.5).shape == (0,)


def test_lemma8_constants_match_paper():
    # Sec. 4.3: a = 0.1, δ = 0.1, K = 64, m = 1500, η = 0.05 → κ ≈ 0.006, κ̄ ≈ 0.0096, W̄ ≈ 0.62.
    k = admission_p_threshold(0.1, 0.1)
    assert abs(k - 0.01 ** (1 / 0.9)) < 1e-12 and round(k, 3) == 0.006
    kb = kappa_bar(1500, k, 0.05)
    assert round(kb, 4) == 0.0096 and abs(64 * kb - 0.62) < 0.01  # 0.615; bài làm tròn "≈ 0.62"


def test_admitted_points_respect_absolute_threshold():
    rng = np.random.default_rng(0)
    k = admission_p_threshold(0.1, 0.1)
    for _ in range(200):
        p = np.r_[rng.uniform(0, 0.01, rng.integers(0, 64)), rng.uniform(0, 1, 64)]
        assert (p[ebh_admit(p, 0.1, 0.1)] <= k + 1e-12).all()


def test_storey_upper_hand_computed():
    p = np.r_[np.full(40, 0.1), np.full(60, 0.9)]
    # π̂_λ = 1 - 60/(100·0.5) = -0.2; slack = 2·sqrt(log(40)/200) = 0.2716.
    assert abs(storey_upper(p, 0.5, 0.05) - (-0.2 + 2 * math.sqrt(math.log(40) / 200))) < 1e-12
    assert storey_upper(np.full(10, 0.01), 0.5, 0.05) == 1.0


def test_cdc_threshold_hand_computed():
    n = 1000
    s, p = np.arange(n, dtype=float), np.linspace(0.001, 1, n)  # p đều: π̂_λ ≈ 0
    pi_up = storey_upper(p, 0.5, 0.05)
    level = 1 - 0.1 * (1 - pi_up) + math.sqrt(math.log(40) / 2000) + 1 / n
    assert cdc_threshold(s, p, 0.1) == s[math.ceil(level * n) - 1]
    # Cửa sổ nhỏ: mức vượt 1 → τ̂ = +∞, ko gắn cờ (Thm. 11).
    assert cdc_threshold(s[:20], p[:20], 0.1) == math.inf


def test_whitener_gives_identity_covariance():
    from sklearn.covariance import LedoitWolf
    rng = np.random.default_rng(0)
    A = rng.normal(size=(6, 6)) * np.r_[2, 1.5, 1.2, 1, 0.8, 0.6]
    f = rng.normal(size=(20000, 6)) @ A + 3
    mu, W = ledoit_wolf_whitener(f)
    S = LedoitWolf().fit(f).covariance_
    assert np.allclose(W.T @ S @ W, np.eye(6), atol=1e-8)  # đúng định nghĩa: Σ_LW được làm trắng thành I
    assert np.abs(np.cov(((f - mu) @ W).T) - np.eye(6)).max() < 0.05


def _toy(seed=0, n_id=640, n_ood=128, burst_at=256, whiten=False, **kw):
    """Benchmark/luồng giả: ID quanh e1, OOD là một cụm chặt quanh e2 đến thành khối liên tiếp."""
    rng = np.random.default_rng(seed)
    d = 16
    def id_pts(m):
        return rng.normal(size=(m, d)) * 0.3 + np.eye(d)[0] * 3
    bench = SimpleNamespace(train_feats=id_pts(2000), cal_feats=id_pts(500), k=5)
    bench.G = l2(bench.train_feats)
    ood = rng.normal(size=(n_ood, d)) * 0.05 + np.eye(d)[1] * 3
    feats = id_pts(n_id)
    feats = np.r_[feats[:burst_at], ood, feats[burst_at:]]
    y = np.r_[np.zeros(burst_at), np.ones(n_ood), np.zeros(n_id - burst_at)]
    m = WARDEN(whiten=whiten, k=5, **kw).fit(bench)
    stream = SimpleNamespace(feats=feats, x=l2(feats), cache={}, bench=bench)
    return m, stream, y, bench


def test_ranking_equals_frozen_base_score():
    m, stream, _y, bench = _toy()
    out = m.score(stream)
    assert np.array_equal(out.scores, kth_dist(l2(stream.feats), bench.G, 5))  # AUROC = nền đóng băng


def test_decide_then_update_and_fifo_cap():
    m, stream, y, _ = _toy(cap=50, batch=64)
    out = m.score(stream)
    adm = m.last_["admitted"]
    assert len(adm) > 50 and y[adm].mean() > 0.9  # khối OOD được nạp, gần như ko nạp nhầm ID
    assert (m.last_["p_dict"][:256 + 64] == 1).all()  # từ điển rỗng tới khi khối đầu tiên đã được quyết định
    for st, d in out.memory:
        assert len(d) <= 50 and (d < st).all()  # chỉ dùng mẫu của batch trước
    last = out.memory[-1][1]
    assert np.array_equal(last, adm[-50:])  # FIFO: giữ 50 mẫu nạp sau cùng


def test_pure_id_stream_admits_almost_nothing():
    m, stream, _y, _ = _toy(n_ood=0, n_id=3200)
    m.score(stream)
    # Lemma 8: ≤ K·κ̄ mẫu ID bị nạp mỗi batch (kỳ vọng); 25 batch × 128 × κ̄(500) ≈ 50 là trần rất lỏng.
    assert len(m.last_["admitted"]) <= 25 * 128 * kappa_bar(500, admission_p_threshold(0.1, 0.1), 0.05)


def test_decision_channels_and_fpr():
    m, stream, y, _ = _toy(n_id=6400, n_ood=128, burst_at=3200)
    m.score(stream)
    L = m.last_
    assert np.array_equal(L["flags"], (L["p_base"] <= 0.05) | (L["p_dict"] <= 0.05))
    for key in ("flags", "flags_cdc", "flags_cdc_dict"):
        assert L[key][y == 0].mean() <= 0.1 + 0.02, key  # FPR ≤ α (cùng phân phối, ko trôi dạt)
    assert L["flags"][y == 1].mean() > 0.9


def test_whitened_variant_runs():
    m, stream, y, _ = _toy(whiten=True)
    out = m.score(stream)
    assert detection_metrics(y, out.scores)["auroc"] > 0.95


SMOKE_REF = Path(__file__).resolve().parents[3] / "results" / "round53_parts" / "in200_textures.json"


def test_smoke_imagenet200_textures(capsys):
    from cure_ood.data import Benchmark

    from .conftest import need_features
    B = Benchmark.load("imagenet200")
    need_features(B)
    stream = B.stream("textures", 0.05, 0)
    au = lambda name, **kw: detection_metrics(stream.y, build(name, **kw).fit(B).score(stream).scores)["auroc"]
    new, new_nowhite, knn = au("warden"), au("warden", whiten=False), au("knn")
    old = max(au("warden_r53", kappa=k, delta=d) for k, d in ((0.5, 0.1), (0.1, 0.2), (0.05, 0.2)))
    if SMOKE_REF.exists():
        row = next(r for r in json.loads(SMOKE_REF.read_text(encoding="utf-8"))["rows"] if (r["pi"], r["seed"]) == (0.05, 0))
        assert old == row["metrics"]["warden_best"]["auroc"]
    assert abs(new_nowhite - knn) < 1e-4  # ko làm trắng: khoảng cách L2 đơn điệu với cosine của kNN
    with capsys.disabled():
        print(f"\n[WARDEN smoke in200/textures pi=0.05 s0] old best-of-3={old:.4f} new={new:.4f} "
              f"new(whiten=False)={new_nowhite:.4f} knn={knn:.4f}")
