"""Kiểm tra từng thành phần của bản DynProto bám bài (Eq. 4, Eq. 5, BIRCH, FIFO, cold start) trên dữ liệu giả."""
from types import SimpleNamespace

import numpy as np
import pytest

from cure_ood.methods import build
from cure_ood.methods.testtime.dynproto import (
    ALPHA_GRID,
    adaptive_alpha,
    birch_prototypes,
    log_msp_odds,
    neglabel_log_odds,
    sigmoid,
)
from cure_ood.ops import l2, softmax


def owttt_variance(os, th):
    """Bản chép compute_os_variance của OWTTT (Yushu-Li/OWTTT) để đối chiếu."""
    t = np.zeros(os.shape)
    t[os >= th] = 1
    w1 = np.count_nonzero(t) / os.size
    w0 = 1 - w1
    if w1 == 0 or w0 == 0:
        return np.inf
    return w0 * np.var(os[t == 0]) + w1 * np.var(os[t == 1])


def toy(n_cls=3, d=8, n_train=60, seed=0):
    """Benchmark giả: lớp c nằm quanh trục c, logit = 10 x tích vô hướng với trục."""
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(n_cls), n_train // n_cls)
    G = l2(np.eye(d)[y] + 0.05 * rng.normal(size=(len(y), d)))
    return SimpleNamespace(G=G.astype(np.float32), train_labels=y, train_logits=10 * G[:, :n_cls])


def toy_stream(bench, n_id, n_ood, seed=1):
    rng = np.random.default_rng(seed)
    n_cls, d = bench.train_logits.shape[1], bench.G.shape[1]
    xi = np.eye(d)[rng.integers(0, n_cls, n_id)] + 0.05 * rng.normal(size=(n_id, d))
    # OOD: hướng trục cuối, lệch nhẹ về lớp 0 để có lớp dự đoán.
    xo = 0.3 * np.eye(d)[0] + np.eye(d)[d - 1] + 0.05 * rng.normal(size=(n_ood, d))
    x = l2(np.r_[xi, xo]).astype(np.float32)
    perm = rng.permutation(len(x))
    x = x[perm]
    return SimpleNamespace(x=x, logits=10 * x[:, :n_cls], cache={}), np.r_[np.zeros(n_id), np.ones(n_ood)][perm]


def test_log_msp_odds_is_logit_of_msp():
    z = np.random.default_rng(0).normal(size=(50, 7)) * 3
    p = softmax(z).max(1)
    assert np.allclose(log_msp_odds(z), np.log(p / (1 - p)))


def test_eq4_score_matches_direct_formula():
    rng = np.random.default_rng(0)
    v, pid, pood = (l2(rng.normal(size=(n, 5))) for n in (10, 3, 4))
    tau, k = 0.5, 5
    a, b = np.exp(v @ pid.T / tau).sum(1), np.exp(v @ pood.T / tau).sum(1)
    assert np.allclose(sigmoid(neglabel_log_odds(v, pid, pood, tau, k)), a / (a + k * b))


def test_eq4_no_overflow_at_paper_temperature():
    v = l2(np.eye(4)[:2].astype(np.float32))
    lo = neglabel_log_odds(v, v[:1], v[1:], 0.01, 5)
    assert np.isfinite(lo).all() and lo[0] > 0 > lo[1]


def test_alpha_owttt_matches_reference_and_splits_bimodal():
    rng = np.random.default_rng(0)
    s = np.clip(np.r_[rng.normal(0.15, 0.05, 30), rng.normal(0.85, 0.05, 98)], 0.001, 0.999)
    ref = ALPHA_GRID[int(np.argmin([owttt_variance(s, th) for th in ALPHA_GRID]))]
    a = adaptive_alpha(s, "owttt")
    lo, hi = s[:30].max(), s[30:].min()  # hai mode tách hẳn: α phải nằm trong khe
    assert a == ref and lo < a <= hi
    assert lo < adaptive_alpha(s, "eq5") <= hi


def test_alpha_rules_differ_on_unbalanced_groups():
    # Eq. 5 như in (ko trọng số) và OWTTT (có trọng số) có thể cho ngưỡng khác nhau.
    rng = np.random.default_rng(3)
    s = np.clip(np.r_[rng.uniform(0.0, 0.5, 5), rng.normal(0.9, 0.03, 120)], 0, 0.999)
    assert adaptive_alpha(s, "owttt") != adaptive_alpha(s, "eq5")


def test_alpha_degenerate_batch_admits_nothing():
    assert adaptive_alpha(np.full(16, 0.5)) == 0.0


def test_birch_one_prototype_per_cluster_as_mean():
    rng = np.random.default_rng(0)
    a, b = np.eye(6)[0] + 0.01 * rng.normal(size=(10, 6)), np.eye(6)[1] + 0.01 * rng.normal(size=(7, 6))
    P = birch_prototypes(l2(np.r_[a, b]), 0.5, 50)
    assert len(P) == 2
    means = np.stack([l2(a).mean(0), l2(b).mean(0)])
    assert np.allclose(np.sort(P, axis=0), np.sort(means, axis=0), atol=1e-6)
    assert np.allclose(birch_prototypes(l2(a[:1]), 0.5, 50), l2(a[:1]))


def test_id_prototypes_use_true_labels():
    B = toy()
    B.train_logits = B.train_logits[:, ::-1].copy()  # nhãn dự đoán sai hết với lớp 0, 2
    m = build("dynproto").fit(B)
    assert np.argmax(m.p_id[0]) == 0 and np.argmax(m.p_id[2]) == 2


def test_theta_is_beta_percentile_of_train_msp():
    B = toy()
    m = build("dynproto", beta=5.0).fit(B)
    assert m.theta == pytest.approx(np.percentile(softmax(B.train_logits).max(1), 5.0))


def test_cold_start_without_candidates_falls_back_to_msp():
    B = toy()
    st, _ = toy_stream(B, 300, 0)
    m = build("dynproto", batch=32, beta=0.0).fit(B)  # θ = min MSP train: gần như ko ai được nạp
    m.theta = -1.0
    out = m.score(st)
    assert np.allclose(out.scores, -log_msp_odds(st.logits))
    assert all(len(idx) == 0 for _, idx in out.memory)


def test_fifo_capacity_and_causality():
    B = toy()
    st, _y = toy_stream(B, 300, 150)
    m = build("dynproto", batch=32, cache_size=4).fit(B)
    out = m.score(st)
    assert all(len(idx) <= 4 * 3 for _, idx in out.memory)
    assert all(max(idx, default=-1) < s + 32 for s, idx in out.memory)  # chỉ dùng mẫu của batch ≤ t
    # Cắt luồng: điểm của các batch đầu ko đổi.
    head = type(st)(x=st.x[:96], logits=st.logits[:96], cache={})
    assert np.allclose(m.score(head).scores, out.scores[:96])


def test_fifo_keeps_most_recent():
    B = toy()
    st, _ = toy_stream(B, 0, 200)
    m = build("dynproto", batch=20, cache_size=3, t_cold=100).fit(B)
    m.theta = 2.0  # mọi mẫu đều qua: cache phải giữ đúng 3 mẫu mới nhất mỗi lớp
    out = m.score(st)
    yhat = st.logits.argmax(1)
    _s_last, idx = out.memory[-1]
    for c in range(3):
        expect = [i for i in range(len(st.x)) if yhat[i] == c][-3:]
        assert [i for i in idx if yhat[i] == c] == expect


def test_dynproto_separates_toy_ood():
    from cure_ood.eval.metrics import detection_metrics
    B = toy()
    st, y = toy_stream(B, 600, 200)
    m = build("dynproto", batch=64, t_cold=2).fit(B)
    assert detection_metrics(y, m.score(st).scores)["auroc"] > 0.95


def test_smoke_imagenet200_textures():
    """Smoke: in AUROC cũ (dynproto_r53) và mới trên một luồng thật."""
    from cure_ood.data import Benchmark
    from cure_ood.eval.metrics import detection_metrics

    from .conftest import need_features
    B = Benchmark.load("imagenet200")
    need_features(B)
    st = B.stream("textures", 0.05, 0)
    res = {name: detection_metrics(st.y, build(name).fit(B).score(st).scores)["auroc"]
           for name in ("dynproto_r53", "dynproto")}
    print(f"\nimagenet200/textures pi=0.05 seed=0 AUROC: {res}")
    assert res["dynproto"] > 0.5
