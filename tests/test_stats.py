"""Tính chất cơ bản của công cụ thống kê, ko cần dữ liệu."""
import numpy as np

from cure_ood.stats import combine, ebh_certifier, loo_p, rank_p


def test_rank_p_is_valid_conformal_pvalue():
    rng = np.random.default_rng(0)
    cal, test = rng.normal(size=999), rng.normal(size=20000)
    p = rank_p(test, cal)
    assert p.min() >= 1 / 1000 and p.max() <= 1
    assert abs(np.mean(p <= 0.1) - 0.1) < 0.01  # cùng phân phối: P(p ≤ α) ≈ α


def test_loo_p_range():
    p = loo_p(np.arange(10.0))
    assert p.max() == 1.0 and p.min() == 0.1


def test_cauchy_combination_monotone():
    a = combine([np.array([0.01, 0.5]), np.array([0.01, 0.5])], [1, 1], "cauchy")
    assert a[0] > a[1]


def test_ebh_keeps_dense_cluster_and_rejects_lone_sample():
    p = np.r_[np.full(15, 1e-4), np.random.default_rng(0).uniform(0.008, 1, 113)]
    keep = ebh_certifier(p)(np.arange(128))
    assert keep[:15].all() and not keep[15:].any()
    lone = np.r_[1e-4, np.random.default_rng(1).uniform(0.008, 1, 127)]
    assert not ebh_certifier(lone)(np.arange(128)).any()


def test_online_ebh_certifies_one_very_strong_sample():
    """Tính tay: p = 1e-6 cho e = 500; ngưỡng của mẫu đầu là 1/(1·alpha·gamma_1), vượt ngay nên k* = 1."""
    from cure_ood.stats import RULES
    r = RULES["online_ebh"](alpha=0.5)
    assert r(np.array([1e-6]))[0]
    # p = 0.5 cho e = 0.707, cần k* ≈ 244 mới đạt: ko được chứng nhận khi mới có 1 phát hiện.
    assert not r(np.array([0.5]))[0]


def test_online_ebh_controls_false_certifications_under_the_null():
    """Toàn mẫu null (p đều): số chứng nhận sai phải rất nhỏ so với alpha."""
    from cure_ood.stats import RULES
    p = np.random.default_rng(0).random(5000)
    keep = np.concatenate([RULES["online_ebh"](alpha=0.2)(p[i:i + 1]) for i in range(len(p))])
    assert keep.sum() <= 1


def test_online_ebh_has_more_power_than_elond():
    """Vùng tín hiệu vừa phải: ngưỡng của online e-BH nới dần nên chứng nhận nhiều hơn, ko thêm mẫu sai."""
    from cure_ood.stats import RULES
    rng = np.random.default_rng(0)
    n = 4000
    ood = rng.random(n) < 0.2
    p = np.where(ood, rng.random(n) * 1e-6, rng.random(n))
    out = {}
    for name in ("elond", "online_ebh"):
        rule = RULES[name](alpha=0.2)
        out[name] = np.concatenate([rule(p[i:i + 1]) for i in range(n)])
    assert out["online_ebh"].sum() > 3 * out["elond"].sum()
    assert not (out["online_ebh"] & ~ood).any() and not (out["elond"] & ~ood).any()


def test_online_ebh_state_is_per_stream():
    """Mỗi luồng phải tạo rule mới: hai rule độc lập cho cùng kết quả."""
    from cure_ood.stats import RULES
    p = np.random.default_rng(1).random(200) * 1e-7
    a = np.concatenate([RULES["online_ebh"]()(p[i:i + 1]) for i in range(len(p))])
    rule = RULES["online_ebh"]()
    b = np.concatenate([rule(p[i:i + 1]) for i in range(len(p))])
    assert a.sum() <= b.sum()  # rule dùng lại tích lũy bằng chứng nên chứng nhận ít nhất bằng
