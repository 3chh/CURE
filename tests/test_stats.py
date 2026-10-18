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
