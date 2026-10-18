"""Bảng phân tách theo benchmark/π, bảng tĩnh, quyết định và hoạt động CDM của report trên kết quả dựng tay."""
import pytest

from cure_ood.eval.report import METRICS, build_report, to_markdown

# AUROC (tỉ lệ) của từng phương pháp theo (benchmark, tập lạ, π); FPR95 = 1 - AUROC cho dễ kiểm.
AUROC = {("b1", "near1", 0.05): {"knn": 0.80, "msp": 0.70, "cure": 0.90},
         ("b1", "far1", 0.05): {"knn": 0.90, "msp": 0.95, "cure": 0.90},
         ("b1", "near1", 0.5): {"knn": 0.60, "msp": 0.50, "cure": 0.80},
         ("b2", "far2", 0.5): {"knn": 0.70, "msp": 0.60, "cure": 0.70}}
NEAR = {"near1"}


def rows():
    out = []
    for (b, d, pi), arms in AUROC.items():
        for seed in (0, 1):
            r = {"benchmark": b, "domain": d, "near": d in NEAR, "pi": pi, "seed": seed, "n": 100,
                 "metrics": {a: {m: v for m in METRICS} | {"fpr95": 1 - v} for a, v in arms.items()}, "dcr": {}}
            r["decisions"] = {"cure": {"0.05": {"fpr": 0.04 + 0.02 * seed, "tpr": 0.5}}}
            r["memory_stats"] = {"cure": {"cdm_mean": 100.0 * (1 + seed), "coverage": 0.5, "certified": 10, "evicted": 30}}
            out.append(r)
    return out


CFG = {"reference": "knn", "static": ["knn", "msp", "cure"],
       "families": {"kNN": {"base": "knn", "method": "cure", "rivals": ["msp"]}}}


def test_breakdown_by_benchmark_and_pi():
    bd = build_report(rows(), CFG)["breakdowns"]["kNN"]
    assert list(bd) == ["b1", "b1 near", "b1 far", "b2", "b2 far", "π = 0.05", "π = 0.5"]
    assert bd["b1"]["n_groups"] == 3 and bd["b1"]["arms"]["cure"]["auroc"] == pytest.approx(100 * (0.9 + 0.9 + 0.8) / 3)
    assert bd["b1 near"]["arms"]["knn"]["auroc"] == pytest.approx(70.0)
    assert bd["π = 0.5"]["arms"]["msp"]["fpr95"] == pytest.approx(45.0)


def test_static_table_best_share_and_order():
    st = build_report(rows(), CFG)["static"]
    assert st["n_groups"] == 4 and list(st["arms"]) == ["cure", "knn", "msp"]
    # Hòa cũng tính là cao nhất: cure ở 3/4 nhóm, knn hòa cure ở 1 nhóm, msp ở 1 nhóm.
    assert [st["arms"][a]["best"] for a in ("cure", "knn", "msp")] == [0.75, 0.25, 0.25]


def test_decision_and_memory_summaries():
    res = build_report(rows(), CFG)
    d = res["decisions"]["cure"]
    assert d["n_groups"] == 4 and d["0.05"]["fpr"] == pytest.approx(5.0) and d["0.05"]["fpr_max"] == pytest.approx(5.0)
    mem = res["memory"]["cure"]
    assert mem["all"]["cdm_mean"] == pytest.approx(150.0) and mem["near"]["n_groups"] == 2 and mem["π = 0.5"]["n_groups"] == 2
    md = to_markdown(res, CFG, "t")
    assert "| cure | 4 | 0.05 | 5.00 | 5.00 | 50.00 |" in md
    assert "| cure | 82.50 | 17.5 | 75.0% |" in md


def test_report_without_new_fields_is_unchanged():
    old = [{k: v for k, v in r.items() if k not in ("decisions", "memory_stats")} for r in rows()]
    res = build_report(old, {"reference": "knn"})
    assert res["decisions"] == {} and res["memory"] == {} and "static" not in res
    assert "## CDM activity" not in to_markdown(res, {}, "t")
