"""Quyết định p ≤ α, certifier ablation và thống kê CDM của CURE trên benchmark tổng hợp nhỏ, ko cần dữ liệu thật."""
import numpy as np
import pytest
import yaml

from cure_ood.data import Benchmark
from cure_ood.eval.metrics import dcr_counts, decision_metrics
from cure_ood.eval.report import build_report, load_rows, to_markdown
from cure_ood.eval.runner import Experiment, run
from cure_ood.methods import build
from cure_ood.stats import all_certifier, oracle_certifier


@pytest.fixture(scope="module")
def bench_cfg(tmp_path_factory):
    """ID: 10 cụm lớp trong 32 chiều; OOD: một cụm chặt riêng, để hậu kiểm có cái để chứng nhận."""
    root = tmp_path_factory.mktemp("feats")
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(10, 32))

    def id_feats(n):
        return (centers[rng.integers(0, 10, n)] + 0.6 * rng.normal(size=(n, 32))).astype(np.float32)

    ood = (rng.normal(size=32) * 1.5 + 0.3 * rng.normal(size=(1500, 32))).astype(np.float32)
    np.savez(root / "syn.npz", id_train_feats=id_feats(3000), id_val_feats=id_feats(1000), id_test_feats=id_feats(2000),
             id_test_labels=np.zeros(2000), ood_feats=ood, ood_labels=np.zeros(1500))
    path = root / "syn.yaml"
    path.write_text(yaml.safe_dump({"name": "syn", "features": "syn.npz", "k": 5, "near": [], "far": ["ood"]}), encoding="utf-8")
    return root, path


@pytest.fixture
def bench(bench_cfg, monkeypatch):
    monkeypatch.setenv("CURE_FEATURES", str(bench_cfg[0]))
    return Benchmark.load(str(bench_cfg[1]))


def test_decision_metrics_hand_computed():
    d = decision_metrics(np.array([0, 0, 1, 1]), np.array([0.01, 0.5, 0.03, 0.2]), [0.05, 0.25])
    assert d == {"0.05": {"fpr": 0.5, "tpr": 0.5}, "0.25": {"fpr": 0.5, "tpr": 1.0}}


def test_ablation_certifiers():
    idx = np.array([3, 0, 2])
    assert all_certifier(np.ones(5))(idx).all()
    assert oracle_certifier(np.array([0, 1, 1, 0, 1]))(idx).tolist() == [False, False, True]


def test_cure_pvalues_decide_and_control_fpr(bench):
    stream = bench.stream("ood", 0.2, 0)
    out = build("cure").fit(bench).score(stream)
    assert out.pvalues.shape == (len(stream),) and (out.pvalues > 0).all() and (out.pvalues <= 1).all()
    assert (out.decide(0.1) == (out.pvalues <= 0.1)).all()
    d = decision_metrics(stream.y, out.pvalues, [0.05, 0.1])
    assert d["0.05"]["fpr"] < 0.1 and d["0.1"]["fpr"] < 0.17  # ID hoán đổi được với C: FPR ≈ α
    assert d["0.1"]["tpr"] > 0.9
    with pytest.raises(ValueError):
        build("knn").fit(bench).score(stream).decide(0.1)


def test_memory_stats_and_oracle_upper_bound(bench):
    stream = bench.stream("ood", 0.2, 0)
    outs = {c: build("cure", certifier=c).fit(bench).score(stream) for c in ("ebh", "all", "oracle_labels")}
    for out in outs.values():
        st = out.memory_stats
        assert set(st) == {"cdm_mean", "coverage", "certified", "evicted"}
        assert 0 <= st["coverage"] <= 1 and 0 <= st["cdm_mean"] <= 512
    assert outs["ebh"].memory_stats["certified"] > 0 and outs["ebh"].memory_stats["coverage"] > 0.5
    assert outs["all"].memory_stats["evicted"] == 0
    assert dcr_counts(outs["oracle_labels"].memory, stream.y)[0] == 0  # oracle: CDM ko có mẫu ID
    assert dcr_counts(outs["all"].memory, stream.y)[0] > 0
    # Số chứng nhận của oracle = số mẫu OOD đã hết hạn (cỡ hàng chờ ko vượt capacity nên ko mẫu nào bị cắt sớm).
    n_expired = outs["all"].memory_stats["certified"]
    assert outs["oracle_labels"].memory_stats["certified"] == int(stream.y[:n_expired].sum())


def test_runner_and_report_record_decisions_and_activity(bench_cfg, bench, tmp_path):
    exp = Experiment(name="syn", benchmarks=[str(bench_cfg[1])], methods={"knn": {}, "cure": {}}, pis=[0.2, 0.5], seeds=[0],
                     output=str(tmp_path))
    run(exp, log=lambda *_: None)
    rows = load_rows(exp.out_dir)
    assert len(rows) == 2 and all(set(r["decisions"]) == {"cure"} and set(r["memory_stats"]) == {"cure"} for r in rows)
    cfg = {"reference": "knn", "static": ["knn", "cure"], "families": {"kNN": {"base": "knn", "method": "cure", "rivals": []}}}
    res = build_report(rows, cfg)
    assert set(res["decisions"]["cure"]) == {"n_groups", "0.05", "0.1"}
    assert res["memory"]["cure"]["all"]["n_groups"] == 2 and set(res["memory"]["cure"]) == {"all", "far", "π = 0.2", "π = 0.5"}
    assert set(res["breakdowns"]["kNN"]) == {"syn", "syn far", "π = 0.2", "π = 0.5"}
    md = to_markdown(res, cfg, "syn")
    for title in ("## Static methods", "## ID/OOD decisions", "## CDM activity", "### kNN: by benchmark and π"):
        assert title in md
