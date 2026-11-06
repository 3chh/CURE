"""Phương pháp tĩnh phải tái hiện đúng AUROC của bản gốc (vòng 21, 27, 33) và K của configs/benchmarks."""
from __future__ import annotations

import json

import pytest

from cure_ood.data import Benchmark
from cure_ood.eval.metrics import detection_metrics
from cure_ood.methods import build
from cure_ood.methods.static.knn import select_k

from .conftest import GOLDEN, need_features

METHODS = ("knn", "msp", "maxlogit", "energy", "gen", "mahalanobis", "vim", "react", "nci", "fdbd", "cadref")
# Vòng 21-33 dùng bản tự viết (nay là *_legacy); tên chuẩn giờ là postprocessor OpenOOD gốc.
LEGACY = {m: f"{m}_legacy" for m in METHODS if m != "knn"}
# File kết quả gốc của từng benchmark (ImageNet-200 ko có APS: vòng 33 dùng siêu tham số mặc định).
SOURCES = {"cifar10": ["round21_parts/cifar10.json"], "cifar10_supcon": ["round21_parts/supcon.json"],
           "cifar100": ["round27_parts/cifar100.json", "round27_parts/cifar100_texture.json"],
           "imagenet200": ["round33_parts/textures.json"]}
CASES = [("cifar10", "cifar100", 0.05, 0), ("cifar10", "svhn", 0.2, 1), ("cifar10_supcon", "tin", 0.05, 2),
         ("cifar100", "texture", 0.2, 0), ("imagenet200", "textures", 0.05, 1)]
# Mahalanobis và ReAct tính có sai số làm tròn số thực (BLAS/float precision) ~1e-7 so với bản gốc.
TOL = {"mahalanobis": 1e-5, "react": 1e-5}
APS = {
    "cifar10": {"energy": {"T": 1}, "gen": {"gamma": 0.01, "M": 5}, "vim": {"dim": 1},
                "react": {"percentile": 85}, "nci": {"alpha": 1e-4}, "fdbd": {"norm": True}},
    "cifar10_supcon": {"energy": {"T": 2.0}, "gen": {"gamma": 0.01, "M": 7}, "vim": {"dim": 128},
                       "react": {"percentile": 95}, "nci": {"alpha": 0.1}, "fdbd": {"norm": False}},
}


def golden(bench, domain, pi, seed):
    for f in SOURCES[bench]:
        path = GOLDEN / f
        if path.exists():
            for r in json.loads(path.read_text(encoding="utf-8"))["rows"]:
                if (r["domain"], r["pi"], r["seed"]) == (domain, pi, seed) and "msp" in r["auroc"]:
                    return r["auroc"]
    pytest.skip(f"thiếu kết quả gốc cho {bench} {domain} {pi} {seed}")


@pytest.fixture(scope="module")
def fitted():
    """Fit một lần mỗi benchmark (APS + thống kê train là phần tốn thời gian)."""
    cache = {}

    def get(name):
        if name not in cache:
            B = Benchmark.load(name)
            need_features(B)
            cache[name] = B, {m: build(LEGACY.get(m, m)).fit(B) for m in METHODS}
        return cache[name]
    return get


@pytest.mark.parametrize("case", CASES, ids=lambda c: "-".join(map(str, c)))
def test_golden_auroc(case, fitted):
    ref = golden(*case)
    B, methods = fitted(case[0])
    st = B.stream(*case[1:])
    bad = {m: (a, ref[m]) for m in METHODS
           if abs((a := detection_metrics(st.y, methods[m].score(st).scores)["auroc"]) - ref[m]) > TOL.get(m, 0)}
    assert not bad


@pytest.mark.parametrize("name", list(APS))
def test_aps_params(name, fitted):
    _, methods = fitted(name)
    assert {m: methods[m].params for m in APS[name]} == APS[name]


def test_no_aps_without_val_ood(fitted):
    _, methods = fitted("imagenet200")
    assert methods["vim"].params == {"dim": 256} and methods["gen"].params == {"gamma": 0.1, "M": 10}


@pytest.mark.parametrize("name", ["cifar10", "cifar10_supcon", "cifar100"])
def test_select_k_matches_config(name):
    B = Benchmark.load(name)
    need_features(B)
    assert select_k(B)[0] == B.k
