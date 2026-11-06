"""AdaODD, WARDEN, RTL++, DCAC, DynProto phải tái hiện kết quả vòng 53 (AUROC và DCR)."""
import pytest

from cure_ood.data import Benchmark
from cure_ood.eval.metrics import dcr_counts, detection_metrics
from cure_ood.methods import build
from cure_ood.methods.legacy.warden import R53_GRID

from .conftest import golden_row, need_features

CASES = [("imagenet200", "textures", 0.5, 0), ("cifar10", "svhn", 0.5, 0), ("cifar100", "texture", 0.2, 2),
         ("cifar10_supcon", "tin", 0.5, 1), ("cifar10_dinov2", "places365", 0.5, 0), ("cifar10", "cifar100", 0.05, 1)]
ARMS = ("adaodd", "rtl_plus_legacy", "dcac_msp_r53", "dcac_energy_r53", "dynproto_r53")
# Vòng 53 chạy bản DynProto cũ (nay là dynproto_r53) dưới khóa "dynproto".
GOLDEN_KEY = {"dynproto_r53": "dynproto", "rtl_plus_legacy": "rtl_plus", "dcac_msp_r53": "dcac_msp", "dcac_energy_r53": "dcac_energy"}
# RTL++ và DynProto phụ thuộc số học ElasticNet/BLAS/threads nên cho phép dung sai số thực nhỏ.
TOL = {"rtl_plus_legacy": 1e-3, "dynproto_r53": 1e-5}
# AdaODD vòng 53 dùng câu "κ = 1.5, γ = 5.0" ở Sec. VI-B của bài; mặc định mới theo Sec. VI-A (vòng 58).
PARAMS = {"adaodd": {"k": 50, "gamma": 5.0, "kappa": 1.5}}


@pytest.mark.parametrize("bench,domain,pi,seed", CASES)
def test_round53_testtime(bench, domain, pi, seed):
    B = Benchmark.load(bench)
    need_features(B)
    ref = golden_row("round53_parts", bench, domain, pi, seed)
    stream = B.stream(domain, pi, seed)
    auroc = lambda m: detection_metrics(stream.y, m.fit(B).score(stream).scores)["auroc"]
    for name in ARMS:
        m = build(name, **PARAMS.get(name, {}))
        if m.requires_logits and not B.has_logits:
            continue
        out = m.fit(B).score(stream)
        key = GOLDEN_KEY.get(name, name)
        assert abs(detection_metrics(stream.y, out.scores)["auroc"] - ref["metrics"][key]["auroc"]) <= TOL.get(name, 0), name
        if key in ref["dcr"]:
            assert dcr_counts(out.memory, stream.y) == ref["dcr"][key], name
    # Bản gốc chọn cấu hình WARDEN theo AUROC test: chỉ đối chiếu cận trên lạc quan đó.
    best = max(auroc(build("warden_r53", kappa=k, delta=d)) for k, d in R53_GRID)
    assert best == ref["metrics"]["warden_best"]["auroc"]
