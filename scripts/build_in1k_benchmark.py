"""Vòng 60c: dựng phía truy vấn của benchmark ImageNet-1k từ file đặc trưng validation đã công bố.

Tách theo đăng ký trước (docs/CURE_Causal_Research_Log.md, vòng 60c), phân tầng theo lớp với seed 0:
5 ảnh/lớp làm tập hiệu chuẩn C, 10 ảnh/lớp làm phần ID của luồng, phần còn lại ko dùng;
1.000 mẫu OpenImage-O tách riêng làm tập OOD để chọn K bằng APS, và bị loại khỏi tập test.

Chạy: py scripts/build_in1k_benchmark.py <features.npz> --out data/features/in1k_query.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

OOD_SETS = ("ssb_hard", "ninco", "inaturalist", "textures", "openimage_o")
N_CAL, N_STREAM, N_VAL_OOD, SEED = 5, 10, 1000, 0


def stratified(labels: np.ndarray, per_class: int, rng) -> np.ndarray:
    """per_class chỉ số mỗi lớp, rút ko lặp; trả về đã sắp tăng dần."""
    out = []
    for c in np.unique(labels):
        i = np.flatnonzero(labels == c)
        out.append(rng.choice(i, min(per_class, len(i)), replace=False))
    return np.sort(np.concatenate(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("features", help="imagenet_resnet50_features_*.npz đã công bố (tập validation + OOD)")
    ap.add_argument("--out", default="data/features/in1k_query.npz")
    ap.add_argument("--report", action="store_true", help="chỉ in thống kê, ko ghi file")
    a = ap.parse_args()
    z = np.load(a.features)
    rng = np.random.default_rng(SEED)
    lab = z["id_labels"]
    cal = stratified(lab, N_CAL, rng)
    rest = np.setdiff1d(np.arange(len(lab)), cal)
    stream = rest[stratified(lab[rest], N_STREAM, rng)]
    assert not set(cal) & set(stream)

    out = {"id_val_feats": z["id_features"][cal], "id_val_labels": lab[cal],
           "id_test_feats": z["id_features"][stream], "id_test_labels": lab[stream]}
    oi = z["openimage_o_features"]
    val_oi = np.sort(rng.choice(len(oi), N_VAL_OOD, replace=False))
    keep = np.setdiff1d(np.arange(len(oi)), val_oi)
    out["val_oi_feats"], out["val_oi_labels"] = oi[val_oi], z["openimage_o_labels"][val_oi]
    for s in OOD_SETS:
        idx = keep if s == "openimage_o" else slice(None)
        out[f"{s}_feats"] = z[f"{s}_features"][idx]
        out[f"{s}_labels"] = z[f"{s}_labels"][idx]

    info = {k: list(v.shape) for k, v in out.items() if k.endswith("_feats")}
    print(json.dumps(info, indent=1))
    print("C và luồng ko giao nhau:", len(cal), len(stream), "| OpenImage-O test còn", len(keep))
    if a.report:
        return
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, **out)
    print("đã ghi", p, round(p.stat().st_size / 1e9, 2), "GB")


if __name__ == "__main__":
    main()
