# -*- coding: utf-8 -*-
"""
eval_resnet50_benchmark.py
--------------------------
Chạy đánh giá đầy đủ cho ResNet-50 (2048-d) tự huấn luyện 100 epochs trên OpenOOD:
- 6 tập OOD: CIFAR-100, TinyImageNet, MNIST, SVHN, Texture, Places365
- 3 tỷ lệ OOD pi: 0.05, 0.20, 0.50
- 3 seeds: 0, 1, 2
- So sánh: kNN, CURE (Batch e-BH), CURE (Online B=1), OODD, WARDEN, AgeQueue.
- Xuất bảng so sánh đối đầu với ResNet-18 gốc của OpenOOD và ResNet-18 tự train.
"""
import sys
import json
import time
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from cure_ood.data.benchmark import Benchmark
from cure_ood.methods.base import build
from cure_ood.eval.metrics import detection_metrics, dcr_counts

def eval_backbone(bench_name, methods=("knn", "cure", "cure_oebh_b1")):
    bench = Benchmark.load(bench_name)
    arms = {}
    for m in methods:
        if m == "knn":
            arms[m] = build("knn").fit(bench)
        elif m == "cure":
            arms[m] = build("cure", rule="fisher").fit(bench)
        elif m == "cure_oebh_b1":
            arms[m] = build("cure", certifier="online_ebh", batch=1, rule="fisher").fit(bench)
        elif m == "oodd":
            arms[m] = build("oodd", gallery="approx").fit(bench)
        elif m == "warden":
            arms[m] = build("warden").fit(bench)
        elif m == "age_queue":
            arms[m] = build("age_queue", ttl=512).fit(bench)

    res = defaultdict(lambda: defaultdict(list))
    domains = bench.domains
    pis = [0.05, 0.2, 0.5]
    seeds = [0, 1, 2]

    for dom in domains:
        is_near = dom in ("cifar100", "tin")
        for pi in pis:
            for s in seeds:
                stream = bench.stream(dom, seed=s, pi=pi)
                y = stream.y
                knn_score = arms["knn"].score(stream).scores
                knn_m = detection_metrics(y, knn_score)
                knn_auroc = knn_m["auroc"] * 100

                for m in methods:
                    out = arms[m].score(stream)
                    dm = detection_metrics(y, out.scores)
                    a = dm["auroc"] * 100
                    f95 = dm["fpr95"] * 100
                    tpr5 = dm["tpr_fpr5"] * 100
                    
                    dcr = 0.0
                    if out.memory is not None:
                        counts = dcr_counts(out.memory, y)
                        if counts and counts[1] > 0:
                            dcr = counts[0] / counts[1] * 100.0

                    res[m]["auroc"].append(a)
                    res[m]["fpr95"].append(f95)
                    res[m]["tpr5"].append(tpr5)
                    res[m]["dcr"].append(dcr)
                    res[m]["delta"].append(a - knn_auroc)
                    if is_near:
                        res[m]["near_auroc"].append(a)
                    else:
                        res[m]["far_auroc"].append(a)

    summary = {}
    for m in methods:
        deltas = np.array(res[m]["delta"])
        harm_rate = np.mean(deltas < -0.5) * 100
        summary[m] = {
            "auroc": float(np.mean(res[m]["auroc"])),
            "near_auroc": float(np.mean(res[m]["near_auroc"])),
            "far_auroc": float(np.mean(res[m]["far_auroc"])),
            "fpr95": float(np.mean(res[m]["fpr95"])),
            "tpr5": float(np.mean(res[m]["tpr5"])),
            "dcr": float(np.mean(res[m]["dcr"])),
            "delta_knn": float(np.mean(deltas)),
            "harm_rate": float(harm_rate),
            "worst_delta": float(np.min(deltas))
        }
    return summary

def main():
    print("=================================================================")
    print("ĐÁNH GIÁ ĐỐI CHỨNG BACKBONE RESNET-50 TỰ TRAIN 100 EPOCHS")
    print("So sánh: ResNet-50 (2048-d) vs ResNet-18 Tự Train vs OpenOOD Gốc")
    print("=================================================================\n")

    methods = ["knn", "cure", "cure_oebh_b1", "oodd", "warden", "age_queue"]

    # 1. ResNet-50 Tự Train
    print("1. Đang chạy 54 luồng cho ResNet-50 (2048-d)...", flush=True)
    t0 = time.time()
    res50 = eval_backbone("r61_resnet50", methods=methods)
    print(f"-> Hoàn tất ResNet-50 trong {(time.time() - t0)/60:.1f} phút.")

    # 2. OpenOOD Gốc ResNet-18
    print("2. Đang chạy 54 luồng cho Backbone GỐC OpenOOD ResNet-18...", flush=True)
    t0 = time.time()
    res18_official = eval_backbone("cifar10", methods=methods)
    print(f"-> Hoàn tất OpenOOD ResNet-18 trong {(time.time() - t0)/60:.1f} phút.")

    # 3. ResNet-18 Tự Train Seed 0
    print("3. Đang chạy 54 luồng cho ResNet-18 Tự Train Seed 0...", flush=True)
    t0 = time.time()
    res18_s0 = eval_backbone("r61_resnet18_s0", methods=methods)
    print(f"-> Hoàn tất ResNet-18 Seed 0 trong {(time.time() - t0)/60:.1f} phút.\n")

    out_data = {
        "r61_resnet50": res50,
        "openood_official_resnet18": res18_official,
        "r61_resnet18_s0": res18_s0
    }

    out_file = REPO_ROOT / "results" / "resnet50_vs_official_benchmark.json"
    out_file.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
    print(f"Đã lưu kết quả tại: {out_file}\n")

    # In bảng tổng hợp
    print("==================================================================================================================")
    print(f"{'Backbone':<30} | {'Method':<20} | {'AUROC':<8} | {'Near':<8} | {'Far':<8} | {'FPR95':<8} | {'Harm Rate':<10} | {'Worst Δ':<8}")
    print("==================================================================================================================")
    for bname, bdict in [("ResNet-50 (Tự train 2048-d)", res50),
                         ("OpenOOD Gốc ResNet-18", res18_official),
                         ("ResNet-18 Seed 0 (Tự train)", res18_s0)]:
        for m in methods:
            d = bdict[m]
            print(f"{bname:<30} | {m:<20} | {d['auroc']:>6.2f}% | {d['near_auroc']:>6.2f}% | {d['far_auroc']:>6.2f}% | {d['fpr95']:>6.2f}% | {d['harm_rate']:>8.1f}% | {d['worst_delta']:>+6.2f}%")
        print("------------------------------------------------------------------------------------------------------------------")

if __name__ == "__main__":
    main()
