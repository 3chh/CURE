# -*- coding: utf-8 -*-
"""
eval_multiseed_auroc.py
------------------------
Chạy đánh giá đầy đủ (54 luồng OpenOOD) cho toàn bộ các seed mới của ResNet-20 và MobileNetV2:
- 6 tập OOD: CIFAR-100, TinyImageNet, MNIST, SVHN, Texture, Places365
- 3 tỷ lệ OOD pi: 0.05, 0.20, 0.50
- 3 seeds luồng: 0, 1, 2
- Phương pháp: kNN, CURE (Batch e-BH), CURE (Online B=1), OODD, WARDEN, AgeQueue.
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

def eval_backbone(bench_name, methods=("knn", "cure", "cure_oebh_b1", "oodd", "warden", "age_queue")):
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
            "auroc": round(float(np.mean(res[m]["auroc"])), 2),
            "near_auroc": round(float(np.mean(res[m]["near_auroc"])), 2),
            "far_auroc": round(float(np.mean(res[m]["far_auroc"])), 2),
            "fpr95": round(float(np.mean(res[m]["fpr95"])), 2),
            "tpr5": round(float(np.mean(res[m]["tpr5"])), 2),
            "dcr": round(float(np.mean(res[m]["dcr"])), 2),
            "delta_knn": round(float(np.mean(deltas)), 2),
            "harm_rate": round(float(harm_rate), 2),
            "worst_delta": round(float(np.min(deltas)), 2)
        }
    return summary

def main():
    print("=================================================================")
    print("ĐÁNH GIÁ ĐA SEED CHO RESNET-20 (S0, S1, S2) & MOBILENETV2 (S0, S1, S2)")
    print("=================================================================\n")

    benchmarks = [
        ("r61_resnet20", "ResNet-20 Seed 0"),
        ("r61_resnet20_s1", "ResNet-20 Seed 1"),
        ("r61_resnet20_s2", "ResNet-20 Seed 2"),
        ("r61_mobilenetv2", "MobileNetV2 Seed 0"),
        ("r61_mobilenetv2_s1", "MobileNetV2 Seed 1"),
        ("r61_mobilenetv2_s2", "MobileNetV2 Seed 2"),
    ]

    all_results = {}
    for bcode, bname in benchmarks:
        print(f"Đang chạy 54 luồng cho {bname} ({bcode})...", flush=True)
        t0 = time.time()
        res = eval_backbone(bcode)
        elapsed = time.time() - t0
        all_results[bcode] = res
        print(f"-> Hoàn tất {bname} trong {elapsed:.1f}s | kNN={res['knn']['auroc']}% | CURE={res['cure']['auroc']}% | CURE_B1={res['cure_oebh_b1']['auroc']}% | WARDEN={res['warden']['auroc']}%", flush=True)

    out_dir = REPO_ROOT / "results" / "multiseed_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "multiseed_auroc_summary.json"
    out_file.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nĐã lưu kết quả tại: {out_file}")

if __name__ == "__main__":
    main()
