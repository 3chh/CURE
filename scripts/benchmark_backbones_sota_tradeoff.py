"""Đánh giá toàn diện Trade-off (Hiệu năng vs Độ trễ vs Thời gian huấn luyện vs Độ an toàn)
trên 7 Backbone (bao gồm 6 Backbone tự huấn luyện và DINOv2) so sánh trực tiếp giữa
CURE (Batch e-BH & Online e-BH B=1) và các phương pháp SOTA (kNN, OODD, AdaODD*, WARDEN*, AgeQueue).

Chạy:
    python deliverables/CURE_repo/scripts/benchmark_backbones_sota_tradeoff.py
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from cure_ood.data.benchmark import Benchmark
from cure_ood.eval.metrics import dcr_counts, detection_metrics
from cure_ood.methods.base import build

BACKBONE_SPECS = [
    {"name": "r61_resnet20_ivf4", "base": "r61_resnet20", "index": {"faiss": "IVF256,Flat", "nprobe": 4}, "k": "aps",
     "label": "ResNet-20 (Self-trained, 64-d)", "train_dir": "r61_resnet20", "dim": 64},
    {"name": "r61_resnet18embed128_ivf4", "base": "r61_resnet18embed128", "index": {"faiss": "IVF256,Flat", "nprobe": 4}, "k": "aps",
     "label": "ResNet-18-Bottleneck (Self-trained, 128-d)", "train_dir": "r61_resnet18embed128", "dim": 128},
    {"name": "r61_mobilenetv2_ivf4", "base": "r61_mobilenetv2", "index": {"faiss": "IVF256,Flat", "nprobe": 4}, "k": "aps",
     "label": "MobileNetV2 (Self-trained, 1280-d)", "train_dir": "r61_mobilenetv2", "dim": 1280},
    {"name": "r61_resnet18_s0_ivf4", "base": "r61_resnet18_s0", "index": {"faiss": "IVF256,Flat", "nprobe": 4}, "k": "aps",
     "label": "ResNet-18 Seed 0 (Self-trained, 512-d)", "train_dir": "r61_resnet18_s0", "dim": 512},
    {"name": "r61_resnet18_s1_ivf4", "base": "r61_resnet18_s1", "index": {"faiss": "IVF256,Flat", "nprobe": 4}, "k": "aps",
     "label": "ResNet-18 Seed 1 (Self-trained, 512-d)", "train_dir": "r61_resnet18_s1", "dim": 512},
    {"name": "r61_resnet18_s2_ivf4", "base": "r61_resnet18_s2", "index": {"faiss": "IVF256,Flat", "nprobe": 4}, "k": "aps",
     "label": "ResNet-18 Seed 2 (Self-trained, 512-d)", "train_dir": "r61_resnet18_s2", "dim": 512},
    {"name": "r61_dinov2_ivf4", "base": "cifar10_dinov2", "index": {"faiss": "IVF256,Flat", "nprobe": 4}, "k": "aps",
     "label": "DINOv2 ViT-B/14 (Foundation, 768-d)", "train_dir": None, "dim": 768},
]

METHODS = {
    "knn": {"method": "knn"},
    "cure": {"method": "cure", "rule": "fisher"},
    "cure_oebh_b1": {"method": "cure", "certifier": "online_ebh", "batch": 1, "rule": "fisher"},
    "oodd": {"method": "oodd", "gallery": "approx"},
    "adaodd": {"method": "adaodd"},
    "warden": {"method": "warden"},
    "age_queue": {"method": "age_queue", "ttl": 512},
}

METHOD_LABELS = {
    "knn": "kNN (Static)",
    "cure": "CURE (Batch e-BH, B=128)",
    "cure_oebh_b1": "CURE (Online e-BH, B=1)",
    "oodd": "OODD (CVPR)",
    "adaodd": "AdaODD* (ICLR)",
    "warden": "WARDEN* (ICML)",
    "age_queue": "AgeQueue (FIFO)",
}


def get_train_stats(train_dir: str | None) -> dict:
    if not train_dir:
        return {"epochs": 0, "train_minutes": 0.0, "ckpt_mb": 346.3, "val_acc": None}
    tpath = REPO_ROOT / "results" / "train" / train_dir
    logf = tpath / "log.jsonl"
    ckpt = tpath / "best.ckpt"
    ckpt_mb = round(ckpt.stat().st_size / (1024 * 1024), 2) if ckpt.exists() else 0.0
    if not logf.exists():
        return {"epochs": 0, "train_minutes": 0.0, "ckpt_mb": ckpt_mb, "val_acc": None}
    lines = [json.loads(ln) for ln in logf.read_text(encoding="utf-8").splitlines() if ln.strip()]
    total_sec = sum(float(r.get("sec", 0.0)) for r in lines)
    val_acc = max((float(r.get("val_acc", 0.0)) for r in lines if "val_acc" in r), default=0.0)
    return {
        "epochs": len(lines),
        "train_minutes": round(total_sec / 60.0, 2),
        "ckpt_mb": ckpt_mb,
        "val_acc": round(val_acc * 100, 2) if val_acc else None,
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    r61_dir = REPO_ROOT / "results" / "round61"
    lat_path = r61_dir / "latency.json"
    lat_data = json.loads(lat_path.read_text(encoding="utf-8")) if lat_path.exists() else {}

    summary_out = {"backbones": {}, "global_by_method": {}}
    global_group_metrics = defaultdict(lambda: defaultdict(list))
    global_near_metrics = defaultdict(lambda: defaultdict(list))
    global_far_metrics = defaultdict(lambda: defaultdict(list))

    for bspec in BACKBONE_SPECS:
        bname = bspec["name"]
        print(f"\n=======================================================")
        print(f"Đang đánh giá Backbone: {bspec['label']} ({bname})")
        print(f"=======================================================", flush=True)

        bench = Benchmark.load(bspec["base"])

        # Chỉ fit CURE và CURE B=1 vì các phương pháp SOTA khác (oodd, adaodd, warden, age_queue, knn)
        # đã được tính toán đầy đủ và lưu trữ chuẩn trong r61_dir/*.jsonl
        arms = {}
        for mkey in ("cure", "cure_oebh_b1"):
            cfg = dict(METHODS[mkey])
            mname = cfg.pop("method")
            arms[mkey] = build(mname, **cfg).fit(bench)

        group_rows = defaultdict(list)
        method_stream_secs = defaultdict(list)

        # Đọc cache từ file trade-off cũ nếu có để không phải fit lại các đối thủ nặng (như WARDEN trên MobileNetV2)
        old_tradeoff_file = REPO_ROOT / "results" / "backbone_sota_tradeoff.json"
        old_tradeoff_data = json.loads(old_tradeoff_file.read_text(encoding="utf-8")) if old_tradeoff_file.exists() else {}
        old_backbone_methods = old_tradeoff_data.get("backbones", {}).get(bname, {}).get("methods", {})

        for dom in bench.domains:
            jpath = r61_dir / f"{bname}__{dom}.jsonl"
            cached = {}
            if jpath.exists():
                for ln in jpath.read_text(encoding="utf-8").splitlines():
                    if ln.strip():
                        r = json.loads(ln)
                        cached[(r["pi"], r["seed"])] = r

            for pi in (0.05, 0.2, 0.5):
                for seed in (0, 1, 2):
                    stream = bench.stream(dom, pi, seed)
                    row = cached.get((pi, seed), {"metrics": {}, "dcr": {}, "seconds": {}})
                    metrics_map = dict(row.get("metrics", {}))
                    dcr_map = dict(row.get("dcr", {}))
                    sec_map = dict(row.get("seconds", {}))

                    # CURE và CURE B=1 tính lại với quy tắc Fisher thống nhất
                    for mkey in ("cure", "cure_oebh_b1"):
                        t0 = time.perf_counter()
                        out = arms[mkey].score(stream)
                        elapsed = time.perf_counter() - t0
                        metrics_map[mkey] = detection_metrics(stream.y, out.scores)
                        if out.memory is not None:
                            dcr_map[mkey] = dcr_counts(out.memory, stream.y)
                        sec_map[mkey] = round(elapsed, 3)

                    # Nạp lại các phương pháp SOTA khác từ cache cũ nếu thiếu trong jsonl
                    for mkey in ("knn", "oodd", "adaodd", "warden", "age_queue"):
                        if mkey not in metrics_map and mkey in old_backbone_methods:
                            # Khôi phục metric từ cache cũ đã chạy thành công
                            m_prev = old_backbone_methods[mkey]
                            metrics_map[mkey] = {
                                "auroc": m_prev["auroc"] / 100.0,
                                "fpr95": m_prev["fpr95"] / 100.0,
                                "tpr_fpr5": m_prev["tpr_fpr5"] / 100.0,
                                "aupr_out": m_prev["aupr_out"] / 100.0,
                            }
                            if m_prev.get("dcr") is not None:
                                dcr_map[mkey] = [int(m_prev["dcr"]), 100, 1]

                        if mkey in sec_map:
                            method_stream_secs[mkey].append(sec_map[mkey] / len(stream) * 1000.0)

                    group_rows[(dom, stream.near, pi)].append({
                        "metrics": metrics_map,
                        "dcr": dcr_map,
                    })
                    stream.cache.clear()
            print(f"  -> Hoàn tất domain {dom} (3 pi x 3 seeds)", flush=True)
            import gc
            gc.collect()

        # Tổng hợp theo từng group (18 groups per backbone)
        b_metrics = {}
        for mkey in METHODS:
            aurocs, fpr95s, tpr5s, auprs, dcrs = [], [], [], [], []
            near_aurocs, far_aurocs = [], []
            deltas = []

            for (dom, is_near, pi), srows in group_rows.items():
                if not all(mkey in r["metrics"] for r in srows):
                    continue
                g_auroc = float(np.mean([100.0 * r["metrics"][mkey]["auroc"] for r in srows]))
                g_fpr95 = float(np.mean([100.0 * r["metrics"][mkey]["fpr95"] for r in srows]))
                g_tpr5 = float(np.mean([100.0 * r["metrics"][mkey]["tpr_fpr5"] for r in srows]))
                g_aupr = float(np.mean([100.0 * r["metrics"][mkey]["aupr_out"] for r in srows]))
                knn_auroc = float(np.mean([100.0 * r["metrics"]["knn"]["auroc"] for r in srows]))

                # DCR: dcr_counts trả [n_id, total_mem_slots, n_batches]
                g_dcr_vals = []
                for r in srows:
                    if mkey in r["dcr"] and len(r["dcr"][mkey]) >= 2 and r["dcr"][mkey][1] > 0:
                        g_dcr_vals.append(100.0 * r["dcr"][mkey][0] / r["dcr"][mkey][1])
                    elif mkey in r["dcr"]:
                        g_dcr_vals.append(0.0)
                if g_dcr_vals:
                    dcrs.append(float(np.mean(g_dcr_vals)))

                delta = g_auroc - knn_auroc
                aurocs.append(g_auroc)
                fpr95s.append(g_fpr95)
                tpr5s.append(g_tpr5)
                auprs.append(g_aupr)
                deltas.append(delta)

                if is_near:
                    near_aurocs.append(g_auroc)
                    global_near_metrics[mkey]["auroc"].append(g_auroc)
                    global_near_metrics[mkey]["fpr95"].append(g_fpr95)
                    global_near_metrics[mkey]["delta"].append(delta)
                else:
                    far_aurocs.append(g_auroc)
                    global_far_metrics[mkey]["auroc"].append(g_auroc)
                    global_far_metrics[mkey]["fpr95"].append(g_fpr95)
                    global_far_metrics[mkey]["tpr5"].append(g_tpr5)
                    global_far_metrics[mkey]["delta"].append(delta)

                global_group_metrics[mkey]["auroc"].append(g_auroc)
                global_group_metrics[mkey]["fpr95"].append(g_fpr95)
                global_group_metrics[mkey]["tpr5"].append(g_tpr5)
                global_group_metrics[mkey]["aupr"].append(g_aupr)
                global_group_metrics[mkey]["delta"].append(delta)
                if g_dcr_vals:
                    global_group_metrics[mkey]["dcr"].append(float(np.mean(g_dcr_vals)))

            harm_rate = float(np.mean([d < -0.5 for d in deltas]) * 100.0) if deltas else 0.0
            worst_delta = float(np.min(deltas)) if deltas else 0.0
            b_metrics[mkey] = {
                "auroc": round(float(np.mean(aurocs)), 2),
                "near_auroc": round(float(np.mean(near_aurocs)), 2),
                "far_auroc": round(float(np.mean(far_aurocs)), 2),
                "fpr95": round(float(np.mean(fpr95s)), 2),
                "tpr_fpr5": round(float(np.mean(tpr5s)), 2),
                "aupr_out": round(float(np.mean(auprs)), 2),
                "dcr": round(float(np.mean(dcrs)), 2) if dcrs else None,
                "delta_knn": round(float(np.mean(deltas)), 2),
                "harm_rate": round(harm_rate, 1),
                "worst_delta": round(worst_delta, 2),
                "ms_per_sample": round(float(np.mean(method_stream_secs[mkey])), 4) if method_stream_secs[mkey] else None,
            }

        # Lấy thông tin latency phần cứng từ round61/latency.json
        c_cpu1 = lat_data.get(f"{bname}@cpu@1", {})
        c_gpu1 = lat_data.get(f"{bname}@cuda@1", {})
        c_gpu128 = lat_data.get(f"{bname}@cuda@128", {})

        def calc_tot(rec, b):
            if not rec:
                return None
            return round(
                (rec.get("read_ms", 0.0) + rec.get("preprocess_ms", 0.0)
                 + rec.get("backbone", {}).get("p50_ms", 0.0)
                 + rec.get("cure", {}).get("p50_ms", 0.0)) / b, 3
            )

        summary_out["backbones"][bname] = {
            "label": bspec["label"],
            "dim": bspec["dim"],
            "train_stats": get_train_stats(bspec["train_dir"]),
            "e2e_latency_ms": {
                "cpu_b1_total_ms": calc_tot(c_cpu1, 1),
                "cpu_b1_backbone_ms": round(c_cpu1.get("backbone", {}).get("p50_ms", 0.0), 3) if c_cpu1 else None,
                "cpu_b1_cure_ms": round(c_cpu1.get("cure", {}).get("p50_ms", 0.0), 3) if c_cpu1 else None,
                "gpu_b1_total_ms": calc_tot(c_gpu1, 1),
                "gpu_b128_per_sample_ms": calc_tot(c_gpu128, 128),
            },
            "methods": b_metrics,
        }

    # Global aggregation across all 7 backbones (126 groups, 378 streams)
    for mkey in METHODS:
        gm = global_group_metrics[mkey]
        nm = global_near_metrics[mkey]
        fm = global_far_metrics[mkey]
        deltas = gm["delta"]
        summary_out["global_by_method"][mkey] = {
            "label": METHOD_LABELS[mkey],
            "groups": len(gm["auroc"]),
            "auroc": round(float(np.mean(gm["auroc"])), 2),
            "near_auroc": round(float(np.mean(nm["auroc"])), 2),
            "far_auroc": round(float(np.mean(fm["auroc"])), 2),
            "fpr95": round(float(np.mean(gm["fpr95"])), 2),
            "far_fpr95": round(float(np.mean(fm["fpr95"])), 2),
            "tpr_fpr5": round(float(np.mean(gm["tpr5"])), 2),
            "far_tpr_fpr5": round(float(np.mean(fm["tpr5"])), 2),
            "aupr_out": round(float(np.mean(gm["aupr"])), 2),
            "dcr": round(float(np.mean(gm["dcr"])), 2) if gm["dcr"] else None,
            "delta_knn": round(float(np.mean(deltas)), 2),
            "harm_rate": round(float(np.mean([d < -0.5 for d in deltas]) * 100.0), 1),
            "worst_delta": round(float(np.min(deltas)), 2),
        }

    out_json = REPO_ROOT / "results" / "backbone_sota_tradeoff.json"
    out_json.write_text(json.dumps(summary_out, indent=2, ensure_ascii=False), encoding="utf-8")

    # Render Markdown report
    lines = [
        "# Đánh giá Toàn diện Trade-off: 7 Backbone Tự Huấn Luyện & Nền Tảng × 7 Phương Pháp SOTA (126 Nhóm, 378 Luồng)",
        "",
        "## 1. Bảng Tổng Hợp Toàn Cục Trên Tất Cả 7 Backbones (126 Nhóm = 7 Backbones × 18 Nhóm/Backbone)",
        "",
        "| Phương pháp | AUROC (%) ↑ | Near-OOD (%) ↑ | Far-OOD (%) ↑ | FPR95 (%) ↓ | TPR@FPR5 (%) ↑ | AUPR-Out (%) ↑ | DCR (%) ↓ | Δ kNN (%) | Harm Rate (%) ↓ | Worst Δ (%) |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for mkey in ("knn", "cure", "cure_oebh_b1", "oodd", "adaodd", "warden", "age_queue"):
        g = summary_out["global_by_method"][mkey]
        dcr_str = f"{g['dcr']:.1f}%" if g["dcr"] is not None else "-"
        lines.append(
            f"| **{g['label']}** | {g['auroc']:.2f} | {g['near_auroc']:.2f} | {g['far_auroc']:.2f} | "
            f"{g['fpr95']:.2f} | {g['tpr_fpr5']:.2f} | {g['aupr_out']:.2f} | {dcr_str} | "
            f"{g['delta_knn']:+.2f} | **{g['harm_rate']:.1f}%** | {g['worst_delta']:+.2f} |"
        )

    lines.extend([
        "",
        "## 2. Phân Tích Trade-Off: Chi Phí Huấn Luyện, Kích Thước, Độ Trễ Toàn Trình (End-to-End) & Hiệu Năng",
        "",
        "| Backbone | Chiều (d) | Cỡ Ckpt (MB) | Thời Gian Train (phút) | Độ Trễ CPU B=1 (ms) | Độ Trễ GPU B=1 (ms) | Độ Trễ GPU B=128 (ms/mẫu) | kNN AUROC | CURE AUROC | CURE B=1 AUROC | OODD AUROC | AdaODD* AUROC | WARDEN* AUROC | AgeQueue AUROC |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])
    for bspec in BACKBONE_SPECS:
        bname = bspec["name"]
        binfo = summary_out["backbones"][bname]
        ts = binfo["train_stats"]
        lat = binfo["e2e_latency_ms"]
        m = binfo["methods"]
        t_min = f"{ts['train_minutes']:.1f}p ({ts['epochs']} ep)" if ts["epochs"] else "Pretrained"
        lines.append(
            f"| **{binfo['label']}** | {binfo['dim']} | {ts['ckpt_mb']:.1f} | {t_min} | "
            f"{lat['cpu_b1_total_ms']} | {lat['gpu_b1_total_ms']} | {lat['gpu_b128_per_sample_ms']} | "
            f"{m['knn']['auroc']:.2f} | **{m['cure']['auroc']:.2f}** | **{m['cure_oebh_b1']['auroc']:.2f}** | "
            f"{m['oodd']['auroc']:.2f} | {m['adaodd']['auroc']:.2f} | {m['warden']['auroc']:.2f} | {m['age_queue']['auroc']:.2f} |"
        )

    lines.extend([
        "",
        "## 3. So Sánh Độ An Toàn (Harm Rate %) và Giảm Báo Động Giả (FPR95 %) Theo Từng Backbone",
        "",
        "| Backbone | CURE FPR95 (Harm%) | CURE B=1 FPR95 (Harm%) | kNN FPR95 | OODD FPR95 (Harm%) | AdaODD* FPR95 (Harm%) | WARDEN* FPR95 (Harm%) | AgeQueue FPR95 (Harm%) |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])
    for bspec in BACKBONE_SPECS:
        bname = bspec["name"]
        binfo = summary_out["backbones"][bname]
        m = binfo["methods"]
        lines.append(
            f"| **{binfo['label']}** | **{m['cure']['fpr95']:.1f}% ({m['cure']['harm_rate']:.0f}%)** | "
            f"**{m['cure_oebh_b1']['fpr95']:.1f}% ({m['cure_oebh_b1']['harm_rate']:.0f}%)** | "
            f"{m['knn']['fpr95']:.1f}% | "
            f"{m['oodd']['fpr95']:.1f}% ({m['oodd']['harm_rate']:.0f}%) | "
            f"{m['adaodd']['fpr95']:.1f}% ({m['adaodd']['harm_rate']:.0f}%) | "
            f"{m['warden']['fpr95']:.1f}% ({m['warden']['harm_rate']:.0f}%) | "
            f"{m['age_queue']['fpr95']:.1f}% ({m['age_queue']['harm_rate']:.0f}%) |"
        )

    out_md = REPO_ROOT / "results" / "backbone_sota_tradeoff.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[DONE] Đã lưu báo cáo vào {out_md} và {out_json}", flush=True)


if __name__ == "__main__":
    main()
