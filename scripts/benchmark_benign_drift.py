"""Đánh giá thực nghiệm Phản chứng: Trôi dạt Lành tính (Benign Concept Drift / Covariate Shift).
Thử nghiệm trên cả 7 Backbones: Luồng dữ liệu xuất hiện sự biến dạng lành tính (ví dụ: nhiễu môi trường,
thay đổi ánh sáng/nhiễu cảm biến) trên chính lớp ID, so sánh xem CURE có nhận diện đúng là ID hay bị
báo động giả hàng loạt (FPR95 vọt lên) như các phương pháp thích nghi không kiểm duyệt khác.

Chạy:
    python deliverables/CURE_repo/scripts/benchmark_benign_drift.py
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
from cure_ood.eval.metrics import detection_metrics
from cure_ood.methods.base import build
from cure_ood.ops import l2

BACKBONE_SPECS = [
    {"name": "r61_resnet20_ivf4", "base": "r61_resnet20", "label": "ResNet-20 (64-d)"},
    {"name": "r61_resnet18embed128_ivf4", "base": "r61_resnet18embed128", "label": "ResNet-18-Bottleneck (128-d)"},
    {"name": "r61_mobilenetv2_ivf4", "base": "r61_mobilenetv2", "label": "MobileNetV2 (1280-d)"},
    {"name": "r61_resnet18_s0_ivf4", "base": "r61_resnet18_s0", "label": "ResNet-18 Seed 0 (512-d)"},
    {"name": "r61_dinov2_ivf4", "base": "cifar10_dinov2", "label": "DINOv2 ViT-B/14 (768-d)"},
]

METHODS = {
    "knn": {"method": "knn"},
    "cure": {"method": "cure", "rule": "fisher"},
    "cure_oebh_b1": {"method": "cure", "certifier": "online_ebh", "batch": 1, "rule": "fisher"},
    "oodd": {"method": "oodd", "gallery": "approx"},
    "warden": {"method": "warden"},
    "age_queue": {"method": "age_queue", "ttl": 512},
}

def make_benign_drift_stream(base_stream, sigma: float = 0.08, drift_frac: float = 0.35, seed: int = 42):
    """Mô phỏng trôi dạt lành tính: một phần mẫu ID bị biến dạng/nhiễu môi trường nhưng vẫn là ID hợp lệ (y=0)."""
    rng = np.random.default_rng(seed)
    x = np.copy(base_stream.x)
    y = np.copy(base_stream.y)
    
    # Chỉ làm trôi dạt trên các mẫu ID (y == 0)
    id_indices = np.flatnonzero(y == 0)
    drift_indices = rng.choice(id_indices, size=int(len(id_indices) * drift_frac), replace=False)
    
    # Thêm nhiễu Gaussian nhẹ (Covariate shift) và chuẩn hóa lại L2
    noise = rng.normal(0.0, sigma, size=(len(drift_indices), x.shape[1])).astype(np.float32)
    x[drift_indices] = l2(x[drift_indices] + noise)
    
    class ShiftedStream:
        def __init__(self, x, y, orig_stream):
            self.x = x
            self.feats = x
            self.y = y
            self.bench = orig_stream.bench
            self.domain = orig_stream.domain
            self.near = orig_stream.near
            self.cache = {}
            self.logits = None
        def __len__(self):
            return len(self.y)
            
    return ShiftedStream(x, y, base_stream)

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("="*70)
    print("KHỞI TẠO THỰC NGHIỆM PHẢN CHỨNG: TRÔI DẠT LÀNH TÍNH (BENIGN COVARIATE SHIFT)")
    print("="*70, flush=True)

    results = {"backbones": {}, "global_summary": {}}
    global_metrics = defaultdict(lambda: defaultdict(list))

    for bspec in BACKBONE_SPECS:
        bname = bspec["name"]
        blabel = bspec["label"]
        print(f"\n[Backbone] {blabel}...", flush=True)
        bench = Benchmark.load(bspec["base"])
        
        arms = {}
        for mkey, mcfg in METHODS.items():
            cfg = dict(mcfg)
            mname = cfg.pop("method")
            if mname == "warden":
                cfg["k"] = bench.k_knn
            print(f"  Fitting {mkey}...", flush=True)
            arms[mkey] = build(mname, **cfg).fit(bench)

        b_res = defaultdict(lambda: defaultdict(list))
        # Thử nghiệm trên 2 domain đại diện: svhn (far) và cifar100 (near) với 3 seed
        for dom in ("svhn", "cifar100"):
            for seed in (0, 1, 2):
                st_orig = bench.stream(dom, 0.2, seed)
                st_drift = make_benign_drift_stream(st_orig, sigma=0.08, drift_frac=0.35, seed=seed)
                
                for mkey, arm in arms.items():
                    out = arm.score(st_drift)
                    det = detection_metrics(st_drift.y, out.scores)
                    b_res[mkey]["auroc"].append(100.0 * det["auroc"])
                    b_res[mkey]["fpr95"].append(100.0 * det["fpr95"])
                    b_res[mkey]["tpr5"].append(100.0 * det["tpr_fpr5"])

        b_summary = {}
        for mkey in METHODS:
            m_auc = float(np.mean(b_res[mkey]["auroc"]))
            m_fpr = float(np.mean(b_res[mkey]["fpr95"]))
            m_tpr = float(np.mean(b_res[mkey]["tpr5"]))
            b_summary[mkey] = {"auroc": round(m_auc, 2), "fpr95": round(m_fpr, 2), "tpr5": round(m_tpr, 2)}
            global_metrics[mkey]["auroc"].append(m_auc)
            global_metrics[mkey]["fpr95"].append(m_fpr)
            global_metrics[mkey]["tpr5"].append(m_tpr)

        results["backbones"][bname] = {"label": blabel, "methods": b_summary}
        print(f"  kNN: AUROC={b_summary['knn']['auroc']:.2f}% | FPR95={b_summary['knn']['fpr95']:.2f}%")
        print(f"  CURE: AUROC={b_summary['cure']['auroc']:.2f}% | FPR95={b_summary['cure']['fpr95']:.2f}%")
        print(f"  OODD: AUROC={b_summary['oodd']['auroc']:.2f}% | FPR95={b_summary['oodd']['fpr95']:.2f}%")

    # Global summary
    for mkey in METHODS:
        results["global_summary"][mkey] = {
            "auroc": round(float(np.mean(global_metrics[mkey]["auroc"])), 2),
            "fpr95": round(float(np.mean(global_metrics[mkey]["fpr95"])), 2),
            "tpr5": round(float(np.mean(global_metrics[mkey]["tpr5"])), 2),
        }

    out_file = REPO_ROOT / "results" / "benign_drift_evaluation.json"
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    
    # Viết báo cáo Markdown
    md = [
        "# Thực Nghiệm Phản Chứng: Đánh Giá Khả Năng Chống Báo Động Giả Khi Trôi Dạt Lành Tính (Benign Covariate Shift)",
        "",
        "## 1. Bảng Tổng Hợp Khi 35% Mẫu ID Bị Trôi Dạt / Nhiễu Cảm Biến Môi Trường",
        "",
        "| Phương pháp | AUROC (%) ↑ | FPR95 (%) ↓ (Báo động giả) | TPR@FPR5 (%) ↑ | Độ tăng FPR95 so với kNN (càng thấp càng tốt) |",
        "|---|:---:|:---:|:---:|:---:|",
    ]
    knn_fpr = results["global_summary"]["knn"]["fpr95"]
    for mkey in METHODS:
        g = results["global_summary"][mkey]
        delta_fpr = g["fpr95"] - knn_fpr
        md.append(f"| **{mkey.upper()}** | {g['auroc']:.2f} | **{g['fpr95']:.2f}%** | {g['tpr5']:.2f}% | **{delta_fpr:+.2f}%** |")
        
    out_md = REPO_ROOT / "results" / "benign_drift_evaluation.md"
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n[DONE] Đã lưu báo cáo phản chứng trôi dạt lành tính vào {out_md}", flush=True)

if __name__ == "__main__":
    main()

