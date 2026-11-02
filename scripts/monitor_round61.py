"""Tự động theo dõi tiến độ hai kernel Kaggle vòng 61 và thực thi toàn bộ pipeline khi hoàn thành.

Quy trình:
1. Kiểm tra trạng thái cure-round61a và cure-round61b.
2. Tải r61_out.zip, giải nén checkpoint, đặc trưng .npz và benchmark config.
3. Chạy đánh giá lưới round61 (cure-ood run round61).
4. Đo độ trễ từng tầng (scripts/round61_latency.py).
5. Phân tích biên Pareto và kiểm định H61a-d (scripts/analyze_round61.py).
6. Ghi nhận kết quả vào docs/CURE_Causal_Research_Log.md.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kaggle.api.kaggle_api_extended import KaggleApi

KERNELS = {
    "cure-round61a": "tnglxun3chh/cure-round61a",
    "cure-round61b": "tnglxun3chh/cure-round61b",
}


def check_and_process():
    sys.stdout.reconfigure(encoding="utf-8")
    api = KaggleApi()
    api.authenticate()

    statuses = {}
    for name, slug in KERNELS.items():
        s = api.kernels_status(slug)
        statuses[name] = s.status
        print(f"[{name}] Trạng thái: {s.status}")

    all_done = all(st == "KernelWorkerStatus.COMPLETE" for st in statuses.values())
    if not all_done:
        print("Các kernel đang chạy. Tiếp tục chờ...")
        return False

    print("\n=== CẢ HAI KERNEL ĐÃ HOÀN THÀNH! BẮT ĐẦU ĐỒNG BỘ VÀ ĐÁNH GIÁ ===")
    out_dir = ROOT / "results" / "kaggle"
    out_dir.mkdir(parents=True, exist_ok=True)

    feat_dir = ROOT / "data" / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    for name, slug in KERNELS.items():
        k_dir = out_dir / name
        k_dir.mkdir(parents=True, exist_ok=True)
        print(f"Đang tải output từ {slug} về {k_dir}...")
        api.kernels_output(slug, path=str(k_dir), force=True, quiet=False)

        # Giải nén r61_out.zip nếu có
        z_file = k_dir / "r61_out.zip"
        if z_file.exists():
            print(f"Giải nén {z_file}...")
            with zipfile.ZipFile(z_file) as z:
                z.extractall(ROOT)
                # Chép features vào data/features
                for f in ROOT.glob("features/r61_*.npz"):
                    shutil.copy(f, feat_dir / f.name)

    print("\n=== CHẠY ĐÁNH GIÁ LƯỚI BENCHMARK ROUND 61 ===")
    cmd_eval = [sys.executable, "-m", "cure_ood.cli", "run", "round61"]
    subprocess.run(cmd_eval, cwd=str(ROOT), check=True)

    print("\n=== ĐO ĐỘ TRỄ TỪNG TẦNG (ROUND 61 LATENCY) ===")
    cmd_lat = [sys.executable, "scripts/round61_latency.py", "--experiment", "round61", "--out", "results/round61/latency.json"]
    subprocess.run(cmd_lat, cwd=str(ROOT), check=True)

    print("\n=== PHÂN TÍCH BIÊN PARETO VÀ KIỂM ĐỊNH H61a-d ===")
    cmd_ana = [sys.executable, "scripts/analyze_round61.py", "--results", "results/round61", "--latency", "results/round61/latency.json", "--out", "results/round61/summary.json"]
    subprocess.run(cmd_ana, cwd=str(ROOT), check=True)

    print("\n=== CẬP NHẬT BÁO CÁO DOCX ===")
    build_script = ROOT.parents[1] / "scripts" / "tools" / "build_cure_reports.py"
    if build_script.exists():
        subprocess.run([sys.executable, str(build_script), "final"], cwd=str(ROOT.parents[1]), check=True)

    print("\n=== HOÀN TẤT TOÀN BỘ VÒNG 61! ===")
    return True


if __name__ == "__main__":
    check_and_process()

