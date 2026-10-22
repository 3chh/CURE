"""Chạy cả một thí nghiệm, mỗi (benchmark, tập lạ, nhóm phương pháp) một tiến trình con để giữ RAM thấp; tự thử lại khi lỗi.

Chạy: py scripts/run_all.py <thí nghiệm> [--group N] [--retries 3] [--benchmark ...] [--method ...] [--force]
Kết quả theo luồng và theo phương pháp nên chạy lại lệnh sẽ tiếp từ chỗ dừng.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load(kind: str, name: str) -> dict:
    """Đọc config trực tiếp: tiến trình cha ko nạp torch (~1,5 GB) vì chỉ cần danh sách việc."""
    path = Path(name) if Path(name).suffix else ROOT / "configs" / kind / f"{name}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--group", type=int, default=0, help="số phương pháp mỗi tiến trình (0 = tất cả)")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--benchmark", nargs="*")
    ap.add_argument("--method", nargs="*")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # log chuyển hướng ra file trên Windows mặc định cp1252
    exp = load("experiments", a.experiment)
    labels = [m for m in exp["methods"] if not a.method or m in a.method]
    size = a.group or len(labels)
    groups = [labels[i:i + size] for i in range(0, len(labels), size)]
    failed = []
    specs = {b if isinstance(b, str) else b["name"]: b for b in exp["benchmarks"]}
    for bname in a.benchmark or specs:
        spec = specs.get(bname, bname)
        bench = load("benchmarks", spec if isinstance(spec, str) else spec["base"])
        for dom in exp.get("domains") or bench["near"] + bench["far"]:
            for g in groups:
                cmd = [sys.executable, "-m", "cure_ood.cli", "run", a.experiment, "--benchmark", bname, "--domain", dom,
                       "--method", *g] + (["--force"] if a.force else [])
                for t in range(1, a.retries + 1):
                    if subprocess.run(cmd, check=False).returncode == 0:
                        break
                    print(f"[run_all] lỗi {bname}/{dom} {g} (lần {t})", flush=True)
                    time.sleep(30)  # chờ RAM/GPU được giải phóng
                else:
                    failed.append((bname, dom, g))
    print("[run_all] xong" + (f"; còn lỗi: {failed}" if failed else ""), flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
