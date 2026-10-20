"""Chép bản tóm tắt kết quả (report.md, summary.json) của các thí nghiệm đã chạy vào reports/ để commit cùng repo.

Chạy: py scripts/export_reports.py [thí nghiệm ...]   (mặc định: mọi thư mục trong results/ có report.md)
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(names):
    for d in sorted((ROOT / "results").iterdir()):
        if not (d / "report.md").exists() or (names and d.name not in names):
            continue
        out = ROOT / "reports" / d.name
        out.mkdir(parents=True, exist_ok=True)
        for f in ["report.md", "summary.json", *(p.name for p in d.glob("latency_*"))]:
            if (d / f).exists():
                shutil.copy2(d / f, out / f)
        print("ok", out)


if __name__ == "__main__":
    main(sys.argv[1:])
