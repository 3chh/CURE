"""Merge seeds 0-19 across all 81 groups into results/master20_81."""
import os
import shutil
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/master20_81"
OUT.mkdir(parents=True, exist_ok=True)

SRC_DIRS = [
    REPO / "results/main81",
    REPO / "results/seed35_81",
    REPO / "results/kaggle/seed69_81/seed69_81",
    REPO / "results/seed69_81",
    REPO / "results/kaggle/seed1019_81/seed1019_81",
    REPO / "results/seed1019_81",
]

# find all unique jsonl file names
all_jsonls = set()
for d in SRC_DIRS:
    if d.exists():
        for f in d.glob("*.jsonl"):
            all_jsonls.add(f.name)

print(f"Found {len(all_jsonls)} distinct benchmark jsonl targets to merge.")

for name in sorted(all_jsonls):
    out_file = OUT / name
    merged_lines = []
    seen_keys = set()
    for d in SRC_DIRS:
        f = d / name
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                # deduplicate by line or seed
                merged_lines.append(line)
    out_file.write_text("\n".join(merged_lines) + "\n", encoding="utf-8")

print(f"Merged {len(all_jsonls)} files into {OUT}.")
print("Running report generation...")
subprocess.run([sys.executable, "-m", "cure_ood.cli", "report", "master20_81"], cwd=REPO, check=False)
print("Done master20_81!")
