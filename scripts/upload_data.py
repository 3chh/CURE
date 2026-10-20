"""Đẩy file dữ liệu mới lên Hugging Face (chỉ người bảo trì): đặc trưng/gallery vào repo dữ liệu, checkpoint vào repo mô hình.

Chạy: py scripts/upload_data.py features <file.npz ...> | checkpoints <file ...>
In sha256 của từng file để ghi vào ghi chú phát hành; repo đích lấy từ CURE_HF_DATA / CURE_HF_CHECKPOINTS.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

DATA_REPO = os.environ.get("CURE_HF_DATA", "3chh/cure-data")
CKPT_REPO = os.environ.get("CURE_HF_CHECKPOINTS", "3chh/cure-checkpoints")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(kind, files):
    repo, typ, prefix = (DATA_REPO, "dataset", "features/") if kind == "features" else (CKPT_REPO, "model", "")
    api = HfApi()
    for f in map(Path, files):
        print(f.name, sha256(f))
        api.upload_file(path_or_fileobj=str(f), path_in_repo=prefix + f.name, repo_id=repo, repo_type=typ,
                        commit_message=f"Add {f.name}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
