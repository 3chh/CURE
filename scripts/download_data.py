"""Tải đặc trưng, checkpoint (và tùy chọn ảnh) từ Hugging Face về data/ và checkpoints/.

Repo dữ liệu để private vì license ImageNet/Places/TinyImageNet: cần `huggingface-cli login` và quyền truy cập.
Chạy: py scripts/download_data.py [features] [checkpoints] [images]   (mặc định: features)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

ROOT = Path(__file__).resolve().parents[1]
DATA_REPO = os.environ.get("CURE_HF_DATA", "3chh/cure-data")
CKPT_REPO = os.environ.get("CURE_HF_CHECKPOINTS", "3chh/cure-checkpoints")
FEATURES = ["resnet18_official_v15.npz", "resnet18_cifar100_v15.npz", "resnet18_in200_v15.npz",  # bảng chính
            "resnet18_supcon_v15.npz", "dino_cifar10_v15.npz",  # độ bền theo backbone
            "cifar10c_official.npz"]  # EDA trôi dạt


def fetch(repo, repo_type, names, dest: Path, strip=""):
    dest.mkdir(parents=True, exist_ok=True)
    for name in names:
        out = dest / name.removeprefix(strip)
        if out.exists():
            print("skip", out)
            continue
        p = hf_hub_download(repo, name, repo_type=repo_type, local_dir=dest / ".hf")
        out.parent.mkdir(parents=True, exist_ok=True)
        os.replace(p, out)
        print("ok  ", out)


def main(what):
    if "features" in what:
        fetch(DATA_REPO, "dataset", [f"features/{f}" for f in FEATURES], ROOT / "data" / "features", "features/")
    if "checkpoints" in what:
        fetch(CKPT_REPO, "model", list_repo_files(CKPT_REPO, repo_type="model"), ROOT / "checkpoints")
    if "images" in what:  # ~15 GB, chỉ cần khi trích lại đặc trưng
        names = [f for f in list_repo_files(DATA_REPO, repo_type="dataset") if f.startswith("images/")]
        fetch(DATA_REPO, "dataset", names, ROOT / "data")


if __name__ == "__main__":
    main(sys.argv[1:] or ["features"])
