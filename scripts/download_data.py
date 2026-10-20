"""Tải dữ liệu từ Hugging Face về đúng chỗ các config đang tìm (data/features, data/images, data/benchmark_imglist, checkpoints).

Repo dữ liệu để private vì license ImageNet/Places/TinyImageNet: cần `huggingface-cli login` và quyền truy cập.
Chạy: py scripts/download_data.py [features] [oodd] [checkpoints] [imglists] [images [tên ...]]   (mặc định: features oodd)
"""
from __future__ import annotations

import os
import sys
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA_REPO = os.environ.get("CURE_HF_DATA", "3chh/cure-data")
CKPT_REPO = os.environ.get("CURE_HF_CHECKPOINTS", "3chh/cure-checkpoints")
FEATURES = ["resnet18_official_v15.npz", "resnet18_cifar100_v15.npz", "resnet18_in200_v15.npz",  # bảng chính
            "resnet18_supcon_v15.npz", "dino_cifar10_v15.npz",  # độ bền theo backbone
            "cifar10c_official.npz"]  # EDA trôi dạt
OODD_GALLERIES = ["oodd_gallery_cifar10.npz", "oodd_gallery_cifar100.npz", "oodd_gallery_cifar10_supcon.npz"]  # 5 crop, scripts/build_oodd_gallery.py


def fetch(repo, repo_type, name, out: Path) -> Path | None:
    """Tải một file về `out`; bỏ qua nếu đã có, trả None nếu repo ko có file đó."""
    if out.exists():
        print("skip", out)
        return out
    try:
        p = hf_hub_download(repo, name, repo_type=repo_type, local_dir=out.parent / ".hf")
    except Exception as e:  # noqa: BLE001  (ko có file trên HF thì báo rồi đi tiếp)
        print(f"MISS {repo}:{name} ({type(e).__name__})")
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    os.replace(p, out)
    print("ok  ", out)
    return out


def untar(tar_path: Path, dest: Path):
    """Giải nén rồi xóa file tar để ko tốn gấp đôi chỗ."""
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as t:
        t.extractall(dest, filter="data")
    tar_path.unlink()
    print("untar", dest)


def images(names):
    """Ảnh OpenOOD: images/openood_<classic|largescale>/<bộ>/part-*.tar → data/images/images_<...>/<bộ>/."""
    files = [f for f in list_repo_files(DATA_REPO, repo_type="dataset") if f.startswith("images/") and f.endswith(".tar")]
    for f in files:
        _, group, name, part = f.split("/")
        if names and name not in names:
            continue
        dest = DATA / "images" / group.replace("openood_", "images_") / name
        if (dest / f".{part}.done").exists():
            continue
        tar = fetch(DATA_REPO, "dataset", f, DATA / "images" / ".tar" / f)
        if tar:
            untar(tar, dest)
            (dest / f".{part}.done").touch()


def main(what):
    if "features" in what:
        for f in FEATURES:
            fetch(DATA_REPO, "dataset", f"features/{f}", DATA / "features" / f)
    if "oodd" in what:  # thiếu thì OODD trên CIFAR báo lỗi thay vì chạy với gallery xấp xỉ
        for f in OODD_GALLERIES:
            fetch(DATA_REPO, "dataset", f"features/{f}", DATA / "features" / f)
    if "checkpoints" in what:
        for f in list_repo_files(CKPT_REPO, repo_type="model"):
            if f.endswith((".ckpt", ".pt", ".pth")):  # openood_official/x.ckpt → checkpoints/x.ckpt, tên config dùng
                fetch(CKPT_REPO, "model", f, ROOT / "checkpoints" / Path(f).name)
    if "imglists" in what:
        tar = fetch(DATA_REPO, "dataset", "benchmark_imglist/part-0000.tar", DATA / ".tar" / "benchmark_imglist.tar")
        if tar:
            untar(tar, DATA / "benchmark_imglist")
    if "images" in what:  # ~22 GB nếu lấy hết; chỉ cần khi trích lại đặc trưng hoặc dựng gallery OODD
        images(what[what.index("images") + 1:])


if __name__ == "__main__":
    main(sys.argv[1:] or ["features", "oodd"])
