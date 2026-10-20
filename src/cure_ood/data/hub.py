"""Tự lấy file thiếu từ Hugging Face vào cache HF lần đầu dùng; tắt bằng CURE_HF_AUTO=0."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

DATA_REPO = os.environ.get("CURE_HF_DATA", "3chh/cure-data")
CKPT_REPO = os.environ.get("CURE_HF_CHECKPOINTS", "3chh/cure-checkpoints")


def enabled() -> bool:
    return os.environ.get("CURE_HF_AUTO", "1") != "0"


def _download(repo: str, repo_type: str, name: str) -> Path | None:
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError
    except ImportError:
        return None
    try:
        return Path(hf_hub_download(repo, name, repo_type=repo_type))
    except EntryNotFoundError:
        return None


def feature_file(name: str, local: Path) -> Path | None:
    """File đặc trưng/gallery: bản local nếu có, ko thì tải từ features/<name> trên HF; None nếu ko có ở đâu."""
    if local.exists():
        return local
    return _download(DATA_REPO, "dataset", f"features/{name}") if enabled() else None


@lru_cache(maxsize=1)
def _checkpoint_index() -> dict[str, str]:
    from huggingface_hub import list_repo_files
    return {Path(f).name: f for f in list_repo_files(CKPT_REPO, repo_type="model")}


def checkpoint_file(name: str) -> Path | None:
    """Checkpoint theo tên file; repo HF xếp theo thư mục con nên tra theo basename."""
    if not enabled():
        return None
    try:
        path = _checkpoint_index().get(Path(name).name)
    except ImportError:
        return None
    return _download(CKPT_REPO, "model", path) if path else None
