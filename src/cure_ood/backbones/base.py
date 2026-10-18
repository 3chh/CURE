"""Giao diện backbone và registry: tên → hàm dựng, để file cấu hình trích đặc trưng chỉ cần ghi tên."""
from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
_BUILDERS: dict[str, Callable[..., Backbone]] = {}

Encode = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor | None]]


@dataclass
class Backbone:
    name: str
    model: torch.nn.Module
    transform: Callable  # ảnh PIL → tensor, đúng tiền xử lý lúc huấn luyện checkpoint
    encode: Encode  # batch đã chuyển lên device → (đặc trưng, logit hoặc None)
    device: torch.device
    fc: tuple[np.ndarray, np.ndarray] | None = None  # (weight, bias) của tầng phân loại

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Trả đặc trưng và logit float32 trên CPU."""
        feats, logits = self.encode(images.to(self.device))
        return feats.float().cpu(), None if logits is None else logits.float().cpu()


def register_backbone(name: str):
    """Decorator đăng ký hàm dựng `fn(checkpoint, device, **options) -> Backbone`."""
    def deco(fn):
        if name in _BUILDERS:
            raise ValueError(f"backbone '{name}' đã được đăng ký")
        _BUILDERS[name] = fn
        return fn
    return deco


def available_backbones() -> list[str]:
    return sorted(_BUILDERS)


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_backbone(name: str, checkpoint: str | Path | None = None, device: str | torch.device | None = None,
                   **options) -> Backbone:
    if name not in _BUILDERS:
        raise KeyError(f"backbone '{name}' chưa đăng ký; có: {available_backbones()}")
    dev = torch.device(device) if device is not None else default_device()
    return _BUILDERS[name](checkpoint=None if checkpoint is None else resolve_checkpoint(checkpoint), device=dev, **options)


def resolve_checkpoint(checkpoint: str | Path) -> Path:
    """Đường dẫn tương đối được tìm trong CURE_CHECKPOINTS (nhiều thư mục, ngăn bởi os.pathsep), mặc định checkpoints/."""
    p = Path(checkpoint)
    if p.is_absolute() or p.exists():
        return p
    dirs = os.environ.get("CURE_CHECKPOINTS", str(REPO_ROOT / "checkpoints")).split(os.pathsep)
    for d in dirs:
        if (Path(d) / p).exists():
            return Path(d) / p
    raise FileNotFoundError(f"ko thấy checkpoint '{checkpoint}' trong {dirs}")


def load_state(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def fc_arrays(fc: torch.nn.Linear) -> tuple[np.ndarray, np.ndarray]:
    return fc.weight.detach().cpu().numpy().astype(np.float32), fc.bias.detach().cpu().numpy().astype(np.float32)
