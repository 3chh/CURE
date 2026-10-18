"""Mô hình torchvision bất kỳ (theo tên) với tầng phân loại riêng, dùng cho checkpoint tự huấn luyện."""
from __future__ import annotations

import torch
import torchvision
from torch import nn

from .base import Backbone, load_state, register_backbone
from .builtin import NORMS, _openood, _require, resize_crop


class TorchvisionNet(nn.Module):
    """Thân torchvision bỏ tầng Linear cuối, `forward` giống mạng OpenOOD để dùng chung đường trích."""

    def __init__(self, arch: str, num_classes: int, weights: str | None = None):
        super().__init__()
        body = torchvision.models.get_model(arch, weights=weights)
        # Đầu phân loại nằm ở chỗ khác nhau tùy họ mạng (fc, classifier.N, heads.head): lấy tầng Linear cuối cùng.
        name, head = [(n, m) for n, m in body.named_modules() if isinstance(m, nn.Linear)][-1]
        parent, _, leaf = name.rpartition(".")
        setattr(body.get_submodule(parent), leaf, nn.Identity())
        self.body, self.fc = body, nn.Linear(head.in_features, num_classes)

    def forward(self, x: torch.Tensor, return_feature: bool = False):
        feature = self.body(x).flatten(1)
        logits = self.fc(feature)
        return (logits, feature) if return_feature else logits


@register_backbone("torchvision")
def torchvision_net(checkpoint, device, arch: str, norm: str = "imagenet", size: int = 224, resize: int = 256) -> Backbone:
    """Checkpoint của `cure-ood train` với model torchvision; `norm`, `size`, `resize` phải khớp cấu hình huấn luyện."""
    state = load_state(_require(checkpoint, "torchvision"))
    model = TorchvisionNet(arch, num_classes=state["fc.weight"].shape[0])
    model.load_state_dict(state, strict=True)
    return _openood("torchvision", model, resize_crop(resize, size, *NORMS[norm]), device)
