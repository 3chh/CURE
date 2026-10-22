"""Các backbone dùng trong bảng kết quả; tiền xử lý khớp từng bit với script trích đặc trưng gốc."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torchvision
from torchvision import transforms

from .base import Backbone, fc_arrays, load_state, register_backbone
from .cifar_nets import CIFAR_NETS
from .openood_nets import ResNet18_32x32, ResNet18_224x224

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
# Thống kê CIFAR như cấu hình OpenOOD v1.5; checkpoint CIFAR-100 được huấn luyện với bộ thứ hai.
CIFAR_NORMS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}
NORMS = {**CIFAR_NORMS, "imagenet": (IMAGENET_MEAN, IMAGENET_STD)}


def _require(checkpoint: Path | None, name: str) -> Path:
    if checkpoint is None:
        raise ValueError(f"backbone '{name}' cần checkpoint")
    return checkpoint


def resize_crop(resize: int, crop: int, mean, std, interpolation=transforms.InterpolationMode.BILINEAR):
    return transforms.Compose([transforms.Resize(resize, interpolation=interpolation), transforms.CenterCrop(crop),
                               transforms.ToTensor(), transforms.Normalize(mean, std)])


def _openood(name: str, model: torch.nn.Module, transform, device: torch.device) -> Backbone:
    model = model.eval().to(device)

    def encode(x):
        logits, feats = model(x, return_feature=True)
        return feats, logits

    return Backbone(name, model, transform, encode, device, fc_arrays(model.fc))


@register_backbone("resnet18_32x32")
def resnet18_32x32(checkpoint, device, norm: str = "cifar10") -> Backbone:
    """ResNet-18 CIFAR của OpenOOD; `norm` chọn thống kê chuẩn hóa của bộ ID (cifar10 | cifar100)."""
    state = load_state(_require(checkpoint, "resnet18_32x32"))
    # Checkpoint cũ (resnet18_cifar10.ckpt, SupCon) đặt tên nhánh tắt là `short`.
    state = {k.replace(".short.", ".shortcut."): v for k, v in state.items()}
    model = ResNet18_32x32(num_classes=state["fc.weight"].shape[0])
    model.load_state_dict(state, strict=True)
    return _openood("resnet18_32x32", model, resize_crop(32, 32, *CIFAR_NORMS[norm]), device)


def _cifar_net(name):
    def builder(checkpoint, device, norm: str = "cifar10") -> Backbone:
        state = load_state(_require(checkpoint, name))
        opts = {"dim": state["embed.0.weight"].shape[0]} if "embed.0.weight" in state else {}
        model = CIFAR_NETS[name](num_classes=state["fc.weight"].shape[0], **opts)
        model.load_state_dict(state, strict=True)
        return _openood(name, model, resize_crop(32, 32, *CIFAR_NORMS[norm]), device)
    builder.__doc__ = f"{name} tự huấn luyện (vòng 61); `norm` như resnet18_32x32."
    return builder


for _name in CIFAR_NETS:
    register_backbone(_name)(_cifar_net(_name))


@register_backbone("resnet18_224x224")
def resnet18_224x224(checkpoint, device) -> Backbone:
    """ResNet-18 ImageNet-200 của OpenOOD."""
    state = load_state(_require(checkpoint, "resnet18_224x224"))
    model = ResNet18_224x224(num_classes=state["fc.weight"].shape[0])
    model.load_state_dict(state, strict=True)
    return _openood("resnet18_224x224", model, resize_crop(256, 224, IMAGENET_MEAN, IMAGENET_STD), device)


@register_backbone("resnet50_in1k")
def resnet50_in1k(checkpoint, device, amp: bool = True) -> Backbone:
    """ResNet-50 torchvision cho ImageNet-1K; `amp` giữ autocast fp16 trên GPU như bản trích gốc."""
    model = torchvision.models.resnet50()
    model.load_state_dict(load_state(_require(checkpoint, "resnet50_in1k")))
    model = model.eval().to(device)
    body = torch.nn.Sequential(*list(model.children())[:-1])
    w, b = fc_arrays(model.fc)

    def encode(x):
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            feats = body(x).flatten(1).float().cpu()
        # Logit tính bằng numpy từ đặc trưng fp32 như bản gốc; tính trên GPU lệch ~1e-6.
        return feats, torch.from_numpy((feats.numpy() @ w.T + b).astype(np.float32))

    return Backbone("resnet50_in1k", model, resize_crop(256, 224, IMAGENET_MEAN, IMAGENET_STD), encode, device, (w, b))


@register_backbone("dinov2_vitb14")
def dinov2_vitb14(checkpoint, device) -> Backbone:
    """DINOv2 ViT-B/14 (token CLS) qua torch.hub; ko có logit."""
    if checkpoint is not None:
        raise ValueError("dinov2_vitb14 lấy trọng số từ torch.hub, ko nhận checkpoint")
    # Ghi rõ nhánh `main` để torch.hub dùng cache mà ko phải hỏi GitHub nhánh mặc định.
    model = torch.hub.load("facebookresearch/dinov2:main", "dinov2_vitb14")
    dtype = torch.float16 if device.type == "cuda" else torch.float32  # CPU chạy fp16 chậm và kém hỗ trợ
    model = model.eval().to(device, dtype)
    transform = resize_crop(224, 224, IMAGENET_MEAN, IMAGENET_STD, transforms.InterpolationMode.BICUBIC)
    return Backbone("dinov2_vitb14", model, transform, lambda x: (model(x.to(dtype)), None), device)
