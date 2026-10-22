"""Registry mô hình huấn luyện; mỗi mô hình gắn với một backbone trong registry trích đặc trưng."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from torch import nn

from ..backbones.cifar_nets import CIFAR_NETS
from ..backbones.openood_nets import ResNet18_32x32
from ..backbones.torchvision_net import TorchvisionNet


@dataclass
class ModelSpec:
    module: nn.Module  # forward(x, return_feature) -> logits | (logits, feature); có thuộc tính `fc`
    backbone: str  # tên trong cure_ood.backbones để trích đặc trưng từ checkpoint
    options: dict  # tùy chọn cho builder backbone, sao cho transform trùng eval_transform lúc huấn luyện


MODELS: dict[str, Callable[..., ModelSpec]] = {}


def register_model(name: str):
    """Decorator đăng ký `fn(num_classes, data, **options) -> ModelSpec`."""
    def deco(fn):
        MODELS[name] = fn
        return fn
    return deco


def build_model(name: str, num_classes: int, data: dict, **options) -> ModelSpec:
    if name not in MODELS:
        raise KeyError(f"model '{name}' chưa đăng ký; có: {sorted(MODELS)}")
    return MODELS[name](num_classes, data, **options)


@register_model("resnet18_32x32")
def resnet18_32x32(num_classes, data) -> ModelSpec:
    # Backbone resnet18_32x32 cố định Resize(32) + CenterCrop(32), nên dữ liệu phải đúng cỡ đó.
    if (data["size"], data["resize"]) != (32, 32) or data["norm"] not in ("cifar10", "cifar100"):
        raise ValueError("resnet18_32x32 cần size = resize = 32 và norm cifar10 | cifar100")
    return ModelSpec(ResNet18_32x32(num_classes=num_classes), "resnet18_32x32", {"norm": data["norm"]})


@register_model("torchvision")
def torchvision_model(num_classes, data, arch: str, weights: str | None = None) -> ModelSpec:
    """Mô hình torchvision theo tên `arch`; `weights` (vd DEFAULT) chỉ dùng để khởi tạo."""
    return ModelSpec(TorchvisionNet(arch, num_classes, weights), "torchvision",
                     {"arch": arch, "norm": data["norm"], "size": data["size"], "resize": data["resize"]})


def _cifar_model(name):
    def fn(num_classes, data, **options) -> ModelSpec:
        if (data["size"], data["resize"]) != (32, 32) or data["norm"] not in ("cifar10", "cifar100"):
            raise ValueError(f"{name} cần size = resize = 32 và norm cifar10 | cifar100")
        return ModelSpec(CIFAR_NETS[name](num_classes=num_classes, **options), name, {"norm": data["norm"]})
    return fn


for _name in CIFAR_NETS:
    register_model(_name)(_cifar_model(_name))
