"""Dữ liệu huấn luyện từ imglist OpenOOD và các bộ tăng cường dữ liệu theo tên."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from ..backbones.builtin import NORMS, resize_crop
from ..data.extract import ListSet, Rows, env_dir, load_rows

AUGMENTS: dict[str, Callable[[int, int], list]] = {}


def register_augment(name: str):
    """Decorator đăng ký `fn(size, resize) -> list phép biến đổi PIL` (chưa gồm ToTensor/Normalize)."""
    def deco(fn):
        AUGMENTS[name] = fn
        return fn
    return deco


@register_augment("openood")
def _openood(size, resize):
    # BasePreprocessor nhánh CIFAR của OpenOOD v1.5 (openood/preprocessors/base_preprocessor.py).
    return [T.Resize(resize, interpolation=T.InterpolationMode.BILINEAR), T.CenterCrop(size),
            T.RandomHorizontalFlip(), T.RandomCrop(size, padding=4)]


@register_augment("openood_imagenet")
def _openood_imagenet(size, resize):
    return [T.RandomResizedCrop(size, interpolation=T.InterpolationMode.BILINEAR), T.RandomHorizontalFlip(0.5)]


@register_augment("supcon")
def _supcon(size, resize):
    # main_supcon.py của SupContrast (Khosla et al. 2020).
    return [T.RandomResizedCrop(size, scale=(0.2, 1.0)), T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8), T.RandomGrayscale(p=0.2)]


@register_augment("linear")
def _linear(size, resize):
    # main_ce.py / main_linear.py của SupContrast.
    return [T.RandomResizedCrop(size, scale=(0.2, 1.0)), T.RandomHorizontalFlip()]


@register_augment("none")
def _none(size, resize):
    return [T.Resize(resize, interpolation=T.InterpolationMode.BILINEAR), T.CenterCrop(size)]


class Views:
    """Áp `transform` độc lập `n` lần → tensor (n, C, H, W); là class để worker DataLoader trên Windows pickle được."""

    def __init__(self, transform, n: int):
        self.transform, self.n = transform, n

    def __call__(self, im):
        return torch.stack([self.transform(im) for _ in range(self.n)])


def train_transform(name: str, data: dict, views: int = 1) -> Views:
    if name not in AUGMENTS:
        raise KeyError(f"augment '{name}' chưa đăng ký; có: {sorted(AUGMENTS)}")
    t = T.Compose([*AUGMENTS[name](data["size"], data["resize"]), T.ToTensor(), T.Normalize(*NORMS[data["norm"]])])
    return Views(t, views)


def eval_transform(data: dict):
    """Phải trùng transform của backbone lúc trích đặc trưng, nên dùng chung resize_crop."""
    return resize_crop(data["resize"], data["size"], *NORMS[data["norm"]])


class CachedSet(ListSet):
    """Giữ ảnh đã giải mã trong RAM: giải mã PNG CIFAR chậm hơn cả bước huấn luyện trên GPU."""

    def __init__(self, rows: Rows, transform):
        super().__init__(rows, transform)
        self.images = []
        for path, _ in rows:
            with Image.open(path) as im:
                self.images.append(np.asarray(im.convert("RGB")))

    def __getitem__(self, i):
        return self.transform(Image.fromarray(self.images[i])), self.rows[i][1]


def split_rows(data: dict, split: str, seed: int) -> Rows:
    """Hàng (đường dẫn, nhãn) của tập `train` | `val` theo imglist trong CURE_IMGLISTS, ảnh trong CURE_IMAGES."""
    lists = env_dir("CURE_IMGLISTS", "data/benchmark_imglist") / data["imglists"]
    root = env_dir("CURE_IMAGES", "data/images") / data["image_root"]
    subsample = data.get("subsample") if split == "train" else data.get("val_subsample")
    return load_rows(lists / f"{data[split]}.txt", root, subsample=subsample, seed=seed)


def make_set(rows: Rows, transform, cache: bool) -> ListSet:
    return CachedSet(rows, transform) if cache else ListSet(rows, transform)
