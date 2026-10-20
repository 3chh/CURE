"""Dựng gallery "informative" của OODD từ ảnh train: 5 crop ngẫu nhiên/ảnh, giữ crop có MSP cao nhất, xếp theo lớp.

Đúng KNNPostprocessor.setup của code OODD chính thức (MultiRandomCrop, normalizer, informative_order đều vendor nguyên văn).
Chạy: py scripts/build_oodd_gallery.py cifar10|cifar100|cifar10_supcon [--image-root DIR] [--imglists DIR]
Ra: <CURE_FEATURES>/oodd_gallery_<bộ>.npz với activation_log (n, d) và idx (thứ tự informative).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from cure_ood.backbones import build_backbone
from cure_ood.backbones.builtin import CIFAR_NORMS
from cure_ood.data.benchmark import data_root
from cure_ood.data.extract import ListSet, load_rows
from cure_ood.methods.testtime._official.oodd_knn import MultiRandomCrop, informative_order, normalizer

ROOT = Path(__file__).resolve().parents[1]
# benchmark → (checkpoint, imglist của bộ ID, chuẩn hóa)
SETS = {"cifar10": ("resnet18_cifar10.ckpt", "cifar10", "cifar10"), "cifar100": ("resnet18_cifar100.ckpt", "cifar100", "cifar100"),
        "cifar10_supcon": ("resnet18_supcon.pt", "cifar10", "cifar10")}
SEED, NUM_CROP = 20261004, 5


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench", choices=sorted(SETS))
    ap.add_argument("--image-root", default=Path(os.environ.get("CURE_IMAGES", ROOT / "data" / "images")) / "images_classic")
    ap.add_argument("--imglists", default=os.environ.get("CURE_IMGLISTS", ROOT / "data" / "benchmark_imglist"))
    a = ap.parse_args()
    torch.manual_seed(SEED)
    ckpt, lists, norm = SETS[a.bench]
    bb = build_backbone("resnet18_32x32", ckpt, norm=norm)
    mean, std = CIFAR_NORMS[norm]
    # Như MulticropPreProcessor: Resize(pre_size) → ToTensor → 5 crop → Normalize.
    tf = transforms.Compose([transforms.Resize(32, interpolation=transforms.InterpolationMode.BILINEAR),
                             transforms.ToTensor(), MultiRandomCrop(num_crop=NUM_CROP, image_size=32),
                             transforms.Normalize(mean, std)])
    rows = load_rows(Path(a.imglists) / lists / f"train_{lists}.txt", Path(a.image_root))
    loader = DataLoader(ListSet(rows, tf), batch_size=64, num_workers=0, shuffle=False)
    feats, msp, labels = [], [], []
    for images, lab in loader:
        b = images.shape[0]
        f, z = bb.forward(images.flatten(0, 1))
        conf = torch.softmax(z, 1).max(1).values.view(b, NUM_CROP)
        best, sel = conf.max(1)
        feats.append(normalizer(f.view(b, NUM_CROP, -1)[torch.arange(b), sel].numpy()))
        msp.append(best)
        labels.append(lab)
    msp, labels = torch.cat(msp), torch.cat(labels)
    out = data_root() / f"oodd_gallery_{a.bench}.npz"
    np.savez(out, activation_log=np.concatenate(feats).astype(np.float32), idx=informative_order(msp, labels),
             msp=msp.numpy(), labels=labels.numpy())
    print("saved", out, "n", len(labels))


if __name__ == "__main__":
    main()
