"""Trích đặc trưng theo imglist OpenOOD ra file .npz; chạy: cure-ood extract <cấu hình>."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..backbones import Backbone, build_backbone
from ..backbones.base import REPO_ROOT
from .benchmark import data_root

CONFIG_DIR = REPO_ROOT / "configs" / "extract"
Rows = list[tuple[Path, int]]


def env_dir(var: str, default: str) -> Path:
    return Path(os.environ.get(var, REPO_ROOT / default))


def read_imglist(path: str | Path) -> list[tuple[str, int]]:
    """Dòng imglist OpenOOD: `đường_dẫn_tương_đối nhãn` (OOD có nhãn -1)."""
    return [(ln.split()[0], int(ln.split()[1])) for ln in Path(path).read_text().splitlines() if ln.strip()]


def load_rows(imglist: str | Path, image_root: Path, roots: dict[str, Path] | None = None, subsample: int | None = None,
              seed: int = 0, dedup: bool = False, skip_missing: bool = False) -> Rows:
    """Đọc imglist rồi lần lượt: bỏ trùng, lấy mẫu con, kiểm tra ảnh tồn tại."""
    rows = read_imglist(imglist)
    if dedup:  # test_imagenet.txt của OpenOOD liệt kê mỗi ảnh hai lần
        seen: set[str] = set()
        rows = [r for r in rows if not (r[0] in seen or seen.add(r[0]))]
    if subsample is not None:  # cùng công thức với bản trích gốc để ra đúng tập con
        rows = [rows[i] for i in np.random.default_rng(seed).permutation(len(rows))[:subsample]]
    # Tiền tố dài nhất thắng, để vd `texture/` lấy từ thư mục ảnh cổ điển trong benchmark ImageNet.
    prefixes = sorted((roots or {}).items(), key=lambda kv: -len(kv[0]))
    out = [(next((d for p, d in prefixes if rel.startswith(p)), image_root) / rel, lab) for rel, lab in rows]
    missing = [p for p, _ in out if not p.exists()]
    if missing and not skip_missing:
        raise FileNotFoundError(f"{imglist}: thiếu {len(missing)}/{len(out)} ảnh, vd {missing[0]}")
    if missing:
        print(f"{Path(imglist).name}: bỏ {len(missing)}/{len(out)} ảnh thiếu, vd {missing[0]}", flush=True)
        miss = set(missing)
        out = [r for r in out if r[0] not in miss]
    return out


class ListSet(Dataset):
    def __init__(self, rows: Rows, transform):
        self.rows, self.transform = rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, label = self.rows[i]
        with Image.open(path) as im:
            return self.transform(im.convert("RGB")), label


def encode_rows(backbone: Backbone, rows: Rows, batch_size: int = 256, num_workers: int = 0):
    feats, logits = [], []
    for x, _ in DataLoader(ListSet(rows, backbone.transform), batch_size=batch_size, num_workers=num_workers):
        f, lo = backbone.forward(x)
        feats.append(f.numpy())
        if lo is not None:
            logits.append(lo.numpy())
    return np.concatenate(feats).astype(np.float32), np.concatenate(logits).astype(np.float32) if logits else None


def extract(backbone: Backbone, sets: dict[str, Rows], out_path: str | Path, batch_size: int = 256,
            num_workers: int = 0) -> dict[str, np.ndarray]:
    """Trích mọi tập rồi ghi một file .npz; trả về chính các mảng đã ghi."""
    arrays: dict[str, np.ndarray] = {}
    if backbone.fc is not None:
        arrays["fc_weight"], arrays["fc_bias"] = backbone.fc
    for name, rows in sets.items():
        feats, logits = encode_rows(backbone, rows, batch_size, num_workers)
        labels = np.array([lab for _, lab in rows], dtype=np.int64)
        arrays[f"{name}_feats"], arrays[f"{name}_labels"] = feats, labels
        msg = f"{name}: {len(rows)} mẫu, d={feats.shape[1]}"
        if logits is not None:
            arrays[f"{name}_logits"] = logits
            if len(labels) and labels.min() >= 0:  # chỉ tập ID có nhãn lớp; độ chính xác để phát hiện lệch tiền xử lý
                msg += f", acc={(logits.argmax(1) == labels).mean():.4f}"
        print(msg, flush=True)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **arrays)
    print("đã ghi", out_path, flush=True)
    return arrays


def load_config(name_or_path: str) -> dict:
    path = Path(name_or_path)
    if not path.suffix:
        path = CONFIG_DIR / f"{name_or_path}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def run(cfg: dict, image_root: str | None = None, out: str | None = None, device: str | None = None) -> dict:
    """Chạy một cấu hình trích; thư mục mặc định đổi được qua CURE_IMAGES, CURE_IMGLISTS, CURE_CHECKPOINTS, CURE_FEATURES."""
    images = env_dir("CURE_IMAGES", "data/images")
    lists = env_dir("CURE_IMGLISTS", "data/benchmark_imglist") / cfg["imglists"]
    root = Path(image_root) if image_root else images / cfg["image_root"]
    roots = {prefix: images / d for prefix, d in (cfg.get("roots") or {}).items()}
    sets = {}
    for name, spec in cfg["sets"].items():
        spec = spec if isinstance(spec, dict) else {"list": spec}
        sets[name] = load_rows(lists / f"{spec['list']}.txt", root, roots, spec.get("subsample"), cfg.get("seed", 0),
                               cfg.get("dedup", False), cfg.get("skip_missing", False))
    backbone = build_backbone(cfg["backbone"], cfg.get("checkpoint"), device, **(cfg.get("options") or {}))
    with torch.inference_mode():
        return extract(backbone, sets, Path(out) if out else data_root() / cfg["output"],
                       cfg.get("batch_size", 256), cfg.get("num_workers", 0))


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")  # console Windows mặc định ko in được tiếng Việt
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config", help="tên trong configs/extract/ hoặc đường dẫn .yaml")
    ap.add_argument("--image-root", help="ghi đè thư mục ảnh chính của cấu hình")
    ap.add_argument("--out", help="ghi đè file .npz đầu ra")
    ap.add_argument("--device")
    args = ap.parse_args(argv)
    run(load_config(args.config), args.image_root, args.out, args.device)


if __name__ == "__main__":
    main()
