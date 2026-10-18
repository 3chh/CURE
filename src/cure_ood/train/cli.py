"""Huấn luyện backbone trên tập train ID của benchmark; chạy: cure-ood train <cấu hình> (hướng dẫn: docs/training.md)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from ..backbones.base import REPO_ROOT
from ..data.benchmark import CONFIG_DIR as BENCH_DIR

CONFIG_DIR = REPO_ROOT / "configs" / "train"
OUTPUT_DIR = REPO_ROOT / "results" / "train"


def load_config(name_or_path: str) -> dict:
    path = Path(name_or_path)
    if not path.suffix:
        path = CONFIG_DIR / f"{name_or_path}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def apply_override(cfg: dict, item: str) -> None:
    """`a.b.0.c=giá_trị`: số nguyên trong đường dẫn là chỉ số danh sách, giá trị đọc như YAML."""
    key, _, value = item.partition("=")
    *path, last = key.split(".")
    node = cfg
    for part in path:
        node = node[int(part)] if isinstance(node, list) else node.setdefault(part, {})
    node[int(last) if isinstance(node, list) else last] = yaml.safe_load(value)


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")  # console Windows mặc định ko in được tiếng Việt
    ap = argparse.ArgumentParser(prog="cure-ood train", description=__doc__.splitlines()[0])
    ap.add_argument("config", help="tên trong configs/train/ hoặc đường dẫn .yaml")
    ap.add_argument("--device")
    ap.add_argument("--output", type=Path, default=OUTPUT_DIR, help="thư mục gốc; run ghi vào <output>/<run>/")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="ghi đè cấu hình, vd --set data.subsample=2000 --set stages.0.epochs=1")
    ap.add_argument("--fresh", action="store_true", help="bỏ trạng thái cũ thay vì chạy tiếp")
    ap.add_argument("--select-k", action="store_true", help="sau khi đã trích đặc trưng: chọn K bằng APS cho benchmark")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    for item in args.set:
        apply_override(cfg, item)
    run = cfg["run"]

    from .export import select_k_into, write_configs  # nhập muộn để --help ko phải nạp torch
    if args.select_k:
        print(f"{run}: k = {select_k_into(BENCH_DIR / f'{run}.yaml')}")
        return
    from .trainer import train
    spec, ckpt = train(cfg, args.output / run, args.device, args.fresh)
    print("checkpoint:", ckpt)
    if "export" in cfg:
        extract_cfg, bench_cfg = write_configs(cfg, spec, ckpt)
        sets = "".join(f"--set {item} " for item in args.set)
        print(f"đã ghi {extract_cfg} và {bench_cfg}. Tiếp theo:\n  cure-ood extract {run}\n"
              f"  cure-ood train {args.config} {sets}--select-k\n  cure-ood run robustness --benchmark {run}")


if __name__ == "__main__":
    main()
