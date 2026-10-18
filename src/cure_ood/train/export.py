"""Sinh cấu hình trích đặc trưng và benchmark cho checkpoint vừa huấn luyện, rồi chọn K bằng APS."""
from __future__ import annotations

from pathlib import Path

import yaml

from ..backbones.base import REPO_ROOT
from ..data.benchmark import CONFIG_DIR as BENCH_DIR
from ..data.benchmark import Benchmark
from ..data.extract import CONFIG_DIR as EXTRACT_DIR
from ..data.extract import load_config
from .models import ModelSpec


def _portable(path: Path) -> str:
    """Đường dẫn tương đối gốc repo nếu được (resolve_checkpoint tìm cả ở đó), ngược lại tuyệt đối."""
    path = path.resolve()
    return path.relative_to(REPO_ROOT).as_posix() if path.is_relative_to(REPO_ROOT) else str(path)


def _dump(path: Path, header: str, cfg: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {header}\n" + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def write_configs(cfg: dict, spec: ModelSpec, checkpoint: Path, extract_dir: Path = EXTRACT_DIR,
                  bench_dir: Path = BENCH_DIR) -> tuple[Path, Path]:
    """Chép tập ảnh từ cấu hình mẫu `export.extract` và giao thức từ `export.benchmark`, thay backbone/checkpoint/file."""
    run, export = cfg["run"], cfg["export"]
    header = f"sinh bởi `cure-ood train` cho run {run}"
    extract = load_config(export["extract"]) if isinstance(export["extract"], str) else dict(export["extract"])
    extract.update(backbone=spec.backbone, checkpoint=_portable(checkpoint), options=spec.options, output=f"{run}.npz")
    template = export["benchmark"]
    bench = dict(yaml.safe_load((bench_dir / f"{template}.yaml").read_text(encoding="utf-8"))
                 if isinstance(template, str) else template)
    bench.update(name=run, description=f"{bench.get('description', '')} | backbone tự huấn luyện, run {run}".lstrip(" |"),
                 backbone=f"{spec.backbone} ({run})", features=f"{run}.npz")
    return (_dump(extract_dir / f"{run}.yaml", header, extract),
            _dump(bench_dir / f"{run}.yaml", f"{header}; k lấy từ mẫu, chọn lại bằng --select-k sau khi trích", bench))


def select_k_into(bench_path: Path) -> int:
    """Chọn K bằng APS (methods/static/knn.py) trên đặc trưng đã trích rồi ghi đè trường `k`."""
    from ..methods.static.knn import K_GRID, select_k  # nhập muộn: kéo theo sklearn và toàn bộ registry phương pháp

    bench = Benchmark.load(str(bench_path))
    k, auroc = select_k(bench, [k for k in K_GRID if k <= len(bench.G)])  # gallery nhỏ hơn K lớn nhất của lưới
    lines = bench_path.read_text(encoding="utf-8").splitlines()
    lines = [f"k: {k}  # APS trên (id_val, val_ood), AUROC {auroc:.4f}" if ln.startswith("k:") else ln for ln in lines]
    bench_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return k
