"""Cấu hình chung cho test: đường dẫn đặc trưng và kết quả gốc dùng để đối chiếu."""
import json
import os
from pathlib import Path

import pytest

from cure_ood.data.benchmark import data_root

# Test ko tự tải từ HF: thiếu dữ liệu thì bỏ qua cho nhanh và chạy được offline.
os.environ.setdefault("CURE_HF_AUTO", "0")

# Kết quả gốc (golden) của các vòng thử nghiệm, chỉ gồm các luồng mà test dùng.
GOLDEN = Path(os.environ.get("CURE_GOLDEN", Path(__file__).resolve().parent / "golden"))
LEGACY_NAME = {"cifar10": "cifar10", "cifar100": "cifar100", "imagenet200": "in200",
               "cifar10_supcon": "supcon", "cifar10_dinov2": "dino"}


def golden_row(round_dir: str, bench: str, domain: str, pi: float, seed: int) -> dict:
    path = GOLDEN / round_dir / f"{LEGACY_NAME[bench]}_{domain}.json"
    if not path.exists():
        pytest.skip(f"thiếu kết quả gốc {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    return next(r for r in rows if (r["pi"], r["seed"]) == (pi, seed))


def need_features(bench):
    if not (data_root() / bench.features).exists():
        pytest.skip(f"thiếu file đặc trưng {bench.features}")


@pytest.fixture(autouse=True, scope="function")
def free_gpu():
    """Giải phóng GPU sau mỗi file test: mô hình của test backbone chiếm gần hết 6 GB nếu để lại."""
    yield
    import gc

    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
