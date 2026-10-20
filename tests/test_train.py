"""Huấn luyện trên dữ liệu tổng hợp nhỏ: loss hữu hạn và giảm, chạy tiếp, checkpoint nạp được vào registry backbone, trích + chọn K."""
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from cure_ood.backbones import build_backbone
from cure_ood.data.extract import run as extract_run
from cure_ood.train.cli import apply_override, load_config
from cure_ood.train.export import select_k_into, write_configs
from cure_ood.train.losses import supcon_loss
from cure_ood.train.models import build_model
from cure_ood.train.trainer import lr_at, train

DATA = {"imglists": "toy", "image_root": "img", "train": "train", "val": "val", "norm": "cifar10",
        "size": 32, "resize": 32, "cache": True}
SETS = {"train": (48, False), "val": (16, False), "test": (16, False), "ood_val": (16, True), "ood": (16, True)}


@pytest.fixture
def toy(tmp_path, monkeypatch):
    """Hai lớp tách được (ảnh tối / sáng + nhiễu) và một tập OOD (nhiễu màu), theo định dạng imglist OpenOOD."""
    rng = np.random.default_rng(0)
    (tmp_path / "lists" / "toy").mkdir(parents=True)
    for name, (n, ood) in SETS.items():
        lines = []
        for i in range(n):
            label = -1 if ood else i % 2
            base = rng.integers(0, 256, (32, 32, 3)) if ood else np.full((32, 32, 3), 60 + 130 * label)
            img = np.clip(base + rng.normal(0, 25, (32, 32, 3)), 0, 255).astype(np.uint8)
            rel = f"toy/{name}/{i}.png"
            (tmp_path / "img" / rel).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(img).save(tmp_path / "img" / rel)
            lines.append(f"{rel} {label}")
        (tmp_path / "lists" / "toy" / f"{name}.txt").write_text("\n".join(lines) + "\n")
    monkeypatch.setenv("CURE_IMAGES", str(tmp_path))
    monkeypatch.setenv("CURE_IMGLISTS", str(tmp_path / "lists"))
    monkeypatch.setenv("CURE_FEATURES", str(tmp_path / "features"))
    return tmp_path


def ce_config(epochs=3):
    return {"run": "toy_ce", "seed": 0, "data": dict(DATA), "model": {"name": "resnet18_32x32"},
            "stages": [{"name": "ce", "loss": {"name": "ce"}, "epochs": epochs, "batch_size": 16, "augment": "openood",
                        "optimizer": {"lr": 0.05, "momentum": 0.9, "weight_decay": 5e-4, "nesterov": True},
                        "schedule": {"name": "cosine", "per": "step", "min_lr": 1e-6}}]}


def read_log(run_dir: Path) -> list[dict]:
    return [json.loads(ln) for ln in (run_dir / "log.jsonl").read_text().splitlines()]


def test_ce_trains_resumes_and_loads(toy):
    run_dir = toy / "runs" / "toy_ce"
    train(ce_config(epochs=1), run_dir, "cpu")
    spec, ckpt = train(ce_config(epochs=3), run_dir, "cpu")  # chạy tiếp từ epoch 1, ko lặp lại
    log = read_log(run_dir)
    assert [r["epoch"] for r in log] == [1, 2, 3]
    assert all(np.isfinite(r["loss"]) for r in log) and log[-1]["loss"] < log[0]["loss"]
    assert ckpt.name == "best.ckpt" and (run_dir / "last.ckpt").exists()
    bb = build_backbone(spec.backbone, ckpt, "cpu", **spec.options)
    feats, logits = bb.forward(torch.randn(4, 3, 32, 32))
    assert feats.shape == (4, 512) and logits.shape == (4, 2)


def test_supcon_with_linear_probe(toy):
    cfg = {**ce_config(), "run": "toy_supcon", "stages": [
        {"name": "supcon", "loss": {"name": "supcon", "temperature": 0.1}, "epochs": 1, "batch_size": 16,
         "augment": "supcon", "validate": False, "optimizer": {"lr": 0.05, "weight_decay": 1e-4},
         "schedule": {"name": "cosine", "per": "epoch", "min_lr": 5e-5, "warmup_epochs": 1, "warmup_from": 0.01}},
        {"name": "linear", "loss": {"name": "linear_ce"}, "epochs": 1, "batch_size": 16, "augment": "linear",
         "optimizer": {"lr": 1.0}, "schedule": {"name": "step", "milestones": [1], "gamma": 0.2}}]}
    spec, ckpt = train(cfg, toy / "runs" / "toy_supcon", "cpu")
    log = read_log(toy / "runs" / "toy_supcon")
    assert [r["stage"] for r in log] == ["supcon", "linear"] and all(np.isfinite(r["loss"]) for r in log)
    assert "val_acc" not in log[0] and "val_acc" in log[1]
    # Giai đoạn tuyến tính chỉ được đổi fc: thân phải trùng bản chạy riêng giai đoạn SupCon (cùng hạt giống).
    _, body_ckpt = train({**cfg, "select": "last", "stages": cfg["stages"][:1]}, toy / "runs" / "toy_body", "cpu")
    full, body = torch.load(ckpt, weights_only=True), torch.load(body_ckpt, weights_only=True)
    assert all(torch.equal(full[k], body[k]) for k in full if not k.startswith("fc."))
    assert not torch.equal(full["fc.weight"], body["fc.weight"])
    assert build_backbone(spec.backbone, ckpt, "cpu", **spec.options).fc[0].shape == (2, 512)


def test_supcon_loss_matches_reference():
    """So với công thức SupConLoss viết theo từng mẫu (SupContrast, contrast_mode='all')."""
    torch.manual_seed(0)
    z = torch.nn.functional.normalize(torch.randn(12, 8), dim=1)
    y = torch.tensor([0, 1, 2, 0, 1, 2] * 2)
    t, bt = 0.1, 0.07
    ref = []
    for i in range(12):
        others = [j for j in range(12) if j != i]
        denom = torch.logsumexp(torch.stack([z[i] @ z[j] / t for j in others]), 0)
        ref.append(torch.stack([z[i] @ z[j] / t - denom for j in others if y[j] == y[i]]).mean())
    assert torch.allclose(supcon_loss(z, y, t, bt), -(t / bt) * torch.stack(ref).mean(), atol=1e-5)


def test_lr_schedules():
    cos = {"name": "cosine", "per": "step", "min_lr": 1e-6}
    assert lr_at(cos, 0.1, 1, 0, 10, 5) == pytest.approx(0.1)
    assert lr_at(cos, 0.1, 3, 5, 10, 5) == pytest.approx(1e-6 + (0.1 - 1e-6) * 0.5)  # nửa đường
    warm = {"name": "cosine", "per": "epoch", "min_lr": 5e-4, "warmup_epochs": 10, "warmup_from": 0.01}
    target = 5e-4 + (0.5 - 5e-4) * 0.5 * (1 + np.cos(np.pi * 10 / 1000))
    assert lr_at(warm, 0.5, 1, 0, 49, 1000) == pytest.approx(0.01)
    assert lr_at(warm, 0.5, 10, 48, 49, 1000) == pytest.approx(0.01 + 489 / 490 * (target - 0.01))
    assert lr_at({"name": "step", "milestones": [60, 75, 90], "gamma": 0.2}, 5.0, 76, 0, 1, 100) == pytest.approx(0.2)


def test_torchvision_model_roundtrip(tmp_path):
    spec = build_model("torchvision", 3, {"norm": "imagenet", "size": 32, "resize": 32}, arch="resnet18")
    torch.save(spec.module.state_dict(), tmp_path / "m.ckpt")
    bb = build_backbone(spec.backbone, tmp_path / "m.ckpt", "cpu", **spec.options)
    x = torch.randn(2, 3, 32, 32)
    logits, feats = spec.module.eval()(x, return_feature=True)
    f, lo = bb.forward(x)
    assert torch.allclose(f, feats.detach()) and torch.allclose(lo, logits.detach()) and f.shape == (2, 512)


def test_export_extract_select_k(toy):
    spec, ckpt = train(ce_config(epochs=1), toy / "runs" / "toy_ce", "cpu")
    cfg = {**ce_config(), "export": {
        "extract": {"imglists": "toy", "image_root": "img", "batch_size": 32,
                    "sets": {"id_train": "train", "id_val": "val", "id_test": "test", "val_ood": "ood_val", "ood": "ood"}},
        "benchmark": {"name": "x", "features": "x.npz", "k": 1, "val_ood": "val_ood", "near": ["ood"], "far": []}}}
    ex_path, bench_path = write_configs(cfg, spec, ckpt, toy / "cfg_extract", toy / "cfg_bench")
    ex = yaml.safe_load(ex_path.read_text(encoding="utf-8"))
    assert ex["backbone"] == "resnet18_32x32" and ex["options"] == {"norm": "cifar10"} and ex["output"] == "toy_ce.npz"
    arrays = extract_run(ex, device="cpu")
    assert arrays["id_test_feats"].shape == (16, 512) and arrays["fc_weight"].shape == (2, 512)
    k = select_k_into(bench_path)
    assert yaml.safe_load(bench_path.read_text(encoding="utf-8"))["k"] == k


def test_shipped_configs_parse():
    for name in ("cifar10_resnet18_ce", "cifar100_resnet18_ce", "cifar10_resnet18_supcon"):
        cfg = load_config(name)
        assert cfg["run"] == name and cfg["stages"]
    cfg = copy.deepcopy(load_config("cifar10_resnet18_ce"))
    apply_override(cfg, "stages.0.epochs=1")
    apply_override(cfg, "data.subsample=2000")
    assert cfg["stages"][0]["epochs"] == 1 and cfg["data"]["subsample"] == 2000


REPO = Path(__file__).resolve().parents[1]
CLASSIC = Path(os.environ.get("CURE_SMOKE_IMAGES", REPO / "data" / "images"))
LISTS = Path(os.environ.get("CURE_SMOKE_IMGLISTS", REPO / "data" / "benchmark_imglist"))


@pytest.mark.skipif(os.environ.get("CURE_TRAIN_SMOKE") != "1" or not (LISTS / "cifar10" / "train_cifar10.txt").exists()
                    or not (CLASSIC / "images_classic" / "cifar10").exists(),
                    reason="smoke CIFAR-10 thật: đặt CURE_TRAIN_SMOKE=1 và cần ảnh + imglist")
@pytest.mark.parametrize("name", ["cifar10_resnet18_ce", "cifar10_resnet18_supcon"])
def test_smoke_cifar10_subset(name, tmp_path, monkeypatch):
    """Một epoch trên 2000 ảnh CIFAR-10 thật; in thời gian/epoch và bộ nhớ GPU để ngoại suy."""
    monkeypatch.setenv("CURE_IMAGES", str(CLASSIC))
    monkeypatch.setenv("CURE_IMGLISTS", str(LISTS))
    cfg = load_config(name)
    for item in ("data.subsample=2000", "data.val_subsample=1000", "stages.0.epochs=1", "stages.-1.epochs=1"):
        apply_override(cfg, item)
    t0 = time.perf_counter()
    train(cfg, tmp_path / name)
    for rec in read_log(tmp_path / name):
        print(name, rec, f"tổng {time.perf_counter() - t0:.0f}s")
        assert np.isfinite(rec["loss"])
