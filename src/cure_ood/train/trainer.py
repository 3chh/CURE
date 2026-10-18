"""Vòng huấn luyện nhiều giai đoạn: AMP, lưu trạng thái mỗi epoch để chạy tiếp, nhật ký results/train/<run>/."""
from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data import eval_transform, make_set, split_rows, train_transform
from .losses import build_loss
from .models import ModelSpec, build_model

DATA_DEFAULTS = {"size": 32, "resize": 32, "cache": False, "num_workers": 0, "val": None, "eval_batch_size": 512}
OPTIM_DEFAULTS = {"momentum": 0.9, "weight_decay": 0.0, "nesterov": False}


def lr_at(sched: dict, base_lr: float, epoch: int, step: int, steps: int, epochs: int) -> float:
    """LR tại bước `step` (từ 0) của epoch `epoch` (từ 1); tính lại mỗi bước nên chạy tiếp ko cần trạng thái scheduler."""
    if sched["name"] == "step":  # MultiStep kiểu SupContrast: giảm khi epoch vượt mốc
        return base_lr * sched["gamma"] ** sum(epoch > m for m in sched["milestones"])
    if sched["name"] != "cosine":
        raise KeyError(f"schedule '{sched['name']}' chưa hỗ trợ (cosine | step)")
    per_step = sched.get("per", "step") == "step"
    min_lr = sched.get("min_lr", 0.0)

    def cos(t):  # t: số bước (per step, như LambdaLR của OpenOOD) hoặc số epoch từ 1 (per epoch, như SupContrast)
        return min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * t / (epochs * steps if per_step else epochs)))

    t = (epoch - 1) * steps + step
    warm = sched.get("warmup_epochs", 0)
    if epoch <= warm:  # khởi động tuyến tính tới giá trị cosine ở cuối pha khởi động
        start, target = sched["warmup_from"], cos(warm * steps if per_step else warm)
        return start + t / (warm * steps) * (target - start)
    return cos(t if per_step else epoch)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _save(obj, path: Path) -> None:
    # Ghi file tạm rồi đổi tên: bị ngắt giữa chừng vẫn còn bản cũ để chạy tiếp.
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def accuracy(model, loader, device, amp: bool) -> float:
    model.eval()
    hit = n = 0
    for x, y in loader:
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            pred = model(x.to(device, non_blocking=True)).argmax(1).cpu()
        hit += int((pred == y).sum())
        n += len(y)
    return hit / n


def _gpu_gb(device) -> dict:
    if device.type != "cuda":
        return {}
    return {"gpu_alloc_gb": round(torch.cuda.max_memory_allocated(device) / 2**30, 2),
            "gpu_reserved_gb": round(torch.cuda.max_memory_reserved(device) / 2**30, 2)}


def train(cfg: dict, run_dir: Path, device: str | None = None, fresh: bool = False) -> tuple[ModelSpec, Path]:
    """Chạy (hoặc chạy tiếp) mọi giai đoạn; trả ModelSpec và checkpoint được chọn theo `select` (best | last)."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = bool(cfg.get("amp", True)) and dev.type == "cuda"
    seed, stages, select = cfg.get("seed", 0), cfg["stages"], cfg.get("select", "best")
    data = {**DATA_DEFAULTS, **cfg["data"]}
    if select == "best" and not (data["val"] and stages[-1].get("validate", True)):
        raise ValueError("select: best cần data.val và giai đoạn cuối có validate")
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path, log_path = run_dir / "state.pt", run_dir / "log.jsonl"
    if fresh:
        for p in (state_path, log_path, run_dir / "best.ckpt", run_dir / "last.ckpt"):
            p.unlink(missing_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    seed_all(seed)
    rows = split_rows(data, "train", seed)
    num_classes = data.get("num_classes") or max(lab for _, lab in rows) + 1
    model_cfg = dict(cfg["model"])
    spec = build_model(model_cfg.pop("name"), num_classes, data, **model_cfg)
    model = spec.module.to(dev)
    feat_dim = model.fc.in_features
    train_set = make_set(rows, None, data["cache"])  # transform gán theo từng giai đoạn để ko giải mã ảnh lại
    loader_kw = {"num_workers": data["num_workers"], "pin_memory": dev.type == "cuda"}
    val_loader = None
    if data["val"]:
        val_set = make_set(split_rows(data, "val", seed), eval_transform(data), data["cache"])
        val_loader = DataLoader(val_set, batch_size=data["eval_batch_size"], **loader_kw)
    print(f"{cfg['run']}: {len(rows)} ảnh train, {num_classes} lớp, model {spec.backbone}, thiết bị {dev}"
          f"{', AMP' if amp else ''}", flush=True)

    state = torch.load(state_path, map_location=dev, weights_only=False) if state_path.exists() else None
    if state:
        model.load_state_dict(state["model"])
        print(f"chạy tiếp từ giai đoạn {state['stage']}, epoch {state['epoch']}", flush=True)
    best_acc = state["best_acc"] if state else -1.0
    torch.backends.cudnn.benchmark = True

    for s, stage in enumerate(stages):
        if state and s < state["stage"]:
            continue
        loss_cfg = dict(stage["loss"])
        objective = build_loss(loss_cfg.pop("name"), feat_dim, num_classes, **loss_cfg).to(dev)
        params = list(model.fc.parameters()) if objective.frozen_backbone else [*model.parameters(), *objective.parameters()]
        opt_cfg = {**OPTIM_DEFAULTS, **stage["optimizer"]}
        optimizer = torch.optim.SGD(params, lr=opt_cfg["lr"], momentum=opt_cfg["momentum"],
                                    weight_decay=opt_cfg["weight_decay"], nesterov=opt_cfg["nesterov"])
        scaler = torch.amp.GradScaler(dev.type, enabled=amp)
        start = 1
        if state and s == state["stage"]:
            objective.load_state_dict(state["objective"])
            optimizer.load_state_dict(state["optimizer"])
            scaler.load_state_dict(state["scaler"])
            start = state["epoch"] + 1
        train_set.transform = train_transform(stage["augment"], data, objective.views)
        final, validate = s == len(stages) - 1, stage.get("validate", True) and val_loader is not None

        for epoch in range(start, stage["epochs"] + 1):
            # Hạt giống theo (giai đoạn, epoch): thứ tự trộn và tăng cường như nhau dù chạy liền hay chạy tiếp.
            g = torch.Generator().manual_seed(seed * 1_000_003 + s * 10_007 + epoch)
            torch.manual_seed(int(torch.randint(2**62, (1,), generator=g)))
            loader = DataLoader(train_set, batch_size=stage["batch_size"], shuffle=True, generator=g, **loader_kw)
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats(dev)
            model.train(not objective.frozen_backbone)
            objective.train()
            t0, total, n = time.perf_counter(), 0.0, 0
            for step, (x, y) in enumerate(loader):
                lr = lr_at(stage["schedule"], opt_cfg["lr"], epoch, step, len(loader), stage["epochs"])
                for group in optimizer.param_groups:
                    group["lr"] = lr
                views = x.shape[1]
                x = x.transpose(0, 1).flatten(0, 1).to(dev, non_blocking=True)  # [khung nhìn 1 | khung nhìn 2 | ...]
                y = y.repeat(views).to(dev, non_blocking=True)
                with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                    loss = objective(model, x, y)
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total += float(loss.detach()) * len(y)
                n += len(y)
            sec = time.perf_counter() - t0
            rec = {"stage": stage.get("name", s), "epoch": epoch, "loss": total / n, "lr": lr, "sec": round(sec, 1),
                   "img_per_sec": round(n / views / sec), **_gpu_gb(dev)}
            if validate:
                rec["val_acc"] = accuracy(model, val_loader, dev, amp)
                if final and rec["val_acc"] >= best_acc:  # >= như BaseRecorder của OpenOOD: hòa thì lấy epoch sau
                    best_acc = rec["val_acc"]
                    torch.save(model.state_dict(), run_dir / "best.ckpt")
            # Lưu trạng thái trước nhật ký: bị ngắt giữa hai bước thì epoch chạy lại ko bị ghi hai dòng.
            _save({"stage": s, "epoch": epoch, "model": model.state_dict(), "objective": objective.state_dict(),
                   "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "best_acc": best_acc}, state_path)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True)
        state = None  # các giai đoạn sau bắt đầu từ đầu

    torch.save(model.state_dict(), run_dir / "last.ckpt")
    ckpt = run_dir / f"{select}.ckpt"
    if val_loader is not None:
        print(f"xong; val_acc ({select}) = {accuracy(load_into(model, ckpt), val_loader, dev, amp):.4f}", flush=True)
    return spec, ckpt


def load_into(model, ckpt: Path):
    model.load_state_dict(torch.load(ckpt, map_location=next(model.parameters()).device, weights_only=True))
    return model
