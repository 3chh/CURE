"""Vòng 61: độ trễ theo tầng (đọc ảnh, tiền xử lý, backbone, kNN, phần còn lại của CURE).

Ba kịch bản theo đăng ký trước: GPU B = 1, GPU B = 128, CPU B = 1. Gallery dùng IVF256 nprobe 4.

    py scripts/round61_latency.py --experiment round61 --out results/round61/latency.json
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cure_ood.backbones import build_backbone
from cure_ood.data import Benchmark
from cure_ood.data.extract import env_dir, load_rows
from cure_ood.data.extract import load_config as load_extract
from cure_ood.eval.runner import Experiment
from cure_ood.methods import build
from cure_ood.ops import l2

SCENARIOS = [("cuda", 1), ("cuda", 128), ("cpu", 1)]


def timeit(fn, n_warm, n_steps, sync: bool):
    t = []
    for i in range(n_warm + n_steps):
        t0 = time.perf_counter()
        fn(i)
        if sync:
            torch.cuda.synchronize()
        if i >= n_warm:
            t.append(time.perf_counter() - t0)
    t = np.array(t) * 1e3
    return {"p50_ms": float(np.percentile(t, 50)), "p99_ms": float(np.percentile(t, 99))}


def sample_images(cfg: dict, n: int) -> list[Path]:
    """Vài ảnh thật của tập test ID, để tầng đọc và tiền xử lý đo trên đúng dữ liệu thật."""
    spec = cfg["sets"]["id_test"]
    name = spec["list"] if isinstance(spec, dict) else spec
    rows = load_rows(env_dir("CURE_IMGLISTS", "data/benchmark_imglist") / cfg["imglists"] / f"{name}.txt",
                     env_dir("CURE_IMAGES", "data/images") / cfg["image_root"], skip_missing=True)
    return [p for p, _ in rows[:n]]


def one(spec, ex: dict | None, device: str, batch: int, n_warm: int, n_steps: int, imgs: list[Path] | None):
    bench = Benchmark.load(spec)
    _ = bench.gallery_index, bench.k_knn
    sync = device == "cuda"
    res = {"device": device, "batch": batch, "k": bench.k_knn, "gallery_size": bench.gallery_index.size,
           "dim": bench.dim}

    if imgs:
        pil = [Image.open(p).convert("RGB") for p in imgs[:8]]
        res["read_ms"] = timeit(lambda i: Image.open(imgs[i % len(imgs)]).convert("RGB"), 20, 200, False)["p50_ms"] * batch
        bb = build_backbone(ex["backbone"], ex.get("checkpoint"), device=device, **(ex.get("options") or {}))
        res["preprocess_ms"] = timeit(lambda i: bb.transform(pil[i % len(pil)]), 20, 200, False)["p50_ms"] * batch
        x = torch.stack([bb.transform(pil[i % len(pil)]) for i in range(batch)])
        res["backbone"] = timeit(lambda i: bb.forward(x), n_warm, n_steps, sync)
        del bb
        gc.collect()

    domain = "svhn" if "svhn" in bench.domains else bench.far[0]
    st = bench.stream(domain, 0.5, 0)
    q = l2(st.feats)
    res["knn"] = timeit(lambda i: bench.knn(q[(i * batch) % 1000:(i * batch) % 1000 + batch]), n_warm, n_steps, sync)

    m = build("cure", certifier="online_ebh", batch=batch).fit(bench)
    sess = {}

    def step(i):
        if i == 0:
            sess["s"] = m.session()
        s = (i * batch) % (len(st.feats) - batch)
        sess["s"].step(st.feats[s:s + batch])

    res["cure"] = timeit(step, n_warm, n_steps, sync)
    res["cure_rest_p50_ms"] = res["cure"]["p50_ms"] - res["knn"]["p50_ms"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="round61")
    ap.add_argument("--variants", nargs="*")
    ap.add_argument("--warm", type=int, default=100)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--images", type=int, default=64, help="số ảnh thật lấy để đo tầng đọc/tiền xử lý")
    ap.add_argument("--out", default="results/round61/latency.json")
    a = ap.parse_args()
    exp = Experiment.load(a.experiment)
    names = a.variants or [b["name"] for b in exp.benchmarks]
    out = Path(a.out)
    res = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    for name in names:
        spec = exp.benchmark_specs([name])[0]
        ex, imgs = None, None
        try:
            ex = load_extract(spec["base"])
            imgs = sample_images(ex, a.images)
        except (OSError, KeyError, FileNotFoundError) as e:  # ko có cấu hình trích hoặc ko có ảnh gốc: chỉ đo kNN và CURE
            print(f"{name}: bỏ tầng ảnh ({e})", flush=True)
        for device, batch in SCENARIOS:
            if device == "cuda" and not torch.cuda.is_available():
                continue
            key = f"{name}@{device}@{batch}"
            if key in res:
                continue
            res[key] = one(spec, ex, device, batch, a.warm, a.steps, imgs)
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            print(key, {k: round(v["p50_ms"], 3) for k, v in res[key].items() if isinstance(v, dict)}, flush=True)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
