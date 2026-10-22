"""Vòng 60: độ trễ trên CPU, B = 1, của bước kNN và cả bước CURE e-LOND cho từng chỉ mục gallery.

Chạy khi máy rảnh: py scripts/edge_latency.py [biến thể ...] --threads 1 4 --out results/edge_gallery/latency.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # kịch bản thiết bị biên: chỉ CPU

import numpy as np
import torch

from cure_ood.data import Benchmark
from cure_ood.eval.runner import Experiment
from cure_ood.methods import build
from cure_ood.ops import l2


def measure(fn, n_warm, n_steps, reps):
    p50, p99 = [], []
    for _ in range(reps):
        t = []
        for i in range(n_warm + n_steps):
            t0 = time.perf_counter()
            fn(i)
            if i >= n_warm:
                t.append(time.perf_counter() - t0)
        p50.append(float(np.percentile(t, 50) * 1e3))
        p99.append(float(np.percentile(t, 99) * 1e3))
    return {"p50_ms": float(np.median(p50)), "p99_ms": float(np.median(p99)), "runs_p50": p50, "runs_p99": p99}


def one(spec, threads, n_warm, n_steps, reps, domain="svhn", pi=0.5, seed=0):
    torch.set_num_threads(threads)
    try:  # FAISS dùng OpenMP riêng, ko theo số luồng của torch
        import faiss
        faiss.omp_set_num_threads(threads)
    except ImportError:
        pass
    t0 = time.perf_counter()
    bench = Benchmark.load(spec)
    _ = bench.gallery_index, bench.k_knn
    build_s = time.perf_counter() - t0
    st = bench.stream(domain, pi, seed)
    x = l2(st.feats)
    res = {"k": bench.k_knn, "gallery_size": bench.gallery_index.size, "build_s": round(build_s, 2), "threads": threads}
    res["knn"] = measure(lambda i: bench.knn(x[i:i + 1]), n_warm, n_steps, reps)
    m = build("cure", certifier="elond", batch=1).fit(bench)
    sessions = {}

    def cure_step(i):
        if i == 0:  # phiên mới cho mỗi lần lặp: trạng thái hàng chờ và CDM giống nhau giữa các lần
            sessions["s"] = m.session()
        sessions["s"].step(st.feats[i:i + 1])

    res["cure"] = measure(cure_step, n_warm, n_steps, reps)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="*", help="tên biến thể trong configs/experiments/edge_gallery.yaml (mặc định: mọi biến thể cifar10)")
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--warm", type=int, default=200)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="results/edge_gallery/latency.json")
    a = ap.parse_args()
    exp = Experiment.load("edge_gallery")
    names = a.variants or [b["name"] for b in exp.benchmarks if b["base"] == "cifar10"]
    out = Path(a.out)
    res = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    for name in names:
        for th in a.threads:
            key = f"{name}@{th}"
            if key in res:
                continue
            res[key] = one(exp.benchmark_specs([name])[0], th, a.warm, a.steps, a.reps)
            print(key, {k: res[key][k] for k in ("k", "gallery_size", "build_s")},
                  "knn p50", round(res[key]["knn"]["p50_ms"], 3), "cure p50", round(res[key]["cure"]["p50_ms"], 3), flush=True)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
