"""Đo thời gian xử lý đầu-cuối cho luồng: ảnh → backbone → phương pháp, quy ra mỗi mẫu và so với ngân sách thời gian thực."""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
import yaml

from ..data.benchmark import Benchmark
from ..methods.base import Method, Output, build, run_session

BUDGETS_FPS = (30, 100)  # video 30 khung/giây và luồng 100 mẫu/giây


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _pct(x):
    x = np.asarray(x) * 1e3
    return {"mean_ms": float(x.mean()), "p50_ms": float(np.percentile(x, 50)), "p95_ms": float(np.percentile(x, 95)),
            "p99_ms": float(np.percentile(x, 99))}


def queue_latency(step_s: np.ndarray, batch: int, n: int, rate: float) -> np.ndarray:
    """Độ trễ (s) của từng mẫu khi mẫu đến đều `rate` mẫu/s: chờ gom đủ batch, chờ batch trước xong, rồi xử lý."""
    arrive = np.arange(n) / rate
    lat, free = np.empty(n), 0.0
    for j, dt in enumerate(step_s):
        s, e = j * batch, min(n, (j + 1) * batch)
        free = max(arrive[e - 1], free) + dt  # batch cuối thiếu mẫu được xử lý khi mẫu cuối của luồng đến
        lat[s:e] = free - arrive[s:e]
    return lat


def timed_session(method: Method, stream, batch: int, rates=BUDGETS_FPS) -> tuple[Output, dict]:
    """Chạy phiên trên luồng, đo từng step (đồng bộ CUDA); trả kết quả và phân vị thời gian step, độ trễ đầu-cuối."""
    z = stream.logits if method.requires_logits else None
    # Một step bỏ đi trên phiên riêng: khởi tạo CUDA/cuBLAS và tải gallery ko tính vào lần đo.
    method.session(stream.domain).step(stream.feats[:batch], None if z is None else z[:batch])
    times = []

    def wrap(step):
        def timed(*args):
            _sync()
            t = time.perf_counter()
            out = step(*args)
            _sync()
            times.append(time.perf_counter() - t)
            return out
        return timed

    out = run_session(method, stream, batch, wrap)
    step_s = np.asarray(times)
    res = {"batch": batch, "steps": len(step_s)} | {f"step_{k}": v for k, v in _pct(step_s).items()}
    for r in rates:
        res |= {f"e2e{r}_{k}": v for k, v in _pct(queue_latency(step_s, batch, len(stream), r)).items()}
    return out, res


def backbone_latency(extract_cfg: str, batch_sizes=(1, 32, 128), n_images=2048, warmup=3) -> dict:
    """Từng batch ảnh ID test: đọc từ đĩa + tiền xử lý (CPU) và chạy mạng (GPU), đo riêng hai phần."""
    # Nhập muộn: torchvision/PIL thuộc gói phụ `extract`, runner nhập module này cả khi ko có chúng.
    from ..backbones import build_backbone
    from ..data.extract import CONFIG_DIR as EXTRACT_DIR
    from ..data.extract import ListSet, env_dir, load_rows
    cfg = yaml.safe_load((EXTRACT_DIR / f"{extract_cfg}.yaml").read_text(encoding="utf-8"))
    bb = build_backbone(cfg["backbone"], cfg.get("checkpoint"), **(cfg.get("options") or {}))
    imglist = env_dir("CURE_IMGLISTS", "data/benchmark_imglist") / cfg["imglists"] / f"{cfg['sets']['id_test']}.txt"
    ds = ListSet(load_rows(imglist, env_dir("CURE_IMAGES", "data/images") / cfg["image_root"])[:n_images], bb.transform)
    out = {}
    for bs in batch_sizes:
        load_t, fwd_t = [], []
        for i, s in enumerate(range(0, len(ds) - bs + 1, bs)):
            t0 = time.perf_counter()
            x = torch.stack([ds[j][0] for j in range(s, s + bs)])
            t1 = time.perf_counter()
            bb.forward(x)
            _sync()
            t2 = time.perf_counter()
            if i >= warmup:  # bỏ vài batch đầu: khởi tạo CUDA/cuDNN
                load_t.append(t1 - t0)
                fwd_t.append(t2 - t1)
        tot = np.add(load_t, fwd_t)
        out[bs] = {"load": _pct(load_t), "forward": _pct(fwd_t), "total": _pct(tot),
                   "per_sample_ms": float(tot.mean() / bs * 1e3), "throughput": float(bs / tot.mean())}
    return out


def method_latency(bench: Benchmark, methods: dict, domain: str, pi: float, seed: int) -> dict:
    """Mỗi phương pháp trên cùng một luồng đặc trưng: thời gian fit, thời gian chấm cả luồng, bộ nhớ đỉnh."""
    import psutil
    stream = bench.stream(domain, pi, seed)
    _ = stream.x, stream.feats, stream.logits  # nạp trước để ko tính vào thời gian phương pháp
    proc, out = psutil.Process(os.getpid()), {}
    for label, spec in methods.items():
        spec = dict(spec or {})
        m = build(spec.pop("method", label), **spec)
        if m.requires_logits and not bench.has_logits:
            continue
        stream.cache.clear()  # ko cho phương pháp sau dùng lại kNN/kênh bộ nhớ đã tính
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        rss0 = proc.memory_info().rss
        t0 = time.perf_counter()
        m.fit(bench)
        _sync()
        t1 = time.perf_counter()
        m.score(stream)
        _sync()
        t2 = time.perf_counter()
        n, batch = len(stream), getattr(m, "batch", None) or 128
        out[label] = {"fit_s": t1 - t0, "score_s": t2 - t1, "per_sample_ms": (t2 - t1) / n * 1e3,
                      "per_batch_ms": (t2 - t1) / -(-n // batch) * 1e3, "batch": batch, "n": n,
                      "gpu_peak_mb": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None,
                      "rss_delta_mb": (proc.memory_info().rss - rss0) / 2**20}
    return out


def to_markdown(res: dict) -> str:
    L = [f"# Latency: {res['benchmark']} / {res['domain']}, π = {res['pi']}", "",
         (f"Device: {res['device']}. Method times are amortised over the whole stream ({res['n']} samples); "
          "per-batch percentiles come from experiments with `streaming: true`."), ""]
    if res.get("backbone"):
        L += ["## Backbone (image → features)", "",
              ("| Batch | Load+preprocess p50 (ms) | Forward p50 (ms) | Total p95 (ms) | Total p99 (ms) | Per sample (ms) | "
               "Throughput (img/s) |"), "|---|---|---|---|---|---|---|"]
        for bs, r in res["backbone"].items():
            L.append(f"| {bs} | {r['load']['p50_ms']:.2f} | {r['forward']['p50_ms']:.2f} | {r['total']['p95_ms']:.2f} | "
                     f"{r['total']['p99_ms']:.2f} | {r['per_sample_ms']:.3f} | {r['throughput']:.0f} |")
        L.append("")
    bb_ms = res["backbone"][max(res["backbone"], key=int)]["per_sample_ms"] if res.get("backbone") else 0.0
    L += ["## Methods (features → score) and end-to-end", "",
          f"End-to-end per sample = backbone at the largest batch ({bb_ms:.3f} ms) + method. "
          + " ".join(f"Budget {f}/s = {1e3 / f:.1f} ms per sample." for f in BUDGETS_FPS), "",
          ("| Method | Fit (s) | Method per sample (ms) | Per batch (ms) | Batch | End-to-end per sample (ms) | "
           "Samples/s | GPU peak (MB) |"), "|---|---|---|---|---|---|---|---|"]
    for label, r in sorted(res["methods"].items(), key=lambda kv: kv[1]["per_sample_ms"]):
        e2e = bb_ms + r["per_sample_ms"]
        gpu = f"{r['gpu_peak_mb']:.0f}" if r["gpu_peak_mb"] is not None else "-"
        L.append(f"| {label} | {r['fit_s']:.2f} | {r['per_sample_ms']:.3f} | {r['per_batch_ms']:.1f} | {r['batch']} | "
                 f"{e2e:.3f} | {1e3 / e2e:.0f} | {gpu} |")
    return "\n".join(L) + "\n"


def run(exp, benchmark: str, domain: str, pi: float, seed: int, extract_cfg: str | None = None, methods=None) -> dict:
    """Đo và ghi results/<thí nghiệm>/latency_<benchmark>_<domain>.{json,md}."""
    bench = Benchmark.load(benchmark)
    chosen = {k: v for k, v in exp.methods.items() if not methods or k in methods}
    res = {"benchmark": benchmark, "domain": domain, "pi": pi, "seed": seed,
           "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"}
    if extract_cfg:
        res["backbone"] = backbone_latency(extract_cfg)
    res["methods"] = method_latency(bench, chosen, domain, pi, seed)
    res["n"] = next(iter(res["methods"].values()))["n"]
    out = exp.out_dir / f"latency_{benchmark}_{domain}"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    md = to_markdown(res)
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    return res | {"markdown": md}
