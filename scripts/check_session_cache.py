"""Vòng 60, H60a: phiên CURE có cache C x CDM phải trùng từng bit với bản tính lại ở mọi bước.

Chạy: py scripts/check_session_cache.py cifar10 [--seeds 0 1 2] [--out ...]
Đo tốc độ hai bản (phần còn lại của H60a): py scripts/check_session_cache.py cifar10 --timing --threads 4
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from cure_ood.data import Benchmark
from cure_ood.methods import build
from cure_ood.methods.cure import CURESession


class NoCache(CURESession):
    """Hành vi trước tầng 0: tính lại C x CDM ở mọi bước."""

    def step(self, feats, logits=None):
        self.cdm_view = None
        return super().step(feats, logits)


def run(m, stream, cls):
    sess = cls(m)
    outs = [sess.step(stream.feats[i:i + 1]) for i in range(len(stream.feats))]
    return np.concatenate([o.scores for o in outs]), np.concatenate([o.pvalues for o in outs]), sess.activity()


def timing(bench, threads, warm, steps, reps, domain, pi, seed):
    """p50/p99 thời gian một bước (B = 1) của bản có cache và bản tính lại mỗi bước, trên cùng luồng."""
    import torch
    torch.set_num_threads(threads)
    B = Benchmark.load(bench)
    m = build("cure", certifier="elond", batch=1).fit(B)
    st = B.stream(domain, pi, seed)
    out = {}
    for name, cls in (("nocache", NoCache), ("cache", CURESession)):
        p50, p99 = [], []
        for _ in range(reps):
            sess, t = cls(m), []
            for i in range(warm + steps):
                t0 = time.perf_counter()
                sess.step(st.feats[i:i + 1])
                if i >= warm:
                    t.append(time.perf_counter() - t0)
            p50.append(float(np.percentile(t, 50) * 1e3))
            p99.append(float(np.percentile(t, 99) * 1e3))
        out[name] = {"p50_ms": float(np.median(p50)), "p99_ms": float(np.median(p99)), "runs_p50": p50}
        print(name, out[name], flush=True)
    out["p50_reduction"] = 1 - out["cache"]["p50_ms"] / out["nocache"]["p50_ms"]
    out["p99_reduction"] = 1 - out["cache"]["p99_ms"] / out["nocache"]["p99_ms"]
    out["H60a_speed_pass"] = bool(out["p50_reduction"] >= 0.15)
    print({k: out[k] for k in ("p50_reduction", "p99_reduction", "H60a_speed_pass")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out")
    ap.add_argument("--timing", action="store_true", help="đo tốc độ hai bản thay vì kiểm trùng bit")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--domain", default="svhn")
    ap.add_argument("--pi", type=float, default=0.5)
    ap.add_argument("--warm", type=int, default=200)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    if a.timing:
        res = timing(a.benchmark, a.threads, a.warm, a.steps, a.reps, a.domain, a.pi, a.seeds[0])
        out = Path(a.out or f"results/edge_gallery/h60a_timing_{a.benchmark}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1), encoding="utf-8")
        return
    out = Path(a.out or f"results/edge_gallery/h60a_{a.benchmark}.json")
    res = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
    done = {(r["domain"], r["pi"], r["seed"]) for r in res}
    B = Benchmark.load(a.benchmark)
    m = build("cure", certifier="elond", batch=1).fit(B)
    for dom in B.domains:
        for pi in (0.05, 0.2, 0.5):
            for seed in a.seeds:
                if (dom, pi, seed) in done:
                    continue
                st = B.stream(dom, pi, seed)
                x, y = run(m, st, NoCache), run(m, st, CURESession)
                same = bool(np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1]) and x[2] == y[2])
                res.append({"benchmark": a.benchmark, "domain": dom, "pi": pi, "seed": seed, "same": same})
                print(res[-1], flush=True)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(res), encoding="utf-8")
    print(f"{a.benchmark}: {sum(r['same'] for r in res)}/{len(res)} luồng trùng từng bit")


if __name__ == "__main__":
    main()
