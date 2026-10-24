"""Vòng 60, H60a: phiên CURE có cache C x CDM phải trùng từng bit với bản tính lại ở mọi bước.

Chạy: py scripts/check_session_cache.py cifar10 [--seeds 0 1 2] [--out results/edge_gallery/h60a_cifar10.json]
"""
from __future__ import annotations

import argparse
import json
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out")
    a = ap.parse_args()
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
