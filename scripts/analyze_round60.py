"""Vòng 60: kiểm H60b-d theo đúng đăng ký trước (docs/CURE_Causal_Research_Log.md) từ results/edge_gallery.

Chạy: py scripts/analyze_round60.py [--latency results/edge_gallery/latency.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

METHODS = {"knn": "kNN", "cure_elond_b1": "CURE"}
HARM = 0.5


def load(out_dir: Path):
    """AUROC (%) và FPR ở α = 0,05 trung bình theo seed cho từng (cấu hình, bộ, tập OOD, π)."""
    acc = defaultdict(list)
    for f in out_dir.glob("*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            bench, cfg = r["benchmark"].split("_", 1)
            g = (cfg, bench, r["domain"], r["pi"])
            acc[g].append(r)
    A = {}
    for g, rows in acc.items():
        A[g] = {m: float(np.mean([100 * r["metrics"][m]["auroc"] for r in rows])) for m in METHODS
                if all(m in r["metrics"] for r in rows)}
        A[g]["fpr05"] = float(np.mean([100 * r["decisions"]["cure_elond_b1"]["0.05"]["fpr"] for r in rows
                                       if "cure_elond_b1" in r.get("decisions", {})] or [np.nan]))
        A[g]["n_seeds"] = len(rows)
    return A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/edge_gallery")
    ap.add_argument("--latency", default="results/edge_gallery/latency.json")
    ap.add_argument("--out", default="results/edge_gallery/round60_summary.json")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    A = load(Path(a.results))
    groups = sorted({g[1:] for g in A})
    cfgs = sorted({g[0] for g in A}, key=lambda c: (c != "g0", c))
    lat = json.loads(Path(a.latency).read_text(encoding="utf-8")) if Path(a.latency).exists() else {}
    summary = {}
    for c in cfgs:
        ks = [g for g in groups if (c, *g) in A and ("g0", *g) in A]
        rec = {"groups": len(ks)}
        for m, name in METHODS.items():
            d = np.array([A[(c, *g)][m] - A[("g0", *g)][m] for g in ks])
            rec[name] = {"auroc": float(np.mean([A[(c, *g)][m] for g in ks])), "drop": float(-d.mean()),
                         "harm": float(np.mean(d < -HARM))}
        f = np.array([A[(c, *g)]["fpr05"] for g in ks])
        rec["fpr05_mean"], rec["fpr05_worst"] = float(np.nanmean(f)), float(np.nanmax(f))
        L = lat.get(f"cifar10_{c}@4")
        if L:
            rec["knn_p50_ms"], rec["cure_p50_ms"], rec["cure_p99_ms"] = L["knn"]["p50_ms"], L["cure"]["p50_ms"], L["cure"]["p99_ms"]
            rec["gallery_size"], rec["k"] = L["gallery_size"], L["k"]
        summary[c] = rec

    g0 = summary.get("g0", {})
    base_ms = g0.get("knn_p50_ms")
    h_b = [c for c, r in summary.items() if c != "g0" and base_ms and r.get("knn_p50_ms", np.inf) <= base_ms / 5
           and r["CURE"]["drop"] <= HARM and r["CURE"]["harm"] <= 0.10]
    hurt = [c for c, r in summary.items() if c != "g0" and r["kNN"]["drop"] >= 0.2]
    comp = [c for c in hurt if summary[c]["CURE"]["drop"] < summary[c]["kNN"]["drop"]]
    h_d = all(r["fpr05_mean"] <= 6 and r["fpr05_worst"] <= 8 for r in summary.values())
    verdict = {
        "H60b": {"pass": bool(h_b), "configs": h_b} if base_ms else "chưa có số đo độ trễ",
        "H60c": {"pass": len(hurt) > 0 and len(comp) >= 2 / 3 * len(hurt), "hurt_configs": hurt, "cure_less": comp},
        "H60d": {"pass": h_d},
    }
    if h_b:
        verdict["recommended"] = min(h_b, key=lambda c: summary[c]["knn_p50_ms"])
    Path(a.out).write_text(json.dumps({"configs": summary, "verdict": verdict}, indent=1), encoding="utf-8")

    print(f"{'cấu hình':10s} {'nhóm':>4s} {'kNN':>7s} {'Δ':>6s} {'CURE':>7s} {'Δ':>6s} {'hại':>5s} {'FPR5%':>6s} {'tệ':>5s} "
          f"{'kNN ms':>7s} {'CURE ms':>8s}")
    for c, r in summary.items():
        print(f"{c:10s} {r['groups']:4d} {r['kNN']['auroc']:7.2f} {-r['kNN']['drop']:+6.2f} {r['CURE']['auroc']:7.2f} "
              f"{-r['CURE']['drop']:+6.2f} {100 * r['CURE']['harm']:4.0f}% {r['fpr05_mean']:6.2f} {r['fpr05_worst']:5.2f} "
              f"{r.get('knn_p50_ms', float('nan')):7.3f} {r.get('cure_p50_ms', float('nan')):8.3f}")
    print(json.dumps(verdict, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
