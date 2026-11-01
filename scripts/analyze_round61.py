"""Vòng 61: biên Pareto (độ trễ, AUROC) theo backbone và kiểm H61a-d theo đăng ký trước.

    py scripts/analyze_round61.py --results results/round61 --latency results/round61/latency.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

METHODS = {"knn": "kNN", "cure": "CURE", "cure_oebh_b1": "CURE B=1"}
BASE = "r61_resnet18_s0_ivf4"          # mốc so sánh: ResNet-18 tự train, seed 0
SEEDS = [f"r61_resnet18_s{i}_ivf4" for i in range(3)]
EMBED128, LIGHTER = "r61_resnet18embed128_ivf4", ("r61_resnet20_ivf4", "r61_mobilenetv2_ivf4")
BUDGETS = (33.0, 10.0)                 # ms/mẫu: 30 và 100 mẫu mỗi giây


def load(out_dir: Path) -> dict:
    """AUROC và FPR95 (%) trung bình theo seed luồng, cho từng (backbone, tập OOD, π)."""
    acc = defaultdict(list)
    for f in out_dir.glob("*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            acc[(r["benchmark"], r["domain"], r["pi"])].append(r)
    A = {}
    for g, rows in acc.items():
        A[g] = {m: {k: float(np.mean([100 * r["metrics"][m][k] for r in rows])) for k in ("auroc", "fpr95")}
                for m in METHODS if all(m in r["metrics"] for r in rows)}
    return A


def spearman(x, y) -> float:
    rx, ry = (np.argsort(np.argsort(v)).astype(float) for v in (np.asarray(x), np.asarray(y)))
    return float(np.corrcoef(rx, ry)[0, 1])


def stage(lat: dict, name: str, device: str, batch: int) -> dict:
    return lat.get(f"{name}@{device}@{batch}", {})


def per_sample(rec: dict, batch: int) -> dict:
    """Thời gian mỗi mẫu của từng tầng (ms); các tầng ảnh đã nhân sẵn theo batch khi đo."""
    if not rec:
        return {}
    out = {k: rec[k] / batch for k in ("read_ms", "preprocess_ms") if k in rec}
    for k in ("backbone", "knn", "cure"):
        if k in rec:
            out[f"{k}_ms"] = rec[k]["p50_ms"] / batch
    if "cure" in rec and "knn" in rec:
        out["cure_rest_ms"] = (rec["cure"]["p50_ms"] - rec["knn"]["p50_ms"]) / batch
    out["total_ms"] = sum(out.get(k, 0.0) for k in ("read_ms", "preprocess_ms", "backbone_ms", "cure_ms"))
    return out


def pareto(points: list[tuple[str, float, float]]) -> list[str]:
    """Giữ backbone ko bị cái nào vừa nhanh hơn vừa tốt hơn."""
    return [n for n, t, a in points if not any(t2 <= t and a2 >= a and (t2, -a2) != (t, -a) for _, t2, a2 in points)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/round61")
    ap.add_argument("--latency", default="results/round61/latency.json")
    ap.add_argument("--out", default="results/round61/summary.json")
    a = ap.parse_args()
    A = load(Path(a.results))
    lat = json.loads(Path(a.latency).read_text(encoding="utf-8")) if Path(a.latency).exists() else {}
    names = sorted({g[0] for g in A})
    groups = sorted({g[1:] for g in A})

    summary = {}
    for n in names:
        ks = [g for g in groups if (n, *g) in A]
        rec = {"groups": len(ks)}
        for m, label in METHODS.items():
            vals = [A[(n, *g)][m] for g in ks if m in A[(n, *g)]]
            if vals:
                rec[label] = {k: float(np.mean([v[k] for v in vals])) for k in ("auroc", "fpr95")}
        for device, batch in (("cpu", 1), ("cuda", 1), ("cuda", 128)):
            s = per_sample(stage(lat, n, device, batch), batch)
            if s:
                rec[f"{device}_b{batch}"] = s
        summary[n] = rec

    def auroc(n, label="CURE B=1"):
        return summary[n].get(label, summary[n].get("CURE", {})).get("auroc", float("nan"))

    def cpu(n, key):
        return summary[n].get("cpu_b1", {}).get(key, float("nan"))

    # H61a: backbone nhẹ hơn, CURE ko kém quá 1,0 AUROC mà tầng backbone nhanh hơn ≥ 2 lần (CPU B = 1).
    h61a = [n for n in LIGHTER if n in summary and auroc(BASE) - auroc(n) <= 1.0
            and cpu(BASE, "backbone_ms") / cpu(n, "backbone_ms") >= 2]
    # H61b: lợi của CURE theo % lỗi giảm được ko tụt khi backbone mạnh lên.
    pts = [(summary[n]["kNN"]["auroc"], (auroc(n) - summary[n]["kNN"]["auroc"]) / (100 - summary[n]["kNN"]["auroc"]))
           for n in names if "kNN" in summary[n]]
    rho = spearman([p[0] for p in pts], [p[1] for p in pts]) if len(pts) > 2 else float("nan")
    # H61c: chênh lệch giữa kiến trúc lớn hơn 3 lần lệch chuẩn giữa 3 seed train của ResNet-18.
    seed_sd = float(np.std([auroc(n) for n in SEEDS if n in summary], ddof=1)) if sum(n in summary for n in SEEDS) == 3 else float("nan")
    arch = [auroc(n) for n in names if n not in SEEDS[1:]]
    spread = float(np.nanmax(arch) - np.nanmin(arch)) if arch else float("nan")
    # H61d: nhúng 128 chiều làm kNN nhanh hơn ≥ 2 lần mà CURE mất ≤ 0,5 AUROC.
    h61d = (EMBED128 in summary and cpu(BASE, "knn_ms") / cpu(EMBED128, "knn_ms") >= 2
            and auroc(BASE) - auroc(EMBED128) <= 0.5)

    front = {}
    for device, batch in (("cpu", 1), ("cuda", 1), ("cuda", 128)):
        key = f"{device}_b{batch}"
        pp = [(n, summary[n][key]["total_ms"], auroc(n)) for n in names if key in summary[n]]
        front[key] = {"pareto": pareto(pp),
                      "khuyen_nghi": {str(b): max((p for p in pp if p[1] <= b), key=lambda p: p[2], default=("ko đạt",))[0]
                                      for b in BUDGETS}}

    verdict = {
        "H61a": {"pass": bool(h61a), "configs": h61a},
        "H61b": {"pass": bool(rho >= -0.3), "spearman": rho},
        "H61c": {"pass": bool(spread > 3 * seed_sd), "seed_sd": seed_sd, "spread": spread},
        "H61d": {"pass": bool(h61d)},
        "pareto": front,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"backbones": summary, "verdict": verdict}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(verdict, ensure_ascii=False, indent=1))
    for n, r in summary.items():
        print(f"{n:30s} kNN {r.get('kNN', {}).get('auroc', float('nan')):6.2f}  CURE {auroc(n):6.2f}  "
              f"CPU B=1 {r.get('cpu_b1', {}).get('total_ms', float('nan')):7.2f} ms")


if __name__ == "__main__":
    main()
