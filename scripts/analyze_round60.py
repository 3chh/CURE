"""Vòng 60/60b: kiểm H60b-e theo đúng đăng ký trước (docs/CURE_Causal_Research_Log.md).

Vòng 60 (gộp CIFAR-10 và CIFAR-100, độ trễ của cifar10):
    py scripts/analyze_round60.py
Vòng 60b, từng bộ một, vd DINOv2:
    py scripts/analyze_round60.py --results results/edge_gallery_b --pool cifar10_dinov2 \\
        --latency results/edge_gallery_b/latency.json --latency-base cifar10_dinov2
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


def split_name(name: str) -> tuple[str, str]:
    """Tên biến thể → (bộ gốc, cấu hình): `cifar10_dinov2@g0` (60b) hoặc `cifar10_g0` (60)."""
    return tuple(name.rsplit("@", 1)) if "@" in name else tuple(name.split("_", 1))


def load(out_dir: Path, pool: list[str]):
    """AUROC (%) và FPR ở α = 0,05 trung bình theo seed cho từng (cấu hình, bộ, tập OOD, π) của các bộ trong pool."""
    acc = defaultdict(list)
    for f in out_dir.glob("*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            bench, cfg = split_name(r["benchmark"])
            if bench in pool:
                acc[(cfg, bench, r["domain"], r["pi"])].append(r)
    A = {}
    for g, rows in acc.items():
        A[g] = {m: float(np.mean([100 * r["metrics"][m]["auroc"] for r in rows])) for m in METHODS
                if all(m in r["metrics"] for r in rows)}
        A[g]["fpr05"] = float(np.mean([100 * r["decisions"]["cure_elond_b1"]["0.05"]["fpr"] for r in rows
                                       if "cure_elond_b1" in r.get("decisions", {})] or [np.nan]))
    return A


def latency_of(lat: dict, base: str, cfg: str, threads: int = 4):
    return lat.get(f"{base}@{cfg}@{threads}") or lat.get(f"{base}_{cfg}@{threads}")


def to_markdown(summary: dict, verdict: dict, title: str | None) -> str:
    """Bảng kết quả và kết luận giả thuyết, dạng Markdown để đưa thẳng vào báo cáo."""
    def ms(r, k):
        return f"{r[k]:.3f}" if k in r else "-"
    lines = [f"# {title or 'Nén gallery: ' + ', '.join(verdict['pool'])}", ""]
    lines += [(f"Bộ: {', '.join(verdict['pool'])}. Δ là chênh lệch AUROC so với g0 (gallery đầy đủ, tìm chính xác); "
               "hại = tỉ lệ nhóm thấp hơn g0 quá 0,5 AUROC. Độ trễ đo trên CPU, B = 1, 4 luồng."), ""]
    lines.append("| Cấu hình | Nhóm | kNN | Δ | CURE | Δ | Hại | FPR 5% | Tệ nhất | kNN p50 (ms) | CURE p50 (ms) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for c, r in summary.items():
        lines.append(f"| {c} | {r['groups']} | {r['kNN']['auroc']:.2f} | {-r['kNN']['drop']:+.2f} | "
                     f"{r['CURE']['auroc']:.2f} | {-r['CURE']['drop']:+.2f} | {100 * r['CURE']['harm']:.0f}% | "
                     f"{r['fpr05_mean']:.2f} | {r['fpr05_worst']:.2f} | {ms(r, 'knn_p50_ms')} | {ms(r, 'cure_p50_ms')} |")
    lines.append("")
    for h, v in verdict.items():
        if h.startswith("H60") and isinstance(v, dict):
            extra = f" ({', '.join(v['configs'])})" if v.get("configs") else ""
            if "hurt_configs" in v:
                extra = f"; CURE mất ít hơn kNN ở {len(v['cure_less'])}/{len(v['hurt_configs'])} cấu hình"
            lines.append(f"- **{h}: {'đạt' if v.get('pass') else 'trượt'}**{extra}")
    if "recommended" in verdict:
        lines.append(f"- Khuyến nghị theo quy tắc ghi trước (rẻ nhất mà đạt): **{verdict['recommended']}**")
    if "max_knn_saving_ms" in verdict:
        lines.append(f"- Tiết kiệm nhiều nhất ở bước kNN so với g0: {verdict['max_knn_saving_ms']:.3f} ms")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/edge_gallery")
    ap.add_argument("--pool", nargs="+", default=["cifar10", "cifar100"], help="các bộ gộp chung khi kiểm giả thuyết")
    ap.add_argument("--latency", nargs="*", default=["results/edge_gallery/latency.json"])
    ap.add_argument("--latency-base", default="cifar10", help="bộ có số đo độ trễ dùng cho pool này")
    ap.add_argument("--out")
    ap.add_argument("--title")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    A = load(Path(a.results), a.pool)
    lat = {}
    for p in a.latency:
        if Path(p).exists():
            lat.update(json.loads(Path(p).read_text(encoding="utf-8")))
    groups = sorted({g[1:] for g in A})
    cfgs = sorted({g[0] for g in A}, key=lambda c: (c != "g0", c))
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
        L = latency_of(lat, a.latency_base, c)
        if L:
            rec["knn_p50_ms"], rec["cure_p50_ms"], rec["cure_p99_ms"] = L["knn"]["p50_ms"], L["cure"]["p50_ms"], L["cure"]["p99_ms"]
            rec["gallery_size"], rec["k"] = L["gallery_size"], L["k"]
        summary[c] = rec

    if not summary:
        print("chưa có kết quả lưới cho", a.pool)
        Path(a.out or Path(a.results) / f"summary_{'+'.join(a.pool)}.json").write_text(
            __import__("json").dumps({"configs": {}, "verdict": {"pool": a.pool, "note": "chưa có kết quả lưới"}}, indent=1),
            encoding="utf-8")
        return
    base_ms = summary.get("g0", {}).get("knn_p50_ms")
    h_b = [c for c, r in summary.items() if c != "g0" and base_ms and r.get("knn_p50_ms", np.inf) <= base_ms / 5
           and r["CURE"]["drop"] <= HARM and r["CURE"]["harm"] <= 0.10]
    hurt = [c for c, r in summary.items() if c != "g0" and r["kNN"]["drop"] >= 0.2]
    comp = [c for c in hurt if summary[c]["CURE"]["drop"] < summary[c]["kNN"]["drop"]]
    verdict = {
        "pool": a.pool,
        "H60b": {"pass": bool(h_b), "configs": h_b} if base_ms else "chưa có số đo độ trễ",
        "H60c": {"pass": len(hurt) > 0 and len(comp) >= 2 / 3 * len(hurt), "hurt_configs": hurt, "cure_less": comp},
        "H60d": {"pass": all(r["fpr05_mean"] <= 6 and r["fpr05_worst"] <= 8 for r in summary.values())},
    }
    if base_ms:
        saving = {c: base_ms - r["knn_p50_ms"] for c, r in summary.items() if "knn_p50_ms" in r}
        verdict["max_knn_saving_ms"] = max(saving.values())  # H60e (đối chứng ImageNet-200): ≤ 1 ms là đạt
    if h_b:
        verdict["recommended"] = min(h_b, key=lambda c: summary[c]["knn_p50_ms"])
    out = Path(a.out or Path(a.results) / f"summary_{'+'.join(a.pool)}.json")
    out.write_text(json.dumps({"configs": summary, "verdict": verdict}, indent=1), encoding="utf-8")
    out.with_suffix(".md").write_text(to_markdown(summary, verdict, a.title), encoding="utf-8")

    print(f"{'cấu hình':10s} {'nhóm':>4s} {'kNN':>7s} {'Δ':>6s} {'CURE':>7s} {'Δ':>6s} {'hại':>5s} {'FPR5%':>6s} {'tệ':>5s} "
          f"{'kNN ms':>7s} {'CURE ms':>8s}")
    for c, r in summary.items():
        print(f"{c:10s} {r['groups']:4d} {r['kNN']['auroc']:7.2f} {-r['kNN']['drop']:+6.2f} {r['CURE']['auroc']:7.2f} "
              f"{-r['CURE']['drop']:+6.2f} {100 * r['CURE']['harm']:4.0f}% {r['fpr05_mean']:6.2f} {r['fpr05_worst']:5.2f} "
              f"{r.get('knn_p50_ms', float('nan')):7.3f} {r.get('cure_p50_ms', float('nan')):8.3f}")
    print(json.dumps(verdict, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
