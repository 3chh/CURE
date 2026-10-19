"""Tổng hợp kết quả từng luồng thành bảng theo mục `report` của file thí nghiệm."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from .latency import BUDGETS_FPS

METRICS = ("auroc", "aupr_in", "aupr_out", "fpr95", "tpr_fpr5")
HEADERS = {"auroc": "AUROC", "aupr_in": "AUPR-In", "aupr_out": "AUPR-Out", "fpr95": "FPR95", "tpr_fpr5": "TPR@FPR5"}
MEMORY_STATS = ("cdm_mean", "coverage", "certified", "evicted")


def load_rows(out_dir: Path, benchmarks=None) -> list[dict]:
    rows = [json.loads(l) for f in sorted(Path(out_dir).glob("*.jsonl")) for l in f.read_text(encoding="utf-8").splitlines()]
    return [r for r in rows if benchmarks is None or r["benchmark"] in benchmarks]


def apply_select(rows, select: dict):
    for label, cands in (select or {}).items():
        for r in rows:
            have = [c for c in cands if c in r["metrics"]]
            if have:
                r["metrics"][label] = r["metrics"][max(have, key=lambda c: r["metrics"][c]["auroc"])]
    return rows


def group(rows) -> dict:
    """{(benchmark, tập lạ, π): {phương pháp: {chỉ số (%), dcr, mem}}}; chỉ giữ phương pháp có đủ mọi seed."""
    g = defaultdict(list)
    for r in rows:
        g[(r["benchmark"], r["domain"], r["pi"])].append(r)
    A = {}
    for k, v in g.items():
        arms = set.intersection(*[set(r["metrics"]) for r in v])
        A[k] = {a: {m: 100 * float(np.mean([r["metrics"][a][m] for r in v])) for m in METRICS} for a in arms}
        A[k]["_near"] = v[0]["near"]
        for a in arms & set.intersection(*[set(r.get("dcr", {})) for r in v]):
            n_id, n, nb = (sum(r["dcr"][a][i] for r in v) for i in range(3))
            A[k][a]["dcr"] = 100 * n_id / n if n else float("nan")
            A[k][a]["mem"] = n / nb if nb else 0.0
        for a in arms & set.intersection(*[set(r.get("decisions", {})) for r in v]):
            A[k][a]["decisions"] = {al: {m: 100 * float(np.mean([r["decisions"][a][al][m] for r in v])) for m in ("fpr", "tpr")}
                                    for al in v[0]["decisions"][a]}
        for a in arms & set.intersection(*[set(r.get("memory_stats", {})) for r in v]):
            A[k][a]["memory_stats"] = {m: float(np.mean([r["memory_stats"][a][m] for r in v])) for m in MEMORY_STATS}
        for a in arms & set.intersection(*[set(r.get("latency", {})) for r in v]):
            A[k][a]["latency"] = {m: float(np.mean([r["latency"][a][m] for r in v])) for m in v[0]["latency"][a]}
    return A


def boot_ci(d, B=10000, seed=0):
    m = np.random.default_rng(seed).choice(d, (B, len(d)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summarize(A, arms, base, keys):
    """Chỉ số trung bình, Δ so với nền, tỉ lệ hại (Δ < -0,5), tỉ lệ an toàn (Δ ≥ -0,5) và hạng trung bình."""
    out = {}
    for a in arms:
        rec = {m: float(np.mean([A[k][a][m] for k in keys])) for m in METRICS}
        d = np.array([A[k][a]["auroc"] - A[k][base]["auroc"] for k in keys])
        rec.update(delta=float(d.mean()), harm=float(np.mean(d < -0.5)), worst=float(d.min()), safe=float(np.mean(d >= -0.5)))
        dc = [A[k][a]["dcr"] for k in keys if np.isfinite(A[k][a].get("dcr", np.nan))]
        if dc:
            rec["dcr"], rec["mem"] = float(np.mean(dc)), float(np.mean([A[k][a]["mem"] for k in keys if "mem" in A[k][a]]))
        rec["rank"] = float(np.mean([sorted(arms, key=lambda b: -A[k][b]["auroc"]).index(a) + 1 for k in keys]))
        out[a] = rec
    return out


def pairs(A, method, rivals, keys):
    res = {}
    for r in rivals:
        res[r] = {}
        for m in ("auroc", "fpr95", "aupr_out", "tpr_fpr5"):
            sgn = -1 if m == "fpr95" else 1
            d = np.array([sgn * (A[k][method][m] - A[k][r][m]) for k in keys])  # dương = phương pháp tốt hơn
            p = float(wilcoxon(d, alternative="greater").pvalue) if np.any(d != 0) else 1.0
            res[r][m] = {"diff": float(d.mean()), "ci95": boot_ci(d), "win": int((d > 1e-3).sum()),
                         "loss": int((d < -1e-3).sum()), "p": p}
    return res


def parts(A, keys) -> dict:
    """Chia nhóm theo benchmark (cả, near, far) và theo π."""
    out = {}
    for b in dict.fromkeys(k[0] for k in keys):
        out[b] = [k for k in keys if k[0] == b]
        out[f"{b} near"] = [k for k in out[b] if A[k]["_near"]]
        out[f"{b} far"] = [k for k in out[b] if not A[k]["_near"]]
    for pi in sorted({k[2] for k in keys}):
        out[f"π = {pi:g}"] = [k for k in keys if k[2] == pi]
    return {p: ks for p, ks in out.items() if ks}


def breakdown(A, arms, keys) -> dict:
    """AUROC / FPR95 trung bình của từng phương pháp trong mỗi phần (benchmark, near/far, π)."""
    return {p: {"n_groups": len(ks), "arms": {a: {m: float(np.mean([A[k][a][m] for k in ks])) for m in ("auroc", "fpr95")}
                                              for a in arms}}
            for p, ks in parts(A, keys).items()}


def static_table(A, arms) -> dict:
    """AUROC, FPR95 trung bình và tỉ lệ nhóm phương pháp đạt AUROC cao nhất; xếp theo AUROC giảm dần."""
    keys = [k for k in A if all(a in A[k] for a in arms)]
    if not keys:
        return {}
    rec = {a: {"auroc": float(np.mean([A[k][a]["auroc"] for k in keys])),
               "fpr95": float(np.mean([A[k][a]["fpr95"] for k in keys])),
               "best": float(np.mean([A[k][a]["auroc"] >= max(A[k][b]["auroc"] for b in arms) - 1e-9 for k in keys]))}
           for a in arms}
    return {"n_groups": len(keys), "arms": dict(sorted(rec.items(), key=lambda t: -t[1]["auroc"]))}


def decision_table(A) -> dict:
    """FPR thực trên ID (trung bình, lớn nhất theo nhóm) và TPR trên OOD của quyết định p ≤ α."""
    arms = dict.fromkeys(a for k in A for a, v in A[k].items() if isinstance(v, dict) and "decisions" in v)
    out = {}
    for a in arms:
        ds = [A[k][a]["decisions"] for k in A if a in A[k] and "decisions" in A[k][a]]
        out[a] = {"n_groups": len(ds)} | {al: {"fpr": float(np.mean([d[al]["fpr"] for d in ds])),
                                               "fpr_max": float(max(d[al]["fpr"] for d in ds)),
                                               "tpr": float(np.mean([d[al]["tpr"] for d in ds]))} for al in ds[0]}
    return out


def memory_table(A) -> dict:
    """Thống kê hoạt động CDM trung bình theo phần (toàn bộ, near/far, π) cho phương pháp có trả thống kê."""
    arms = dict.fromkeys(a for k in A for a, v in A[k].items() if isinstance(v, dict) and "memory_stats" in v)
    out = {}
    for a in arms:
        keys = [k for k in A if a in A[k] and "memory_stats" in A[k][a]]
        sel = {"all": keys, "near": [k for k in keys if A[k]["_near"]], "far": [k for k in keys if not A[k]["_near"]]}
        sel |= {f"π = {pi:g}": [k for k in keys if k[2] == pi] for pi in sorted({k[2] for k in keys})}
        out[a] = {p: {"n_groups": len(ks)} | {m: float(np.mean([A[k][a]["memory_stats"][m] for k in ks])) for m in MEMORY_STATS}
                  for p, ks in sel.items() if ks}
    return out


def realtime_table(A, arms, ref) -> dict:
    """Chất lượng (so với ref) và thời gian của từng cấu hình chạy phiên trực tuyến: phân vị step, độ trễ đầu-cuối."""
    keys = [k for k in A if ref in A[k] and all(a in A[k] and "latency" in A[k][a] for a in arms)]
    if not keys:
        return {}
    rec = summarize(A, arms, ref, keys)
    for a in arms:
        lat = [A[k][a]["latency"] for k in keys]
        rec[a] |= {m: float(np.mean([x[m] for x in lat])) for m in lat[0]}
        rec[a]["step_p99_worst_ms"] = float(max(x["step_p99_ms"] for x in lat))
    return {"n_groups": len(keys), "arms": rec}


def build_report(rows, cfg: dict) -> dict:
    rows = apply_select(rows, cfg.get("select"))
    A = group(rows)
    ref = cfg.get("reference", "knn")
    res = {"n_streams": len(rows), "families": {}, "breakdowns": {}}
    ranking = [a for a in cfg.get("ranking", []) if all(a in A[k] for k in A)]
    if ranking:
        keys = list(A)
        res["main"] = {"n_groups": len(keys), "arms": summarize(A, ranking, ref, keys)}
    for fam, f in (cfg.get("families") or {}).items():
        arms = [f["base"], f["method"], *f["rivals"]]
        keys = [k for k in A if all(a in A[k] for a in arms)]
        if not keys:
            continue
        res["families"][fam] = {
            sub: {"n_groups": len(ks), "arms": summarize(A, arms, f["base"], ks)} | ({"pairs": pairs(A, f["method"], f["rivals"], ks)} if sub == "all" else {})
            for sub, ks in (("all", keys), ("near", [k for k in keys if A[k]["_near"]]), ("far", [k for k in keys if not A[k]["_near"]])) if ks}
        res["breakdowns"][fam] = breakdown(A, arms, keys)
    if cfg.get("static"):
        res["static"] = static_table(A, cfg["static"])
    if cfg.get("realtime"):
        res["realtime"] = realtime_table(A, cfg["realtime"], ref)
    res["decisions"], res["memory"] = decision_table(A), memory_table(A)
    return res


def to_markdown(res: dict, cfg: dict, title: str) -> str:
    names = cfg.get("labels", {})

    def nm(a):
        return names.get(a, a)

    intro = (f"{res['n_streams']} streams. Metrics in %, averaged over seeds per group "
             "(benchmark, OOD set, π), then over groups. FPR95 lower is better; all others higher is better.")
    L = [f"# {title}", "", intro, ""]
    if "main" in res:
        m = res["main"]
        L += [f"## Main table ({m['n_groups']} groups)", "", "| Method | AUROC | FPR95 | Safe groups | Mean rank |", "|---|---|---|---|---|"]
        L += [f"| {nm(a)} | {r['auroc']:.2f} | {r['fpr95']:.1f} | {100 * r['safe']:.1f}% | {r['rank']:.2f} |" for a, r in m["arms"].items()]
        L += ["", f"Safe group: AUROC no more than 0.5 below {nm(cfg.get('reference', 'knn'))}. Ties in rank are broken by row order.", ""]
    for fam, subs in res["families"].items():
        L += [f"## Same-base comparison: {fam}", ""]
        for sub, t in subs.items():
            L += [f"### {sub} ({t['n_groups']} groups)", "",
                  "| Method | " + " | ".join(HEADERS[m] for m in METRICS) + " | DCR | Memory | Δ base | Harm rate | Worst Δ | Rank |",
                  "|---" * 12 + "|"]
            for a, r in t["arms"].items():
                dcr = f"{r['dcr']:.1f}%" if "dcr" in r else "-"
                mem = f"{r['mem']:.0f}" if "mem" in r else "-"
                L.append(f"| {nm(a)} | " + " | ".join(f"{r[m]:.2f}" for m in METRICS) +
                         f" | {dcr} | {mem} | {r['delta']:+.2f} | {100 * r['harm']:.1f}% | {r['worst']:+.2f} | {r['rank']:.2f} |")
            L.append("")
            if "pairs" in t:
                L += ["Paired comparison (positive = CURE better; FPR95 sign flipped; 95% bootstrap CI over groups; one-sided Wilcoxon).", "",
                      "| Rival | Metric | Diff | 95% CI | Win / loss | p |", "|---|---|---|---|---|---|"]
                for r, pr in t["pairs"].items():
                    L += [f"| {nm(r)} | {HEADERS[m]} | {v['diff']:+.2f} | [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}] | "
                          f"{v['win']} / {v['loss']} | {v['p']:.1e} |" for m, v in pr.items()]
                L.append("")
        bd = res["breakdowns"].get(fam)
        if bd:
            arms = list(next(iter(bd.values()))["arms"])
            L += [f"### {fam}: by benchmark and π (AUROC / FPR95)", "",
                  "| Part | Groups | " + " | ".join(nm(a) for a in arms) + " |", "|---" * (len(arms) + 2) + "|"]
            L += [f"| {p} | {t['n_groups']} | " + " | ".join(f"{t['arms'][a]['auroc']:.2f} / {t['arms'][a]['fpr95']:.1f}" for a in arms)
                  + " |" for p, t in bd.items()]
            L.append("")
    if res.get("static"):
        t = res["static"]
        L += [f"## Static methods ({t['n_groups']} groups)", "", "| Method | AUROC | FPR95 | Best in % of groups |", "|---|---|---|---|"]
        L += [f"| {nm(a)} | {r['auroc']:.2f} | {r['fpr95']:.1f} | {100 * r['best']:.1f}% |" for a, r in t["arms"].items()]
        L.append("")
    if res.get("realtime"):
        t, rates = res["realtime"], BUDGETS_FPS
        L += [f"## Real-time streaming ({t['n_groups']} groups)", "",
              (f"Each stream is fed to the method one batch of B samples at a time. Harm / safe: AUROC more than 0.5 below / "
               f"no more than 0.5 below {nm(cfg.get('reference', 'knn'))}. Step: processing time of one batch on features "
               "(CUDA-synchronised), percentiles per stream then averaged over groups; worst = largest per-group p99. "
               "End-to-end per sample at a uniform arrival rate = waiting for the batch to fill + waiting for the previous "
               "batch + processing (backbone not included)."), "",
              "| Method | B | AUROC | FPR95 | Harm | Safe | Step p50 (ms) | p95 | p99 | Worst p99 | "
              + " | ".join(f"E2E @{r}/s mean / p99 (ms)" for r in rates) + " |", "|---" * (10 + len(rates)) + "|"]
        L += [f"| {nm(a)} | {r['batch']:.0f} | {r['auroc']:.2f} | {r['fpr95']:.1f} | {100 * r['harm']:.1f}% | "
              f"{100 * r['safe']:.1f}% | {r['step_p50_ms']:.2f} | {r['step_p95_ms']:.2f} | {r['step_p99_ms']:.2f} | "
              f"{r['step_p99_worst_ms']:.2f} | " + " | ".join(f"{r[f'e2e{x}_mean_ms']:.0f} / {r[f'e2e{x}_p99_ms']:.0f}" for x in rates)
              + " |" for a, r in t["arms"].items()]
        L.append("")
    if res.get("decisions"):
        L += ["## ID/OOD decisions (flag OOD when p ≤ α)", "",
              ("Realised FPR on ID and TPR on OOD in %. p is conformal against the calibration set, so FPR should stay near α or "
               "below when the ID part of the stream is exchangeable with it."), "",
              "| Method | Groups | α | FPR (mean) | FPR (worst group) | TPR |", "|---|---|---|---|---|---|"]
        for a, d in res["decisions"].items():
            L += [f"| {nm(a)} | {d['n_groups']} | {al} | {v['fpr']:.2f} | {v['fpr_max']:.2f} | {v['tpr']:.2f} |"
                  for al, v in d.items() if al != "n_groups"]
        L.append("")
    if res.get("memory"):
        L += ["## CDM activity", "",
              ("CDM: mean CDM size per batch. Coverage: share of the stream scored with a non-empty memory. "
               "Certified / evicted: samples per stream that left the review queue into CDM / out of the system."), "",
              "| Method | Part | Groups | CDM | Coverage | Certified | Evicted |", "|---|---|---|---|---|---|---|"]
        for a, by_part in res["memory"].items():
            L += [f"| {nm(a)} | {p} | {r['n_groups']} | {r['cdm_mean']:.0f} | {100 * r['coverage']:.1f}% | {r['certified']:.0f} | "
                  f"{r['evicted']:.0f} |" for p, r in by_part.items()]
        L.append("")
    return "\n".join(L)
