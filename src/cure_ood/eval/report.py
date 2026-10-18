"""Tổng hợp kết quả từng luồng thành bảng theo mục `report` của file thí nghiệm."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

METRICS = ("auroc", "aupr_in", "aupr_out", "fpr95", "tpr_fpr5")
HEADERS = {"auroc": "AUROC", "aupr_in": "AUPR-In", "aupr_out": "AUPR-Out", "fpr95": "FPR95", "tpr_fpr5": "TPR@FPR5"}


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


def build_report(rows, cfg: dict) -> dict:
    rows = apply_select(rows, cfg.get("select"))
    A = group(rows)
    ref = cfg.get("reference", "knn")
    res = {"n_streams": len(rows), "families": {}}
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
    return "\n".join(L)
