"""Ablation công tắc bộ nhớ: CURE có hơn một công tắc đơn giản không?

Các nhánh (cùng host bộ nhớ run_memory, B=128, cùng siêu tham số, chỉ khác luật cho mẫu vào CDM):
- knn            : tĩnh, ko bộ nhớ.
- cure           : CURE chính thức (e-BH, gộp Cauchy) - cấu hình chốt từ trước.
- cure_fisher    : như cure nhưng gộp Fisher (bản đổi sau khi xem Bộ 2 - chỉ để đối chiếu).
- all            : nạp mọi mẫu hết hạn (bỏ hậu kiểm) = "luôn bật".
- raw            : hậu kiểm Beta-Binomial từng mẫu ở mức 0.05, KO hiệu chỉnh đa kiểm định.
- scoregate      : công tắc ngây thơ - nạp mẫu có điểm tĩnh s0 thuộc 5% lạ nhất của C (ko dùng mật độ).
- oracle_cert    : chứng nhận bằng nhãn thật (cận trên của bộ hậu kiểm, đọc nhãn).
Hậu xử lý (đọc nhãn, chỉ làm cận trên):
- oracle_switch_all  : mỗi nhóm chọn max(knn, all)  - công tắc biết trước nên bật hay tắt.
- oracle_switch_cure : mỗi nhóm chọn max(knn, cure).

Đo thêm cho mọi nhánh có bộ nhớ: coverage (phần luồng được chấm khi CDM khác rỗng),
cdm_mean (cỡ CDM trung bình), DCR (phần slot bộ nhớ là mẫu ID).

Chạy:
    python deliverables/CURE_repo/scripts/switch_ablation.py
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from cure_ood.data.benchmark import Benchmark
from cure_ood.eval.metrics import dcr_counts, detection_metrics
from cure_ood.methods.base import build, knn_context
from cure_ood.methods.cure import CURE
from cure_ood.methods.memory import run_memory
from cure_ood.stats import CERTIFIERS

# Hậu kiểm từng mẫu ko hiệu chỉnh đa kiểm định.
CERTIFIERS["raw"] = lambda p, alpha=0.05: (lambda expired: p[expired] <= 0.05)


class ScoreGateCURE(CURE):
    """Công tắc ngây thơ: mẫu hết hạn vào CDM nếu s0 nằm trong gate_q lạ nhất của C (ko kiểm định mật độ)."""

    def __init__(self, gate_q=0.05, **kw):
        super().__init__(certifier="scoregate", **kw)
        self.gate_q = gate_q

    def memory_channel(self, stream):
        key = ("cure_mem_scoregate", self.gate_q, self.window, self.batch, self.capacity, self.k_mem)
        if key not in stream.cache:
            ctx = knn_context(stream, self.screen_q)
            thr = float(np.quantile(ctx.cb, self.gate_q))
            b = ctx.b
            stream.cache[key] = run_memory(
                stream.x, b, ctx.passes, ctx.shadow, batch=self.batch, capacity=self.capacity,
                k_mem=self.k_mem, ttl=self.window, certifier=lambda expired: b[expired] <= thr,
                queue_in_memory=False, k_shadow=self.k_shadow, calib=(stream.bench.C, ctx.cb))
        return stream.cache[key]


ARMS = ["knn", "cure", "cure_fisher", "all", "raw", "scoregate", "oracle_cert"]
LABELS = {
    "knn": "kNN tĩnh",
    "cure": "CURE chính thức (e-BH, Cauchy)",
    "cure_fisher": "CURE (e-BH, Fisher) - đổi hậu nghiệm",
    "all": "Luôn bật (nạp mọi mẫu)",
    "raw": "Hậu kiểm ko hiệu chỉnh (p≤0.05)",
    "scoregate": "Công tắc ngây thơ theo điểm tĩnh",
    "oracle_cert": "Chứng nhận oracle (đọc nhãn)",
    "oracle_switch_all": "Công tắc oracle max(kNN, Luôn bật)",
    "oracle_switch_cure": "Công tắc oracle max(kNN, CURE)",
}


def make_arms():
    return {
        "knn": build("knn"),
        "cure": build("cure"),
        "cure_fisher": build("cure", rule="fisher"),
        "all": build("cure", certifier="all"),
        "raw": build("cure", certifier="raw"),
        "scoregate": ScoreGateCURE(),
        "oracle_cert": build("cure", certifier="oracle_labels"),
    }


def eval_stream(arms, stream):
    row = {}
    for k, arm in arms.items():
        out = arm.score(stream)
        d = detection_metrics(stream.y, out.scores)
        r = {"auroc": 100 * d["auroc"], "fpr95": 100 * d["fpr95"], "tpr5": 100 * d["tpr_fpr5"]}
        if out.memory_stats:
            r["coverage"] = 100 * out.memory_stats["coverage"]
            r["cdm_mean"] = out.memory_stats["cdm_mean"]
            c = dcr_counts(out.memory, stream.y) if out.memory else [0, 0]
            r["dcr"] = 100 * c[0] / c[1] if c[1] > 0 else 0.0
        row[k] = r
    return row


def summarize(groups):
    """groups: {gid: {"near": bool|None, "rows": [row,...]}} -> bảng theo nhánh."""
    gm = {}
    for gid, g in groups.items():
        m = {}
        for k in ARMS:
            vals = [r[k] for r in g["rows"]]
            m[k] = {f: float(np.mean([v[f] for v in vals])) for f in vals[0]}
        m["oracle_switch_all"] = dict(max(m["knn"], m["all"], key=lambda v: v["auroc"]))
        m["oracle_switch_cure"] = dict(max(m["knn"], m["cure"], key=lambda v: v["auroc"]))
        gm[gid] = m
    table = {}
    for k in ARMS + ["oracle_switch_all", "oracle_switch_cure"]:
        deltas = [gm[g][k]["auroc"] - gm[g]["knn"]["auroc"] for g in gm]
        row = {
            "auroc": np.mean([gm[g][k]["auroc"] for g in gm]),
            "fpr95": np.mean([gm[g][k]["fpr95"] for g in gm]),
            "tpr5": np.mean([gm[g][k]["tpr5"] for g in gm]),
            "delta": np.mean(deltas),
            "harm": 100 * np.mean([d < -0.5 for d in deltas]),
            "worst": np.min(deltas),
            "wins": 100 * np.mean([d > 0.5 for d in deltas]),
        }
        near = [g for g in gm if groups[g]["near"] is True]
        far = [g for g in gm if groups[g]["near"] is False]
        if near:
            row["near"] = np.mean([gm[g][k]["auroc"] for g in near])
            row["far"] = np.mean([gm[g][k]["auroc"] for g in far])
        for f in ("coverage", "cdm_mean", "dcr"):
            vs = [gm[g][k][f] for g in gm if f in gm[g][k]]
            if vs:
                row[f] = np.mean(vs)
        table[k] = {a: round(float(b), 2) for a, b in row.items()}
    return table, gm


def fmt_table(table, with_near):
    head = "| Nhánh | AUROC | " + ("Near | Far | " if with_near else "") + \
           "FPR95 | TPR@FPR5 | Δ kNN | Thắng >0.5 (%) | Hại <-0.5 (%) | Worst Δ | Coverage (%) | CDM TB | DCR (%) |"
    sep = "|---|" + "---:|" * (head.count("|") - 2)
    lines = [head, sep]
    for k, r in table.items():
        g = lambda f, fmt="{:.2f}": fmt.format(r[f]) if f in r else "-"
        lines.append(
            f"| {LABELS[k]} | {g('auroc')} | " + (f"{g('near')} | {g('far')} | " if with_near else "") +
            f"{g('fpr95')} | {g('tpr5')} | {r['delta']:+.2f} | {g('wins', '{:.1f}')} | {g('harm', '{:.1f}')} | "
            f"{r['worst']:+.2f} | {g('coverage', '{:.1f}')} | {g('cdm_mean', '{:.1f}')} | {g('dcr', '{:.1f}')} |")
    return lines


def run_suite1(backbones):
    groups = {}
    for bname in backbones:
        bench = Benchmark.load(bname)
        arms = {k: a.fit(bench) for k, a in make_arms().items()}
        for dom in bench.domains:
            t0 = time.perf_counter()
            for pi in (0.05, 0.2, 0.5):
                rows, near = [], None
                for seed in (0, 1, 2):
                    st = bench.stream(dom, pi, seed)
                    near = bool(st.near)
                    rows.append(eval_stream(arms, st))
                groups[f"{bname}|{dom}|{pi}"] = {"near": near, "rows": rows}
            print(f"  [{bname}] {dom}: {time.perf_counter() - t0:.1f}s", flush=True)
    return groups


def run_suite2():
    from scripts.benchmark_cyber_and_realworld_sota import CustomStream, build_scenarios
    groups = {}
    for sc in build_scenarios():
        arms = {k: a.fit(sc["bench"]) for k, a in make_arms().items()}
        rows = []
        for fx, fy in sc["streams"]:
            rows.append(eval_stream(arms, CustomStream(bench=sc["bench"], domain=sc["id"], feats=fx, y=fy)))
        groups[sc["id"]] = {"near": None, "rows": rows, "label": sc["label"]}
        print(f"  [suite2] {sc['id']} xong", flush=True)
    return groups


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    backbones = ["r61_resnet20", "r61_resnet18_s0"]
    print("== Bộ 1: OpenOOD CIFAR-10 trên", backbones, flush=True)
    g1 = run_suite1(backbones)
    t1, gm1 = summarize(g1)
    print("== Bộ 2: 14 kịch bản", flush=True)
    g2 = run_suite2()
    t2, gm2 = summarize(g2)

    out = {"suite1": {"table": t1, "groups": gm1}, "suite2": {"table": t2, "groups": gm2}}
    (REPO_ROOT / "results" / "switch_ablation.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False, default=float), encoding="utf-8")

    md = ["# Ablation công tắc bộ nhớ: CURE so với các công tắc đơn giản", "",
          f"## Bộ 1 - OpenOOD CIFAR-10, backbone {', '.join(backbones)} ({len(g1)} nhóm, {3 * len(g1)} luồng)", ""]
    md += fmt_table(t1, True)
    md += ["", f"## Bộ 2 - 14 kịch bản mạng/công nghiệp/forensics ({len(g2)} nhóm)", ""]
    md += fmt_table(t2, False)
    md += ["", "## Bộ 2 - tỉ lệ bật bộ nhớ của CURE chính thức theo từng kịch bản", "",
           "| Kịch bản | kNN AUROC | CURE AUROC | Δ | Coverage (%) | CDM TB | DCR (%) | Luôn bật Δ | Ngây thơ Δ |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for gid, m in gm2.items():
        k0 = m["knn"]["auroc"]
        md.append(f"| {g2[gid]['label']} | {k0:.2f} | {m['cure']['auroc']:.2f} | {m['cure']['auroc'] - k0:+.2f} | "
                  f"{m['cure']['coverage']:.1f} | {m['cure']['cdm_mean']:.1f} | {m['cure']['dcr']:.1f} | "
                  f"{m['all']['auroc'] - k0:+.2f} | {m['scoregate']['auroc'] - k0:+.2f} |")
    md += ["", "## Bộ 1 - tỉ lệ bật bộ nhớ của CURE theo nhóm (near/far, π)", "",
           "| Nhóm | kNN | CURE | Δ | Coverage (%) | CDM TB | Luôn bật Δ | Ngây thơ Δ |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for gid, m in gm1.items():
        k0 = m["knn"]["auroc"]
        md.append(f"| {gid}{' (near)' if g1[gid]['near'] else ''} | {k0:.2f} | {m['cure']['auroc']:.2f} | "
                  f"{m['cure']['auroc'] - k0:+.2f} | {m['cure']['coverage']:.1f} | {m['cure']['cdm_mean']:.1f} | "
                  f"{m['all']['auroc'] - k0:+.2f} | {m['scoregate']['auroc'] - k0:+.2f} |")
    (REPO_ROOT / "results" / "switch_ablation.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("[DONE] results/switch_ablation.md", flush=True)


if __name__ == "__main__":
    main()

