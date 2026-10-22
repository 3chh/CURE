"""Chạy một thí nghiệm: mọi phương pháp × benchmark × tập lạ × π × seed, ghi kết quả từng luồng (tiếp tục được)."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..data.benchmark import Benchmark
from ..methods.base import build, has_session
from .latency import timed_session
from .metrics import dcr_counts, decision_metrics, detection_metrics

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs" / "experiments"


@dataclass
class Experiment:
    name: str
    benchmarks: list[str | dict]  # tên, hoặc biến thể {name, base, ...ghi đè} (vd đổi chỉ mục gallery)
    methods: dict[str, dict]  # nhãn → {method: tên đăng ký (mặc định = nhãn), ...tham số}
    pis: list[float] = field(default_factory=lambda: [0.05, 0.2, 0.5])
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    domains: list[str] | None = None  # None = mọi tập lạ của benchmark
    decision_alphas: list[float] = field(default_factory=lambda: [0.05, 0.1])  # mức α của quyết định p ≤ α
    output: str = "results"
    streaming: bool = False  # chấm bằng phiên trực tuyến từng batch (cỡ = tham số batch) và đo thời gian từng step
    report: dict = field(default_factory=dict)  # cấu hình bảng, xem eval/report.py

    @classmethod
    def load(cls, name_or_path: str) -> Experiment:
        path = Path(name_or_path)
        if not path.suffix:
            path = CONFIG_DIR / f"{name_or_path}.yaml"
        return cls(**yaml.safe_load(path.read_text(encoding="utf-8")))

    def benchmark_specs(self, names=None) -> list[str | dict]:
        specs = {b if isinstance(b, str) else b["name"]: b for b in self.benchmarks}
        return [specs.get(n, n) for n in names] if names else list(specs.values())

    @property
    def out_dir(self) -> Path:
        return Path(self.output) / self.name


def run(exp: Experiment, benchmarks=None, domains=None, methods=None, log=print, force=False) -> None:
    """Lọc theo benchmarks/domains/methods để chia việc cho nhiều tiến trình; luồng đã có kết quả thì bỏ qua."""
    exp.out_dir.mkdir(parents=True, exist_ok=True)
    for spec in exp.benchmark_specs(benchmarks):
        bench = Benchmark.load(spec)
        arms = {}
        for label, spec in exp.methods.items():
            if methods and label not in methods:
                continue
            spec = dict(spec or {})
            m = build(spec.pop("method", label), **spec)
            if why := m.supports(bench):
                log(f"bỏ qua {label} trên {bench.name}: {why}")
                continue
            arms[label] = m.fit(bench)
        for dom in domains or exp.domains or bench.domains:
            if dom not in bench.domains:
                continue
            path = exp.out_dir / f"{bench.name}__{dom}.jsonl"
            rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
            done = {(r["pi"], r["seed"]): r for r in rows}
            for pi in exp.pis:
                for seed in exp.seeds:
                    row = done.get((pi, seed))
                    # force: tính lại các phương pháp đã chọn (vd sau khi thêm chỉ số mới), giữ nguyên phương pháp khác.
                    todo = [a for a in arms if force or row is None or a not in row["metrics"]]
                    if not todo:
                        continue
                    stream = bench.stream(dom, pi, seed)
                    row = row or {"benchmark": bench.name, "domain": dom, "near": stream.near, "pi": pi, "seed": seed,
                                  "n": len(stream), "metrics": {}, "dcr": {}, "seconds": {}}
                    for a in todo:
                        t = time.perf_counter()
                        if exp.streaming and has_session(arms[a]):
                            out, row.setdefault("latency", {})[a] = timed_session(arms[a], stream, arms[a].batch)
                        else:
                            out = arms[a].score(stream)
                        row["seconds"][a] = round(time.perf_counter() - t, 3)
                        row["metrics"][a] = detection_metrics(stream.y, out.scores)
                        if out.memory is not None:
                            row["dcr"][a] = dcr_counts(out.memory, stream.y)
                        if out.pvalues is not None:
                            row.setdefault("decisions", {})[a] = decision_metrics(stream.y, out.pvalues, exp.decision_alphas)
                        if out.memory_stats is not None:
                            row.setdefault("memory_stats", {})[a] = out.memory_stats
                    done[(pi, seed)] = row
                    path.write_text("".join(json.dumps(r) + "\n" for r in done.values()), encoding="utf-8")
                    log(f"{bench.name} {dom} pi={pi} seed={seed} " +
                        " ".join(f"{a}={100 * row['metrics'][a]['auroc']:.2f}" for a in arms))
