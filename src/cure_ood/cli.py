"""Dòng lệnh: cure-ood extract | run | report | list."""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None):
    p = argparse.ArgumentParser(prog="cure-ood")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run an experiment config (resumable)")
    r.add_argument("experiment", help="name in configs/experiments or path to a YAML file")
    r.add_argument("--benchmark", nargs="*", help="only these benchmarks")
    r.add_argument("--domain", nargs="*", help="only these OOD sets")
    r.add_argument("--method", nargs="*", help="only these method labels")
    r.add_argument("--output", help="override the output root")
    s = sub.add_parser("report", help="aggregate results into Markdown + JSON tables")
    s.add_argument("experiment")
    s.add_argument("--benchmark", nargs="*", help="restrict the report to these benchmarks")
    s.add_argument("--output")
    sub.add_parser("list", help="list registered methods and benchmark configs")
    sub.add_parser("extract", help="extract features from images (see docs/adding_a_backbone.md)", add_help=False)
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["extract"]:
        from .data.extract import main as extract_main
        return extract_main(argv[1:])
    a = p.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    from .eval.runner import Experiment
    if a.cmd == "list":
        from .data.benchmark import CONFIG_DIR
        from .methods.base import REGISTRY, build
        build("knn")  # nạp registry
        for name, cls in sorted(REGISTRY.items()):
            print(f"{name:22s} {'(cần logit)' if cls.requires_logits else ''}")
        print("Nguồn và độ trung thành của từng phương pháp: README.md, mục Methods.")
        print("benchmarks:", ", ".join(sorted(f.stem for f in CONFIG_DIR.glob("*.yaml"))))
        return
    exp = Experiment.load(a.experiment)
    if a.output:
        exp.output = a.output
    if a.cmd == "run":
        from .eval.runner import run
        run(exp, a.benchmark, a.domain, a.method)
    else:
        from .eval.report import build_report, load_rows, to_markdown
        res = build_report(load_rows(exp.out_dir, a.benchmark), exp.report)
        tag = "" if not a.benchmark else "_" + "_".join(a.benchmark)
        (exp.out_dir / f"summary{tag}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        md = to_markdown(res, exp.report, f"{exp.name}{tag}")
        (exp.out_dir / f"report{tag}.md").write_text(md, encoding="utf-8")
        print(md)


if __name__ == "__main__":
    main()
