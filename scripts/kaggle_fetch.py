"""Tải output một notebook Kaggle và dựng lại notebook kèm output từ log (API ko trả notebook đã chạy).

Chạy: py scripts/kaggle_fetch.py <user>/<kernel> --source notebooks/kaggle_round60.ipynb --out results/kaggle/<tên>
Ô code nào bắt đầu bằng print("@@CELL k@@") thì nhận đúng phần log của nó; ko có dấu mốc thì cả log gắn vào ô đầu.
"""
from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path

MARK = re.compile(r"@@CELL (\d+)@@\n?")


def split_log(events: list[dict]) -> dict[int, list[dict]]:
    """Gom các đoạn stdout/stderr theo ô, giữ thứ tự và tên luồng."""
    out, cell = {}, 0
    for e in events:
        data, name = e.get("data", ""), e.get("stream_name", "stdout")
        pos = 0
        for m in MARK.finditer(data):
            if data[pos:m.start()]:
                out.setdefault(cell, []).append({"name": name, "text": data[pos:m.start()]})
            cell, pos = int(m.group(1)), m.end()
        if data[pos:]:
            out.setdefault(cell, []).append({"name": name, "text": data[pos:]})
    return out


def rebuild(source: Path, events: list[dict]) -> dict:
    nb = json.loads(source.read_text(encoding="utf-8"))
    parts = split_log(events)
    k = 0
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        streams = []
        for p in parts.get(k, []):  # gộp các đoạn liền nhau cùng luồng
            if streams and streams[-1]["name"] == p["name"]:
                streams[-1]["text"] += p["text"]
            else:
                streams.append(dict(p))
        c["outputs"] = [{"output_type": "stream", "name": s["name"], "text": s["text"].splitlines(keepends=True)}
                        for s in streams]
        c["execution_count"] = k + 1
        k += 1
    return nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kernel")
    ap.add_argument("--source", required=True, help="notebook đã đẩy lên (để lấy mã nguồn từng ô)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    api.kernels_output(a.kernel, path=str(out), force=True, quiet=True)
    log = next(out.glob("*.log"), None)
    events = json.loads(log.read_text(encoding="utf-8")) if log else []
    name = Path(a.source).stem
    (out / f"{name}_executed.ipynb").write_text(json.dumps(rebuild(Path(a.source), events), ensure_ascii=False, indent=1),
                                                encoding="utf-8")
    for z in out.glob("*.zip"):  # kết quả đóng gói: giải nén cạnh file zip để đọc thẳng
        with zipfile.ZipFile(z) as f:
            f.extractall(out / z.stem)
    status = api.kernels_status(a.kernel)
    print(a.kernel, status.status, "| log", len(events), "đoạn |", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
