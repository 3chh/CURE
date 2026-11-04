"""Đẩy hai kernel Kaggle chạy song song với máy local:
1. tnglxun3chh/cure-round61b: Huấn luyện 2 backbone còn lại (ResNet18-LogitNorm, ResNet-50) và trích đặc trưng.
2. tnglxun3chh/cure-round62: Chạy đánh giá toàn bộ 7 phương pháp SOTA với chỉ mục FAISS IVF256 trên 7 backbone của round61.
"""
import json
import sys
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parents[1]

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    api = KaggleApi()
    api.authenticate()

    # 1. Cập nhật & push cure-round61b
    p61b = ROOT / "results" / "kaggle" / "pull_61b" / "cure-round61b.ipynb"
    if p61b.exists():
        nb61b = json.loads(p61b.read_text(encoding="utf-8"))
        nb61b["cells"][1]["source"] = [
            'RUNS = ["r61_resnet18_logitnorm", "r61_resnet50"]\n',
            "WORKERS = 4\n",
            "EPOCHS = 50\n",
            "SUBSAMPLE = None\n",
        ]
        p61b.write_text(json.dumps(nb61b, ensure_ascii=False, indent=1), encoding="utf-8")
        res61b = api.kernels_push(str(p61b.parent))
        print("Đã push cure-round61b:", res61b.ref, getattr(res61b, "url", ""))

    # 2. Cập nhật & push cure-round62 để chạy round61 SOTA với FAISS-CPU trên Kaggle (CPU mode vì GPU quota 30h đã hết)
    p62_dir = ROOT / "results" / "kaggle" / "pull_62"
    p62_meta = p62_dir / "kernel-metadata.json"
    p62_nb = p62_dir / "cure-round62.ipynb"
    if p62_meta.exists() and p62_nb.exists():
        meta = json.loads(p62_meta.read_text(encoding="utf-8"))
        meta["enable_gpu"] = False
        meta.pop("machine_shape", None)
        if "tnglxun3chh/cure-round61a" not in meta.get("kernel_sources", []):
            meta.setdefault("kernel_sources", []).append("tnglxun3chh/cure-round61a")
        p62_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        nb62 = json.loads(p62_nb.read_text(encoding="utf-8"))
        nb62["cells"][1]["source"] = [
            'EXPERIMENT = "round61"\n',
            "WORKERS = 4\n",
        ]
        # Bổ sung bước giải nén r61_out.zip từ cure-round61a và ghi đè round61.yaml với đầy đủ 7 phương pháp SOTA
        r61_yaml = (ROOT / "configs" / "experiments" / "round61.yaml").read_text(encoding="utf-8")
        # Chỉ giữ 7 backbone đã có trong r61_out.zip của 61a + 60b
        r61_yaml_7 = "\n".join(
            l for l in r61_yaml.splitlines()
            if "r61_resnet50" not in l and "r61_resnet18_logitnorm" not in l
        )
        patch_code = (
            '\n# Giải nén r61_out.zip nếu có trong INPUT\n'
            'for zf in INPUT.rglob("r61_out.zip"):\n'
            '    with zipfile.ZipFile(zf) as z:\n'
            '        z.extractall(REPO)\n'
            '    for f in (REPO / "features").glob("*.npz") if (REPO / "features").exists() else []:\n'
            '        if not (FEAT / f.name).exists():\n'
            '            shutil.copy(f, FEAT / f.name)\n'
            f'(REPO / "configs/experiments/round61.yaml").write_text({r61_yaml_7!r}, encoding="utf-8")\n'
        )
        cell2_src = nb62["cells"][2]["source"]
        if isinstance(cell2_src, list):
            cell2_src = "".join(cell2_src)
        if "r61_out.zip" not in cell2_src:
            nb62["cells"][2]["source"] = cell2_src + patch_code
        p62_nb.write_text(json.dumps(nb62, ensure_ascii=False, indent=1), encoding="utf-8")
        res62 = api.kernels_push(str(p62_dir))
        print("Đã push cure-round62:", res62.ref, getattr(res62, "url", ""))

if __name__ == "__main__":
    main()
