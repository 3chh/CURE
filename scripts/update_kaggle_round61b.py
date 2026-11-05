import json
import base64
from pathlib import Path

p_nb = Path('deliverables/CURE_repo/results/kaggle/pull_61b/cure-round61b.ipynb')
nb = json.loads(p_nb.read_text(encoding='utf-8'))

b64 = Path('deliverables/CURE_repo/results/kaggle/pull_61b/code_b64.txt').read_text().strip()

# Update cell 1
nb['cells'][1]['source'] = [
    'RUNS = ["r61_resnet50", "r61_convnext"]\n',
    'WORKERS = 4\n',
    'EPOCHS = 100\n',
    'SUBSAMPLE = None\n'
]

# Update cell 2
nb['cells'][2]['source'] = [
    'print("@@CELL 1@@", flush=True)\n',
    'import os, sys, time, json, shutil, subprocess, zipfile, io, base64\n',
    'from pathlib import Path\n',
    'INPUT, WORK = Path("/kaggle/input"), Path("/kaggle/working")\n',
    'T0 = time.time()\n',
    'def tick(m): print(f"[{(time.time() - T0) / 60.0:.1f} min] {m}", flush=True)\n',
    '\n',
    'REPO = Path("/tmp/CURE")\n',
    'if REPO.exists(): shutil.rmtree(REPO)\n',
    'REPO.mkdir(parents=True, exist_ok=True)\n',
    f'CODE_B64 = """{b64}"""\n',
    'with zipfile.ZipFile(io.BytesIO(base64.b64decode(CODE_B64))) as z:\n',
    '    z.extractall(REPO)\n',
    'tick("Đã giải nén mã nguồn mới nhất (ResNet-50 và ConvNeXt-Tiny) vào /tmp/CURE")\n',
    '\n',
    '(WORK / "results").mkdir(exist_ok=True)\n',
    'shutil.rmtree(REPO / "results", ignore_errors=True)\n',
    '(REPO / "results").symlink_to(WORK / "results")\n',
    'subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(REPO)], check=True)\n',
    'sys.path.insert(0, str(REPO / "src"))\n',
    '\n',
    'IMG = next(p for p in INPUT.rglob("images_classic") if (p / "cifar10").is_dir())\n',
    'LIST = next(p for p in INPUT.rglob("benchmark_imglist") if (p / "cifar10").is_dir())\n',
    'FEAT = WORK / "features"; FEAT.mkdir(exist_ok=True)\n',
    'os.environ.update(CURE_IMAGES=str(IMG.parent), CURE_IMGLISTS=str(LIST), CURE_FEATURES=str(FEAT), CURE_HF_AUTO="0")\n',
    'os.chdir(REPO)\n',
    '\n',
    'import torch\n',
    'print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "KHÔNG CÓ")\n',
    'assert torch.cuda.is_available(), "Cần GPU để huấn luyện"\n'
]

p_nb.write_text(json.dumps(nb, indent=1), encoding='utf-8')
print('Updated cure-round61b.ipynb successfully!')

