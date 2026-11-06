# -*- coding: utf-8 -*-
"""
auto_multiseed_pipeline.py
--------------------------
Tự động hóa toàn bộ chuỗi huấn luyện Multi-Seed trên GPU local:
1. Giám sát r61_resnet20_s1 cho đến khi hoàn tất 100 epochs.
2. Trích xuất đặc trưng cho r61_resnet20_s1.
3. Huấn luyện r61_resnet20_s2 (Seed 2) -> Trích xuất đặc trưng.
4. Huấn luyện r61_mobilenetv2_s1 (Seed 1) -> Trích xuất đặc trưng.
5. Huấn luyện r61_mobilenetv2_s2 (Seed 2) -> Trích xuất đặc trưng.
6. Cập nhật và tái tạo Báo cáo Word (.docx) & Bảng số liệu Excel (.xlsx).
"""

import os
import sys
import time
import subprocess
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
os.environ["PYTHONPATH"] = str(REPO / "src")

def log(msg):
    t = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{t}] {msg}", flush=True)

def wait_for_completion(run_name, max_epochs=100):
    log_file = REPO / "results" / "train" / run_name / "log.jsonl"
    log(f"Đang theo dõi tiến trình {run_name} tại {log_file}...")
    while True:
        if log_file.exists():
            try:
                lines = log_file.read_text(encoding="utf-8").strip().split("\n")
                lines = [l for l in lines if l.strip()]
                if len(lines) >= max_epochs:
                    log(f"✓ {run_name} ĐÃ HOÀN TẤT ĐỦ {max_epochs} EPOCHS!")
                    break
                elif len(lines) > 0:
                    last_line = lines[-1]
                    import json
                    d = json.loads(last_line)
                    log(f"  {run_name}: Epoch {d.get('epoch')}/{max_epochs} | loss={d.get('loss', 0):.4f} | val_acc={d.get('val_acc', 0):.3f}")
            except Exception as e:
                pass
        time.sleep(20)

def run_train(cfg_path):
    log(f"--- Bắt đầu huấn luyện {cfg_path.name} ---")
    cmd = [sys.executable, "-m", "cure_ood.train.cli", str(cfg_path)]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src")
    subprocess.run(cmd, check=True, env=env)
    log(f"✓ Hoàn thành huấn luyện {cfg_path.name}")

def run_extract(run_name):
    cfg_extract = REPO / "configs" / "extract" / f"{run_name}.yaml"
    if not cfg_extract.exists():
        # Tạo config extract tương tự r61_resnet20
        base_name = run_name.split("_s")[0]
        base_cfg = REPO / "configs" / "extract" / f"{base_name}.yaml"
        if base_cfg.exists():
            text = base_cfg.read_text(encoding="utf-8")
            text = text.replace(base_name, run_name)
            cfg_extract.write_text(text, encoding="utf-8")
            log(f"Đã tự động sinh {cfg_extract}")
            
    if cfg_extract.exists():
        log(f"--- Bắt đầu trích xuất đặc trưng cho {run_name} ---")
        cmd = [sys.executable, "-m", "cure_ood.data.extract", str(cfg_extract)]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO / "src")
        subprocess.run(cmd, check=True, env=env)
        log(f"✓ Đã trích xuất xong đặc trưng cho {run_name}")

def check_colab_checkpoints():
    dl_dir = Path(os.path.expanduser("~/Downloads"))
    zip_path = dl_dir / "cure_colab_checkpoints.zip"
    if zip_path.exists():
        log(f"Phát hiện tệp kết quả tải về từ Colab: {zip_path}! Đang tự động nạp...")
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as z:
            for member in z.namelist():
                if "mobilenetv2_s1" in member:
                    target = REPO / "results" / "train" / "r61_mobilenetv2_s1"
                    target.mkdir(parents=True, exist_ok=True)
                    (target / "best.ckpt").write_bytes(z.read(member))
                    log(f"✓ Đã nạp checkpoint Colab: {target / 'best.ckpt'}")
                elif "mobilenetv2_s2" in member:
                    target = REPO / "results" / "train" / "r61_mobilenetv2_s2"
                    target.mkdir(parents=True, exist_ok=True)
                    (target / "best.ckpt").write_bytes(z.read(member))
                    log(f"✓ Đã nạp checkpoint Colab: {target / 'best.ckpt'}")
        return True
    return False

def main():
    log("KÍCH HOẠT HỆ THỐNG HUẤN LUYỆN MULTI-SEED TỰ ĐỘNG...")
    
    # 1. Chờ ResNet-20 Seed 1
    wait_for_completion("r61_resnet20_s1", 100)
    time.sleep(5)
    run_extract("r61_resnet20_s1")
    
    # 2. Huấn luyện ResNet-20 Seed 2
    cfg_r20_s2 = REPO / "configs" / "train" / "r61_resnet20_s2.yaml"
    if cfg_r20_s2.exists():
        run_train(cfg_r20_s2)
        run_extract("r61_resnet20_s2")
        
    # Kiểm tra xem Colab đã tải về MobileNetV2 chưa
    colab_ready = check_colab_checkpoints()
    
    # 3. MobileNetV2 Seed 1
    if not (REPO / "results" / "train" / "r61_mobilenetv2_s1" / "best.ckpt").exists():
        cfg_mb_s1 = REPO / "configs" / "train" / "r61_mobilenetv2_s1.yaml"
        if cfg_mb_s1.exists():
            run_train(cfg_mb_s1)
    run_extract("r61_mobilenetv2_s1")
        
    # 4. MobileNetV2 Seed 2
    if not (REPO / "results" / "train" / "r61_mobilenetv2_s2" / "best.ckpt").exists():
        cfg_mb_s2 = REPO / "configs" / "train" / "r61_mobilenetv2_s2.yaml"
        if cfg_mb_s2.exists():
            run_train(cfg_mb_s2)
    run_extract("r61_mobilenetv2_s2")
        
    # 5. Tái tạo Báo cáo Word & Excel
    log("--- Tái tạo Báo cáo Word (.docx) và Excel (.xlsx) với dữ liệu Multi-seed ---")
    subprocess.run([sys.executable, "scripts/generate_final_report_docx_excel.py"], check=True)
    log("=== TOÀN BỘ CHUỖI MULTI-SEED ĐÃ HOÀN TẤT THÀNH CÔNG ===")

if __name__ == "__main__":
    main()

