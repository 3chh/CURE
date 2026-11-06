#!/bin/bash
# Chờ Kaggle xử lý xong dataset ảnh rồi đẩy vòng 61a (train 4 backbone đầu).
K="C:/Users/Hp Victus/AppData/Local/Programs/Python/Python312/Scripts/kaggle.exe"
D="C:/Users/HPVICT~1/AppData/Local/Temp/claude/c--Users-Hp-Victus-Desktop-DeepFakeDetection/2dc319e2-d3d3-45a6-9e0e-98692adce634/scratchpad/kaggle/kernel_61a"
for i in $(seq 1 60); do
  if "$K" datasets list --mine 2>/dev/null | grep -q "cure-images-classic"; then
    out=$(cd "$D" && "$K" kernels push -p . 2>&1 | tail -1)
    echo "$(date +%H:%M) $out"
    exit 0
  fi
  sleep 120
done
echo "$(date +%H:%M) hết giờ chờ dataset ảnh"
