#!/bin/bash
# Chờ một phiên GPU Kaggle rảnh rồi đẩy bản trích ImageNet-1k đầy đủ (giới hạn 2 phiên GPU cùng lúc).
K="C:/Users/Hp Victus/AppData/Local/Programs/Python/Python312/Scripts/kaggle.exe"
D="C:/Users/HPVICT~1/AppData/Local/Temp/claude/c--Users-Hp-Victus-Desktop-DeepFakeDetection/2dc319e2-d3d3-45a6-9e0e-98692adce634/scratchpad/kaggle/kernel_in1k_full"
st() { "$K" kernels status "tnglxun3chh/$1" 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z_]*'; }
while true; do
  busy=0
  for k in cure-round60-gpu cure-round60b-gpu; do
    case "$(st $k)" in *RUNNING*|*QUEUED*) busy=$((busy+1));; esac
  done
  if [ "$busy" -lt 2 ]; then
    out=$(cd "$D" && "$K" kernels push -p . 2>&1 | tail -1)
    echo "$(date +%H:%M) $out"
    case "$out" in *success*) exit 0;; esac
  fi
  sleep 180
done
