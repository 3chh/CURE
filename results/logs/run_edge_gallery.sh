#!/bin/bash
# Vòng 60: chạy các biến thể gallery được truyền vào, mỗi biến thể một tiến trình CPU; chạy lại sẽ tiếp từ chỗ dừng.
export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export CURE_FEATURES="C:/Users/Hp Victus/Desktop/DeepFakeDetection/data/openood_features"
for v in "$@"; do
  for t in 1 2 3; do py -m cure_ood.cli run edge_gallery --benchmark $v && break; echo "[retry $v $t]"; sleep 30; done
done
echo "[xong: $*]"
