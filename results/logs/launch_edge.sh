#!/bin/bash
cd "$(dirname "$0")/../.."
V=$(py -c "import yaml;print(' '.join(b['name'] for b in yaml.safe_load(open('configs/experiments/edge_gallery.yaml',encoding='utf-8'))['benchmarks']))")
set -- $V
nohup bash results/logs/run_edge_gallery.sh "${@:1:17}" > results/logs/edge_gallery_A.log 2>&1 &
nohup bash results/logs/run_edge_gallery.sh "${@:18:9}" > results/logs/edge_gallery_B.log 2>&1 &
nohup bash results/logs/run_edge_gallery.sh "${@:27:8}" > results/logs/edge_gallery_C.log 2>&1 &
echo "A: ${@:1:17}"; echo "B: ${@:18:9}"; echo "C: ${@:27:8}"
