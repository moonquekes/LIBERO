#!/usr/bin/env bash
# 渲染后端对比：集显（Mesa D3D12 默认 Intel UHD770）vs 独显（4090），同一 bddl 同条件 mini 评测计时
trap '' PIPE
set -u
D=/home/x/vla/libero/data/suction_dataset_multi_part_sorting
PY=/home/x/miniforge3/envs/vla-adapter/bin/python
cd /home/x/vla/libero
export MUJOCO_GL=egl
B=${1:-pick_up_the_triangular_steel_plate_and_place_it_in_the_yellow_bin.bddl}
mkdir -p "$D/eval_r2"
echo "== iGPU Intel UHD770 =="
t0=$(date +%s)
$PY scripts/eval_sorting_diag.py --bddls "$B" --episodes 4 --workers 3 --max_steps 600 \
  --out_json "$D/eval_r2/bench_igpu.json" 2>&1 | tail -n 3
t1=$(date +%s); echo "IGPU_SECONDS=$((t1-t0))"
echo "== dGPU NVIDIA 4090 =="
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
t0=$(date +%s)
$PY scripts/eval_sorting_diag.py --bddls "$B" --episodes 4 --workers 3 --max_steps 600 \
  --out_json "$D/eval_r2/bench_dgpu.json" 2>&1 | tail -n 3
t1=$(date +%s); echo "DGPU_SECONDS=$((t1-t0))"
