#!/usr/bin/env bash
# 补 ckpt-6500 的 tri_swap1 6 条（B 其余 25/25 已有）
trap '' PIPE
set -u
D=/home/x/vla/libero/data/suction_dataset_multi_part_sorting
PY=/home/x/miniforge3/envs/vla-adapter/bin/python
cd /home/x/vla/libero
export MUJOCO_GL=egl
$PY scripts/eval_sorting_diag.py \
  --bddls pick_up_the_triangular_steel_plate_and_place_it_in_the_yellow_bin_swap1.bddl \
  --episodes 6 --workers 3 --max_steps 600 \
  --out_json "$D/eval_r2/diag_trisw1_r4_ckpt-6500.json" 2>&1 | tail -n 8
