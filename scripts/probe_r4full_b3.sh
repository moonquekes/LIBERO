#!/usr/bin/env bash
# ckpt 扫描探针：只测 r4full ckpt-7500 在 B 崩掉的 3 个任务（rect_swap2/tri_swap0/tri_swap1）各 6 条
trap '' PIPE
set -u
CKPT=${1:?usage: probe_r4full_b3.sh <ckpt-step>}
D=/home/x/vla/libero/data/suction_dataset_multi_part_sorting
PY=/home/x/miniforge3/envs/vla-adapter/bin/python
cd /home/x/vla/libero
export MUJOCO_GL=egl
BD="pick_up_the_rectangular_steel_plate_and_place_it_in_the_red_bin_swap2.bddl,pick_up_the_triangular_steel_plate_and_place_it_in_the_yellow_bin_swap0.bddl,pick_up_the_triangular_steel_plate_and_place_it_in_the_yellow_bin_swap1.bddl"
$PY scripts/eval_sorting_diag.py --bddls "$BD" --episodes 6 --workers 4 --max_steps 600 \
  --out_json "$D/eval_r2/probe_b3_r4full_$CKPT.json" 2>&1 | tail -n 10
