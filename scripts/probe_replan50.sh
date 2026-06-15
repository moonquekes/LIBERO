#!/usr/bin/env bash
# replan_every=50 折中探针：rect-A（欠冲点）、tri-A（被replan25打断点）、rect-swap1（被replan25打断点）
trap '' PIPE
set -u
D=/home/x/vla/libero/data/suction_dataset_multi_part_sorting
PY=/home/x/miniforge3/envs/vla-adapter/bin/python
cd /home/x/vla/libero
export MUJOCO_GL=egl
for B in pick_up_the_rectangular_steel_plate_and_place_it_in_the_red_bin.bddl \
         pick_up_the_triangular_steel_plate_and_place_it_in_the_yellow_bin.bddl \
         pick_up_the_rectangular_steel_plate_and_place_it_in_the_red_bin_swap1.bddl; do
  $PY scripts/eval_sorting_diag.py --bddls "$B" --episodes 6 --workers 3 --max_steps 600 \
    --replan_every 50 \
    --out_json "$D/eval_r2/probe50_$B.json" > "$D/probe50_$B.log" 2>&1
done
{ echo PROBE50_DONE
  grep -h '总成功率' "$D"/probe50_*.log
} | tee "$D/probe50_done.txt"
