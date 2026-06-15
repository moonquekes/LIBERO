#!/usr/bin/env bash
# replan_every=100 探针：rect-A（看修复是否保留）、tri-A（看打断是否消失）
trap '' PIPE
set -u
D=/home/x/vla/libero/data/suction_dataset_multi_part_sorting
PY=/home/x/miniforge3/envs/vla-adapter/bin/python
cd /home/x/vla/libero
export MUJOCO_GL=egl
for B in pick_up_the_rectangular_steel_plate_and_place_it_in_the_red_bin.bddl \
         pick_up_the_triangular_steel_plate_and_place_it_in_the_yellow_bin.bddl; do
  $PY scripts/eval_sorting_diag.py --bddls "$B" --episodes 6 --workers 3 --max_steps 600 \
    --replan_every 100 \
    --out_json "$D/eval_r2/probe100_$B.json" > "$D/probe100_$B.log" 2>&1
done
{ echo PROBE100_DONE
  grep -h '总成功率' "$D"/probe100_*.log
} | tee "$D/probe100_done.txt"
