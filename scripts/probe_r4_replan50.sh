#!/usr/bin/env bash
# r4 + replan50 探针：三个 no_contact 失败点（rect-A/round-A/tri_swap1）+ tri-A 守门
trap '' PIPE
set -u
D=/home/x/vla/libero/data/suction_dataset_multi_part_sorting
PY=/home/x/miniforge3/envs/vla-adapter/bin/python
cd /home/x/vla/libero
export MUJOCO_GL=egl
for B in pick_up_the_rectangular_steel_plate_and_place_it_in_the_red_bin.bddl \
         pick_up_the_round_steel_plate_and_place_it_in_the_blue_bin.bddl \
         pick_up_the_triangular_steel_plate_and_place_it_in_the_yellow_bin_swap1.bddl \
         pick_up_the_triangular_steel_plate_and_place_it_in_the_yellow_bin.bddl; do
  $PY scripts/eval_sorting_diag.py --bddls "$B" --episodes 6 --workers 3 --max_steps 600 \
    --replan_every 50 \
    --out_json "$D/eval_r2/probe_r4_50_$B.json" > "$D/probe_r4_50_$B.log" 2>&1
done
{ echo PROBE_R4_50_DONE
  grep -H '总成功率' "$D"/probe_r4_50_*.log
} | tee "$D/probe_r4_50_done.txt"
