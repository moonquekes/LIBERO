#!/usr/bin/env bash
# 脚本专家采集：rect/round 8cm 评测分布各 15 + round 4cm 补 12（无需渲染/h100，纯 CPU）
set -u
D=/home/x/vla/libero/data/suction_dataset_multi_part_sorting
PY=/home/x/miniforge3/envs/vla-adapter/bin/python
cd /home/x/vla/libero
$PY -u scripts/scripted_expert_collect.py \
  --bddl pick_up_the_rectangular_steel_plate_and_place_it_in_the_red_bin.bddl \
  --num 15 --workers 4 --out-dir "$D/raw_hdf5/rectangular_red_bin_selfc8" \
  > "$D/selfc8_rect.log" 2>&1
$PY -u scripts/scripted_expert_collect.py \
  --bddl pick_up_the_round_steel_plate_and_place_it_in_the_blue_bin.bddl \
  --num 15 --workers 4 --out-dir "$D/raw_hdf5/round_blue_bin_selfc8" \
  > "$D/selfc8_round.log" 2>&1
$PY -u scripts/scripted_expert_collect.py \
  --bddl pick_up_the_round_steel_plate_and_place_it_in_the_blue_bin_diag4cm.bddl \
  --num 12 --workers 4 --out-dir "$D/raw_hdf5/round_blue_bin_selfc4" \
  > "$D/selfc8_round4cm.log" 2>&1
echo SELFC8_DONE
for d in rectangular_red_bin_selfc8 round_blue_bin_selfc8 round_blue_bin_selfc4; do
  echo "$d: $(ls "$D/raw_hdf5/$d" | wc -l)"
done
