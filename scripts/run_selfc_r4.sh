#!/usr/bin/env bash
# r4 脚本专家采集：tri swap0/swap1 各 15（治 B-tri 抓错）+ rect-D 再 15（治 A-rect 开环欠冲）
trap '' PIPE
set -u
D=/home/x/vla/libero/data/suction_dataset_multi_part_sorting
PY=/home/x/miniforge3/envs/vla-adapter/bin/python
cd /home/x/vla/libero
$PY -u scripts/scripted_expert_collect.py \
  --bddl pick_up_the_triangular_steel_plate_and_place_it_in_the_yellow_bin_swap0.bddl \
  --num 15 --workers 4 --out-dir "$D/raw_hdf5/triangular_yellow_bin_sw0selfc8" \
  > "$D/selfc_r4_trisw0.log" 2>&1
$PY -u scripts/scripted_expert_collect.py \
  --bddl pick_up_the_triangular_steel_plate_and_place_it_in_the_yellow_bin_swap1.bddl \
  --num 15 --workers 4 --out-dir "$D/raw_hdf5/triangular_yellow_bin_sw1selfc8" \
  > "$D/selfc_r4_trisw1.log" 2>&1
$PY -u scripts/scripted_expert_collect.py \
  --bddl pick_up_the_rectangular_steel_plate_and_place_it_in_the_red_bin.bddl \
  --num 15 --workers 4 --out-dir "$D/raw_hdf5/rectangular_red_bin_selfc8b" \
  > "$D/selfc_r4_rect.log" 2>&1
{ echo SELFC_R4_DONE
  for d in triangular_yellow_bin_sw0selfc8 triangular_yellow_bin_sw1selfc8 rectangular_red_bin_selfc8b; do
    echo "$d: $(ls "$D/raw_hdf5/$d" 2>/dev/null | wc -l)"
  done
} | tee "$D/selfc_r4_done.txt"
