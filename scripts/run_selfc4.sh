#!/usr/bin/env bash
# 4cm 布局自采：rect 15 + round 15（r2bal 当老师，需 8000 端口服务在线）
set -u
D=/home/x/vla/libero/data/suction_dataset_multi_part_sorting
PY=/home/x/miniforge3/envs/vla-adapter/bin/python
export MUJOCO_GL=egl
cd /home/x/vla/libero
$PY -u scripts/selfcollect_rollout.py \
  --bddl pick_up_the_rectangular_steel_plate_and_place_it_in_the_red_bin_diag4cm.bddl \
  --num 15 --workers 4 --out-dir "$D/raw_hdf5/rectangular_red_bin_selfc4" \
  > "$D/selfc_rect.log" 2>&1
$PY -u scripts/selfcollect_rollout.py \
  --bddl pick_up_the_round_steel_plate_and_place_it_in_the_blue_bin_diag4cm.bddl \
  --num 15 --workers 4 --out-dir "$D/raw_hdf5/round_blue_bin_selfc4" \
  > "$D/selfc_round.log" 2>&1
echo SELFC_DONE
tail -2 "$D/selfc_rect.log" "$D/selfc_round.log"
