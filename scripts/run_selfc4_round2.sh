#!/usr/bin/env bash
# round 重采（25° 吸盘，修复 10° 吸不上的问题）：已有 3 条，补 12 条
set -u
D=/home/x/vla/libero/data/suction_dataset_multi_part_sorting
PY=/home/x/miniforge3/envs/vla-adapter/bin/python
export MUJOCO_GL=egl
cd /home/x/vla/libero
$PY -u scripts/selfcollect_rollout.py \
  --bddl pick_up_the_round_steel_plate_and_place_it_in_the_blue_bin_diag4cm.bddl \
  --num 12 --workers 4 --out-dir "$D/raw_hdf5/round_blue_bin_selfc4" \
  > "$D/selfc_round2.log" 2>&1
echo SELFC_ROUND2_DONE
ls "$D/raw_hdf5/round_blue_bin_selfc4" | wc -l
tail -3 "$D/selfc_round2.log"
