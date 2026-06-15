#!/usr/bin/env bash
# 录 r4full ckpt-7500 的典型成功视频：round-A（全量修复点）+ tri-A，开环官方协议
trap '' PIPE
set -u
D=/home/x/vla/libero/data/suction_dataset_multi_part_sorting
PY=/home/x/miniforge3/envs/vla-adapter/bin/python
cd /home/x/vla/libero
export MUJOCO_GL=egl
VID="$D/eval_r2/typical_videos_r4full"
mkdir -p "$VID"
$PY scripts/eval_sorting_diag.py --bddls pick_up_the_round_steel_plate_and_place_it_in_the_blue_bin.bddl \
  --episodes 3 --workers 3 --max_steps 600 --save_video_dir "$VID" --save_success \
  --out_json "$VID/diag_round_succ.json" 2>&1 | tail -n 4
$PY scripts/eval_sorting_diag.py --bddls pick_up_the_triangular_steel_plate_and_place_it_in_the_yellow_bin.bddl \
  --episodes 2 --workers 2 --max_steps 600 --save_video_dir "$VID" --save_success \
  --out_json "$VID/diag_tri_succ.json" 2>&1 | tail -n 4
echo TYPICAL_REC_DONE
