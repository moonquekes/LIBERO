#!/usr/bin/env bash
# 录失败 case 视频：跑指定 bddl 6 条（开环官方协议），失败集自动存 mp4+拼图
trap '' PIPE
set -u
B=${1:?usage: record_fail_videos.sh <bddl>}
D=/home/x/vla/libero/data/suction_dataset_multi_part_sorting
PY=/home/x/miniforge3/envs/vla-adapter/bin/python
VID="$D/eval_r2/fail_videos_r4"
cd /home/x/vla/libero
export MUJOCO_GL=egl
mkdir -p "$VID"
$PY scripts/eval_sorting_diag.py --bddls "$B" --episodes 6 --workers 3 --max_steps 600 \
  --save_video_dir "$VID" \
  --out_json "$VID/diag_$B.json" 2>&1 | tail -n 6
