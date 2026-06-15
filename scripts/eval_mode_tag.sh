#!/usr/bin/env bash
# 通用单模式评测：$1=A|B，$2=tag（产物 diag_/eval_/fail_videos_ 带 tag）
trap '' PIPE
set -u
M=${1:?usage: eval_mode_tag.sh A|B <tag>}
TAG=${2:?usage: eval_mode_tag.sh A|B <tag>}
D=/home/x/vla/libero/data/suction_dataset_multi_part_sorting
PY=/home/x/miniforge3/envs/vla-adapter/bin/python
cd /home/x/vla/libero
export MUJOCO_GL=egl
VID="$D/eval_r2/fail_videos_$TAG"
mkdir -p "$VID"
if [ "$M" = "A" ]; then EP=8; else EP=6; fi
$PY scripts/eval_sorting_diag.py --mode "$M" --episodes "$EP" --workers 4 --max_steps 600 \
  --save_video_dir "$VID" \
  --out_json "$D/eval_r2/diag_${M}_$TAG.json" 2>&1 | tee "$D/eval_r2/eval_${M}_$TAG.log" | tail -n 4
echo "EVAL_${M}_${TAG}_DONE"
