#!/usr/bin/env bash
# r4full 正式开环评测单模式：$1=A|B，$2=ckpt tag（如 2500，产物带后缀防覆盖）；失败视频自动存
trap '' PIPE
set -u
M=${1:?usage: eval_r4full_mode.sh A|B [tag]}
TAG=${2:+_$2}
D=/home/x/vla/libero/data/suction_dataset_multi_part_sorting
PY=/home/x/miniforge3/envs/vla-adapter/bin/python
cd /home/x/vla/libero
export MUJOCO_GL=egl
VID="$D/eval_r2/fail_videos_r4full$TAG"
mkdir -p "$VID"
if [ "$M" = "A" ]; then EP=8; else EP=6; fi
$PY scripts/eval_sorting_diag.py --mode "$M" --episodes "$EP" --workers 4 --max_steps 600 \
  --save_video_dir "$VID" \
  --out_json "$D/eval_r2/diag_${M}_r4full$TAG.json" 2>&1 | tee "$D/eval_r2/eval_${M}_r4full$TAG.log" | tail -n 8
echo "EVAL_${M}${TAG}_DONE" | tee -a "$D/eval_r2/r4full_eval_done.txt"
