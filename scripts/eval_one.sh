#!/usr/bin/env bash
# 可复用单次评测：$1=mode(A/B) $2=tag。落 diag json + done 文件(供扫 ckpt 用)。
trap '' PIPE
set -u
M=$1; TAG=$2; EXTRA=${3:-}
D=/home/x/vla/libero/data/suction_dataset_multi_part_sorting
PY=/home/x/miniforge3/envs/vla-adapter/bin/python
cd /home/x/vla/libero
export MUJOCO_GL=egl
mkdir -p "$D/eval_r2"
EP=$([ "$M" = A ] && echo 8 || echo 6)
rm -f "$D/eval_r2/done_${M}_${TAG}.txt"
echo "=== eval $M $TAG (episodes=$EP) extra=[$EXTRA] ==="
$PY scripts/eval_sorting_diag.py --mode "$M" --episodes "$EP" --workers 4 --max_steps 600 $EXTRA \
  --out_json "$D/eval_r2/diag_${M}_${TAG}.json" 2>&1 \
  | tee "$D/eval_r2/eval_${M}_${TAG}.log" | tail -n 8
echo "DONE_${M}_${TAG}" | tee "$D/eval_r2/done_${M}_${TAG}.txt"
