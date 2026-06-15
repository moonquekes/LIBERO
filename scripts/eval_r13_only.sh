#!/usr/bin/env bash
# r13 评测（部署已就绪）：A/B 开环(r13cf) + A/B 末端下压(r13cftd)，落 done 文件。
trap '' PIPE
set -u
D=/home/x/vla/libero/data/suction_dataset_multi_part_sorting
PY=/home/x/miniforge3/envs/vla-adapter/bin/python
cd /home/x/vla/libero
export MUJOCO_GL=egl
mkdir -p "$D/eval_r2"
rm -f "$D/eval_r2/r13_eval_done.txt"

run_one () {  # $1=mode $2=tag $3=extra
  local M=$1 TAG=$2 EXTRA=$3
  local EP; EP=$([ "$M" = A ] && echo 8 || echo 6)
  echo "=== eval $M $TAG (episodes=$EP) ==="
  $PY scripts/eval_sorting_diag.py --mode "$M" --episodes "$EP" --workers 4 --max_steps 600 $EXTRA \
    --out_json "$D/eval_r2/diag_${M}_${TAG}.json" 2>&1 | tee "$D/eval_r2/eval_${M}_${TAG}.log" | tail -n 6
}

run_one A r13cf  ""
run_one B r13cf  ""
run_one A r13cftd "--terminal_descent"
run_one B r13cftd "--terminal_descent"
echo R13_EVAL_DONE | tee "$D/eval_r2/r13_eval_done.txt"
