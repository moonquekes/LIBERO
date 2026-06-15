#!/usr/bin/env bash
# r5press 复测：A/B 开环复现最好成绩(目标 A 66.7 / B 100 / 合计 86.7)。落 done 文件。
trap '' PIPE
set -u
D=/home/x/vla/libero/data/suction_dataset_multi_part_sorting
PY=/home/x/miniforge3/envs/vla-adapter/bin/python
cd /home/x/vla/libero
export MUJOCO_GL=egl
mkdir -p "$D/eval_r2"
rm -f "$D/eval_r2/r5press_verify_done.txt"

for M in A B; do
  EP=$([ "$M" = A ] && echo 8 || echo 6)
  echo "=== eval $M r5press_verify (episodes=$EP) ==="
  $PY scripts/eval_sorting_diag.py --mode "$M" --episodes "$EP" --workers 4 --max_steps 600 \
    --out_json "$D/eval_r2/diag_${M}_r5press_verify.json" 2>&1 \
    | tee "$D/eval_r2/eval_${M}_r5press_verify.log" | tail -n 6
done
echo R5PRESS_VERIFY_DONE | tee "$D/eval_r2/r5press_verify_done.txt"
