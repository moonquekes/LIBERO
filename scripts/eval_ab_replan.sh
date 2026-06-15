#!/usr/bin/env bash
# 全量 A+B 闭环评测（--replan_every 25），服务须已就绪
trap '' PIPE
set -u
CKPT=${1:?usage: eval_ab_replan.sh ckpt-7500}
D=/home/x/vla/libero/data/suction_dataset_multi_part_sorting
PY=/home/x/miniforge3/envs/vla-adapter/bin/python
cd /home/x/vla/libero
export MUJOCO_GL=egl
$PY scripts/eval_sorting_diag.py --mode A --episodes 8 --workers 4 --max_steps 600 \
  --replan_every 25 \
  --out_json "$D/eval_r2/diag_A_r3_${CKPT}_replan.json" > "$D/eval_A_r3_${CKPT}_replan.log" 2>&1
$PY scripts/eval_sorting_diag.py --mode B --episodes 6 --workers 4 --max_steps 600 \
  --replan_every 25 \
  --out_json "$D/eval_r2/diag_B_r3_${CKPT}_replan.json" > "$D/eval_B_r3_${CKPT}_replan.log" 2>&1
{ echo "REPLAN_DONE $CKPT"
  tail -n 10 "$D/eval_A_r3_${CKPT}_replan.log"
  tail -n 12 "$D/eval_B_r3_${CKPT}_replan.log"
} | tee "$D/replan_done_$CKPT.txt"
