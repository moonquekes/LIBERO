#!/usr/bin/env bash
# r5press 复测补跑 B（A 已得 15/24=62.5%）。落 done 文件。
trap '' PIPE
set -u
D=/home/x/vla/libero/data/suction_dataset_multi_part_sorting
PY=/home/x/miniforge3/envs/vla-adapter/bin/python
cd /home/x/vla/libero
export MUJOCO_GL=egl
rm -f "$D/eval_r2/r5press_B_done.txt"
echo "=== eval B r5press_verify (episodes=6) ==="
$PY scripts/eval_sorting_diag.py --mode B --episodes 6 --workers 4 --max_steps 600 \
  --out_json "$D/eval_r2/diag_B_r5press_verify.json" 2>&1 \
  | tee "$D/eval_r2/eval_B_r5press_verify.log" | tail -n 8
echo R5PRESS_B_DONE | tee "$D/eval_r2/r5press_B_done.txt"
