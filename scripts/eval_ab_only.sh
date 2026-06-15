#!/usr/bin/env bash
# 仅评测 A+B（h100 服务已就绪时用）。防断管：忽略 SIGPIPE，结果全部落 WSL 文件。
trap '' PIPE
set -u
CKPT=${1:?usage: eval_ab_only.sh ckpt-5000}
D=/home/x/vla/libero/data/suction_dataset_multi_part_sorting
PY=/home/x/miniforge3/envs/vla-adapter/bin/python
cd /home/x/vla/libero
export MUJOCO_GL=egl
$PY scripts/eval_sorting_diag.py --mode A --episodes 8 --workers 4 --max_steps 600 \
  --out_json "$D/eval_r2/diag_A_r3_$CKPT.json" > "$D/eval_A_r3_$CKPT.log" 2>&1
$PY scripts/eval_sorting_diag.py --mode B --episodes 6 --workers 4 --max_steps 600 \
  --out_json "$D/eval_r2/diag_B_r3_$CKPT.json" > "$D/eval_B_r3_$CKPT.log" 2>&1
{ echo "SCAN_DONE $CKPT"
  tail -n 8 "$D/eval_A_r3_$CKPT.log"
  tail -n 8 "$D/eval_B_r3_$CKPT.log"
} | tee "$D/scan_done_$CKPT.txt"
