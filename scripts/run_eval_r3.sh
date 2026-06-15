#!/usr/bin/env bash
# r3selfc ckpt-7500 完整 A+B 评测（口径与 r2bal 对齐：A 3x8=24，B 6x6=36）
set -u
D=/home/x/vla/libero/data/suction_dataset_multi_part_sorting
PY=/home/x/miniforge3/envs/vla-adapter/bin/python
export MUJOCO_GL=egl
cd /home/x/vla/libero
timeout 2400 $PY -u scripts/eval_sorting_diag.py --mode A --episodes 8 --workers 4 \
  --max_steps 600 --out_json "$D/eval_r2/diag_A_r3selfc.json" > "$D/eval_A_r3.log" 2>&1
echo A_DONE
timeout 2400 $PY -u scripts/eval_sorting_diag.py --mode B --episodes 6 --workers 4 \
  --max_steps 600 --out_json "$D/eval_r2/diag_B_r3selfc.json" > "$D/eval_B_r3.log" 2>&1
echo EVAL_R3_DONE
grep -h '总成功率\|success_rate' "$D/eval_r2/diag_A_r3selfc.json" "$D/eval_r2/diag_B_r3selfc.json" 2>/dev/null
tail -3 "$D/eval_A_r3.log" "$D/eval_B_r3.log"
