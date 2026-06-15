#!/usr/bin/env bash
# r5 双臂全协议评测：r5press(r8) 与 r5pr16(r16) 各自 deploy→A→B，产物分 tag 落盘
trap '' PIPE
set -u
D=/home/x/vla/libero/data/suction_dataset_multi_part_sorting
cd /home/x/vla/libero
for ARM in r5press r5pr16; do
  bash scripts/deploy_lora.sh "$ARM" 7500 || { echo "DEPLOY_FAIL_$ARM" | tee -a "$D/eval_r2/r5_eval_done.txt"; exit 1; }
  bash scripts/eval_mode_tag.sh A "$ARM" >> "$D/eval_r2/r5_eval_progress.txt" 2>&1
  bash scripts/eval_mode_tag.sh B "$ARM" >> "$D/eval_r2/r5_eval_progress.txt" 2>&1
  echo "ARM_${ARM}_DONE" | tee -a "$D/eval_r2/r5_eval_done.txt"
done
echo R5_EVAL_ALL_DONE | tee -a "$D/eval_r2/r5_eval_done.txt"
