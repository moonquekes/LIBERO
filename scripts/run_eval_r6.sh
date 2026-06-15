#!/usr/bin/env bash
# r6press 全协议评测：deploy ckpt-7500 → A → B，产物 tag=r6press
trap '' PIPE
set -u
D=/home/x/vla/libero/data/suction_dataset_multi_part_sorting
cd /home/x/vla/libero
bash scripts/deploy_lora.sh r6press 7500 || { echo DEPLOY_FAIL | tee -a "$D/eval_r2/r6_eval_done.txt"; exit 1; }
bash scripts/eval_mode_tag.sh A r6press >> "$D/eval_r2/r6_eval_progress.txt" 2>&1
bash scripts/eval_mode_tag.sh B r6press >> "$D/eval_r2/r6_eval_progress.txt" 2>&1
echo R6_EVAL_DONE | tee -a "$D/eval_r2/r6_eval_done.txt"
