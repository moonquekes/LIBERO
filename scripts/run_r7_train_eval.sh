#!/usr/bin/env bash
# r7 一体化收尾：等 h100 训练退出 → 校验终点 ckpt → deploy → A/B 评测（哨兵折叠进管线，按新规范全程 WSL 包装）
trap '' PIPE
set -u
D=/home/x/vla/libero/data/suction_dataset_multi_part_sorting
cd /home/x/vla/libero
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -p 10036 root@10.31.118.28"
$SSH "timeout 7200 bash -c 'while pgrep -f \"[p]eft_train\" >/dev/null; do sleep 30; done'; echo TRAIN_EXITED; tail -n 2 /root/data/train_r7.log" | tee "$D/r7_train_done.txt"
$SSH "ls /root/data/X-VLA-sorting-ckpt-r7press/ckpt-7500/state.json" >/dev/null || { echo NO_FINAL_CKPT | tee -a "$D/r7_train_done.txt"; exit 1; }
bash scripts/deploy_lora.sh r7press 7500 || { echo DEPLOY_FAIL | tee -a "$D/eval_r2/r7_eval_done.txt"; exit 1; }
bash scripts/eval_mode_tag.sh A r7press >> "$D/eval_r2/r7_eval_progress.txt" 2>&1
bash scripts/eval_mode_tag.sh B r7press >> "$D/eval_r2/r7_eval_progress.txt" 2>&1
echo R7_ALL_DONE | tee -a "$D/eval_r2/r7_eval_done.txt"
