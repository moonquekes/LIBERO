#!/usr/bin/env bash
# r8 一体化收尾：等 h100 训练退出 → 校验终点 ckpt → deploy → A/B 评测
# 运行方式：Windows Start-Process 隐藏 wsl.exe（脱离工具会话中继）；完成与否看产物文件
trap '' PIPE
set -u
D=/home/x/vla/libero/data/suction_dataset_multi_part_sorting
cd /home/x/vla/libero
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -p 10036 root@10.31.118.28"
$SSH "timeout 7200 bash -c 'while pgrep -f \"[p]eft_train\" >/dev/null; do sleep 30; done'; echo TRAIN_EXITED; tail -n 2 /root/data/train_r8.log" | tee "$D/r8_train_done.txt"
$SSH "ls /root/data/X-VLA-sorting-ckpt-r8div/ckpt-7500/state.json" >/dev/null || { echo NO_FINAL_CKPT | tee -a "$D/r8_train_done.txt"; exit 1; }
bash scripts/deploy_lora.sh r8div 7500 || { echo DEPLOY_FAIL | tee -a "$D/eval_r2/r8_eval_done.txt"; exit 1; }
bash scripts/eval_mode_tag.sh A r8div >> "$D/eval_r2/r8_eval_progress.txt" 2>&1
bash scripts/eval_mode_tag.sh B r8div >> "$D/eval_r2/r8_eval_progress.txt" 2>&1
echo R8_ALL_DONE | tee -a "$D/eval_r2/r8_eval_done.txt"
