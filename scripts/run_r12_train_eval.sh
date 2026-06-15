#!/usr/bin/env bash
# r12 一体化收尾：等训练 → 部署 → 开环 A/B(r12cf) + 带末端下压 A/B(r12cftd)
trap '' PIPE
set -u
D=/home/x/vla/libero/data/suction_dataset_multi_part_sorting
PY=/home/x/miniforge3/envs/vla-adapter/bin/python
cd /home/x/vla/libero
export MUJOCO_GL=egl
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -p 10036 root@10.31.118.28"
$SSH "timeout 7200 bash -c 'while pgrep -f \"[p]eft_train\" >/dev/null; do sleep 30; done'; echo TRAIN_EXITED; tail -n 2 /root/data/train_r12.log" | tee "$D/r12_train_done.txt"
$SSH "ls /root/data/X-VLA-sorting-ckpt-r12cf/ckpt-7500/state.json" >/dev/null || { echo NO_CKPT | tee -a "$D/eval_r2/r12_eval_done.txt"; exit 1; }
bash scripts/deploy_lora.sh r12cf 7500 || { echo DEPLOY_FAIL | tee -a "$D/eval_r2/r12_eval_done.txt"; exit 1; }
for M in A B; do
  EP=$([ "$M" = A ] && echo 8 || echo 6)
  $PY scripts/eval_sorting_diag.py --mode "$M" --episodes "$EP" --workers 4 --max_steps 600 \
    --out_json "$D/eval_r2/diag_${M}_r12cf.json" 2>&1 | tee "$D/eval_r2/eval_${M}_r12cf.log" | tail -n 4
done
for M in A B; do
  EP=$([ "$M" = A ] && echo 8 || echo 6)
  $PY scripts/eval_sorting_diag.py --mode "$M" --episodes "$EP" --workers 4 --max_steps 600 --terminal_descent \
    --out_json "$D/eval_r2/diag_${M}_r12cftd.json" 2>&1 | tee "$D/eval_r2/eval_${M}_r12cftd.log" | tail -n 4
done
echo R12_ALL_DONE | tee -a "$D/eval_r2/r12_eval_done.txt"
