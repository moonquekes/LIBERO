#!/usr/bin/env bash
# 部署 r4full 全量微调 ckpt（默认 7500）到 h100:8000，前台等就绪；顺带确保本地 8000 隧道
trap '' PIPE
set -u
CKPT=${1:-7500}
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=10 -p 10036 root@10.31.118.28"
$SSH "pkill -f '[d]eploy'" || true
sleep 2
# & 只包 nohup 简单命令，前置步骤用 ; 收尾
timeout 30 $SSH "cd /root/data/X-VLA; rm -rf /root/data/deploy_out_full; nohup /root/miniconda3/envs/lingbot-vla/bin/python deploy.py --model_path /root/data/X-VLA-sorting-ckpt-r4full/ckpt-$CKPT --processor_path /root/data/X-VLA-Libero --output_dir /root/data/deploy_out_full --port 8000 --disable_slurm > /root/data/deploy_full.log 2>&1 </dev/null & disown; exit 0"
for i in $(seq 1 30); do
  sleep 5
  $SSH "grep -q 'Uvicorn running' /root/data/deploy_full.log" && {
    pgrep -f "ssh -f -N -L 8000" >/dev/null || ssh -f -N -L 8000:localhost:8000 -p 10036 root@10.31.118.28
    echo DEPLOY_READY; exit 0; }
done
echo DEPLOY_TIMEOUT
$SSH "tail -n 20 /root/data/deploy_full.log"
exit 1
