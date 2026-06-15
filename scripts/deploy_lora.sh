#!/usr/bin/env bash
# 部署 LoRA ckpt 到 h100:8000：$1=ckpt 目录后缀（如 r5press），$2=step；顺带确保本地隧道
trap '' PIPE
set -u
ARM=${1:?usage: deploy_lora.sh <arm> <step>}
STEP=${2:?usage: deploy_lora.sh <arm> <step>}
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=10 -p 10036 root@10.31.118.28"
$SSH "pkill -f '[d]eploy'" || true
sleep 2
# & 只包 nohup 简单命令，前置步骤用 ; 收尾
timeout 30 $SSH "cd /root/data/X-VLA; rm -rf /root/data/deploy_out_scan; nohup /root/miniconda3/envs/lingbot-vla/bin/python deploy.py --model_path /root/data/X-VLA-Libero --LoRA_path /root/data/X-VLA-sorting-ckpt-$ARM/ckpt-$STEP --output_dir /root/data/deploy_out_scan --port 8000 --disable_slurm > /root/data/deploy_scan.log 2>&1 </dev/null & disown; exit 0"
for i in $(seq 1 30); do
  sleep 5
  $SSH "grep -q 'Uvicorn running' /root/data/deploy_scan.log" && {
    # 隧道走 Windows 侧常驻 ssh（mirrored 网络共享 localhost），只验通不自建
    curl -s -o /dev/null -m 5 http://localhost:8000/docs || { echo TUNNEL_DOWN; exit 1; }
    echo DEPLOY_READY; exit 0; }
done
echo DEPLOY_TIMEOUT
exit 1
