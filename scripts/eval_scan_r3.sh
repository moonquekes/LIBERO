#!/usr/bin/env bash
# 扫描 r3selfc 中段 ckpt：重启 h100 服务 → A 评测 → B 评测 → 汇总（有确定终点，可挂后台）
set -u
CKPT=${1:?usage: eval_scan_r3.sh ckpt-5000}
D=/home/x/vla/libero/data/suction_dataset_multi_part_sorting
PY=/home/x/miniforge3/envs/vla-adapter/bin/python
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=10 -p 10036 root@10.31.118.28"

# 杀旧服务（单独一跳）
$SSH "pkill -f '[d]eploy'" || true
sleep 3
# 起新服务（disown 秒回；& 只能包 nohup 简单命令，前置步骤用 ; 收尾，否则子壳攥管道挂死 ssh）
timeout 30 $SSH "cd /root/data/X-VLA; rm -rf /root/data/deploy_out_scan; nohup /root/miniconda3/envs/lingbot-vla/bin/python deploy.py --model_path /root/data/X-VLA-Libero --LoRA_path /root/data/X-VLA-sorting-ckpt-r3selfc/$CKPT --output_dir /root/data/deploy_out_scan --port 8000 --disable_slurm > /root/data/deploy_scan.log 2>&1 </dev/null & disown; exit 0"
# 等服务就绪
for i in $(seq 1 30); do
  sleep 5
  $SSH "grep -q 'Uvicorn running' /root/data/deploy_scan.log" && break
done
$SSH "tail -n 1 /root/data/deploy_scan.log"

cd /home/x/vla/libero
export MUJOCO_GL=egl
$PY scripts/eval_sorting_diag.py --mode A --episodes 8 --workers 4 --max_steps 600 \
  --out_json "$D/eval_r2/diag_A_r3_$CKPT.json" > "$D/eval_A_r3_$CKPT.log" 2>&1
echo "A_DONE $CKPT"
$PY scripts/eval_sorting_diag.py --mode B --episodes 6 --workers 4 --max_steps 600 \
  --out_json "$D/eval_r2/diag_B_r3_$CKPT.json" > "$D/eval_B_r3_$CKPT.log" 2>&1
echo "B_DONE $CKPT"
for f in "$D/eval_A_r3_$CKPT.log" "$D/eval_B_r3_$CKPT.log"; do
  echo "== $f"
  tail -n 14 "$f"
done
echo "SCAN_DONE $CKPT"
