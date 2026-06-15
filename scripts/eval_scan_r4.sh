#!/usr/bin/env bash
# r4 终评：重启 h100 服务（r4 ckpt）→ 开环 A+B → 汇总落盘
trap '' PIPE
set -u
CKPT=${1:?usage: eval_scan_r4.sh ckpt-7500}
D=/home/x/vla/libero/data/suction_dataset_multi_part_sorting
PY=/home/x/miniforge3/envs/vla-adapter/bin/python
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=10 -p 10036 root@10.31.118.28"

$SSH "pkill -f '[d]eploy'" || true
sleep 3
timeout 30 $SSH "cd /root/data/X-VLA; rm -rf /root/data/deploy_out_scan; nohup /root/miniconda3/envs/lingbot-vla/bin/python deploy.py --model_path /root/data/X-VLA-Libero --LoRA_path /root/data/X-VLA-sorting-ckpt-r4selfc/$CKPT --output_dir /root/data/deploy_out_scan --port 8000 --disable_slurm > /root/data/deploy_scan.log 2>&1 </dev/null & disown; exit 0"
for i in $(seq 1 30); do
  sleep 5
  $SSH "grep -q 'Uvicorn running' /root/data/deploy_scan.log" && break
done

cd /home/x/vla/libero
export MUJOCO_GL=egl
$PY scripts/eval_sorting_diag.py --mode A --episodes 8 --workers 4 --max_steps 600 \
  --out_json "$D/eval_r2/diag_A_r4_$CKPT.json" > "$D/eval_A_r4_$CKPT.log" 2>&1
$PY scripts/eval_sorting_diag.py --mode B --episodes 6 --workers 4 --max_steps 600 \
  --out_json "$D/eval_r2/diag_B_r4_$CKPT.json" > "$D/eval_B_r4_$CKPT.log" 2>&1
{ echo "R4_EVAL_DONE $CKPT"
  tail -n 12 "$D/eval_A_r4_$CKPT.log"
  tail -n 14 "$D/eval_B_r4_$CKPT.log"
} | tee "$D/r4_eval_done_$CKPT.txt"
