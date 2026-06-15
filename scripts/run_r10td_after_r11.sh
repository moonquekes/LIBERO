#!/usr/bin/env bash
# Plan A 测试：等 r11 链完成（避端口冲突）→ 部署 r10lang → 带末端下压跑 A/B（r10td）。
# r10 全场零抓错、失败全是 no_contact 欠冲 → 末端下压应把多数欠冲转成功。
trap '' PIPE
set -u
D=/home/x/vla/libero/data/suction_dataset_multi_part_sorting
PY=/home/x/miniforge3/envs/vla-adapter/bin/python
cd /home/x/vla/libero
export MUJOCO_GL=egl
# 1) 等 r11 链完成（最多 70 分钟）
for i in $(seq 1 70); do
  [ -f "$D/eval_r2/r11_eval_done.txt" ] && break
  sleep 60
done
# 2) 部署 r10lang（deploy 脚本自带 pkill 旧 deploy）
bash scripts/deploy_lora.sh r10lang 7500 || { echo DEPLOY_FAIL | tee "$D/eval_r2/r10td_done.txt"; exit 1; }
VID="$D/eval_r2/fail_videos_r10td"
mkdir -p "$VID"
# 3) 带 --terminal_descent 跑 A/B
$PY scripts/eval_sorting_diag.py --mode A --episodes 8 --workers 4 --max_steps 600 --terminal_descent \
  --save_video_dir "$VID" --out_json "$D/eval_r2/diag_A_r10td.json" 2>&1 | tee "$D/eval_r2/eval_A_r10td.log" | tail -n 5
$PY scripts/eval_sorting_diag.py --mode B --episodes 6 --workers 4 --max_steps 600 --terminal_descent \
  --save_video_dir "$VID" --out_json "$D/eval_r2/diag_B_r10td.json" 2>&1 | tee "$D/eval_r2/eval_B_r10td.log" | tail -n 5
echo R10TD_DONE | tee "$D/eval_r2/r10td_done.txt"
