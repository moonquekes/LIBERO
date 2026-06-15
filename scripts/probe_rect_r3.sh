#!/usr/bin/env bash
# rect-A 顽固失败三探针（ckpt-7500）：①开环+录像 ②闭环 replan ③wide 散布对照
trap '' PIPE
set -u
D=/home/x/vla/libero/data/suction_dataset_multi_part_sorting
PY=/home/x/miniforge3/envs/vla-adapter/bin/python
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=10 -p 10036 root@10.31.118.28"
RECT=pick_up_the_rectangular_steel_plate_and_place_it_in_the_red_bin.bddl
RECTW=pick_up_the_rectangular_steel_plate_and_place_it_in_the_red_bin_diagwide.bddl

$SSH "pkill -f '[d]eploy'" || true
sleep 3
$SSH "cd /root/data/X-VLA && rm -rf /root/data/deploy_out_scan && nohup /root/miniconda3/envs/lingbot-vla/bin/python deploy.py --model_path /root/data/X-VLA-Libero --LoRA_path /root/data/X-VLA-sorting-ckpt-r3selfc/ckpt-7500 --output_dir /root/data/deploy_out_scan --port 8000 --disable_slurm > /root/data/deploy_scan.log 2>&1 </dev/null & disown; exit 0"
for i in $(seq 1 30); do
  sleep 5
  $SSH "grep -q 'Uvicorn running' /root/data/deploy_scan.log" && break
done

cd /home/x/vla/libero
export MUJOCO_GL=egl
mkdir -p "$D/eval_r2/vid_rect_open"
$PY scripts/eval_sorting_diag.py --bddls "$RECT" --episodes 4 --workers 2 --max_steps 600 \
  --save_video_dir "$D/eval_r2/vid_rect_open" \
  --out_json "$D/eval_r2/probe_rect_open.json" > "$D/probe_rect_open.log" 2>&1
$PY scripts/eval_sorting_diag.py --bddls "$RECT" --episodes 4 --workers 2 --max_steps 600 \
  --replan_every 25 \
  --out_json "$D/eval_r2/probe_rect_replan.json" > "$D/probe_rect_replan.log" 2>&1
$PY scripts/eval_sorting_diag.py --bddls "$RECTW" --episodes 4 --workers 2 --max_steps 600 \
  --out_json "$D/eval_r2/probe_rect_wide.json" > "$D/probe_rect_wide.log" 2>&1
{ echo PROBE_DONE
  for t in open replan wide; do
    echo "== $t"
    grep -E '总成功率|SUCCESS|NO_GRASP|WRONG|PLACED' "$D/probe_rect_$t.log" | grep -v 'ep[0-9]' | tail -n 6
  done
  ls "$D/eval_r2/vid_rect_open" 2>/dev/null
} | tee "$D/probe_rect_done.txt"
