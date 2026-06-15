#!/usr/bin/env bash
# B 模式半场评测（前台防断管）：$1=1|2，$2=tag
trap '' PIPE
set -u
H=${1:?usage: eval_b_half.sh 1|2 <tag>}
TAG=${2:?usage: eval_b_half.sh 1|2 <tag>}
D=/home/x/vla/libero/data/suction_dataset_multi_part_sorting
PY=/home/x/miniforge3/envs/vla-adapter/bin/python
cd /home/x/vla/libero
export MUJOCO_GL=egl
if [ "$H" = "1" ]; then
  BD="pick_up_the_rectangular_steel_plate_and_place_it_in_the_red_bin_swap1.bddl,pick_up_the_rectangular_steel_plate_and_place_it_in_the_red_bin_swap2.bddl,pick_up_the_round_steel_plate_and_place_it_in_the_blue_bin_swap0.bddl"
else
  BD="pick_up_the_round_steel_plate_and_place_it_in_the_blue_bin_swap2.bddl,pick_up_the_triangular_steel_plate_and_place_it_in_the_yellow_bin_swap0.bddl,pick_up_the_triangular_steel_plate_and_place_it_in_the_yellow_bin_swap1.bddl"
fi
VID="$D/eval_r2/fail_videos_$TAG"
mkdir -p "$VID"
$PY scripts/eval_sorting_diag.py --bddls "$BD" --episodes 6 --workers 4 --max_steps 600 \
  --save_video_dir "$VID" \
  --out_json "$D/eval_r2/diag_B${H}_$TAG.json" 2>&1 | tee "$D/eval_r2/eval_B${H}_$TAG.log" | tail -n 6
