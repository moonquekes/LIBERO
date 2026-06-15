#!/usr/bin/env bash
# r7 全对称管线：补齐 5 个缺居中怼穿 demo 的形状×布局组合（各15条）→ audit → 增量打包 → r7 meta → 上传
trap '' PIPE
set -u
D=/home/x/vla/libero/data/suction_dataset_multi_part_sorting
PY=/home/x/miniforge3/envs/vla-adapter/bin/python
cd /home/x/vla/libero
DIRS="rectangular_red_bin_sw1selfc8 rectangular_red_bin_sw2selfc8 round_blue_bin_sw0selfc8 round_blue_bin_sw2selfc8 triangular_yellow_bin_selfc8"
bddl_of() {
  case "$1" in
    rectangular_red_bin_sw1selfc8) echo pick_up_the_rectangular_steel_plate_and_place_it_in_the_red_bin_swap1.bddl;;
    rectangular_red_bin_sw2selfc8) echo pick_up_the_rectangular_steel_plate_and_place_it_in_the_red_bin_swap2.bddl;;
    round_blue_bin_sw0selfc8) echo pick_up_the_round_steel_plate_and_place_it_in_the_blue_bin_swap0.bddl;;
    round_blue_bin_sw2selfc8) echo pick_up_the_round_steel_plate_and_place_it_in_the_blue_bin_swap2.bddl;;
    triangular_yellow_bin_selfc8) echo pick_up_the_triangular_steel_plate_and_place_it_in_the_yellow_bin.bddl;;
  esac
}

# 1) 采集（守卫：≥15 跳过）
for d in $DIRS; do
  n=$(ls "$D/raw_hdf5/$d" 2>/dev/null | wc -l)
  if [ "$n" -ge 15 ]; then echo "skip collect $d ($n)"; continue; fi
  $PY -u scripts/scripted_expert_collect.py --bddl "$(bddl_of $d)" \
    --num 15 --workers 4 --out-dir "$D/raw_hdf5/$d" > "$D/selfc_r7_$d.log" 2>&1
  echo "collected $d: $(ls "$D/raw_hdf5/$d" | wc -l)"
done

# 2) audit（并行）
for d in $DIRS; do
  $PY scripts/audit_success.py --raw-root "$D/raw_hdf5/$d" > "$D/audit_r7_$d.log" 2>&1 &
done
wait
grep -H '审查' "$D"/audit_r7_*.log
if grep -q FAILED "$D"/audit_r7_*.log; then echo AUDIT_HAS_FAILURE | tee "$D/r7_pipeline_done.txt"; exit 1; fi

# 3) 增量打包
bash scripts/run_pack_r2.sh > "$D/pack_r7_runner.log" 2>&1
# 4) r7 meta
$PY scripts/mk_meta_r7.py
# 5) 上传
ssh -n -o BatchMode=yes -o ConnectTimeout=10 -p 10036 root@10.31.118.28 "cd /root/data/sorting/xvla_hdf5 && rm -rf $DIRS"
tar cf - -C "$D/xvla_hdf5" $DIRS | ssh -o BatchMode=yes -p 10036 root@10.31.118.28 "tar xf - -C /root/data/sorting/xvla_hdf5"
scp -P 10036 "$D/sorting_meta_r7_press.json" root@10.31.118.28:/root/data/sorting/
echo R7_PIPELINE_DONE | tee "$D/r7_pipeline_done.txt"
