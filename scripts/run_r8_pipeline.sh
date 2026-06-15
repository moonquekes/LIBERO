#!/usr/bin/env bash
# r8 多样性管线：9 个形状×布局组合各采 15 条"随机化脚本专家"demo → audit → 打包 → r8 meta → 上传
trap '' PIPE
set -u
D=/home/x/vla/libero/data/suction_dataset_multi_part_sorting
PY=/home/x/miniforge3/envs/vla-adapter/bin/python
cd /home/x/vla/libero
DIRS="rectangular_red_bin_div rectangular_red_bin_sw1div rectangular_red_bin_sw2div round_blue_bin_div round_blue_bin_sw0div round_blue_bin_sw2div triangular_yellow_bin_div triangular_yellow_bin_sw0div triangular_yellow_bin_sw1div"
bddl_of() {
  case "$1" in
    rectangular_red_bin_div) echo pick_up_the_rectangular_steel_plate_and_place_it_in_the_red_bin.bddl;;
    rectangular_red_bin_sw1div) echo pick_up_the_rectangular_steel_plate_and_place_it_in_the_red_bin_swap1.bddl;;
    rectangular_red_bin_sw2div) echo pick_up_the_rectangular_steel_plate_and_place_it_in_the_red_bin_swap2.bddl;;
    round_blue_bin_div) echo pick_up_the_round_steel_plate_and_place_it_in_the_blue_bin.bddl;;
    round_blue_bin_sw0div) echo pick_up_the_round_steel_plate_and_place_it_in_the_blue_bin_swap0.bddl;;
    round_blue_bin_sw2div) echo pick_up_the_round_steel_plate_and_place_it_in_the_blue_bin_swap2.bddl;;
    triangular_yellow_bin_div) echo pick_up_the_triangular_steel_plate_and_place_it_in_the_yellow_bin.bddl;;
    triangular_yellow_bin_sw0div) echo pick_up_the_triangular_steel_plate_and_place_it_in_the_yellow_bin_swap0.bddl;;
    triangular_yellow_bin_sw1div) echo pick_up_the_triangular_steel_plate_and_place_it_in_the_yellow_bin_swap1.bddl;;
  esac
}

# 1) 采集（守卫：≥15 跳过）
for d in $DIRS; do
  n=$(ls "$D/raw_hdf5/$d" 2>/dev/null | wc -l)
  if [ "$n" -ge 15 ]; then echo "skip collect $d ($n)"; continue; fi
  $PY -u scripts/scripted_expert_collect.py --bddl "$(bddl_of $d)" \
    --num 15 --workers 4 --out-dir "$D/raw_hdf5/$d" > "$D/selfc_r8_$d.log" 2>&1
  echo "collected $d: $(ls "$D/raw_hdf5/$d" | wc -l)"
done

# 2) audit（并行）
for d in $DIRS; do
  $PY scripts/audit_success.py --raw-root "$D/raw_hdf5/$d" > "$D/audit_r8_$d.log" 2>&1 &
done
wait
grep -H '审查' "$D"/audit_r8_*.log
if grep -q FAILED "$D"/audit_r8_*.log; then echo AUDIT_HAS_FAILURE | tee "$D/r8_pipeline_done.txt"; exit 1; fi

# 3) 增量打包（旧目录原生 skip）
bash scripts/run_pack_r2.sh > "$D/pack_r8_runner.log" 2>&1
# 4) r8 meta
$PY scripts/mk_meta_r8.py
# 5) 上传
ssh -n -o BatchMode=yes -o ConnectTimeout=10 -p 10036 root@10.31.118.28 "cd /root/data/sorting/xvla_hdf5 && rm -rf $DIRS"
tar cf - -C "$D/xvla_hdf5" $DIRS | ssh -o BatchMode=yes -p 10036 root@10.31.118.28 "tar xf - -C /root/data/sorting/xvla_hdf5"
scp -P 10036 "$D/sorting_meta_r8_div.json" root@10.31.118.28:/root/data/sorting/
echo R8_PIPELINE_DONE | tee "$D/r8_pipeline_done.txt"
