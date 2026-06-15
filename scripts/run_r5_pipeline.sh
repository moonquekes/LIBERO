#!/usr/bin/env bash
# r5 流水线：重采 5 个脚本专家目录（怼穿式下压）→ audit → 增量打包 → r5 meta → 上传 h100
# 各阶段产物落盘可断点续跑：清旧仅首次（r5_cleared.txt 哨兵）、采集按目录文件数守卫、打包原生续打
trap '' PIPE
set -u
D=/home/x/vla/libero/data/suction_dataset_multi_part_sorting
PY=/home/x/miniforge3/envs/vla-adapter/bin/python
cd /home/x/vla/libero
DIRS="rectangular_red_bin_selfc8 rectangular_red_bin_selfc8b round_blue_bin_selfc8 triangular_yellow_bin_sw0selfc8 triangular_yellow_bin_sw1selfc8"
bddl_of() {
  case "$1" in
    rectangular_red_bin_selfc8|rectangular_red_bin_selfc8b) echo pick_up_the_rectangular_steel_plate_and_place_it_in_the_red_bin.bddl;;
    round_blue_bin_selfc8) echo pick_up_the_round_steel_plate_and_place_it_in_the_blue_bin.bddl;;
    triangular_yellow_bin_sw0selfc8) echo pick_up_the_triangular_steel_plate_and_place_it_in_the_yellow_bin_swap0.bddl;;
    triangular_yellow_bin_sw1selfc8) echo pick_up_the_triangular_steel_plate_and_place_it_in_the_yellow_bin_swap1.bddl;;
  esac
}

# 0) 清旧（raw + 下游孤儿，防陈旧文件混进打包），仅首次
if [ ! -f "$D/r5_cleared.txt" ]; then
  for d in $DIRS; do
    rm -rf "$D/raw_hdf5/$d" "$D/converted_hdf5/$d" "$D/xvla_hdf5/$d"
  done
  echo cleared | tee "$D/r5_cleared.txt"
fi

# 1) 采集（守卫：目录内 ≥15 个文件即跳过）
for d in $DIRS; do
  n=$(ls "$D/raw_hdf5/$d" 2>/dev/null | wc -l)
  if [ "$n" -ge 15 ]; then echo "skip collect $d ($n)"; continue; fi
  $PY -u scripts/scripted_expert_collect.py --bddl "$(bddl_of $d)" \
    --num 15 --workers 4 --out-dir "$D/raw_hdf5/$d" > "$D/selfc_r5_$d.log" 2>&1
  echo "collected $d: $(ls "$D/raw_hdf5/$d" | wc -l)"
done

# 2) audit（5 目录并行，纯 CPU 回放）
for d in $DIRS; do
  $PY scripts/audit_success.py --raw-root "$D/raw_hdf5/$d" > "$D/audit_r5_$d.log" 2>&1 &
done
wait
grep -H '审查' "$D"/audit_r5_*.log
if grep -q FAILED "$D"/audit_r5_*.log; then echo AUDIT_HAS_FAILURE | tee "$D/r5_pipeline_done.txt"; exit 1; fi

# 3) 增量打包（全目录扫，旧目录原生 skip）
bash scripts/run_pack_r2.sh > "$D/pack_r5_runner.log" 2>&1
# 4) r5 meta
$PY scripts/mk_meta_r5.py
# 5) 上传（先清远端同名目录防陈旧残留）
ssh -n -o BatchMode=yes -o ConnectTimeout=10 -p 10036 root@10.31.118.28 "cd /root/data/sorting/xvla_hdf5 && rm -rf $DIRS"
tar cf - -C "$D/xvla_hdf5" $DIRS | ssh -o BatchMode=yes -p 10036 root@10.31.118.28 "tar xf - -C /root/data/sorting/xvla_hdf5"
scp -P 10036 "$D/sorting_meta_r5_press.json" root@10.31.118.28:/root/data/sorting/
ssh -n -o BatchMode=yes -o ConnectTimeout=10 -p 10036 root@10.31.118.28 "ls /root/data/sorting/xvla_hdf5 | wc -l"
echo R5_PIPELINE_DONE | tee "$D/r5_pipeline_done.txt"
