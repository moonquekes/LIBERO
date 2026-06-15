#!/usr/bin/env bash
# r9 数据：30 条 round-D"几何多样+速度节奏冻结"demo → audit → 打包 → r9 meta（不上传，上传走 Windows scp）
trap '' PIPE
set -u
D=/home/x/vla/libero/data/suction_dataset_multi_part_sorting
PY=/home/x/miniforge3/envs/vla-adapter/bin/python
cd /home/x/vla/libero
export SE_SPEED_JITTER=0
NEW=round_blue_bin_d9div
BDDL=pick_up_the_round_steel_plate_and_place_it_in_the_blue_bin.bddl
n=$(ls "$D/raw_hdf5/$NEW" 2>/dev/null | wc -l)
if [ "$n" -lt 30 ]; then
  $PY -u scripts/scripted_expert_collect.py --bddl "$BDDL" --num 30 --workers 4 \
    --out-dir "$D/raw_hdf5/$NEW" > "$D/selfc_r9_$NEW.log" 2>&1
fi
echo "collected: $(ls "$D/raw_hdf5/$NEW" | wc -l)"
$PY scripts/audit_success.py --raw-root "$D/raw_hdf5/$NEW" > "$D/audit_r9_$NEW.log" 2>&1
grep -H '审查' "$D/audit_r9_$NEW.log"
if grep -q FAILED "$D/audit_r9_$NEW.log"; then echo AUDIT_FAIL | tee "$D/r9_data_done.txt"; exit 1; fi
bash scripts/run_pack_r2.sh > "$D/pack_r9_runner.log" 2>&1
$PY scripts/mk_meta_r9.py
echo R9_DATA_DONE | tee "$D/r9_data_done.txt"
