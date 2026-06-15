#!/usr/bin/env bash
# r6 修复管线：round-D 补 30 条怼穿居中 demo（D 层居中覆盖拉平 45:45）→ audit → 增量打包 → r6 meta → 上传
trap '' PIPE
set -u
D=/home/x/vla/libero/data/suction_dataset_multi_part_sorting
PY=/home/x/miniforge3/envs/vla-adapter/bin/python
cd /home/x/vla/libero
NEW=round_blue_bin_selfc8b
BDDL=pick_up_the_round_steel_plate_and_place_it_in_the_blue_bin.bddl

# 1) 采集（守卫：≥30 跳过）
n=$(ls "$D/raw_hdf5/$NEW" 2>/dev/null | wc -l)
if [ "$n" -lt 30 ]; then
  $PY -u scripts/scripted_expert_collect.py --bddl "$BDDL" \
    --num 30 --workers 4 --out-dir "$D/raw_hdf5/$NEW" > "$D/selfc_r6_$NEW.log" 2>&1
fi
echo "collected: $(ls "$D/raw_hdf5/$NEW" | wc -l)"

# 2) audit
$PY scripts/audit_success.py --raw-root "$D/raw_hdf5/$NEW" > "$D/audit_r6_$NEW.log" 2>&1
grep -H '审查' "$D/audit_r6_$NEW.log"
if grep -q FAILED "$D/audit_r6_$NEW.log"; then echo AUDIT_HAS_FAILURE | tee "$D/r6_pipeline_done.txt"; exit 1; fi

# 3) 增量打包（旧目录原生 skip）
bash scripts/run_pack_r2.sh > "$D/pack_r6_runner.log" 2>&1
# 4) r6 meta
$PY scripts/mk_meta_r6.py
# 5) 上传
ssh -n -o BatchMode=yes -o ConnectTimeout=10 -p 10036 root@10.31.118.28 "cd /root/data/sorting/xvla_hdf5 && rm -rf $NEW"
tar cf - -C "$D/xvla_hdf5" "$NEW" | ssh -o BatchMode=yes -p 10036 root@10.31.118.28 "tar xf - -C /root/data/sorting/xvla_hdf5"
scp -P 10036 "$D/sorting_meta_r6_press.json" root@10.31.118.28:/root/data/sorting/
echo R6_PIPELINE_DONE | tee "$D/r6_pipeline_done.txt"
