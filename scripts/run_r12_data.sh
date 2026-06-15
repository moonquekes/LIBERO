#!/usr/bin/env bash
# r12 数据：30 组反事实配对(90 条)→ audit → 打包 → r12 meta（上传走 Windows scp）
trap '' PIPE
set -u
D=/home/x/vla/libero/data/suction_dataset_multi_part_sorting
PY=/home/x/miniforge3/envs/vla-adapter/bin/python
cd /home/x/vla/libero
export MUJOCO_GL=egl
DIRS="rectangular_red_bin_cf round_blue_bin_cf triangular_yellow_bin_cf"
# 1) 采集（守卫：三目录都 ≥28 才跳过）
need=0
for d in $DIRS; do n=$(ls "$D/raw_hdf5/$d" 2>/dev/null | wc -l); [ "$n" -lt 28 ] && need=1; done
if [ "$need" = 1 ]; then
  $PY -u scripts/scripted_collect_triples.py --triples 30 --workers 4 \
    --out-root "$D/raw_hdf5" > "$D/cf_r12.log" 2>&1
fi
for d in $DIRS; do echo "collected $d: $(ls "$D/raw_hdf5/$d" 2>/dev/null | wc -l)"; done
# 2) audit（并行）
for d in $DIRS; do
  $PY scripts/audit_success.py --raw-root "$D/raw_hdf5/$d" > "$D/audit_r12_$d.log" 2>&1 &
done
wait
grep -H '审查' "$D"/audit_r12_*.log
if grep -q FAILED "$D"/audit_r12_*.log; then echo AUDIT_FAIL | tee "$D/r12_data_done.txt"; exit 1; fi
# 3) 打包（旧目录原生 skip）
bash scripts/run_pack_r2.sh > "$D/pack_r12_runner.log" 2>&1
# 4) r12 meta
$PY scripts/mk_meta_r12.py
echo R12_DATA_DONE | tee "$D/r12_data_done.txt"
