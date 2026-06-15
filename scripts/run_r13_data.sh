#!/usr/bin/env bash
# r13 数据：P/Q/R 各 30 组反事实配对(270 条)→ audit → 打包 → r13 meta（上传走 Windows scp）
trap '' PIPE
set -u
D=/home/x/vla/libero/data/suction_dataset_multi_part_sorting
PY=/home/x/miniforge3/envs/vla-adapter/bin/python
cd /home/x/vla/libero
export MUJOCO_GL=egl
rm -rf "$D"/raw_hdf5/_cfP_smoke
DIRS="rectangular_red_bin_cfP round_blue_bin_cfP triangular_yellow_bin_cfP rectangular_red_bin_cfQ round_blue_bin_cfQ triangular_yellow_bin_cfQ rectangular_red_bin_cfR round_blue_bin_cfR triangular_yellow_bin_cfR"
# 1) 采集 P/Q/R（守卫：该布局首目录 ≥28 才跳过）
for L in P Q R; do
  guard="$D/raw_hdf5/rectangular_red_bin_cf$L"
  n=$(ls "$guard" 2>/dev/null | wc -l)
  case "$L" in P) BS=21000;; Q) BS=22000;; R) BS=23000;; esac
  if [ "$n" -lt 28 ]; then
    $PY -u scripts/scripted_collect_triples.py --layout "$L" --triples 30 --workers 4 \
      --base-seed "$BS" --out-root "$D/raw_hdf5" > "$D/cf_r13_$L.log" 2>&1
  fi
  echo "layout $L done"
done
for d in $DIRS; do echo "$d: $(ls "$D/raw_hdf5/$d" 2>/dev/null | wc -l)"; done
# 2) audit（并行）
for d in $DIRS; do
  $PY scripts/audit_success.py --raw-root "$D/raw_hdf5/$d" > "$D/audit_r13_$d.log" 2>&1 &
done
wait
grep -H '审查' "$D"/audit_r13_*.log
if grep -q FAILED "$D"/audit_r13_*.log; then echo AUDIT_FAIL | tee "$D/r13_data_done.txt"; exit 1; fi
# 3) 打包（旧目录原生 skip）
bash scripts/run_pack_r2.sh > "$D/pack_r13_runner.log" 2>&1
# 4) r13 meta
$PY scripts/mk_meta_r13.py
echo R13_DATA_DONE | tee "$D/r13_data_done.txt"
