#!/usr/bin/env bash
# r4 流水线：audit 新采 3 目录 → 增量打包 → r4 均衡 meta → 上传 h100（训练另起）
trap '' PIPE
set -u
D=/home/x/vla/libero/data/suction_dataset_multi_part_sorting
PY=/home/x/miniforge3/envs/vla-adapter/bin/python
cd /home/x/vla/libero

for d in triangular_yellow_bin_sw0selfc8 triangular_yellow_bin_sw1selfc8 rectangular_red_bin_selfc8b; do
  $PY scripts/audit_success.py --raw-root "$D/raw_hdf5/$d" \
    > "$D/audit_$d.log" 2>&1 &
done
wait
grep -H '审查' "$D"/audit_triangular_yellow_bin_sw0selfc8.log \
              "$D"/audit_triangular_yellow_bin_sw1selfc8.log \
              "$D"/audit_rectangular_red_bin_selfc8b.log
if grep -q FAILED "$D"/audit_triangular_yellow_bin_sw0selfc8.log \
                  "$D"/audit_triangular_yellow_bin_sw1selfc8.log \
                  "$D"/audit_rectangular_red_bin_selfc8b.log; then
  echo AUDIT_HAS_FAILURE; exit 1
fi

bash scripts/run_pack_r2.sh > "$D/pack_r4_runner.log" 2>&1
$PY scripts/mk_meta_r4.py

tar cf - -C "$D/xvla_hdf5" \
  triangular_yellow_bin_sw0selfc8 triangular_yellow_bin_sw1selfc8 rectangular_red_bin_selfc8b \
| ssh -o BatchMode=yes -p 10036 root@10.31.118.28 "tar xf - -C /root/data/sorting/xvla_hdf5"
scp -P 10036 "$D/sorting_meta_r4_bal.json" root@10.31.118.28:/root/data/sorting/
ssh -n -o BatchMode=yes -p 10036 root@10.31.118.28 "ls /root/data/sorting/xvla_hdf5 | wc -l"
echo R4_PIPELINE_DONE | tee "$D/r4_pipeline_done.txt"
