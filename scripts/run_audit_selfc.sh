#!/usr/bin/env bash
# 复验 4 个自采目录——4 路并行（audit 无渲染纯 CPU，每路 ~1GB）
set -u
D=/home/x/vla/libero/data/suction_dataset_multi_part_sorting
PY=/home/x/miniforge3/envs/vla-adapter/bin/python
cd /home/x/vla/libero
printf '%s\n' rectangular_red_bin_selfc4 round_blue_bin_selfc4 \
              rectangular_red_bin_selfc8 round_blue_bin_selfc8 \
| xargs -P 4 -I{} bash -c "$PY -u scripts/audit_success.py --raw-root '$D/raw_hdf5/{}' > '$D/audit_{}.log' 2>&1"
echo AUDIT_SELFC_DONE
for d in rectangular_red_bin_selfc4 round_blue_bin_selfc4 \
         rectangular_red_bin_selfc8 round_blue_bin_selfc8; do
  echo "== $d"
  tail -2 "$D/audit_$d.log" | head -1
done
