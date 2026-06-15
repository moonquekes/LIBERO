#!/usr/bin/env bash
# Round-2 全量审计：9 个子目录并行回放,各自写日志
set -u
base=/home/x/vla/libero/data/suction_dataset_multi_part_sorting
cd /home/x/vla/libero
ls "$base/raw_hdf5" | xargs -P 9 -I{} bash -c "/home/x/miniforge3/envs/vla-adapter/bin/python scripts/audit_success.py --raw-root '$base/raw_hdf5/{}' > '$base/audit_r2_{}.log' 2>&1"
echo AUDIT_ALL_DONE
for f in "$base"/audit_r2_*.log; do
  echo "== $(basename $f)"
  grep -E '^审查|^失败|FAILED' "$f" | head -8
done
