#!/usr/bin/env bash
# 盘点各轮数据：raw 子目录条数 + 各 meta 实际 entry 数
D=/home/x/vla/libero/data/suction_dataset_multi_part_sorting
PY=/home/x/miniforge3/envs/vla-adapter/bin/python
echo "== raw_hdf5 per-dir counts =="
for d in "$D"/raw_hdf5/*/; do
  printf "%-55s %s\n" "$(basename "$d")" "$(ls "$d" 2>/dev/null | wc -l)"
done
echo "== total raw =="
find "$D/raw_hdf5" -name '*.hdf5' | wc -l
echo "== total xvla =="
find "$D/xvla_hdf5" -name '*.h5' | wc -l
echo "== meta entry counts =="
for m in "$D"/*meta*.json; do
  n=$("$PY" - "$m" <<'EOF'
import json,sys
d=json.load(open(sys.argv[1]))
dl=d.get("datalist",d) if isinstance(d,dict) else d
print(len(dl))
EOF
)
  echo "$(basename "$m") : $n"
done
