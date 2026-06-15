#!/usr/bin/env python3
"""合并 run_pack_r2.sh 产出的 9 个分片 meta -> xvla_meta.json,并校验计数。"""
import glob
import json
import os
import sys

base = "/home/x/vla/libero/data/suction_dataset_multi_part_sorting"
parts = sorted(glob.glob(os.path.join(base, "xvla_meta_part_*.json")))
if not parts:
    sys.exit("no part metas found")

merged = None
datalist = []
for p in parts:
    m = json.load(open(p))
    if merged is None:
        merged = {k: v for k, v in m.items() if k != "datalist"}
    datalist += m["datalist"]

missing = [e for e in datalist if not os.path.isfile(e)]
if missing:
    sys.exit(f"missing files in datalist: {missing[:5]} ...")

merged["datalist"] = sorted(datalist)
out = os.path.join(base, "xvla_meta.json")
json.dump(merged, open(out, "w"), indent=1)

from collections import Counter
cnt = Counter(e.split("/xvla_hdf5/")[1].split("/")[0] for e in datalist)
for k in sorted(cnt):
    print(f"{k}: {cnt[k]}")
print(f"TOTAL: {len(datalist)} -> {out}")
