#!/usr/bin/env python3
import h5py, glob, os
from collections import Counter
D = "/home/x/vla/libero/data/suction_dataset_multi_part_sorting/xvla_hdf5"
c = Counter()
for f in glob.glob(D + "/**/*.h5", recursive=True):
    try:
        h = h5py.File(f, "r"); v = h["language_instruction"][()]; h.close()
        s = v.decode() if hasattr(v, "decode") else v[0].decode()
        c[s] += 1
    except Exception:
        pass
print("唯一指令字符串数:", len(c))
for s, n in c.most_common():
    print(f"  {n:4d}  {s!r}")
