#!/usr/bin/env python3
"""验证多样性注入：3 条 smoke demo 的轨迹统计应当显著不同。"""
import glob
import h5py
import numpy as np


def find_actions(h):
    found = []
    h.visititems(lambda n, o: found.append(n) if isinstance(o, h5py.Dataset) and n.endswith("actions") else None)
    return found[0]


for f in sorted(glob.glob("/home/x/vla/libero/data/suction_dataset_multi_part_sorting/raw_hdf5/_smoke_div/*.hdf5")):
    with h5py.File(f, "r") as h:
        a = h[find_actions(h)][:]
    z = a[:, 2]
    g = a[:, -1]
    on = np.where(g > 0)[0]
    t = int(on[0]) if len(on) else -1
    print(f"T={len(a):4d} z_max={z.max():.3f} z_min={z.min():.3f} 吸合帧={t:4d} "
          f"末段均速={np.mean(np.abs(np.diff(z[max(0, t - 9):t + 1]))) * 1000:.2f}mm/帧")
