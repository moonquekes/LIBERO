#!/usr/bin/env python3
"""核对打包后 xvla 数据的全程 z 画像（不截断在首次开吸帧）。"""
import glob, os
import h5py
import numpy as np

ROOT = "/home/x/vla/libero/data/suction_dataset_multi_part_sorting/xvla_hdf5"
DIRS = ["rectangular_red_bin_selfc8", "rectangular_red_bin_selfc8b",
        "round_blue_bin_selfc8", "triangular_yellow_bin_sw0selfc8",
        "triangular_yellow_bin_sw1selfc8"]
for d in DIRS:
    mins, lows, mt = [], [], 0
    for f in sorted(glob.glob(os.path.join(ROOT, d, "*.h5"))):
        mt = max(mt, os.path.getmtime(f))
        with h5py.File(f, "r") as h:
            a = h["abs_action_6d"][:]
        z = a[:, 2]
        mins.append(float(z.min()))
        lows.append(int((z < 0.945).sum()))
    import datetime
    ts = datetime.datetime.fromtimestamp(mt).strftime("%m-%d %H:%M")
    print(f"{d:36s} n={len(mins):2d} 全程z_min={np.mean(mins):.4f}±{np.std(mins):.4f} "
          f"低位帧均值={np.mean(lows):5.1f} 最新文件={ts}")
