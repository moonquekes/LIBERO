#!/usr/bin/env python3
"""扫描 xvla_hdf5 全部 demo 的末端贴近段统计——验证"数据集本身有没有欠冲/贴面帧够不够"。
abs_action_6d[T,10]: xyz + rot6d + grip(idx9, >0=吸)。纯读 action,不碰图像,秒级。"""
import h5py, glob, os
import numpy as np

ROOT = "/home/x/vla/libero/data/suction_dataset_multi_part_sorting/xvla_hdf5"

print(f"{'dir':52s} {'n':>3s} {'z_min(m)':>15s} {'z_on(m)':>8s} {'slowfr':>6s} {'v_last(mm/fr)':>13s}")
for d in sorted(os.listdir(ROOT)):
    dd = os.path.join(ROOT, d)
    if not os.path.isdir(dd):
        continue
    stats = []
    for f in sorted(glob.glob(os.path.join(dd, "*.h5"))):
        try:
            with h5py.File(f, "r") as h:
                a = h["abs_action_6d"][:]
        except Exception:
            continue
        z, g = a[:, 2], a[:, 9]
        on = np.where(g > 0)[0]
        if len(on) == 0:
            continue
        t = int(on[0])
        pre = z[: t + 1]
        if len(pre) < 3:
            continue
        zmin, z_on = float(pre.min()), float(z[t])
        slow = int((pre < zmin + 0.005).sum())          # 吸合前位于最低点5mm内的帧数
        k = min(8, t)
        v = float(np.mean(np.abs(np.diff(pre[-k - 1:])))) * 1000  # 末8帧平均速度 mm/帧
        stats.append((zmin, z_on, slow, v))
    if not stats:
        continue
    arr = np.array(stats)
    print(f"{d[:52]:52s} {len(stats):3d} {arr[:,0].mean():7.4f}±{arr[:,0].std():.4f} {arr[:,1].mean():8.4f} {arr[:,2].mean():6.1f} {arr[:,3].mean():13.2f}")
