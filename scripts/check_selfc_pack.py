#!/usr/bin/env python3
"""抽查自采打包产物完整性（smoke 用）。"""
import glob
import sys

import cv2
import h5py
import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "/tmp/selfc_smoke_xvla"
for fp in sorted(glob.glob(root + "/**/*.h5", recursive=True)):
    f = h5py.File(fp, "r")
    a = f["abs_action_6d"][()]
    n_img = len(f["observations"]["agentview_image"])
    im = cv2.imdecode(np.frombuffer(f["observations"]["agentview_image"][0], np.uint8), 1)
    print(fp.split("/")[-1][:60])
    print("  T:", a.shape, "imgs:", n_img, "lang:", f["language_instruction"][()][:70])
    print("  img:", im.shape, "mean", round(float(im.mean()), 1))
    print("  pos min/max:", np.round(a[:, :3].min(0), 3), np.round(a[:, :3].max(0), 3))
    print("  grip vals:", sorted(set(a[:, 9].tolist())))
    f.close()
