#!/usr/bin/env bash
# r5 冒烟：用改后的脚本专家采 2 条 rect-D，验证怼穿段写入 + 回放成功
trap '' PIPE
set -u
D=/home/x/vla/libero/data/suction_dataset_multi_part_sorting
PY=/home/x/miniforge3/envs/vla-adapter/bin/python
cd /home/x/vla/libero
TMP="$D/raw_hdf5_smoke_r5"
rm -rf "$TMP"
$PY -u scripts/scripted_expert_collect.py \
  --bddl pick_up_the_rectangular_steel_plate_and_place_it_in_the_red_bin.bddl \
  --num 2 --workers 2 --out-dir "$TMP" 2>&1 | tail -n 3
$PY - "$TMP" <<'EOF'
import sys, glob
import h5py, numpy as np
for f in sorted(glob.glob(sys.argv[1] + "/*.h*5")):
    with h5py.File(f, "r") as h:
        a = h["trajectory"]["actions"][:]
    z, g = a[:, 2], a[:, 6]
    on = np.where(g > 0)[0]
    t = int(on[0])
    print(f"{f.split('/')[-1]}  z_on={z[t]:.4f}  z_min={z.min():.4f}  低于0.945的帧数={(z<0.945).sum()}")
EOF
$PY scripts/audit_success.py --raw-root "$TMP" 2>&1 | tail -n 3
rm -rf "$TMP"
echo SMOKE_DONE
