#!/usr/bin/env bash
# 生成 mk_meta_r12.py = r5 配方 + 3 个反事实配对目录(cf)入布局 D；排除 r6/r9 的 round 补充与 d9div
set -eu
cd /home/x/vla/libero/scripts
sed 's/EXCLUDE = {"round_blue_bin_selfc4", "round_blue_bin_selfc8b"}/EXCLUDE = {"round_blue_bin_selfc4", "round_blue_bin_selfc8b", "round_blue_bin_d9div"}/;
     s/"triangular_yellow_bin": "D",/"triangular_yellow_bin": "D", "rectangular_red_bin_cf": "D", "round_blue_bin_cf": "D", "triangular_yellow_bin_cf": "D",/;
     s/r9_div/r12_cf/g' mk_meta_r9.py > mk_meta_r12.py
/home/x/miniforge3/envs/vla-adapter/bin/python -m py_compile mk_meta_r12.py
echo "--- 核对 ---"
grep -nE 'EXCLUDE =|_cf"|sorting_meta_r12' mk_meta_r12.py
echo MK_META_R12_OK
