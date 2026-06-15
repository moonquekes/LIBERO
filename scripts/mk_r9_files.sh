#!/usr/bin/env bash
# 生成 mk_meta_r9.py：= r6 配方（修好 round-A 的那版），但把同质 round_blue_bin_selfc8b 换成多样 round_blue_bin_d9div
set -eu
cd /home/x/vla/libero/scripts
sed 's/EXCLUDE = {"round_blue_bin_selfc4"}/EXCLUDE = {"round_blue_bin_selfc4", "round_blue_bin_selfc8b"}/;
     s/"round_blue_bin_selfc8": "D", "round_blue_bin_selfc8b": "D",/"round_blue_bin_selfc8": "D", "round_blue_bin_selfc8b": "D", "round_blue_bin_d9div": "D",/;
     s/r6_press/r9_div/g' mk_meta_r6.py > mk_meta_r9.py
/home/x/miniforge3/envs/vla-adapter/bin/python -m py_compile mk_meta_r9.py
echo "--- EXCLUDE/LAYOUT/OUT 核对 ---"
grep -nE 'EXCLUDE =|d9div|selfc8b|sorting_meta_r9' mk_meta_r9.py
echo MK_META_R9_OK
