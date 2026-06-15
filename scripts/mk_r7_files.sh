#!/usr/bin/env bash
# 生成 mk_meta_r7.py（补 5 个新居中目录的 LAYOUT 映射）并校验
set -eu
cd /home/x/vla/libero/scripts
sed 's/"triangular_yellow_bin": "D",/"triangular_yellow_bin": "D", "triangular_yellow_bin_selfc8": "D",/;
     s/"rectangular_red_bin_swap1": "P", "round_blue_bin_swap0": "P",/"rectangular_red_bin_swap1": "P", "round_blue_bin_swap0": "P", "rectangular_red_bin_sw1selfc8": "P", "round_blue_bin_sw0selfc8": "P",/;
     s/"triangular_yellow_bin_sw0selfc8": "Q",/"triangular_yellow_bin_sw0selfc8": "Q", "rectangular_red_bin_sw2selfc8": "Q",/;
     s/"triangular_yellow_bin_sw1selfc8": "R",/"triangular_yellow_bin_sw1selfc8": "R", "round_blue_bin_sw2selfc8": "R",/;
     s/r6_press/r7_press/' mk_meta_r6.py > mk_meta_r7.py
/home/x/miniforge3/envs/vla-adapter/bin/python -m py_compile mk_meta_r7.py
grep -c 'selfc8' mk_meta_r7.py
echo MK_META_R7_OK
