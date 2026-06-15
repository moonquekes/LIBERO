#!/usr/bin/env bash
# 生成 4cm 槽宽的 D 布局诊断 BDDL（介于旧 2cm 宽散布与新 8cm 窄散布之间）
set -eu
cd /home/x/vla/libero/libero/libero/bddl_files/custom
rm -f pick_up_the__diag4cm.bddl
for s in rectangular_steel_plate_and_place_it_in_the_red_bin round_steel_plate_and_place_it_in_the_blue_bin; do
  sed -e 's/(-0.3050 -0.3250 -0.2250 -0.2450)/(-0.2850 -0.3050 -0.2450 -0.2650)/' \
      -e 's/(-0.1850 -0.3250 -0.1050 -0.2450)/(-0.1650 -0.3050 -0.1250 -0.2650)/' \
      -e 's/(-0.0650 -0.3250 0.0150 -0.2450)/(-0.0450 -0.3050 -0.0050 -0.2650)/' \
      "pick_up_the_${s}.bddl" > "pick_up_the_${s}_diag4cm.bddl"
done
ls -la *diag4cm*
grep -n '0.2850\|0.1650\|0.0450' pick_up_the_rectangular_steel_plate_and_place_it_in_the_red_bin_diag4cm.bddl | head -5
