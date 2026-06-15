#!/usr/bin/env bash
set -eu
cd /home/x/vla/libero/scripts
sed 's/"round_blue_bin_selfc8": "D",/"round_blue_bin_selfc8": "D", "round_blue_bin_selfc8b": "D",/; s/r5_press/r6_press/' mk_meta_r5.py > mk_meta_r6.py
/home/x/miniforge3/envs/vla-adapter/bin/python -m py_compile mk_meta_r6.py
grep -n 'selfc8b\|r6_press' mk_meta_r6.py | head -n 5
sed -i 's/\r$//' run_r6_pipeline.sh
bash -n run_r6_pipeline.sh
echo ALL_OK
