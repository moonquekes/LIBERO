#!/usr/bin/env bash
# 上传 r3 增量：4 个自采打包目录 + 均衡 meta
set -eu
D=/home/x/vla/libero/data/suction_dataset_multi_part_sorting
H="ssh -p 10036 root@10.31.118.28"
tar cf - -C "$D/xvla_hdf5" \
  rectangular_red_bin_selfc4 round_blue_bin_selfc4 \
  rectangular_red_bin_selfc8 round_blue_bin_selfc8 \
| $H "tar xf - -C /root/data/sorting/xvla_hdf5"
scp -P 10036 "$D/sorting_meta_r3_bal.json" root@10.31.118.28:/root/data/sorting/
$H "ls /root/data/sorting/xvla_hdf5 | wc -l; python3 -c \"import json;print(len(json.load(open('/root/data/sorting/sorting_meta_r3_bal.json'))['datalist']))\""
echo UPLOAD_R3_DONE
