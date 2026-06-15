#!/usr/bin/env bash
# Round-2 全量重打包：9 子目录并行,4090 渲染
# 必须全量重渲染:三角 2026-06-08 放大 s=1.15,旧 rect/round 包里干扰三角还是小号
# ⚠️ 内存上限决定并行数(每 worker ~2GB):12GB 时 5 路 OOM 静默全灭(2026-06-10 实测)最多 3 路;
#    24GB(临时调高 .wslconfig)可跑 6 路,打包完把 .wslconfig 改回 12GB。
# raw2xvla 已支持断点续打包(跳过已存在且完整的输出),重启安全。
set -u
base=/home/x/vla/libero/data/suction_dataset_multi_part_sorting
xvla=/mnt/c/Users/x/Desktop/华工科技/VLA/X-VLA/evaluation/libero/raw2xvla.py
export MUJOCO_GL=egl
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
cd /home/x/vla/libero
mkdir -p "$base/xvla_hdf5"
rm -f "$base"/xvla_meta_part_*.json
ls "$base/raw_hdf5" | xargs -P 3 -I{} bash -c "/home/x/miniforge3/envs/vla-adapter/bin/python -u '$xvla' --raw-root '$base/raw_hdf5/{}' --out-dir '$base/xvla_hdf5/{}' --meta-out '$base/xvla_meta_part_{}.json' --libero-root /home/x/vla/libero > '$base/pack_r2_{}.log' 2>&1"
echo PACK_ALL_DONE
for f in "$base"/pack_r2_*.log; do
  echo "== $(basename $f)"
  tail -2 "$f"
done
