#!/usr/bin/env python3
"""r13 meta：r5 基础 + 反事实配对(cf)铺到全部 4 布局(D/P/Q/R)。
均衡策略升级为"每个(布局×形状)单元统一到 N 条"——12 单元全等，彻底防层间/形状偏斜
(r12 把 cf 全堆 D 导致 D 占 61%、swap 饿死、B 崩；这里每层每形状都有 cf 信号且权重相同)。
不在 LAYOUT 的目录(r7/r8 残留、selfc8b、d9div)自动跳过。"""
import glob
import json
import os
import random

D = "/home/x/vla/libero/data/suction_dataset_multi_part_sorting"
REMOTE_PREFIX = "/root/data/sorting/xvla_hdf5"
OUT = os.path.join(D, "sorting_meta_r13_cf.json")
N = 50   # 每个(布局×形状)单元目标条数

LAYOUT = {
    # D
    "rectangular_red_bin": "D", "rectangular_red_bin_selfc4": "D",
    "rectangular_red_bin_selfc8": "D", "rectangular_red_bin_selfc8b": "D",
    "round_blue_bin": "D", "round_blue_bin_selfc8": "D",
    "triangular_yellow_bin": "D",
    "rectangular_red_bin_cf": "D", "round_blue_bin_cf": "D", "triangular_yellow_bin_cf": "D",
    # P
    "rectangular_red_bin_swap1": "P", "round_blue_bin_swap0": "P",
    "rectangular_red_bin_cfP": "P", "round_blue_bin_cfP": "P", "triangular_yellow_bin_cfP": "P",
    # Q
    "rectangular_red_bin_swap2": "Q", "triangular_yellow_bin_swap0": "Q", "triangular_yellow_bin_sw0selfc8": "Q",
    "rectangular_red_bin_cfQ": "Q", "round_blue_bin_cfQ": "Q", "triangular_yellow_bin_cfQ": "Q",
    # R
    "round_blue_bin_swap2": "R", "triangular_yellow_bin_swap1": "R", "triangular_yellow_bin_sw1selfc8": "R",
    "rectangular_red_bin_cfR": "R", "round_blue_bin_cfR": "R", "triangular_yellow_bin_cfR": "R",
}


def shape_of(dirname):
    if dirname.startswith("rectangular"):
        return "rect"
    if dirname.startswith("round"):
        return "round"
    return "tri"


def main():
    groups = {}
    for part in sorted(glob.glob(os.path.join(D, "xvla_meta_part_*.json"))):
        dirname = os.path.basename(part)[len("xvla_meta_part_"):-len(".json")]
        if dirname not in LAYOUT:
            print(f"SKIP {dirname}")
            continue
        with open(part) as f:
            files = json.load(f)["datalist"]
        key = (LAYOUT[dirname], shape_of(dirname))
        groups.setdefault(key, []).extend(sorted(files))

    rng = random.Random(0)
    datalist = []
    print(f"每(布局×形状)单元 → {N} 条：")
    for layout in ["D", "P", "Q", "R"]:
        for shape in ["rect", "round", "tri"]:
            fs = groups.get((layout, shape), [])
            if not fs:
                print(f"  ⚠ {layout}/{shape}: 空！")
                continue
            if len(fs) >= N:
                sel = rng.sample(fs, N)
            else:
                sel = [fs[i % len(fs)] for i in range(N)]
            datalist.extend(sel)
            print(f"  {layout}/{shape}: {len(fs)} -> {N}")

    remote = [p.replace(os.path.join(D, "xvla_hdf5"), REMOTE_PREFIX) for p in datalist]
    meta = {
        "dataset_name": "libero",
        "observation_key": ["observations/agentview_image",
                            "observations/robot0_eye_in_hand_image"],
        "language_instruction_key": "language_instruction",
        "datalist": remote,
    }
    with open(OUT, "w") as f:
        json.dump(meta, f, indent=1)
    print(f"\n共 {len(remote)} 条（去重 {len(set(remote))} 实际文件）→ {OUT}")


if __name__ == "__main__":
    main()
