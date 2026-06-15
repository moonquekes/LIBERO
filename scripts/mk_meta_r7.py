#!/usr/bin/env python3
"""
mk_meta_r5.py — Round-5 均衡 meta（r4 基础 + 5 个脚本专家目录重采为怼穿式下压）。

与 r4 的差异：
1) 剔除 round_blue_bin_selfc4（策略自采、高位悬停风格 z~0.947——与重采后的
   怼穿式脚本数据风格冲突，正是 rect-D 双峰欠冲的同款病灶）。
2) 输出 sorting_meta_r7_press.json。
两级均衡逻辑与 r4 完全相同：布局内拉平 + P 层翻倍补形状全局配比。

用法：python scripts/mk_meta_r5.py   # 输出 $D/sorting_meta_r7_press.json
"""
import glob
import json
import os

D = "/home/x/vla/libero/data/suction_dataset_multi_part_sorting"
REMOTE_PREFIX = "/root/data/sorting/xvla_hdf5"
OUT = os.path.join(D, "sorting_meta_r7_press.json")

EXCLUDE = {"round_blue_bin_selfc4"}

LAYOUT = {
    "rectangular_red_bin": "D", "rectangular_red_bin_selfc4": "D",
    "rectangular_red_bin_selfc8": "D", "rectangular_red_bin_selfc8b": "D",
    "round_blue_bin": "D", "round_blue_bin_selfc4": "D", "round_blue_bin_selfc8": "D", "round_blue_bin_selfc8b": "D",
    "triangular_yellow_bin": "D", "triangular_yellow_bin_selfc8": "D",
    "rectangular_red_bin_swap1": "P", "round_blue_bin_swap0": "P", "rectangular_red_bin_sw1selfc8": "P", "round_blue_bin_sw0selfc8": "P",
    "rectangular_red_bin_swap2": "Q", "triangular_yellow_bin_swap0": "Q",
    "triangular_yellow_bin_sw0selfc8": "Q", "rectangular_red_bin_sw2selfc8": "Q",
    "round_blue_bin_swap2": "R", "triangular_yellow_bin_swap1": "R",
    "triangular_yellow_bin_sw1selfc8": "R", "round_blue_bin_sw2selfc8": "R",
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
        if dirname in EXCLUDE:
            print(f"剔除 {dirname}（高位悬停风格，防双峰）")
            continue
        if dirname not in LAYOUT:
            raise SystemExit(f"未知子目录 {dirname}，先补 LAYOUT 表")
        with open(part) as f:
            files = json.load(f)["datalist"]
        key = (LAYOUT[dirname], shape_of(dirname))
        groups.setdefault(key, []).extend(sorted(files))

    datalist = []
    shape_total = {}
    print("布局内均衡（复制条目=过采样）：")
    for layout in ["D", "P", "Q", "R"]:
        shapes = {s: fs for (lay, s), fs in groups.items() if lay == layout}
        target = max(len(fs) for fs in shapes.values())
        rep = 2 if layout == "P" else 1   # P 整层翻倍补形状全局配比
        for s, fs in sorted(shapes.items()):
            dup = [fs[i % len(fs)] for i in range(target * rep)]
            datalist.extend(dup)
            shape_total[s] = shape_total.get(s, 0) + len(dup)
            print(f"  {layout}/{s}: {len(fs)} -> {target * rep}")

    print("形状总量：", shape_total)
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
    uniq = len(set(remote))
    print(f"\n共 {len(remote)} 条（去重 {uniq} 实际文件）→ {OUT}")


if __name__ == "__main__":
    main()
