#!/usr/bin/env python3
"""
mk_meta_r4.py — Round-4 均衡 meta（r3 基础 + tri swap 自采 + rect-D 补采）。

两级均衡：
1) 布局内拉平（r2bal 验证有效）：每个物理布局内各指令 demo 条数复制到该布局最大值。
2) 形状全局补齐：r4 起 tri 在 Q/R 各 +15 导致形状总数失衡（tri 多 ~13%），
   而 P 是唯一不含 tri 的布局 → 把 P 整层翻倍，rect/round 各 +15，
   既补齐形状总量又不破坏任何布局的内部均衡。

用法：python scripts/mk_meta_r4.py   # 输出 $D/sorting_meta_r4_bal.json
"""
import glob
import json
import os

D = "/home/x/vla/libero/data/suction_dataset_multi_part_sorting"
REMOTE_PREFIX = "/root/data/sorting/xvla_hdf5"
OUT = os.path.join(D, "sorting_meta_r4_bal.json")

LAYOUT = {
    "rectangular_red_bin": "D", "rectangular_red_bin_selfc4": "D",
    "rectangular_red_bin_selfc8": "D", "rectangular_red_bin_selfc8b": "D",
    "round_blue_bin": "D", "round_blue_bin_selfc4": "D", "round_blue_bin_selfc8": "D",
    "triangular_yellow_bin": "D",
    "rectangular_red_bin_swap1": "P", "round_blue_bin_swap0": "P",
    "rectangular_red_bin_swap2": "Q", "triangular_yellow_bin_swap0": "Q",
    "triangular_yellow_bin_sw0selfc8": "Q",
    "round_blue_bin_swap2": "R", "triangular_yellow_bin_swap1": "R",
    "triangular_yellow_bin_sw1selfc8": "R",
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
