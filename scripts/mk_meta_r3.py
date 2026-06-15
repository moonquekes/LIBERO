#!/usr/bin/env python3
"""
mk_meta_r3.py — 生成 Round-3 布局内均衡 meta（含脚本专家自采，远端前缀）。

均衡原则（r2bal 验证有效）：模型策略≈"识别物理布局→抓该布局 demo 最多的目标"，
故在每个物理布局内把三种（或两种）指令的 demo 条数用复制条目拉平。
布局：D=默认（rect@0/round@1/tri@2，含 selfc4/selfc8 自采）；P=rect_swap1+round_swap0；
Q=rect_swap2+tri_swap0；R=round_swap2+tri_swap1。
本轮起每形状恰好 D+2swap → 布局内拉平后跨形状总数自动相等。

用法：python scripts/mk_meta_r3.py   # 输出 $D/sorting_meta_r3_bal.json
"""
import glob
import json
import os

D = "/home/x/vla/libero/data/suction_dataset_multi_part_sorting"
REMOTE_PREFIX = "/root/data/sorting/xvla_hdf5"
OUT = os.path.join(D, "sorting_meta_r3_bal.json")

LAYOUT = {
    "rectangular_red_bin": "D", "rectangular_red_bin_selfc4": "D",
    "rectangular_red_bin_selfc8": "D",
    "round_blue_bin": "D", "round_blue_bin_selfc4": "D", "round_blue_bin_selfc8": "D",
    "triangular_yellow_bin": "D",
    "rectangular_red_bin_swap1": "P", "round_blue_bin_swap0": "P",
    "rectangular_red_bin_swap2": "Q", "triangular_yellow_bin_swap0": "Q",
    "round_blue_bin_swap2": "R", "triangular_yellow_bin_swap1": "R",
}


def shape_of(dirname):
    if dirname.startswith("rectangular"):
        return "rect"
    if dirname.startswith("round"):
        return "round"
    return "tri"


def main():
    # 按 (layout, shape) 聚合所有打包产物
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
    print("布局内均衡（复制条目=过采样）：")
    for layout in ["D", "P", "Q", "R"]:
        shapes = {s: fs for (lay, s), fs in groups.items() if lay == layout}
        target = max(len(fs) for fs in shapes.values())
        for s, fs in sorted(shapes.items()):
            dup = [fs[i % len(fs)] for i in range(target)]
            datalist.extend(dup)
            print(f"  {layout}/{s}: {len(fs)} -> {target}")

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
