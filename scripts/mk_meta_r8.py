#!/usr/bin/env python3
"""
mk_meta_r8.py — Round-8 meta：人工 demo + 9 个"多样化脚本专家"目录（每形状×布局 15 条）。

与 r7 的差异：
1) 剔除全部旧 selfc4/selfc8 目录（同质刻板轨迹批量堆叠 → 居中场景视觉不可区分 →
   检索塌缩出 slot1 吸引子，r5→r6→r7 总分单调下降的病因）。
2) 改用 div 目录：逐条随机化路径/速度/高度/下压深度的脚本专家重采。
3) 取消 P 层翻倍 hack：每形状恰好出现在 3 个布局，布局内拉平后全局自动均衡。

用法：python scripts/mk_meta_r8.py   # 输出 $D/sorting_meta_r8_div.json
"""
import glob
import json
import os

D = "/home/x/vla/libero/data/suction_dataset_multi_part_sorting"
REMOTE_PREFIX = "/root/data/sorting/xvla_hdf5"
OUT = os.path.join(D, "sorting_meta_r8_div.json")

EXCLUDE = {
    "rectangular_red_bin_selfc4", "rectangular_red_bin_selfc8", "rectangular_red_bin_selfc8b",
    "round_blue_bin_selfc4", "round_blue_bin_selfc8", "round_blue_bin_selfc8b",
    "triangular_yellow_bin_selfc8", "triangular_yellow_bin_sw0selfc8", "triangular_yellow_bin_sw1selfc8",
    "rectangular_red_bin_sw1selfc8", "rectangular_red_bin_sw2selfc8",
    "round_blue_bin_sw0selfc8", "round_blue_bin_sw2selfc8",
}

LAYOUT = {
    "rectangular_red_bin": "D", "round_blue_bin": "D", "triangular_yellow_bin": "D",
    "rectangular_red_bin_div": "D", "round_blue_bin_div": "D", "triangular_yellow_bin_div": "D",
    "rectangular_red_bin_swap1": "P", "round_blue_bin_swap0": "P",
    "rectangular_red_bin_sw1div": "P", "round_blue_bin_sw0div": "P",
    "rectangular_red_bin_swap2": "Q", "triangular_yellow_bin_swap0": "Q",
    "rectangular_red_bin_sw2div": "Q", "triangular_yellow_bin_sw0div": "Q",
    "round_blue_bin_swap2": "R", "triangular_yellow_bin_swap1": "R",
    "round_blue_bin_sw2div": "R", "triangular_yellow_bin_sw1div": "R",
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
            print(f"剔除 {dirname}（同质脚本旧数据）")
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
        for s, fs in sorted(shapes.items()):
            dup = [fs[i % len(fs)] for i in range(target)]
            datalist.extend(dup)
            shape_total[s] = shape_total.get(s, 0) + len(dup)
            print(f"  {layout}/{s}: {len(fs)} -> {target}")

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
