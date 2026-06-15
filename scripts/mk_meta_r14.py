#!/usr/bin/env python3
"""r14 meta：定向修补 r12（热启动用）。
策略：cf 仅留 D（保 round-A 识别，避免 r13 把 cf 铺 swap 导致的位置→形状坍缩），
同时把 swap 层(P/Q/R)加权上来防饿死(r12 的 B 崩根因)。
非对称均衡：D 每形状单元 → 60(cf 占~半，强识别信号)；swap 每形状单元 → 30。
→ D 总 180 / swap 总 180，份额 50:50（r12 是 D 61%）。配低 lr 短步热启动用。"""
import glob, json, os, random

D = "/home/x/vla/libero/data/suction_dataset_multi_part_sorting"
REMOTE_PREFIX = "/root/data/sorting/xvla_hdf5"
OUT = os.path.join(D, "sorting_meta_r14.json")
N_D, N_SWAP = 60, 30

# 与 r12 同：cf 仅在 D；swap 来自 P/Q/R 且无 cf
LAYOUT = {
    "rectangular_red_bin": "D", "rectangular_red_bin_selfc4": "D",
    "rectangular_red_bin_selfc8": "D", "rectangular_red_bin_selfc8b": "D",
    "round_blue_bin": "D", "round_blue_bin_selfc8": "D",
    "triangular_yellow_bin": "D",
    "rectangular_red_bin_cf": "D", "round_blue_bin_cf": "D", "triangular_yellow_bin_cf": "D",
    "rectangular_red_bin_swap1": "P", "round_blue_bin_swap0": "P",
    "rectangular_red_bin_swap2": "Q", "triangular_yellow_bin_swap0": "Q", "triangular_yellow_bin_sw0selfc8": "Q",
    "round_blue_bin_swap2": "R", "triangular_yellow_bin_swap1": "R", "triangular_yellow_bin_sw1selfc8": "R",
}
# 标注哪些目录是 cf（用于核对 D 单元里 cf 占比）
CF_DIRS = {"rectangular_red_bin_cf", "round_blue_bin_cf", "triangular_yellow_bin_cf"}


def shape_of(d):
    return "rect" if d.startswith("rectangular") else ("round" if d.startswith("round") else "tri")


def main():
    groups, cf_count = {}, {}
    for part in sorted(glob.glob(os.path.join(D, "xvla_meta_part_*.json"))):
        dirname = os.path.basename(part)[len("xvla_meta_part_"):-len(".json")]
        if dirname not in LAYOUT:
            continue
        files = json.load(open(part))["datalist"]
        key = (LAYOUT[dirname], shape_of(dirname))
        groups.setdefault(key, []).extend(sorted(files))
        if dirname in CF_DIRS:
            cf_count[key] = cf_count.get(key, 0) + len(files)

    rng = random.Random(0)
    datalist = []
    print("每单元均衡（D→60 / swap→30；括号内 cf 唯一条数）：")
    for layout in ["D", "P", "Q", "R"]:
        N = N_D if layout == "D" else N_SWAP
        for shape in ["rect", "round", "tri"]:
            fs = groups.get((layout, shape), [])
            if not fs:
                continue
            sel = rng.sample(fs, N) if len(fs) >= N else [fs[i % len(fs)] for i in range(N)]
            datalist.extend(sel)
            print(f"  {layout}/{shape}: {len(fs)}uniq -> {N}  (cf_uniq={cf_count.get((layout, shape), 0)})")

    remote = [p.replace(os.path.join(D, "xvla_hdf5"), REMOTE_PREFIX) for p in datalist]
    meta = {
        "dataset_name": "libero",
        "observation_key": ["observations/agentview_image", "observations/robot0_eye_in_hand_image"],
        "language_instruction_key": "language_instruction",
        "datalist": remote,
    }
    json.dump(meta, open(OUT, "w"), indent=1)
    nD = sum(1 for r in remote if any(c in r for c in ["/rectangular_red_bin", "/round_blue_bin/", "/round_blue_bin_selfc", "/round_blue_bin_cf", "/triangular_yellow_bin/", "/triangular_yellow_bin_cf", "/rectangular_red_bin_selfc", "/rectangular_red_bin_cf"]))
    print(f"\n共 {len(remote)} 条（去重 {len(set(remote))} 文件）→ {OUT}")


if __name__ == "__main__":
    main()
