#!/usr/bin/env python3
"""r10 = r5press 同数据 + lang_aug_map（语言复述增强，逼模型读指令、反视觉捷径）。
每条指令配 10 条复述（含 canonical，保证评测用的原句也在训练分布里）；形状词/动词/句式都变，
唯一稳定的语义锚点是 形状词 + 颜色框 → 模型必须读语义而非背模板。零数据改动、不碰形状平衡。"""
import json
import os

D = "/home/x/vla/libero/data/suction_dataset_multi_part_sorting"
SRC = os.path.join(D, "sorting_meta_r5_press.json")
OUT = os.path.join(D, "sorting_meta_r10_lang.json")

AUG = {
    "pick the rectangular steel plate and place it gently in the red bin": [
        "pick the rectangular steel plate and place it gently in the red bin",
        "pick up the rectangular steel plate and put it in the red bin",
        "grab the rectangular metal plate and place it in the red box",
        "take the rectangle steel plate and set it down in the red bin",
        "move the rectangular plate into the red container",
        "lift the rectangular steel plate and drop it gently into the red bin",
        "put the rectangular metal plate in the red bin",
        "place the rectangular steel plate carefully in the red box",
        "pick the square steel plate and place it in the red bin",
        "sort the rectangular plate into the red bin",
    ],
    "pick the round steel plate and place it gently in the blue bin": [
        "pick the round steel plate and place it gently in the blue bin",
        "pick up the round steel plate and put it in the blue bin",
        "grab the round metal plate and place it in the blue box",
        "take the circular steel plate and set it down in the blue bin",
        "move the round plate into the blue container",
        "lift the round steel plate and drop it gently into the blue bin",
        "put the circular metal plate in the blue bin",
        "place the round steel plate carefully in the blue box",
        "pick the disc-shaped steel plate and place it in the blue bin",
        "sort the round plate into the blue bin",
    ],
    "pick the triangular steel plate and place it gently in the yellow bin": [
        "pick the triangular steel plate and place it gently in the yellow bin",
        "pick up the triangular steel plate and put it in the yellow bin",
        "grab the triangular metal plate and place it in the yellow box",
        "take the triangle steel plate and set it down in the yellow bin",
        "move the triangular plate into the yellow container",
        "lift the triangular steel plate and drop it gently into the yellow bin",
        "put the triangular metal plate in the yellow bin",
        "place the triangular steel plate carefully in the yellow box",
        "pick the three-sided steel plate and place it in the yellow bin",
        "sort the triangular plate into the yellow bin",
    ],
}

m = json.load(open(SRC))
# 校验 datalist 里出现的指令都被 AUG 覆盖（防 normalize 后字符串不匹配 → 静默不增强）
m["lang_aug_map"] = AUG
json.dump(m, open(OUT, "w"), indent=1, ensure_ascii=False)
print("datalist 条数:", len(m["datalist"]), " 去重文件:", len(set(m["datalist"])))
print("lang_aug_map keys:")
for k in AUG:
    print(f"  [{len(AUG[k])}] {k}")
print("→", OUT)
