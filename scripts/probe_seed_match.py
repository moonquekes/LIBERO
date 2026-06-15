#!/usr/bin/env python3
"""验证：同一 np.random 种子两次 reset 是否给出相同工件布局（反事实配对的前提）。"""
import os
import numpy as np
import scripted_expert_collect as se

bddl = os.path.join(se.CUSTOM, "pick_up_the_round_steel_plate_and_place_it_in_the_blue_bin.bddl")
env = se.make_env(bddl)
robots = se.find_robots(env)
PLATES = ["steel_plate_1", "steel_plate_round_1", "steel_plate_triangle_1"]


def positions():
    out = {}
    for b in PLATES:
        try:
            out[b] = np.asarray(se.body_pos(env, b))[:2].round(4).tolist()
        except Exception as e:
            out[b] = f"ERR:{e}"
    return out


for seed in [7, 7, 9, 9]:
    np.random.seed(seed)
    env.reset()
    print(f"seed={seed}: {positions()}")
