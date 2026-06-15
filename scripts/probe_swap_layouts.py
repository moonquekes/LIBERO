#!/usr/bin/env python3
"""确认 P/Q/R 换位物理场景里三块板各在哪个槽（slot0≈-0.28 / slot1≈-0.146 / slot2≈-0.016, x 坐标）。"""
import os
import numpy as np
import scripted_expert_collect as se

SCENES = {
    "P=rect_swap1": "pick_up_the_rectangular_steel_plate_and_place_it_in_the_red_bin_swap1.bddl",
    "Q=rect_swap2": "pick_up_the_rectangular_steel_plate_and_place_it_in_the_red_bin_swap2.bddl",
    "R=round_swap2": "pick_up_the_round_steel_plate_and_place_it_in_the_blue_bin_swap2.bddl",
}
PLATES = ["steel_plate_1", "steel_plate_round_1", "steel_plate_triangle_1"]


def slot(x):
    return min(enumerate([-0.28, -0.146, -0.016]), key=lambda kv: abs(kv[1] - x))[0]


for tag, bddl in SCENES.items():
    env = se.make_env(os.path.join(se.CUSTOM, bddl))
    np.random.seed(123)
    env.reset()
    line = []
    for p in PLATES:
        x = float(np.asarray(se.body_pos(env, p))[0])
        line.append(f"{p.split('_')[2][:5]}@slot{slot(x)}(x={x:.3f})")
    print(f"{tag}: {'  '.join(line)}")
    try:
        env.env.close()
    except Exception:
        pass
