#!/usr/bin/env python3
"""Tier 3 反事实配对采集（CAST 式），支持全部物理布局 D/P/Q/R。
同一随机种子 → 同一初始画面，分别抓 矩/圆/三角：位置不再判别，只有指令→目标 → 逼模型读语言。
干净怼穿抓取（无横向抖动）。每条按其"指令 bddl"存（goal/content），状态覆盖位置故场景由采集 env 决定。"""
import os
import argparse
import time
import numpy as np
import multiprocessing as mp
import scripted_expert_collect as se

T = se.T
BDDLUtils = se.BDDLUtils

RECT = "pick_up_the_rectangular_steel_plate_and_place_it_in_the_red_bin.bddl"
ROUND = "pick_up_the_round_steel_plate_and_place_it_in_the_blue_bin.bddl"
TRI = "pick_up_the_triangular_steel_plate_and_place_it_in_the_yellow_bin.bddl"
RECT_S1 = "pick_up_the_rectangular_steel_plate_and_place_it_in_the_red_bin_swap1.bddl"
RECT_S2 = "pick_up_the_rectangular_steel_plate_and_place_it_in_the_red_bin_swap2.bddl"
ROUND_S0 = "pick_up_the_round_steel_plate_and_place_it_in_the_blue_bin_swap0.bddl"
ROUND_S2 = "pick_up_the_round_steel_plate_and_place_it_in_the_blue_bin_swap2.bddl"
TRI_S0 = "pick_up_the_triangular_steel_plate_and_place_it_in_the_yellow_bin_swap0.bddl"
TRI_S1 = "pick_up_the_triangular_steel_plate_and_place_it_in_the_yellow_bin_swap1.bddl"

# shape → (target_base, target_bin, dir_prefix)
RECT_T = ("steel_plate_1", "red_bin_1", "rectangular_red_bin")
ROUND_T = ("steel_plate_round_1", "blue_bin_1", "round_blue_bin")
TRI_T = ("steel_plate_triangle_1", "yellow_bin_1", "triangular_yellow_bin")

# layout → (scene_bddl, [(shape_tuple, instr_bddl), ...])  已用 probe_swap_layouts.py 验证排布
LAYOUTS = {
    "D": (ROUND,    [(RECT_T, RECT),  (ROUND_T, ROUND),    (TRI_T, TRI)]),
    "P": (RECT_S1,  [(RECT_T, RECT_S1), (ROUND_T, ROUND_S0), (TRI_T, TRI)]),
    "Q": (RECT_S2,  [(RECT_T, RECT_S2), (ROUND_T, ROUND),    (TRI_T, TRI_S0)]),
    "R": (ROUND_S2, [(RECT_T, RECT),  (ROUND_T, ROUND_S2),  (TRI_T, TRI_S1)]),
}


def placed_ok(env, target_base, target_bin, thresh=0.07):
    p = np.asarray(se.body_pos(env, target_base))[:2]
    b = np.asarray(se.body_pos(env, target_bin))[:2]
    return float(np.linalg.norm(p - b)) < thresh


def grasp_clean(env, robots, target_base, target_bin):
    env.reset()
    for r in robots:
        r.controller.use_delta = True
    for _ in range(10):
        env.step([0, 0, 0, 0, 0, 0, -1])
    for r in robots:
        r.controller.use_delta = False
    ctrl = robots[0].controller
    aa0 = T.quat2axisangle(T.mat2quat(np.asarray(ctrl.ee_ori_mat)))
    rec = se.Recorder(env, ctrl, aa0)
    rec.hold(se.HOLD_FRAMES_HEAD, -1.0)
    plate = se.body_pos(env, target_base)
    tbin = se.body_pos(env, target_bin)
    attached = lambda: env.attached_body_id is not None
    rec.servo([plate[0], plate[1], plate[2] + se.HOVER_DZ], -1.0, se.TRAVEL_STEP)
    rec.servo([plate[0], plate[1], plate[2] + 0.06], -1.0, se.TRAVEL_STEP)
    hit = rec.servo([plate[0], plate[1], plate[2]], 1.0, se.DESCEND_STEP,
                    settle=0, max_extra=80, stop_fn=attached)
    if not hit:
        return None
    nm = env.sim.model.body_id2name(env.attached_body_id) or ""
    if not nm.startswith(target_base):
        return None
    rec.servo([plate[0], plate[1], plate[2] - se.PRESS_DZ], 1.0, se.PRESS_STEP)
    rec.hold(3, 1.0)
    rec.servo([plate[0], plate[1], plate[2] + se.LIFT_DZ], 1.0, se.TRAVEL_STEP)
    if not attached():
        return None
    rec.servo([tbin[0], tbin[1], plate[2] + se.LIFT_DZ], 1.0, se.TRAVEL_STEP)
    if not attached():
        return None
    rec.servo([tbin[0], tbin[1], tbin[2] + se.DROP_DZ], 1.0, se.TRAVEL_STEP)
    rec.hold(15, -1.0)
    rec.servo([tbin[0], tbin[1], tbin[2] + se.DROP_DZ + 0.08], -1.0, se.TRAVEL_STEP)
    if not placed_ok(env, target_base, target_bin):
        return None
    rec.hold(se.HOLD_FRAMES_TAIL, -1.0)
    if not placed_ok(env, target_base, target_bin):
        return None
    return np.stack(rec.states), np.stack(rec.actions)


def worker(job):
    seeds, layout, out_root = job["seeds"], job["layout"], job["out_root"]
    scene_bddl, specs = LAYOUTS[layout]
    suffix = "" if layout == "D" else layout
    env = se.make_env(os.path.join(se.CUSTOM, scene_bddl))
    robots = se.find_robots(env)
    targets = []
    for (tb, tbin, prefix), instr_bddl in specs:
        bp = os.path.join(se.CUSTOM, instr_bddl)
        info = BDDLUtils.get_problem_info(bp)
        targets.append(dict(tb=tb, tbin=tbin, pn=info["problem_name"],
                            goal=info.get("language_instruction", "").strip(),
                            content=open(bp).read(), path=bp,
                            out_dir=os.path.join(out_root, f"{prefix}_cf{suffix}")))
    saved = 0
    for seed in seeds:
        for tg in targets:
            np.random.seed(seed)
            out = grasp_clean(env, robots, tg["tb"], tg["tbin"])
            if out is not None:
                se.save_raw(tg["out_dir"], tg["goal"].replace(" ", "_"), out[0], out[1],
                            tg["pn"], tg["goal"], tg["path"], tg["content"])
                saved += 1
                print(f"  [{layout} pid{os.getpid()}] seed{seed} {tg['tb']} ✅", flush=True)
            else:
                print(f"  [{layout} pid{os.getpid()}] seed{seed} {tg['tb']} ❌", flush=True)
    try:
        env.env.close()
    except Exception:
        pass
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", required=True, choices=list(LAYOUTS))
    ap.add_argument("--triples", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--base-seed", type=int, default=20000)
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args()
    seeds = [args.base_seed + i for i in range(args.triples)]
    chunks = [seeds[i::args.workers] for i in range(args.workers)]
    jobs = [{"seeds": c, "layout": args.layout, "out_root": args.out_root}
            for c in chunks if c]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(jobs)) as pool:
        results = pool.map(worker, jobs)
    print(f"[{args.layout}] 配对采集完成：共存 {sum(results)} 条（目标 {args.triples*3}）→ {args.out_root}")


if __name__ == "__main__":
    main()
