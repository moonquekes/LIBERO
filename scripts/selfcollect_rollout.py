#!/usr/bin/env python3
"""
selfcollect_rollout.py — 模型自采（self-collection / 自蒸馏）。

背景：D 布局 rect/round 的人工 demo 是旧版宽散布（±2.5cm），新评测散布全居中，
位置分布失配导致模型在 D 布局无视指令奔三角（2026-06-11 宽散布诊断 12/12 坐实）。
r2bal 在 4cm 槽（中间散布）上 91.7%——用它当老师，在 4cm 布局上 rollout，
录 states+actions 存成与人工采集同构的 raw_hdf5，audit 复验后走 raw2xvla 正常打包。

与人工 raw 的唯一差异：动作是**绝对位姿**（eval 推理模式），attrs 标 actions_absolute=True，
raw2xvla / audit_success 读到该标记会关掉 controller 的 delta 解释（已配套修改）。
开头垫 5 帧 hold（绝对位姿原地保持），对齐 raw2xvla cap_index=5 的 settle 帧裁剪。
吸盘参数用 25°/0.7/2（与评测/部署的 wrapper 默认一致；10° 下 round 盘吸不上，3/48 实测）。
参数写入 h5 attrs，raw2xvla / audit_success 按 attrs 还原回放物理，保证录制=回放。

用法（先确保 r2bal 服务在 8000 端口）：
  MUJOCO_GL=egl python scripts/selfcollect_rollout.py \
    --bddl pick_up_the_rectangular_steel_plate_and_place_it_in_the_red_bin_diag4cm.bddl \
    --num 15 --out-dir $D/raw_hdf5/rectangular_red_bin_selfc4 --workers 4
"""
import argparse, json, os, sys, time
import multiprocessing as mp

import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_sorting_diag import (  # noqa: E402
    CUSTOM, XVLAClient, find_robots, check_success, mat_to_rotate6d, parse_target,
)
import libero.libero.envs.bddl_utils as BDDLUtils  # noqa: E402
from libero.libero.envs import TASK_MAPPING  # noqa: E402
from libero.libero.envs.suction_sticky_wrapper import SuctionStickyWrapper  # noqa: E402
from robosuite import load_controller_config  # noqa: E402
import robosuite.utils.transform_utils as T  # noqa: E402

# 与评测/部署一致（wrapper 默认）。回放端按 h5 attrs 还原同款参数，无需全局一致。
# 教训：10° 下策略带轻微倾角接近 round 盘永远吸不上（3/48），rect 平面大不受影响。
SUCTION_KWARGS = dict(normal_max_angle_deg=25.0, effective_radius_ratio=0.7,
                      detach_grace_steps=2)
HOLD_FRAMES_HEAD = 5    # 对齐 raw2xvla --cap-index 5
HOLD_FRAMES_TAIL = 10   # 成功后再录几帧静止，给回放的接触结算留余量


def make_env(bddl_path, resolution=256):
    cc = load_controller_config(default_controller="OSC_POSE")
    pn = BDDLUtils.get_problem_info(bddl_path)["problem_name"]
    env = TASK_MAPPING[pn](
        bddl_file_name=bddl_path, robots=["SuctionPanda"], controller_configs=cc,
        has_renderer=False, has_offscreen_renderer=True, render_camera="agentview",
        ignore_done=True, use_camera_obs=True, reward_shaping=True, control_freq=20,
        camera_names=["robot0_eye_in_hand", "agentview"],
        camera_heights=resolution, camera_widths=resolution,
    )
    return SuctionStickyWrapper(env, **SUCTION_KWARGS)


def hold_action(ctrl, grip):
    """绝对位姿模式下的"原地保持"动作：目标=当前末端位姿。"""
    aa = T.quat2axisangle(T.mat2quat(np.asarray(ctrl.ee_ori_mat)))
    return np.concatenate([np.asarray(ctrl.ee_pos), aa, [grip]]).astype(np.float64)


def record_episode(env, robots, policy, goal, max_steps):
    """rollout 一条；成功返回 (states, actions)，失败返回 None。"""
    policy.reset()
    obs = env.reset()
    for r in robots:
        r.controller.use_delta = True
    for _ in range(10):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
    for r in robots:
        r.controller.use_delta = False

    ctrl = robots[0].controller
    states, actions = [], []

    def rec(a):
        states.append(env.sim.get_state().flatten())
        return env.step(np.asarray(a).tolist())

    for _ in range(HOLD_FRAMES_HEAD):
        a = hold_action(ctrl, -1.0)
        obs, _, _, _ = rec(a)
        actions.append(a)

    success = False
    grip = -1.0
    for _ in range(max_steps):
        av = obs["agentview_image"]; wr = obs["robot0_eye_in_hand_image"]
        pos = np.asarray(ctrl.ee_pos, dtype=np.float32)
        ori6d = mat_to_rotate6d(np.asarray(ctrl.ee_ori_mat, dtype=np.float32))
        a = policy.step(av, wr, pos, ori6d, goal).astype(np.float64)
        obs, _, _, _ = rec(a)
        actions.append(a)
        grip = float(a[-1])
        if check_success(env):
            success = True
            break
    if not success:
        return None

    for _ in range(HOLD_FRAMES_TAIL):
        a = hold_action(ctrl, grip)
        obs, _, _, _ = rec(a)
        actions.append(a)
    if not check_success(env):     # 静置后仍需成立，过滤"擦边一瞬间"的假成功
        return None
    return np.stack(states), np.stack(actions)


def save_raw(out_dir, task_tag, states, actions, problem_name, goal,
             bddl_path, bddl_content):
    os.makedirs(out_dir, exist_ok=True)
    fn = (f"robosuite_ln_libero_tabletop_manipulation_{task_tag}"
          f"_ep_{int(time.time() * 10) % 10**9}_{os.getpid()}.hdf5")
    path = os.path.join(out_dir, fn)
    with h5py.File(path, "w") as f:
        g = f.create_group("trajectory")
        g.create_dataset("states", data=states)
        g.create_dataset("actions", data=actions)
        f.attrs["problem_info"] = json.dumps(
            {"problem_name": problem_name, "language_instruction": goal})
        f.attrs["bddl_file_name"] = bddl_path
        f.attrs["bddl_file_content"] = bddl_content
        f.attrs["actions_absolute"] = True       # 关键标记：回放需关 delta
        f.attrs["collected_by"] = "selfcollect_rollout.py(r2bal)"
        # 回放物理须与录制一致：raw2xvla / audit_success 读这组 attrs 覆盖吸盘参数
        f.attrs["suction_normal_max_angle_deg"] = float(SUCTION_KWARGS["normal_max_angle_deg"])
        f.attrs["suction_effective_radius_ratio"] = float(SUCTION_KWARGS["effective_radius_ratio"])
        f.attrs["suction_detach_grace_steps"] = int(SUCTION_KWARGS["detach_grace_steps"])
    return path


def worker(job):
    quota, A = job["quota"], job["args"]
    bddl_path = os.path.join(CUSTOM, A["bddl"])
    pn = BDDLUtils.get_problem_info(bddl_path)["problem_name"]
    goal = BDDLUtils.get_problem_info(bddl_path).get("language_instruction", "").strip()
    bddl_content = open(bddl_path).read()
    task_tag = goal.replace(" ", "_")
    env = make_env(bddl_path, A["resolution"])
    robots = find_robots(env)
    policy = XVLAClient(A["ip"], A["port"], steps=A["flow_steps"])
    saved, attempts = [], 0
    while len(saved) < quota and attempts < quota * 4:
        attempts += 1
        t0 = time.time()
        out = record_episode(env, robots, policy, goal, A["max_steps"])
        if out is not None:
            p = save_raw(A["out_dir"], task_tag, out[0], out[1], pn, goal,
                         bddl_path, bddl_content)
            saved.append(p)
            print(f"  [pid{os.getpid()}] ✅ {len(saved)}/{quota} "
                  f"T={len(out[1])} ({time.time()-t0:.0f}s) {os.path.basename(p)}", flush=True)
        else:
            print(f"  [pid{os.getpid()}] ❌ 失败重试 ({time.time()-t0:.0f}s)", flush=True)
    try:
        env.env.close()
    except Exception:
        pass
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl", required=True, help="CUSTOM 下的 bddl 文件名（采集布局）")
    ap.add_argument("--num", type=int, default=15, help="目标成功条数")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max_steps", type=int, default=600)
    ap.add_argument("--flow_steps", type=int, default=10)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--server_ip", default="127.0.0.1")
    ap.add_argument("--server_port", type=int, default=8000)
    args = ap.parse_args()

    A = dict(bddl=args.bddl, out_dir=args.out_dir, max_steps=args.max_steps,
             flow_steps=args.flow_steps, resolution=args.resolution,
             ip=args.server_ip, port=args.server_port)
    base, extra = divmod(args.num, args.workers)
    jobs = [{"quota": base + (1 if i < extra else 0), "args": A}
            for i in range(args.workers) if base + (1 if i < extra else 0) > 0]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(jobs)) as pool:
        results = pool.map(worker, jobs)
    total = sum(len(r) for r in results)
    print(f"\n自采完成：{total}/{args.num} → {args.out_dir}")
    if total < args.num:
        print("⚠️ 未达目标条数（成功率低于预期），可重跑补齐")


if __name__ == "__main__":
    main()
