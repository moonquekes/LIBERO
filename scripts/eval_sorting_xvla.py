#!/usr/bin/env python3
"""
eval_sorting_xvla.py — 用 X-VLA server 评测「多零件按形状分拣」微调模型。

client-server：模型 server 在 h100（deploy.py + LoRA ckpt），本脚本在 WSL 跑 LIBERO 仿真
（自定义分拣 BDDL + SuctionPanda + SuctionStickyWrapper），通过 HTTP 查询 server 拿绝对位姿动作。

评测 A（默认布局）: 3 个默认 BDDL。
评测 B（换位反捷径）: 6 个换位 BDDL。
每个 BDDL 跑 N 个 episode，用 env._check_success() 判成功，报 per-BDDL + 总体成功率。

用法（WSL，vla-adapter 环境，先建好到 h100 的端口转发 localhost:PORT）：
  MUJOCO_GL=egl python eval_sorting_xvla.py --mode A --episodes 10 --server_port 8000
"""
import argparse, collections, os, sys, time
from typing import Deque, Dict, List, Optional

import numpy as np
import imageio
import json_numpy
import requests

LIBERO_ROOT = "/home/x/vla/libero"
if LIBERO_ROOT not in sys.path:
    sys.path.insert(0, LIBERO_ROOT)

import libero.libero.envs.robots  # noqa: F401  注册 SuctionPanda
import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import TASK_MAPPING
from libero.libero.envs.suction_sticky_wrapper import SuctionStickyWrapper
from robosuite import load_controller_config
import robosuite.utils.transform_utils as T

EPS = 1e-6
CUSTOM = os.path.join(LIBERO_ROOT, "libero/libero/bddl_files/custom")

MODE_BDDL = {
    "A": [
        "pick_up_the_rectangular_steel_plate_and_place_it_in_the_red_bin.bddl",
        "pick_up_the_round_steel_plate_and_place_it_in_the_blue_bin.bddl",
        "pick_up_the_triangular_steel_plate_and_place_it_in_the_yellow_bin.bddl",
    ],
    "B": [
        "pick_up_the_rectangular_steel_plate_and_place_it_in_the_red_bin_swap1.bddl",
        "pick_up_the_rectangular_steel_plate_and_place_it_in_the_red_bin_swap2.bddl",
        "pick_up_the_round_steel_plate_and_place_it_in_the_blue_bin_swap0.bddl",
        "pick_up_the_round_steel_plate_and_place_it_in_the_blue_bin_swap2.bddl",
        "pick_up_the_triangular_steel_plate_and_place_it_in_the_yellow_bin_swap0.bddl",
        "pick_up_the_triangular_steel_plate_and_place_it_in_the_yellow_bin_swap1.bddl",
    ],
}


def rotate6d_to_axisangle(r6d):
    single = r6d.ndim == 1
    if single:
        r6d = r6d[None, :]
    a1, a2 = r6d[:, 0:3], r6d[:, 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + EPS)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + EPS)
    b3 = np.cross(b1, b2, axis=-1)
    R = np.stack([b1, b2, b3], axis=-1)
    out = [T.quat2axisangle(T.mat2quat(R[i])) for i in range(R.shape[0])]
    out = np.stack(out, axis=0)
    return out[0] if single else out


def mat_to_rotate6d(R):
    return np.concatenate([R[:3, 0], R[:3, 1]], axis=-1)


def flip_agentview(img):
    return np.flip(np.flip(img, 0), 1)


def make_suction_env(bddl_file, resolution=256):
    cc = load_controller_config(default_controller="OSC_POSE")
    pn = BDDLUtils.get_problem_info(bddl_file)["problem_name"]
    env = TASK_MAPPING[pn](
        bddl_file_name=bddl_file, robots=["SuctionPanda"], controller_configs=cc,
        has_renderer=False, has_offscreen_renderer=True, render_camera="agentview",
        ignore_done=True, use_camera_obs=True, reward_shaping=True, control_freq=20,
        camera_names=["robot0_eye_in_hand", "agentview"],
        camera_heights=resolution, camera_widths=resolution,
    )
    env = SuctionStickyWrapper(env)
    env.seed(0)
    return env


def find_robots(env):
    cur = env
    for _ in range(8):
        if hasattr(cur, "robots") and getattr(cur, "robots"):
            return cur.robots
        cur = getattr(cur, "env", None)
        if cur is None:
            break
    raise RuntimeError("找不到 robots")


def check_success(env):
    cur = env
    for _ in range(8):
        if hasattr(cur, "_check_success"):
            try:
                return bool(cur._check_success())
            except Exception:
                pass
        cur = getattr(cur, "env", None)
        if cur is None:
            break
    return False


class XVLAClient:
    def __init__(self, host, port, steps=10, domain_id=3):
        self.url = f"http://{host}:{port}/act"
        self.steps = steps
        self.domain_id = domain_id
        self.reset()

    def reset(self):
        self.proprio: Optional[np.ndarray] = None
        self.action_plan: Deque[List[float]] = collections.deque()

    def _query(self, av, wr, pos, ori6d, goal):
        main = flip_agentview(av)
        closed = np.concatenate([pos, ori6d, np.array([0.0])], axis=-1)
        closed = np.concatenate([closed, np.zeros_like(closed)], axis=-1)
        if self.proprio is None:
            self.proprio = closed
        return {
            "proprio": json_numpy.dumps(self.proprio),
            "language_instruction": goal,
            "image0": json_numpy.dumps(main),
            "image1": json_numpy.dumps(wr),
            "domain_id": self.domain_id,
            "steps": self.steps,
        }

    def step(self, av, wr, pos, ori6d, goal):
        if not self.action_plan:
            r = requests.post(self.url, json=self._query(av, wr, pos, ori6d, goal), timeout=60)
            r.raise_for_status()
            action = np.array(r.json()["action"])  # (T,10)
            self.proprio[:9] = action[-1, :9].copy()
            eef = action[:, :3]
            aa = rotate6d_to_axisangle(action[:, 3:9])
            grip = action[:, 9:10]
            for row in np.concatenate([eef, aa, grip], axis=-1).tolist():
                self.action_plan.append(row)
        a = np.array(self.action_plan.popleft(), dtype=np.float32)
        a[-1] = 1.0 if a[-1] > 0.5 else -1.0
        return a


def run_episode(env, robots, policy, goal, max_steps=300, num_wait=10, frames_out=None):
    policy.reset()
    obs = env.reset()
    for r in robots:
        r.controller.use_delta = True
    for _ in range(num_wait):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
    for r in robots:
        r.controller.use_delta = False
    success = False
    for _ in range(max_steps):
        av = obs["agentview_image"]; wr = obs["robot0_eye_in_hand_image"]
        if frames_out is not None:
            frames_out.append(np.hstack([flip_agentview(av), flip_agentview(wr)]))
        ctrl = robots[0].controller
        pos = np.asarray(ctrl.ee_pos, dtype=np.float32)
        ori6d = mat_to_rotate6d(np.asarray(ctrl.ee_ori_mat, dtype=np.float32))
        a = policy.step(av, wr, pos, ori6d, goal)
        obs, _, _, _ = env.step(a.tolist())
        if check_success(env):
            success = True
            break
    return success


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["A", "B"], default="A")
    p.add_argument("--episodes", type=int, default=10, help="每个 BDDL 跑多少 episode")
    p.add_argument("--server_ip", default="127.0.0.1")
    p.add_argument("--server_port", type=int, default=8000)
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--save_video_dir", default=None, help="非空则保存失败 episode 的 mp4+montage")
    p.add_argument("--save_success", action="store_true", help="同时保存成功 episode 的视频")
    args = p.parse_args()

    policy = XVLAClient(args.server_ip, args.server_port)
    total_s = total_n = 0
    print(f"=== 评测模式 {args.mode}，每 BDDL {args.episodes} 条 ===")
    for bf in MODE_BDDL[args.mode]:
        path = os.path.join(CUSTOM, bf)
        goal = BDDLUtils.get_problem_info(path).get("language_instruction", "").strip()
        env = make_suction_env(path, args.resolution)
        robots = find_robots(env)
        s = 0
        for ep in range(args.episodes):
            # save_video_dir 模式下每条都录帧；只保存「失败」episode（失败分析用），文件名带 ep 号
            frames = [] if args.save_video_dir else None
            ok = run_episode(env, robots, policy, goal, args.max_steps, frames_out=frames)
            s += int(ok)
            print(f"  [{bf[:48]}] ep{ep+1}/{args.episodes}: {'成功' if ok else '失败'}")
            if frames and (not ok or args.save_success):
                os.makedirs(args.save_video_dir, exist_ok=True)
                tag = ("OK" if ok else "FAIL") + f"_ep{ep+1}"
                base = bf.replace('.bddl', '') + f"__{tag}"
                imageio.mimsave(os.path.join(args.save_video_dir, base + ".mp4"),
                                frames, fps=30, output_params=["-pix_fmt", "yuv420p"])
                n = len(frames)
                idxs = [int(k * (n - 1) / 5) for k in range(6)] if n > 1 else [0]
                montage = np.vstack([frames[i] for i in idxs])
                imageio.imwrite(os.path.join(args.save_video_dir, base + "_montage.png"), montage)
        try:
            env.env.close()
        except Exception:
            pass
        total_s += s; total_n += args.episodes
        print(f"  >>> {bf[:48]}: {s}/{args.episodes}")
    print(f"\n=== 模式 {args.mode} 总成功率: {total_s}/{total_n} = {100.0*total_s/max(1,total_n):.1f}% ===")


if __name__ == "__main__":
    main()
