#!/usr/bin/env python3
"""eval_sorting_diag.py — 诊断版分拣评测 + 多进程并行。

在 eval_sorting_xvla.py 基础上，对每个 episode 输出失败归因：
  NO_GRASP:<reason>   全程没吸住任何板（reason 来自 SuctionStickyWrapper：no_contact/
                      angle_exceeded/outside_effective_radius/suction_never_on）
  WRONG_OBJECT        吸住了非目标板（背 slot 捷径）
  WRONG_BIN:<bin>     吸住目标板但放进了别的框
  GRASPED_TARGET_DROPPED  吸住目标板但最后落在桌面/框外（中途松掉/超时）
  PLACED_CORRECT_BUT_FAIL 目标板几何上在正确框内但 _check_success=False（放得太高/未判定）
  SUCCESS

并行：--workers N，每个 worker 自建 env + client，共用 h100 server（端口转发 localhost:port）。
本地 RTX4090 + EGL 离屏渲染，多进程各自一个 EGL context；瓶颈是逐 chunk 的 HTTP 往返，
并行可把各 worker 的仿真步与 server 推理重叠 → 大幅提速。

用法（WSL vla-adapter env，先建好到 h100 的 localhost:port 转发）：
  MUJOCO_GL=egl python eval_sorting_diag.py --mode A --episodes 10 --workers 4 \
      --server_port 8000 --max_steps 600 --save_video_dir ~/diag_videos --out_json ~/diag_A.json
"""
import argparse, collections, json, os, sys, time
os.environ.setdefault("MUJOCO_GL", "egl")
from typing import Deque, List, Optional

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
BINS = ["red_bin_1", "blue_bin_1", "yellow_bin_1"]
BIN_THRESH = 0.06  # m：目标板水平距框心小于此值算"在框里"

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


def parse_target(bddl_fn):
    fn = bddl_fn.lower()
    if "rectangular" in fn:
        shape, base = "rect", "steel_plate_1"
    elif "round" in fn:
        shape, base = "round", "steel_plate_round_1"
    elif "triangular" in fn:
        shape, base = "triangle", "steel_plate_triangle_1"
    else:
        shape, base = "?", "?"
    if "red_bin" in fn:
        tbin = "red_bin_1"
    elif "blue_bin" in fn:
        tbin = "blue_bin_1"
    elif "yellow_bin" in fn:
        tbin = "yellow_bin_1"
    else:
        tbin = "?"
    return shape, base, tbin


def shape_of(name):
    n = (name or "").lower()
    if "bin" in n:
        return "bin"
    if "round" in n:
        return "round"
    if "triangle" in n:
        return "triangle"
    if "steel_plate" in n:
        return "rect"
    return "other"


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


def resolve_body(env, obj_base):
    model = env.sim.model
    want = obj_base + "_main"
    fallback = None
    for bid in range(model.nbody):
        nm = model.body_id2name(bid)
        if nm == want:
            return bid
        if fallback is None and nm and nm.startswith(obj_base):
            fallback = bid
    return fallback


class XVLAClient:
    def __init__(self, host, port, steps=10, domain_id=3, replan_every=0):
        self.url = f"http://{host}:{port}/act"
        self.steps = steps            # flow 积分步数（越大动作越精确）
        self.replan_every = replan_every  # >0：每执行这么多步就重查（闭环）；0：整段开环
        self.domain_id = domain_id
        self.reset()

    def reset(self):
        self.proprio: Optional[np.ndarray] = None
        self.action_plan: Deque[List[float]] = collections.deque()
        self._last_grip = 0.0

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
            if self.replan_every > 0:
                # 闭环：proprio 用当前真实状态（而非上一段预测的末位姿）
                closed = np.concatenate([pos, ori6d, np.array([self._last_grip])], axis=-1)
                self.proprio = np.concatenate([closed, np.zeros_like(closed)], axis=-1)
            r = requests.post(self.url, json=self._query(av, wr, pos, ori6d, goal), timeout=120)
            r.raise_for_status()
            action = np.array(r.json()["action"])  # (T,10)
            if self.replan_every <= 0:
                self.proprio[:9] = action[-1, :9].copy()
            eef = action[:, :3]
            aa = rotate6d_to_axisangle(action[:, 3:9])
            grip = action[:, 9:10]
            rows = np.concatenate([eef, aa, grip], axis=-1)
            keep = self.replan_every if self.replan_every > 0 else len(rows)
            for row in rows[:keep].tolist():
                self.action_plan.append(row)
        a = np.array(self.action_plan.popleft(), dtype=np.float32)
        a[-1] = 1.0 if a[-1] > 0.5 else -1.0
        self._last_grip = float(a[-1])
        return a


def run_episode_diag(env, robots, policy, goal, target_base, target_bin,
                     max_steps=600, num_wait=10, frames_out=None,
                     td_enable=False, td_step=0.004, td_max=0.05):
    policy.reset()
    obs = env.reset()
    for r in robots:
        r.controller.use_delta = True
    for _ in range(num_wait):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
    for r in robots:
        r.controller.use_delta = False

    target_bid = resolve_body(env, target_base)
    bin_bids = {b: resolve_body(env, b) for b in BINS}

    ever_on = False
    reasons = collections.Counter()
    grasped = set()
    first_grasp = None
    grasped_target_steps = 0
    success = False

    # Plan A 末端下压状态：策略已开吸却 no_contact（欠冲）时，硬压 Z 直到吸上或到限位。
    # 这不是 naive replan（重问策略得同样偏短答案），而是绕过策略对最后几毫米的判断。
    td_accum = 0.0
    prev_undershoot = False
    td_steps_used = 0
    td_xy = None

    for t in range(max_steps):
        av = obs["agentview_image"]; wr = obs["robot0_eye_in_hand_image"]
        if frames_out is not None:
            frames_out.append(np.hstack([flip_agentview(av), flip_agentview(wr)]))
        ctrl = robots[0].controller
        pos = np.asarray(ctrl.ee_pos, dtype=np.float32)
        ori6d = mat_to_rotate6d(np.asarray(ctrl.ee_ori_mat, dtype=np.float32))
        a = policy.step(av, wr, pos, ori6d, goal)
        if td_enable and prev_undershoot and td_accum < td_max:
            if td_xy is None:
                td_xy = (float(pos[0]), float(pos[1]))   # 记录欠冲起点 XY
            a = np.asarray(a, dtype=np.float32).copy()
            a[0], a[1] = td_xy               # 冻结 XY：纯垂直下压，别让策略搬运段把 XY 拐向错板
            a[2] = float(pos[2]) - td_step   # 从当前高度再下压 td_step（绝对位姿目标）
            a[6] = 1.0                        # 维持吸盘开
            td_accum += td_step
            td_steps_used += 1
        obs, _, _, _ = env.step(a.tolist())

        on = env.is_suction_on()
        reason_label = None
        if on:
            ever_on = True
            reason_label = env.get_suction_diagnostics()["reason_label"]
            reasons[reason_label] += 1
        att = env.attached_body_id
        if att is not None:
            nm = env.sim.model.body_id2name(att)
            grasped.add(nm)
            if first_grasp is None:
                first_grasp = t
            if nm and nm.startswith(target_base):
                grasped_target_steps += 1
            td_accum = 0.0   # 吸上了，重置下压预算
            td_xy = None
        prev_undershoot = bool(td_enable and on and att is None and reason_label == "no_contact")
        if not prev_undershoot:
            td_xy = None
            td_accum = 0.0

        if check_success(env):
            success = True
            break

    # ---- 末位姿态 ----
    data = env.sim.data
    def xy(bid):
        p = data.body_xpos[bid]
        return np.array([p[0], p[1]]), float(p[2])

    tgt_xy, tgt_z = (xy(target_bid) if target_bid is not None else (np.array([np.nan, np.nan]), np.nan))
    bin_dist = {}
    for b, bid in bin_bids.items():
        if bid is not None and np.all(np.isfinite(tgt_xy)):
            bin_dist[b] = float(np.linalg.norm(tgt_xy - xy(bid)[0]))
    nearest_bin = min(bin_dist, key=bin_dist.get) if bin_dist else None
    nearest_d = bin_dist.get(nearest_bin, float("nan")) if nearest_bin else float("nan")
    in_bin = nearest_bin if (nearest_bin and nearest_d < BIN_THRESH) else None

    grasped_target = any(n and n.startswith(target_base) for n in grasped)
    wrong_plates = sorted({n for n in grasped if n and not n.startswith(target_base)
                           and shape_of(n) in ("rect", "round", "triangle")})

    # ---- 归因 ----
    if success:
        cat = "SUCCESS"
    elif len(grasped) == 0:
        dom = reasons.most_common(1)[0][0] if reasons else "suction_never_on"
        cat = "NO_GRASP:" + dom
    elif not grasped_target:
        cat = "WRONG_OBJECT"
    elif in_bin is None:
        cat = "GRASPED_TARGET_DROPPED"
    elif in_bin == target_bin:
        cat = "PLACED_CORRECT_BUT_FAIL"
    else:
        cat = "WRONG_BIN:" + in_bin

    diag = {
        "success": bool(success),
        "category": cat,
        "ever_suction_on": bool(ever_on),
        "grasped_bodies": sorted(grasped),
        "grasped_target": bool(grasped_target),
        "wrong_plates_grasped": wrong_plates,
        "suction_reason_hist": dict(reasons),
        "first_grasp_step": first_grasp,
        "grasped_target_steps": grasped_target_steps,
        "target_final_xy": [float(tgt_xy[0]), float(tgt_xy[1])],
        "target_final_z": tgt_z,
        "nearest_bin": nearest_bin,
        "nearest_bin_dist": nearest_d,
        "target_in_bin": in_bin,
        "target_bin": target_bin,
        "steps": t + 1,
    }
    return success, diag


def run_job(job):
    bf = job["bddl"]; ep = job["ep"]; A = job["args"]
    path = os.path.join(CUSTOM, bf)
    goal = BDDLUtils.get_problem_info(path).get("language_instruction", "").strip()
    shape, base, tbin = parse_target(bf)
    env = make_suction_env(path, A["resolution"])
    robots = find_robots(env)
    policy = XVLAClient(A["ip"], A["port"], steps=A["flow_steps"], replan_every=A["replan_every"])
    frames = [] if A["save_video_dir"] else None
    t0 = time.time()
    success, diag = run_episode_diag(env, robots, policy, goal, base, tbin,
                                     A["max_steps"], frames_out=frames,
                                     td_enable=A.get("terminal_descent", False),
                                     td_step=A.get("td_step", 0.004),
                                     td_max=A.get("td_max", 0.05))
    diag["bddl"] = bf; diag["ep"] = ep; diag["shape"] = shape
    diag["wall_s"] = round(time.time() - t0, 1)

    if frames and (not success or A["save_success"]):
        os.makedirs(A["save_video_dir"], exist_ok=True)
        tag = ("OK" if success else "FAIL") + f"_ep{ep+1}_" + diag["category"].replace(":", "-").replace("/", "-")
        b = bf.replace(".bddl", "") + "__" + tag
        try:
            imageio.mimsave(os.path.join(A["save_video_dir"], b + ".mp4"),
                            frames, fps=30, output_params=["-pix_fmt", "yuv420p"])
            n = len(frames)
            idxs = [int(k * (n - 1) / 5) for k in range(6)] if n > 1 else [0]
            imageio.imwrite(os.path.join(A["save_video_dir"], b + "_montage.png"),
                            np.vstack([frames[i] for i in idxs]))
        except Exception as e:
            diag["video_error"] = str(e)
    try:
        env.env.close()
    except Exception:
        pass
    print(f"  [{bf[:46]}] ep{ep+1}: {diag['category']:<28} ({diag['wall_s']}s)", flush=True)
    return diag


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["A", "B"], default="A")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--server_ip", default="127.0.0.1")
    p.add_argument("--server_port", type=int, default=8000)
    p.add_argument("--max_steps", type=int, default=600)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--save_video_dir", default=None)
    p.add_argument("--save_success", action="store_true")
    p.add_argument("--out_json", default=None)
    p.add_argument("--flow_steps", type=int, default=10, help="flow 积分步数，越大动作越精确")
    p.add_argument("--replan_every", type=int, default=0, help=">0 每这么多步重查(闭环)；0 整段开环")
    p.add_argument("--bddls", default="", help="逗号分隔的 bddl 文件名，覆盖 --mode 的列表")
    p.add_argument("--terminal_descent", action="store_true",
                   help="Plan A：策略已开吸却 no_contact 时硬压 Z 直到吸上（部署侧脚本末端抓取）")
    p.add_argument("--td_step", type=float, default=0.004, help="末端下压每步 m")
    p.add_argument("--td_max", type=float, default=0.05, help="末端下压总预算 m（防戳穿桌面）")
    args = p.parse_args()

    A = {"ip": args.server_ip, "port": args.server_port, "max_steps": args.max_steps,
         "resolution": args.resolution, "save_video_dir": args.save_video_dir,
         "save_success": args.save_success,
         "flow_steps": args.flow_steps, "replan_every": args.replan_every,
         "terminal_descent": args.terminal_descent, "td_step": args.td_step, "td_max": args.td_max}
    bddl_list = [b.strip() for b in args.bddls.split(",") if b.strip()] or MODE_BDDL[args.mode]
    jobs = [{"bddl": bf, "ep": ep, "args": A}
            for bf in bddl_list for ep in range(args.episodes)]

    print(f"=== 诊断评测 模式 {args.mode}，每 BDDL {args.episodes} 条，workers={args.workers} ===", flush=True)
    t0 = time.time()
    if args.workers <= 1:
        results = [run_job(j) for j in jobs]
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers) as pool:
            results = pool.map(run_job, jobs)
    dt = time.time() - t0

    # ---- 汇总 ----
    per_bddl = collections.defaultdict(lambda: [0, 0])
    cat_hist = collections.Counter()
    for d in results:
        per_bddl[d["bddl"]][0] += int(d["success"])
        per_bddl[d["bddl"]][1] += 1
        cat_hist[d["category"]] += 1
    tot_s = sum(int(d["success"]) for d in results)
    tot_n = len(results)

    print("\n=== 每 BDDL 成功率 ===")
    for bf in bddl_list:
        s, n = per_bddl[bf]
        print(f"  {bf[:52]:<52} {s}/{n}")
    print("\n=== 失败/成功 归因直方图 ===")
    for cat, c in cat_hist.most_common():
        print(f"  {cat:<32} {c}")
    print(f"\n=== 模式 {args.mode} 总成功率 {tot_s}/{tot_n} = {100.0*tot_s/max(1,tot_n):.1f}%"
          f" | 用时 {dt:.0f}s | {tot_n/max(dt,1e-6):.2f} ep/s | workers={args.workers} ===")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({"mode": args.mode, "episodes": args.episodes, "workers": args.workers,
                       "wall_s": dt, "total": [tot_s, tot_n],
                       "per_bddl": {k: v for k, v in per_bddl.items()},
                       "cat_hist": dict(cat_hist), "results": results}, f, indent=2)
        print(f"📝 写出 {args.out_json}")


if __name__ == "__main__":
    main()
