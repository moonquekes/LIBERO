#!/usr/bin/env python3
"""
scripted_expert_collect.py — 脚本专家采集（真值航点，不依赖策略/h100）。

背景：r2bal 自蒸馏在 round@4cm 真实散布上成功率仅 ~6%（评测 5/6 是 seed(0) 单布局假象，
eval_sorting_diag 每条 episode 都重建 env 并 seed(0)，同 BDDL 实为同一布局反复滚），
弱老师采不动。改用脚本专家：从 MuJoCo 读目标板/框真值位姿，OSC 绝对位姿走
悬停→下压(贴面再吸)→提起→平移→放框 航点，成功率接近 100%，可直接采 8cm 评测分布（全居中）。

与 selfcollect_rollout.py 同构：动作=绝对位姿（attrs actions_absolute=True），
吸盘 25°/0.7/2（评测同款，参数写 attrs，raw2xvla/audit 按 attrs 还原回放物理），
开头 5 帧 hold（对齐 cap_index=5）+ 成功后 10 帧 hold 复验。
不开相机/渲染（raw 只存 states+actions，图像打包时重渲）→ 纯 CPU，每条秒级。

用法：
  python scripts/scripted_expert_collect.py \
    --bddl pick_up_the_round_steel_plate_and_place_it_in_the_blue_bin.bddl \
    --num 15 --workers 4 --out-dir $D/raw_hdf5/round_blue_bin_selfc8
"""
import argparse, json, os, sys, time
import multiprocessing as mp

import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_sorting_diag import (  # noqa: E402
    CUSTOM, find_robots, check_success, parse_target, resolve_body,
)
import libero.libero.envs.bddl_utils as BDDLUtils  # noqa: E402
from libero.libero.envs import TASK_MAPPING  # noqa: E402
from libero.libero.envs.suction_sticky_wrapper import SuctionStickyWrapper  # noqa: E402
from robosuite import load_controller_config  # noqa: E402
import robosuite.utils.transform_utils as T  # noqa: E402

# 与评测/部署一致（wrapper 默认）；回放端按 h5 attrs 还原
SUCTION_KWARGS = dict(normal_max_angle_deg=25.0, effective_radius_ratio=0.7,
                      detach_grace_steps=2)
HOLD_FRAMES_HEAD = 5     # 对齐 raw2xvla --cap-index 5
HOLD_FRAMES_TAIL = 10    # 成功后静置复验，滤"擦边一瞬"假成功
TRAVEL_STEP = 0.0035     # 平移速度 m/step（7cm/s @20Hz，接近人工遥操作手感）
DESCEND_STEP = 0.0015    # 下压速度 m/step（贴面阶段放慢）
PRESS_DZ = 0.03          # 吸附后指令继续下压的深度（怼穿式余量，对齐人工 demo 风格）
PRESS_STEP = 0.003       # 下压段速度 m/step
HOVER_DZ = 0.20          # 悬停高度（板上方）
LIFT_DZ = 0.18           # 提起高度
DROP_DZ = 0.10           # 框上方释放高度


def make_env(bddl_path):
    cc = load_controller_config(default_controller="OSC_POSE")
    pn = BDDLUtils.get_problem_info(bddl_path)["problem_name"]
    env = TASK_MAPPING[pn](
        bddl_file_name=bddl_path, robots=["SuctionPanda"], controller_configs=cc,
        has_renderer=False, has_offscreen_renderer=False, ignore_done=True,
        use_camera_obs=False, reward_shaping=True, control_freq=20,
    )
    return SuctionStickyWrapper(env, **SUCTION_KWARGS)


def body_pos(env, base):
    bid = resolve_body(env, base)
    return np.array(env.sim.data.body_xpos[bid], dtype=np.float64)


class Recorder:
    """逐步记录 (state-before, action)，动作=绝对位姿 [pos3, axisangle3, grip]。"""

    def __init__(self, env, ctrl, aa_fixed):
        self.env, self.ctrl, self.aa = env, ctrl, aa_fixed
        self.states, self.actions = [], []

    def step_to(self, pos, grip):
        a = np.concatenate([np.asarray(pos, dtype=np.float64), self.aa, [grip]])
        self.states.append(self.env.sim.get_state().flatten())
        self.env.step(a.tolist())
        self.actions.append(a)

    def hold(self, n, grip):
        for _ in range(n):
            self.step_to(np.array(self.ctrl.ee_pos), grip)

    def servo(self, target, grip, step_size, settle=8, max_extra=80,
              stop_fn=None):
        """胡萝卜点从当前 ee 匀速走到 target；到达后再补 settle 帧收敛。
        stop_fn 返回 True 时提前结束（如吸附成功）。返回是否被 stop_fn 截停。"""
        target = np.asarray(target, dtype=np.float64)
        carrot = np.array(self.ctrl.ee_pos, dtype=np.float64)
        n_max = int(np.linalg.norm(target - carrot) / step_size) + settle + max_extra
        for _ in range(n_max):
            d = target - carrot
            dist = np.linalg.norm(d)
            carrot = target if dist <= step_size else carrot + d / dist * step_size
            self.step_to(carrot, grip)
            if stop_fn is not None and stop_fn():
                return True
            if np.array_equal(carrot, target):
                settle -= 1
                if settle <= 0:
                    break
        return False


def record_episode(env, robots, target_base, target_bin):
    obs = env.reset()
    for r in robots:
        r.controller.use_delta = True
    for _ in range(10):
        env.step([0, 0, 0, 0, 0, 0, -1])
    for r in robots:
        r.controller.use_delta = False

    ctrl = robots[0].controller
    aa0 = T.quat2axisangle(T.mat2quat(np.asarray(ctrl.ee_ori_mat)))
    rec = Recorder(env, ctrl, aa0)
    rec.hold(HOLD_FRAMES_HEAD, -1.0)

    plate = body_pos(env, target_base)
    tbin = body_pos(env, target_bin)

    # —— r8 多样性注入：逐条随机化路径/速度/高度/下压深度。
    # 同质脚本 demo 批量堆叠会让居中场景视觉不可区分 → 检索塌缩、slot 吸引子
    # （r5→r6→r7 总分单调下降的病因假设）；人工 demo 的天然多样性正在于此。
    rng = np.random.default_rng()
    # SE_SPEED_JITTER=0 → 只随机几何(悬停/提起/下压深度/横向绕行/切换高度)，速度节奏冻结。
    # 修 r8"速度随机化伤开环模仿节奏一致性"的回归：可变的是路径形状，不能乱的是节奏。
    spd = rng.uniform if os.environ.get("SE_SPEED_JITTER", "1") != "0" else (lambda a, b: 1.0)
    hover = HOVER_DZ * rng.uniform(0.7, 1.3)
    travel = TRAVEL_STEP * spd(0.7, 1.4)
    descend = DESCEND_STEP * spd(0.8, 1.6)
    press_dz = float(rng.uniform(0.02, 0.045))
    press_step = PRESS_STEP * spd(0.8, 1.3)
    lift = LIFT_DZ * rng.uniform(0.85, 1.25)
    drop = DROP_DZ * rng.uniform(0.8, 1.3)
    off = rng.uniform(-0.03, 0.03, size=2)   # 高空横向绕行偏移
    near = float(rng.uniform(0.04, 0.09))    # 快降→慢压切换高度

    # 1) 经随机偏移中途点绕行，再回到板正上方（路径多样性）
    rec.servo([plate[0] + off[0], plate[1] + off[1], plate[2] + hover], -1.0, travel)
    rec.servo([plate[0], plate[1], plate[2] + hover * rng.uniform(0.5, 0.9)], -1.0, travel)
    # 2) 快降到接近，再贴面慢压 + 开吸，直到吸附（自适应探 pad 高度，不假设吸盘长度）
    attached = lambda: env.attached_body_id is not None
    rec.servo([plate[0], plate[1], plate[2] + near], -1.0, travel)
    hit = rec.servo([plate[0], plate[1], plate[2]], 1.0, descend,
                    settle=0, max_extra=80, stop_fn=attached)
    if not hit:
        return None                      # 没吸上（极少），整条重来
    nm = env.sim.model.body_id2name(env.attached_body_id) or ""
    if not nm.startswith(target_base):
        return None                      # 吸错件（理论不可能，保险）
    # 吸附后指令继续下压 press_dz：实际臂被板面挡住，阻抗只是压实。
    # 治零余量悬停吸取——旧版 stop_fn 一吸上就停，写进数据的目标 z 恰在接触点，
    # 模仿幅度衰减 1-2mm 即 no_contact（r4full/r4 欠冲实锤病因）。
    rec.servo([plate[0], plate[1], plate[2] - press_dz], 1.0, press_step)
    rec.hold(int(rng.integers(2, 7)), 1.0)
    # 3) 提起
    rec.servo([plate[0], plate[1], plate[2] + lift], 1.0, travel)
    if not attached():
        return None
    # 4) 平移到框上方
    rec.servo([tbin[0], tbin[1], plate[2] + lift], 1.0, travel)
    if not attached():
        return None
    # 5) 降到释放高度 → 松吸 → 静置
    rec.servo([tbin[0], tbin[1], tbin[2] + drop], 1.0, travel)
    rec.hold(15, -1.0)
    # 6) 撤离
    rec.servo([tbin[0], tbin[1], tbin[2] + drop + 0.08], -1.0, travel)

    if not check_success(env):
        return None
    rec.hold(HOLD_FRAMES_TAIL, -1.0)
    if not check_success(env):
        return None
    return np.stack(rec.states), np.stack(rec.actions)


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
        f.attrs["actions_absolute"] = True
        f.attrs["collected_by"] = "scripted_expert_collect.py"
        f.attrs["suction_normal_max_angle_deg"] = float(SUCTION_KWARGS["normal_max_angle_deg"])
        f.attrs["suction_effective_radius_ratio"] = float(SUCTION_KWARGS["effective_radius_ratio"])
        f.attrs["suction_detach_grace_steps"] = int(SUCTION_KWARGS["detach_grace_steps"])
    return path


def worker(job):
    quota, A = job["quota"], job["args"]
    bddl_path = os.path.join(CUSTOM, A["bddl"])
    info = BDDLUtils.get_problem_info(bddl_path)
    pn = info["problem_name"]
    goal = info.get("language_instruction", "").strip()
    bddl_content = open(bddl_path).read()
    task_tag = goal.replace(" ", "_")
    _, target_base, target_bin = parse_target(A["bddl"])
    env = make_env(bddl_path)
    robots = find_robots(env)
    saved, attempts = [], 0
    while len(saved) < quota and attempts < quota * 4:
        attempts += 1
        t0 = time.time()
        out = record_episode(env, robots, target_base, target_bin)
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
    ap.add_argument("--num", type=int, default=15)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    A = dict(bddl=args.bddl, out_dir=args.out_dir)
    base, extra = divmod(args.num, args.workers)
    jobs = [{"quota": base + (1 if i < extra else 0), "args": A}
            for i in range(args.workers) if base + (1 if i < extra else 0) > 0]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(jobs)) as pool:
        results = pool.map(worker, jobs)
    total = sum(len(r) for r in results)
    print(f"\n脚本专家采集完成：{total}/{args.num} → {args.out_dir}")
    if total < args.num:
        print("⚠️ 未达目标条数，可重跑补齐（文件名时间戳+pid，追加安全）")


if __name__ == "__main__":
    main()
