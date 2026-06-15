#!/usr/bin/env python3
"""
audit_success.py — 只读成功率审查（不写任何训练数据）。

复用 raw2xvla 的 env 思路：用 set_state_from_flattened 覆盖 qpos（能读带 center_marker
的早期数据 + 失效 bddl 路径），replay 完整轨迹后调 env._check_success()，可选追加 settle
帧（对齐 create_dataset 的 success_settle_steps）。专门用于审查 create_dataset 读不了的
早期 shared_stable_v1 数据。

用法：
  python audit_success.py --files a.hdf5 b.hdf5 ...
  python audit_success.py --raw-root <dir>          # 审整目录
"""
import argparse, glob, json, os, sys, tempfile
import numpy as np
import h5py


def _decode(v):
    if isinstance(v, bytes):
        return v.decode("utf-8", "ignore")
    if isinstance(v, np.ndarray) and v.dtype.kind in ("S", "O"):
        return _decode(v.tobytes() if v.dtype.kind == "S" else v.item())
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=[])
    ap.add_argument("--raw-root", default=None)
    ap.add_argument("--libero-root", default="/home/x/vla/libero")
    ap.add_argument("--settle-steps", type=int, default=40)
    ap.add_argument("--suction-normal-max-angle-deg", type=float, default=10.0)
    ap.add_argument("--suction-effective-radius-ratio", type=float, default=0.7)
    ap.add_argument("--suction-detach-grace-steps", type=int, default=2)
    args = ap.parse_args()

    if args.libero_root not in sys.path:
        sys.path.insert(0, args.libero_root)
    import libero.libero.envs.robots  # noqa: F401
    from libero.libero.envs import TASK_MAPPING
    from libero.libero.envs.suction_sticky_wrapper import SuctionStickyWrapper
    from robosuite import load_controller_config

    files = list(args.files)
    if args.raw_root:
        files += sorted(glob.glob(os.path.join(args.raw_root, "**", "*.hdf5"), recursive=True))
    if not files:
        print("no files"); sys.exit(1)

    suction_kwargs = dict(
        normal_max_angle_deg=args.suction_normal_max_angle_deg,
        effective_radius_ratio=args.suction_effective_radius_ratio,
        detach_grace_steps=args.suction_detach_grace_steps,
    )
    results = []
    for fp in files:
        with h5py.File(fp, "r") as f:
            src = f["trajectory"] if "trajectory" in f else f["data"][list(f["data"].keys())[0]]
            states = src["states"][()]
            actions = np.asarray(src["actions"][()])
            problem = json.loads(_decode(f.attrs["problem_info"]))
            problem_name = problem["problem_name"]
            bddl_content = _decode(f.attrs["bddl_file_content"])
            # selfcollect_rollout.py 自采的动作是绝对位姿（eval 推理模式），回放须关 delta
            actions_absolute = bool(f.attrs.get("actions_absolute", False))
            # 回放物理须与录制一致：自采 25°（attrs 标注），人工 demo 无 attrs 走命令行默认
            file_suction = dict(
                normal_max_angle_deg=float(
                    f.attrs.get("suction_normal_max_angle_deg", suction_kwargs["normal_max_angle_deg"])),
                effective_radius_ratio=float(
                    f.attrs.get("suction_effective_radius_ratio", suction_kwargs["effective_radius_ratio"])),
                detach_grace_steps=int(
                    f.attrs.get("suction_detach_grace_steps", suction_kwargs["detach_grace_steps"])),
            )

        # 每条单独建 env：_check_success 依赖该 demo 自己 bddl 的 :goal，不能跨任务复用
        tmp_bddl = os.path.join(tempfile.gettempdir(), f"_audit_env_{os.getpid()}.bddl")
        with open(tmp_bddl, "w") as bf:
            bf.write(bddl_content)
        cc = load_controller_config(default_controller="OSC_POSE")
        env = SuctionStickyWrapper(
            TASK_MAPPING[problem_name](
                bddl_file_name=tmp_bddl, robots=["SuctionPanda"], controller_configs=cc,
                has_renderer=False, has_offscreen_renderer=False, ignore_done=True,
                use_camera_obs=False, reward_shaping=True, control_freq=20,
            ), **file_suction)

        for _ in range(5):
            try:
                env.reset(); break
            except Exception:
                continue
        env.sim.set_state_from_flattened(states[0]); env.sim.forward()
        if actions_absolute:
            env.env.robots[0].controller.use_delta = False

        last = None
        for a in actions:
            env.step(a.tolist()); last = a
        ok = bool(env._check_success())
        if not ok and args.settle_steps > 0 and last is not None:
            settle = np.array(last, dtype=np.float64)
            if not actions_absolute:
                settle[:6] = 0.0   # delta 模式零位移=悬停；绝对模式复用最后目标位姿即悬停
            for _ in range(args.settle_steps):
                env.step(settle.tolist())
                if env._check_success():
                    ok = True; break
        results.append((fp, ok))
        print(f"  {'success' if ok else 'FAILED ':8}  {os.path.basename(fp)}")
        try:
            env.env.close()
        except Exception:
            pass

    s = sum(1 for _, ok in results if ok)
    print(f"\n审查 {len(results)} 条：成功 {s}，失败 {len(results)-s}")
    fails = [fp for fp, ok in results if not ok]
    if fails:
        print("失败：")
        for fp in fails:
            print("  ", fp)


if __name__ == "__main__":
    main()
