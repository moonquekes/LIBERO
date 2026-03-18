from collections import Counter
from pathlib import Path
import argparse
import json

import cv2
import h5py
import numpy as np

import init_path
import libero.libero.utils.utils as libero_utils
from libero.libero.envs import TASK_MAPPING
from libero.libero.envs.suction_sticky_wrapper import SuctionStickyWrapper


def set_grip_cylinder_visibility(env, alpha=0.0):
    try:
        current = env
        model = None
        for _ in range(10):
            if hasattr(current, "sim") and getattr(current, "sim") is not None:
                model = current.sim.model
                break
            if not hasattr(current, "env"):
                break
            current = current.env
        if model is None:
            return
        for site_id in range(model.nsite):
            name = model.site_id2name(site_id)
            if name and name.endswith("grip_site_cylinder"):
                model.site_rgba[site_id, 3] = float(alpha)
    except Exception:
        pass


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def get_suction_diagnostics(env) -> dict:
    default = {
        "constraint_ok": False,
        "attached": False,
        "reason_code": 0,
        "reason_label": "off",
        "contact_angle_deg": np.nan,
        "contact_radial_offset_m": np.nan,
        "contact_body_id": -1,
    }
    if hasattr(env, "get_suction_diagnostics"):
        default.update(env.get_suction_diagnostics())
    return default


def format_constraint_status(diagnostics: dict) -> str:
    if diagnostics.get("attached") and diagnostics.get("constraint_ok"):
        return "attached"

    reason_code = int(diagnostics.get("reason_code", 0))
    status_map = {
        0: "off",
        2: "no_contact",
        3: "rejected:body",
        4: "rejected:angle",
        5: "rejected:radius",
        6: "contact_lost",
    }
    return status_map.get(reason_code, diagnostics.get("reason_label", "unknown"))


def summarize_finite(values, reducer):
    finite = [float(value) for value in values if np.isfinite(value)]
    if not finite:
        return None
    return float(reducer(finite))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--output-dir", default="/home/x/vla/libero/data/suction_dataset/replay_mp4"
    )
    parser.add_argument(
        "--suction-normal-max-angle-deg",
        type=float,
        default=None,
        help="覆盖数据集内记录的吸盘法向量最大夹角阈值（度）",
    )
    parser.add_argument(
        "--suction-effective-radius-ratio",
        type=float,
        default=None,
        help="覆盖数据集内记录的吸盘有效支撑半径系数",
    )
    parser.add_argument(
        "--suction-detach-grace-steps",
        type=int,
        default=None,
        help="覆盖数据集内记录的吸盘脱附缓冲步数",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    dataset_stem = dataset_path.stem
    if args.output_dir is None:
        output_dir = dataset_path.parent / f"{dataset_stem}_replay"
    else:
        output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(dataset_path, "r") as f:
        grp = f["data"]
        env_args = json.loads(grp.attrs["env_args"])
        problem_name = env_args["problem_name"]
        env_kwargs = env_args["env_kwargs"]
        suction_wrapper_kwargs = dict(env_args.get("suction_wrapper_kwargs", {}))
        if not suction_wrapper_kwargs and "suction_wrapper_kwargs_json" in grp.attrs:
            suction_wrapper_kwargs = json.loads(grp.attrs["suction_wrapper_kwargs_json"])
        if args.suction_normal_max_angle_deg is not None:
            suction_wrapper_kwargs["normal_max_angle_deg"] = float(args.suction_normal_max_angle_deg)
        if args.suction_effective_radius_ratio is not None:
            suction_wrapper_kwargs["effective_radius_ratio"] = float(args.suction_effective_radius_ratio)
        if args.suction_detach_grace_steps is not None:
            suction_wrapper_kwargs["detach_grace_steps"] = int(args.suction_detach_grace_steps)

        env = TASK_MAPPING[problem_name](**env_kwargs)
        env = SuctionStickyWrapper(env, **suction_wrapper_kwargs)
        reason_codes = env.get_suction_reason_codes()
        reason_labels = {int(code): name for name, code in reason_codes.items()}
        summary = []

        try:
            for ep in sorted(grp.keys()):
                ep_grp = grp[ep]
                states = ep_grp["states"][()]
                actions = ep_grp["actions"][()]
                model_xml = ep_grp.attrs["model_file"]
                model_xml = libero_utils.postprocess_model_xml(model_xml, {})

                reset_success = False
                while not reset_success:
                    try:
                        env.reset()
                        set_grip_cylinder_visibility(env, alpha=0.0)
                        reset_success = True
                    except Exception:
                        pass

                env.reset_from_xml_string(model_xml)
                env.sim.reset()
                env.sim.set_state_from_flattened(states[0])
                env.sim.forward()

                sample_obs, *_ = (
                    env.step(actions[0]) if len(actions) else (None, None, None, None)
                )
                if len(actions):
                    env.reset_from_xml_string(model_xml)
                    env.sim.reset()
                    env.sim.set_state_from_flattened(states[0])
                    env.sim.forward()

                if sample_obs is None:
                    summary.append(f"{ep}: skipped (empty actions)")
                    continue

                agent_h, agent_w = sample_obs["agentview_image"].shape[:2]
                wrist_h, wrist_w = sample_obs["robot0_eye_in_hand_image"].shape[:2]
                canvas_h = max(agent_h, wrist_h) + 64
                canvas_w = agent_w + wrist_w

                out_path = output_dir / f"{dataset_stem}__{ep}_replay.mp4"
                ensure_parent(out_path)
                writer = cv2.VideoWriter(
                    str(out_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    args.fps,
                    (canvas_w, canvas_h),
                )

                divergences = []
                reason_counter = Counter()
                attached_steps = 0
                angle_values = []
                radial_values = []
                for i, action in enumerate(actions):
                    obs, reward, done, info = env.step(action)
                    diagnostics = get_suction_diagnostics(env)
                    reason_code = int(diagnostics["reason_code"])
                    reason_counter[reason_code] += 1
                    attached_steps += int(bool(diagnostics["attached"]))
                    angle_values.append(float(diagnostics["contact_angle_deg"]))
                    radial_values.append(float(diagnostics["contact_radial_offset_m"]))

                    sim_state = env.sim.get_state().flatten()
                    target_state = states[min(i + 1, len(states) - 1)]
                    err = float(np.linalg.norm(target_state - sim_state))
                    divergences.append(err)

                    agent = obs["agentview_image"][::-1]
                    wrist = obs["robot0_eye_in_hand_image"][::-1]
                    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
                    canvas[:agent_h, :agent_w] = agent
                    canvas[:wrist_h, agent_w : agent_w + wrist_w] = wrist

                    status = format_constraint_status(diagnostics)
                    angle_deg = diagnostics["contact_angle_deg"]
                    radial_offset = diagnostics["contact_radial_offset_m"]
                    angle_text = "na" if not np.isfinite(angle_deg) else f"{float(angle_deg):.1f}"
                    radial_text = (
                        "na"
                        if not np.isfinite(radial_offset)
                        else f"{float(radial_offset):.4f}"
                    )
                    line1 = (
                        f"{ep}  step={i + 1}/{len(actions)}  div={err:.4f}  "
                        f"reward={float(reward):.3f}  done={bool(done)}"
                    )
                    line2 = (
                        f"status={status}  attached={bool(diagnostics['attached'])}  "
                        f"angle_deg={angle_text}  radial_m={radial_text}"
                    )
                    cv2.putText(
                        canvas,
                        line1,
                        (8, canvas_h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        canvas,
                        line2,
                        (8, canvas_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

                writer.release()
                json_path = out_path.with_suffix(".json")
                json_payload = {
                    "episode": ep,
                    "output_mp4": str(out_path),
                    "frames": int(len(actions)),
                    "attached_steps": int(attached_steps),
                    "reason_code_counts": {
                        str(code): int(count) for code, count in sorted(reason_counter.items())
                    },
                    "reason_label_counts": {
                        reason_labels.get(code, str(code)): int(count)
                        for code, count in sorted(reason_counter.items())
                    },
                    "mean_contact_angle_deg": summarize_finite(angle_values, np.mean),
                    "max_contact_angle_deg": summarize_finite(angle_values, np.max),
                    "mean_contact_radial_offset_m": summarize_finite(radial_values, np.mean),
                    "max_contact_radial_offset_m": summarize_finite(radial_values, np.max),
                    "mean_divergence": summarize_finite(divergences, np.mean),
                    "max_divergence": summarize_finite(divergences, np.max),
                }
                json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
                summary.append(
                    f"{ep}: frames={len(actions)}, mean_div={float(np.mean(divergences)):.4f}, "
                    f"max_div={float(np.max(divergences)):.4f}, json={json_path}, out={out_path}"
                )
                print(summary[-1], flush=True)
        finally:
            env.close()

    summary_path = output_dir / f"{dataset_stem}__summary.txt"
    summary_path.write_text("\n".join(summary), encoding="utf-8")
    print(f"summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
