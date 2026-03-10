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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--output-dir", default='/home/x/vla/libero/data/suction_dataset/replay_mp4')
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    if args.output_dir is None:
        output_dir = dataset_path.parent / f"{dataset_path.stem}_replay"
    else:
        output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(dataset_path, "r") as f:
        grp = f["data"]
        env_args = json.loads(grp.attrs["env_args"])
        problem_name = env_args["problem_name"]
        env_kwargs = env_args["env_kwargs"]

        env = TASK_MAPPING[problem_name](**env_kwargs)
        env = SuctionStickyWrapper(env)
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

                sample_obs, *_ = env.step(actions[0]) if len(actions) else (None, None, None, None)
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
                canvas_h = max(agent_h, wrist_h) + 40
                canvas_w = agent_w + wrist_w

                out_path = output_dir / f"{ep}_replay.mp4"
                ensure_parent(out_path)
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (canvas_w, canvas_h))

                divergences = []
                for i, action in enumerate(actions):
                    obs, reward, done, info = env.step(action)
                    sim_state = env.sim.get_state().flatten()
                    target_state = states[min(i + 1, len(states) - 1)]
                    err = float(np.linalg.norm(target_state - sim_state))
                    divergences.append(err)

                    agent = obs["agentview_image"][::-1]
                    wrist = obs["robot0_eye_in_hand_image"][::-1]
                    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
                    canvas[:agent_h, :agent_w] = agent
                    canvas[:wrist_h, agent_w:agent_w + wrist_w] = wrist

                    text = (
                        f"{ep}  step={i + 1}/{len(actions)}  div={err:.4f}  "
                        f"reward={float(reward):.3f}  done={bool(done)}"
                    )
                    cv2.putText(
                        canvas,
                        text,
                        (8, canvas_h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

                writer.release()
                summary.append(
                    f"{ep}: frames={len(actions)}, mean_div={float(np.mean(divergences)):.4f}, "
                    f"max_div={float(np.max(divergences)):.4f}, out={out_path}"
                )
                print(summary[-1], flush=True)
        finally:
            env.close()

    summary_path = output_dir / "summary.txt"
    summary_path.write_text("\n".join(summary), encoding="utf-8")
    print(f"summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
