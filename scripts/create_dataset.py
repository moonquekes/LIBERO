import argparse
import os
from pathlib import Path
import h5py
import numpy as np
import json
import robosuite
import robosuite.utils.transform_utils as T
import robosuite.macros as macros

import init_path
import libero.libero.utils.utils as libero_utils
import cv2
from PIL import Image
from robosuite.utils import camera_utils

from libero.libero.envs import *
from libero.libero import get_libero_path
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


def resolve_output_dataset_path(dataset_path: str, bddl_file_name: str) -> str:
    bddl_rel_path = bddl_file_name.split("bddl_files/")[-1]
    flat_dataset_name = bddl_rel_path.replace("/", "__").replace(".bddl", ".hdf5")

    if dataset_path:
        output_path = Path(dataset_path).expanduser()
        if not output_path.is_absolute():
            output_path = (Path(__file__).resolve().parents[1] / output_path).resolve()
    else:
        output_path = Path(__file__).resolve().parents[1] / "datasets" / "converted"

    if output_path.suffix == ".hdf5":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return str(output_path)

    output_path.mkdir(parents=True, exist_ok=True)
    return str(output_path / flat_dataset_name)


def clip_vector_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    if max_norm <= 0:
        return vector
    norm = np.linalg.norm(vector)
    if norm <= max_norm or norm == 0:
        return vector
    return vector * (max_norm / norm)


def is_noop_action(action: np.ndarray, prev_action: np.ndarray | None, threshold: float) -> bool:
    if prev_action is None:
        return np.linalg.norm(action[:-1]) < threshold
    return np.linalg.norm(action[:-1]) < threshold and action[-1] == prev_action[-1]


def transform_action(action: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    transformed = np.asarray(action, dtype=np.float64).copy()
    transformed[:3] *= args.replay_translation_scale
    transformed[3:6] *= args.replay_rotation_scale
    transformed[:3] = clip_vector_norm(transformed[:3], args.max_translation_norm)
    transformed[3:6] = clip_vector_norm(transformed[3:6], args.max_rotation_norm)
    return transformed


def split_action(action: np.ndarray, args: argparse.Namespace) -> list[np.ndarray]:
    if not args.split_large_actions:
        return [action]

    translation_norm = np.linalg.norm(action[:3])
    rotation_norm = np.linalg.norm(action[3:6])
    translation_steps = 1
    rotation_steps = 1

    if args.substep_translation_norm > 0:
        translation_steps = max(1, int(np.ceil(translation_norm / args.substep_translation_norm)))
    if args.substep_rotation_norm > 0:
        rotation_steps = max(1, int(np.ceil(rotation_norm / args.substep_rotation_norm)))

    substeps = max(translation_steps, rotation_steps)
    if args.max_action_substeps > 0:
        substeps = min(substeps, args.max_action_substeps)

    if substeps <= 1:
        return [action]

    split_actions = []
    for _ in range(substeps):
        step_action = action.copy()
        step_action[:6] = action[:6] / substeps
        split_actions.append(step_action)
    return split_actions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-file", default="demo.hdf5")

    parser.add_argument(
        "--use-actions",
        action="store_true",
    )
    parser.add_argument("--use-camera-obs", action="store_true")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="datasets/",
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        default="training_set",
    )

    parser.add_argument("--no-proprio", action="store_true")

    parser.add_argument(
        "--use-depth",
        action="store_true",
    )
    parser.add_argument(
        "--filter-noop",
        action="store_true",
        help="过滤动作范数接近 0 的样本，用于对齐 *_no_noops 数据",
    )
    parser.add_argument(
        "--noop-threshold",
        type=float,
        default=1e-4,
        help="no-op 判定阈值（官方风格：前 6 维接近 0 且 gripper 未变化）",
    )
    parser.add_argument(
        "--camera-resolution",
        type=int,
        default=256,
        help="离线渲染图像分辨率（官方 RLDS 为 256）",
    )
    parser.add_argument(
        "--replay-translation-scale",
        type=float,
        default=1.0,
        help="离线重播前对平移动作额外缩放",
    )
    parser.add_argument(
        "--replay-rotation-scale",
        type=float,
        default=1.0,
        help="离线重播前对旋转动作额外缩放",
    )
    parser.add_argument(
        "--max-translation-norm",
        type=float,
        default=0.0,
        help="离线重播时平移动作最大范数；<= 0 表示不裁剪",
    )
    parser.add_argument(
        "--max-rotation-norm",
        type=float,
        default=0.0,
        help="离线重播时旋转动作最大范数；<= 0 表示不裁剪",
    )
    parser.add_argument(
        "--split-large-actions",
        action="store_true",
        help="将超过阈值的大动作拆成多个更小的重播子步",
    )
    parser.add_argument(
        "--substep-translation-norm",
        type=float,
        default=0.3,
        help="启用大动作拆分时，每个子步允许的平移范数上限",
    )
    parser.add_argument(
        "--substep-rotation-norm",
        type=float,
        default=0.2,
        help="启用大动作拆分时，每个子步允许的旋转范数上限",
    )
    parser.add_argument(
        "--max-action-substeps",
        type=int,
        default=16,
        help="单个动作最多拆分成多少个子步",
    )

    args = parser.parse_args()

    hdf5_path = args.demo_file
    f = h5py.File(hdf5_path, "r")
    env_name = f["data"].attrs["env"]

    env_args = f["data"].attrs["env_info"]
    env_kwargs = json.loads(f["data"].attrs["env_info"])

    problem_info = json.loads(f["data"].attrs["problem_info"])
    problem_info["domain_name"]
    problem_name = problem_info["problem_name"]
    language_instruction = problem_info["language_instruction"]

    # list of all demonstrations episodes
    demos = list(f["data"].keys())

    bddl_file_name = f["data"].attrs["bddl_file_name"]

    hdf5_path = resolve_output_dataset_path(args.dataset_path, bddl_file_name)

    output_parent_dir = Path(hdf5_path).parent
    output_parent_dir.mkdir(parents=True, exist_ok=True)

    h5py_f = h5py.File(hdf5_path, "w")

    grp = h5py_f.create_group("data")

    grp.attrs["env_name"] = env_name
    grp.attrs["problem_info"] = f["data"].attrs["problem_info"]
    grp.attrs["macros_image_convention"] = macros.IMAGE_CONVENTION

    libero_utils.update_env_kwargs(
        env_kwargs,
        bddl_file_name=bddl_file_name,
        has_renderer=not args.use_camera_obs,
        has_offscreen_renderer=args.use_camera_obs,
        ignore_done=True,
        use_camera_obs=args.use_camera_obs,
        camera_depths=args.use_depth,
        camera_names=[
            "robot0_eye_in_hand",
            "agentview",
        ],
        reward_shaping=True,
        control_freq=20,
        camera_heights=args.camera_resolution,
        camera_widths=args.camera_resolution,
        camera_segmentations=None,
    )

    grp.attrs["bddl_file_name"] = bddl_file_name
    grp.attrs["bddl_file_content"] = open(bddl_file_name, "r").read()
    print(grp.attrs["bddl_file_content"])

    env = TASK_MAPPING[problem_name](
        **env_kwargs,
    )
    env = SuctionStickyWrapper(env)

    env_args = {
        "type": 1,
        "env_name": env_name,
        "problem_name": problem_name,
        "bddl_file": f["data"].attrs["bddl_file_name"],
        "env_kwargs": env_kwargs,
    }

    grp.attrs["env_args"] = json.dumps(env_args)
    print(grp.attrs["env_args"])
    total_len = 0
    demos = demos

    cap_index = 5

    for (i, ep) in enumerate(demos):
        print("Playing back random episode... (press ESC to quit)")

        # # select an episode randomly
        # read the model xml, using the metadata stored in the attribute for this episode
        model_xml = f["data/{}".format(ep)].attrs["model_file"]
        reset_success = False
        while not reset_success:
            try:
                env.reset()
                set_grip_cylinder_visibility(env, alpha=0.0)
                reset_success = True
            except:
                continue

        model_xml = libero_utils.postprocess_model_xml(model_xml, {})

        if not args.use_camera_obs:
            env.viewer.set_camera(0)

        # load the flattened mujoco states
        states = f["data/{}/states".format(ep)][()]
        actions = np.array(f["data/{}/actions".format(ep)][()])

        num_actions = actions.shape[0]

        init_idx = 0
        env.reset_from_xml_string(model_xml)
        env.sim.reset()
        env.sim.set_state_from_flattened(states[init_idx])
        env.sim.forward()
        model_xml = env.sim.model.get_xml()

        ee_states = []
        gripper_states = []
        joint_states = []
        robot_states = []

        agentview_images = []
        eye_in_hand_images = []

        agentview_depths = []
        eye_in_hand_depths = []

        agentview_seg = {0: [], 1: [], 2: [], 3: [], 4: []}

        rewards = []
        dones = []

        noop_skipped = 0
        split_steps_added = 0
        replay_states = []
        replay_actions = []
        prev_kept_action = None
        can_check_alignment = (
            args.replay_translation_scale == 1.0
            and args.replay_rotation_scale == 1.0
            and args.max_translation_norm <= 0
            and args.max_rotation_norm <= 0
            and not args.split_large_actions
        )

        for j, action in enumerate(actions):
            # Skip recording because the force sensor is not stable in
            # the beginning
            if j < cap_index:
                continue

            transformed_action = transform_action(action, args)

            if args.filter_noop and is_noop_action(transformed_action, prev_kept_action, args.noop_threshold):
                noop_skipped += 1
                continue

            sub_actions = split_action(transformed_action, args)
            split_steps_added += max(0, len(sub_actions) - 1)

            for sub_action in sub_actions:
                replay_states.append(env.sim.get_state().flatten())
                obs, reward, done, info = env.step(sub_action)

                if can_check_alignment and j < num_actions - 1 and sub_action is sub_actions[-1]:
                    state_playback = env.sim.get_state().flatten()
                    err = np.linalg.norm(states[j + 1] - state_playback)
                    if err > 0.01:
                        print(
                            f"[warning] playback diverged by {err:.2f} for ep {ep} at step {j}"
                        )

                replay_actions.append(sub_action.copy())

                if not args.no_proprio:
                    if "robot0_gripper_qpos" in obs:
                        gripper_states.append(obs["robot0_gripper_qpos"])

                    joint_states.append(obs["robot0_joint_pos"])

                    ee_states.append(
                        np.hstack(
                            (
                                obs["robot0_eef_pos"],
                                T.quat2axisangle(obs["robot0_eef_quat"]),
                            )
                        )
                    )

                robot_states.append(env.get_robot_state_vector(obs))

                if args.use_camera_obs:
                    if args.use_depth:
                        agentview_depths.append(obs["agentview_depth"])
                        eye_in_hand_depths.append(obs["robot0_eye_in_hand_depth"])

                    agentview_images.append(obs["agentview_image"])
                    eye_in_hand_images.append(obs["robot0_eye_in_hand_image"])
                else:
                    env.render()

            prev_kept_action = transformed_action.copy()

        # end of one trajectory
        states = np.asarray(replay_states)
        actions = np.asarray(replay_actions)
        if len(actions) == 0:
            print(f"[warning] demo_{i} 在过滤后无有效样本，已跳过")
            continue
        dones = np.zeros(len(actions)).astype(np.uint8)
        dones[-1] = 1
        rewards = np.zeros(len(actions)).astype(np.uint8)
        rewards[-1] = 1
        if args.use_camera_obs:
            print(len(actions), len(agentview_images))
            assert len(actions) == len(agentview_images)
        print(
            f"[info] demo_{i}: kept={len(actions)}, noop_skipped={noop_skipped}, cap_skipped={cap_index}, split_steps_added={split_steps_added}"
        )

        ep_data_grp = grp.create_group(f"demo_{i}")

        obs_grp = ep_data_grp.create_group("obs")
        if not args.no_proprio:
            obs_grp.create_dataset(
                "gripper_states", data=np.stack(gripper_states, axis=0)
            )
            obs_grp.create_dataset("joint_states", data=np.stack(joint_states, axis=0))
            obs_grp.create_dataset("ee_states", data=np.stack(ee_states, axis=0))
            obs_grp.create_dataset("ee_pos", data=np.stack(ee_states, axis=0)[:, :3])
            obs_grp.create_dataset("ee_ori", data=np.stack(ee_states, axis=0)[:, 3:])

        if args.use_camera_obs:
            obs_grp.create_dataset(
                "agentview_rgb", data=np.stack(agentview_images, axis=0)
            )
            obs_grp.create_dataset(
                "eye_in_hand_rgb", data=np.stack(eye_in_hand_images, axis=0)
            )
            if args.use_depth:
                obs_grp.create_dataset(
                    "agentview_depth", data=np.stack(agentview_depths, axis=0)
                )
                obs_grp.create_dataset(
                    "eye_in_hand_depth", data=np.stack(eye_in_hand_depths, axis=0)
                )

        ep_data_grp.create_dataset("actions", data=actions)
        ep_data_grp.create_dataset("states", data=states)
        ep_data_grp.create_dataset("robot_states", data=np.stack(robot_states, axis=0))
        ep_data_grp.create_dataset("rewards", data=rewards)
        ep_data_grp.create_dataset("dones", data=dones)
        num_samples = len(agentview_images) if args.use_camera_obs else len(actions)
        ep_data_grp.attrs["num_samples"] = num_samples
        ep_data_grp.attrs["model_file"] = model_xml
        ep_data_grp.attrs["init_state"] = states[init_idx]
        total_len += num_samples

    grp.attrs["num_demos"] = len(demos)
    grp.attrs["total"] = total_len
    env.close()

    h5py_f.close()
    f.close()

    print("The created dataset is saved in the following path: ")
    print(hdf5_path)


if __name__ == "__main__":
    main()
