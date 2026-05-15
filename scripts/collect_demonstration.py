import argparse
import cv2
import datetime
import h5py
import init_path
import json
import numpy as np
import os
import re
import robosuite as suite
import shutil
import time
from glob import glob
from robosuite import load_controller_config
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
from robosuite.utils.input_utils import input2action


import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import *
from libero.libero.envs.suction_sticky_wrapper import SuctionStickyWrapper


PREVIEW_WINDOW_NAME = "Wrist Camera (robot0_eye_in_hand)"
WINDOW_SIZE = 512  # 腕部摄像头预览窗口及主窗口统一尺寸


def clip_vector_norm(vector, max_norm):
    if max_norm <= 0:
        return vector
    norm = np.linalg.norm(vector)
    if norm <= max_norm or norm == 0:
        return vector
    return vector * (max_norm / norm)


def get_data_collection_wrapper(env):
    current = env
    for _ in range(10):
        if isinstance(current, DataCollectionWrapper):
            return current
        if not hasattr(current, "env"):
            break
        current = current.env
    return None


def finalize_data_collection_episode(env):
    wrapper = get_data_collection_wrapper(env)
    if wrapper is None:
        return False
    if not getattr(wrapper, "has_interaction", False):
        return True

    wrapper._flush()
    wrapper.has_interaction = False
    wrapper.states = []
    wrapper.action_infos = []
    wrapper.successful = False
    return True


def postprocess_input_action(
    action,
    invert_controls="none",
    translation_scale=1.0,
    rotation_scale=1.0,
    translation_deadzone=0.0,
    rotation_deadzone=0.0,
    max_translation_norm=0.0,
    max_rotation_norm=0.0,
):
    action = np.asarray(action, dtype=np.float64).copy()

    if invert_controls in ["x", "xy"]:
        action[0] = -action[0]
    if invert_controls in ["y", "xy"]:
        action[1] = -action[1]

    action[:3] *= translation_scale
    action[3:6] *= rotation_scale

    if translation_deadzone > 0:
        action[:3][np.abs(action[:3]) < translation_deadzone] = 0.0
    if rotation_deadzone > 0:
        action[3:6][np.abs(action[3:6]) < rotation_deadzone] = 0.0

    action[:3] = clip_vector_norm(action[:3], max_translation_norm)
    action[3:6] = clip_vector_norm(action[3:6], max_rotation_norm)
    return action


def register_keyboard_callbacks(viewer, device):
    def _register(method_name, callback):
        if callback is None or not hasattr(viewer, method_name):
            return
        method = getattr(viewer, method_name)
        try:
            method("any", callback)
        except TypeError:
            method(callback)

    _register("add_keypress_callback", device.on_press)
    _register("add_keyup_callback", getattr(device, "on_release", None))
    _register("add_keyrepeat_callback", device.on_press)


def try_resize_main_window(env, size):
    try:
        import glfw

        viewer = getattr(env, "viewer", None)
        if viewer is None and hasattr(env, "env"):
            viewer = getattr(env.env, "viewer", None)
        if viewer is None:
            return False

        candidates = [
            getattr(viewer, "window", None),
            getattr(getattr(viewer, "viewer", None), "window", None),
        ]
        for window in candidates:
            if window is not None:
                glfw.set_window_size(window, int(size), int(size))
                return True
    except Exception:
        return False
    return False


def set_suction_indicator_visibility(env, alpha=0.7):
    """开启/关闭 suction_indicator site 的透明度（仅用于人工采集时的可视化）。"""
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
            return False
        for site_id in range(model.nsite):
            name = model.site_id2name(site_id)
            if name and name.endswith("suction_indicator"):
                model.site_rgba[site_id, 3] = float(alpha)
        return True
    except Exception:
        return False


def set_grip_cylinder_visibility(env, alpha=0.3):
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
            return False
        ids = []
        for site_id in range(model.nsite):
            name = model.site_id2name(site_id)
            if name and name.endswith("grip_site_cylinder"):
                ids.append(site_id)
        for site_id in ids:
            model.site_rgba[site_id, 0] = 0.0
            model.site_rgba[site_id, 1] = 1.0
            model.site_rgba[site_id, 2] = 0.0
            model.site_rgba[site_id, 3] = float(alpha)
        return len(ids) > 0
    except Exception:
        return False


def set_workpiece_center_marker_visibility(env, alpha=0.95):
    """显示工件中心点，仅用于人工采集时辅助对准吸盘。"""
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
            return False

        ids = []
        center_site_re = re.compile(
            r"^steel_plate(?:_round|_triangle|_large)?_1_center_marker$"
        )
        for site_id in range(model.nsite):
            name = model.site_id2name(site_id)
            if name and center_site_re.match(name):
                ids.append(site_id)

        for site_id in ids:
            model.site_size[site_id, 0] = 0.008
            model.site_size[site_id, 1] = 0.008
            model.site_size[site_id, 2] = 0.008
            model.site_rgba[site_id, 0] = 1.0
            model.site_rgba[site_id, 1] = 0.0
            model.site_rgba[site_id, 2] = 1.0
            model.site_rgba[site_id, 3] = float(alpha)
        return len(ids) > 0
    except Exception:
        return False


def get_suction_diagnostics(env):
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


def get_suction_display_context(env):
    suction_on = bool(env.is_suction_on()) if hasattr(env, "is_suction_on") else False
    nearest_distance = np.inf
    nearest_body_id = -1
    if hasattr(env, "get_nearest_attachable_distance"):
        nearest_distance, nearest_body_id = env.get_nearest_attachable_distance()

    probe = {
        "available": False,
        "constraint_ok": False,
        "reason_code": 2,
        "reason_label": "no_contact",
        "contact_angle_deg": np.nan,
        "contact_radial_offset_m": np.nan,
        "contact_body_id": -1,
    }
    if hasattr(env, "get_current_constraint_probe"):
        probe.update(env.get_current_constraint_probe())

    return {
        "suction_on": suction_on,
        "nearest_attachable_distance_m": float(nearest_distance),
        "nearest_attachable_body_id": int(nearest_body_id),
        "probe": probe,
    }


def format_suction_status(diagnostics):
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


def get_model_from_env(env):
    current = env
    for _ in range(10):
        if hasattr(current, "sim") and getattr(current, "sim") is not None:
            return current.sim.model
        if not hasattr(current, "env"):
            break
        current = current.env
    return None


def get_suction_body_name(env, body_id):
    if body_id is None or int(body_id) < 0:
        return "none"
    model = get_model_from_env(env)
    if model is None or int(body_id) >= model.nbody:
        return str(body_id)
    name = model.body_id2name(int(body_id))
    return name or str(body_id)


def get_suction_debug_lines(env, proximity_threshold_m):
    diagnostics = get_suction_diagnostics(env)
    display_context = get_suction_display_context(env)
    suction_on = bool(display_context["suction_on"])
    nearest_distance = float(display_context["nearest_attachable_distance_m"])
    nearest_body_id = int(display_context["nearest_attachable_body_id"])
    nearest_body_name = get_suction_body_name(env, nearest_body_id)
    near_candidate = bool(
        np.isfinite(nearest_distance) and nearest_distance <= float(proximity_threshold_m)
    )
    display_context["near_candidate"] = near_candidate

    if not diagnostics.get("attached") and not near_candidate:
        distance_text = "na" if not np.isfinite(nearest_distance) else f"{nearest_distance:.4f}"
        suction_label = "on" if suction_on else "off"
        line1 = f"suction={suction_label} attached=False near=False"
        line2 = f"dist_m={distance_text} body={nearest_body_name}"
        signature = (suction_label, "far", distance_text, nearest_body_name)
        return diagnostics, line1, line2, signature, display_context

    if diagnostics.get("attached"):
        status = format_suction_status(diagnostics)
        source = diagnostics
    else:
        probe = display_context["probe"]
        status = format_suction_status(probe)
        source = probe

    angle_deg = source.get("contact_angle_deg", np.nan)
    radial_offset = source.get("contact_radial_offset_m", np.nan)
    body_id = int(source.get("contact_body_id", -1))
    body_name = get_suction_body_name(env, body_id)

    angle_text = "na" if not np.isfinite(angle_deg) else f"{float(angle_deg):.1f}"
    radial_text = "na" if not np.isfinite(radial_offset) else f"{float(radial_offset):.4f}"
    suction_label = "on" if suction_on else "off"
    line1 = (
        f"suction={suction_label} status={status} attached={bool(diagnostics['attached'])} "
        f"ok={bool(source.get('constraint_ok', False))}"
    )
    line2 = f"angle_deg={angle_text} radial_m={radial_text} body={body_name}"
    signature = (
        suction_label,
        status,
        bool(diagnostics["attached"]),
        bool(source.get("constraint_ok", False)),
        int(source.get("reason_code", 0)),
        angle_text,
        radial_text,
        body_name,
    )
    return diagnostics, line1, line2, signature, display_context


def get_suction_debug_color(diagnostics, display_context):
    if diagnostics.get("attached") and diagnostics.get("constraint_ok"):
        return (80, 255, 80)
    if not display_context.get("suction_on"):
        return (80, 80, 255)
    if not display_context.get("near_candidate"):
        return (255, 200, 80)
    return (0, 215, 255)


def get_success_debug_lines(raw_success, success_ready):
    if success_ready:
        line1 = "raw_success=True save_ready=True"
        line2 = "goal=in_correct_bin release=complete"
        signature = ("ready",)
    elif raw_success:
        line1 = "raw_success=True save_ready=False"
        line2 = "goal=in_correct_bin release=waiting"
        signature = ("waiting_release",)
    else:
        line1 = "raw_success=False save_ready=False"
        line2 = "goal=not_reached release=not_ready"
        signature = ("not_ready",)
    return line1, line2, signature


def get_success_debug_color(raw_success, success_ready):
    if success_ready:
        return (80, 255, 80)
    if raw_success:
        return (0, 215, 255)
    return (180, 180, 180)


def collect_human_trajectory(
    env,
    device,
    arm,
    env_configuration,
    problem_info,
    remove_directory=[],
    record_cameras=None,
    preview_window_x=120,
    preview_window_y=80,
    preview_flip="none",
    show_grip_cylinder=True,
    invert_controls="none",
    translation_scale=1.0,
    rotation_scale=1.0,
    translation_deadzone=0.0,
    rotation_deadzone=0.0,
    max_translation_norm=0.0,
    max_rotation_norm=0.0,
    suction_display_proximity_threshold_m=0.05,
    require_suction_release_for_success=True,
    show_workpiece_center_marker=True,
):
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment to control
        device (Device): to receive controls from the device
        arms (str): which arm to control (eg bimanual) 'right' or 'left'
        env_configuration (str): specified environment configuration
    """

    reset_success = False
    while not reset_success:
        try:
            env.reset()
            if show_grip_cylinder:
                set_grip_cylinder_visibility(env, alpha=0.3)
            set_suction_indicator_visibility(env, alpha=0.7)
            if show_workpiece_center_marker:
                set_workpiece_center_marker_visibility(env, alpha=0.95)
            reset_success = True
        except:
            continue

    # ID = 2 always corresponds to agentview
    env.render()
    try_resize_main_window(env, WINDOW_SIZE)

    task_completion_hold_count = (
        -1
    )  # counter to collect 10 timesteps after reaching goal
    device.start_control()

    # Loop until we get a reset from the input or the task completes
    saving = True
    count = 0

    if record_cameras is None:
        record_cameras = []

    camera_frames = {cam: [] for cam in record_cameras}
    preview_window_moved = False
    last_suction_signature = None
    last_success_signature = None
    waiting_for_release_logged = False

    while True:
        count += 1
        # Set active robot
        active_robot = (
            env.robots[0]
            if env_configuration == "bimanual"
            else env.robots[arm == "left"]
        )

        # Get the newest action
        action, grasp = input2action(
            device=device,
            robot=active_robot,
            active_arm=arm,
            env_configuration=env_configuration,
        )

        # If action is none, then this a reset so we should break
        if action is None:
            print("[info] 手动重置当前回合（该回合不保存）")
            saving = False
            break

        action = postprocess_input_action(
            action,
            invert_controls=invert_controls,
            translation_scale=translation_scale,
            rotation_scale=rotation_scale,
            translation_deadzone=translation_deadzone,
            rotation_deadzone=rotation_deadzone,
            max_translation_norm=max_translation_norm,
            max_rotation_norm=max_rotation_norm,
        )

        # Run environment step

        obs, reward, done, _ = env.step(action)
        suction_diagnostics, suction_line1, suction_line2, suction_signature, suction_display_context = (
            get_suction_debug_lines(env, suction_display_proximity_threshold_m)
        )
        raw_success = env._check_success()
        success_ready = raw_success
        if require_suction_release_for_success:
            success_ready = (
                raw_success
                and not bool(suction_display_context.get("suction_on"))
                and not bool(suction_diagnostics.get("attached"))
            )
        success_line1, success_line2, success_signature = get_success_debug_lines(
            raw_success, success_ready
        )
        if suction_signature != last_suction_signature:
            print(f"[suction] {suction_line1}; {suction_line2}")
            last_suction_signature = suction_signature
        if success_signature != last_success_signature:
            print(f"[success] {success_line1}; {success_line2}")
            last_success_signature = success_signature
        env.render()

        for cam in record_cameras:
            frame = obs.get(f"{cam}_image")
            if frame is None:
                continue
            camera_frames[cam].append(frame.copy())

        # 仅显示腕部摄像头预览（支持翻转，不影响保存图像）
        wrist_frame = obs.get("robot0_eye_in_hand_image")
        if wrist_frame is not None:
            try:
                display_frame = wrist_frame[::-1]
                if preview_flip in ["x", "xy"]:
                    display_frame = display_frame[:, ::-1]
                if preview_flip in ["y", "xy"]:
                    display_frame = display_frame[::-1]
                bgr = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
                bgr = cv2.resize(bgr, (WINDOW_SIZE, WINDOW_SIZE))
                debug_color = get_suction_debug_color(suction_diagnostics, suction_display_context)
                cv2.putText(
                    bgr,
                    suction_line1,
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    debug_color,
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    bgr,
                    suction_line2,
                    (10, 46),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    debug_color,
                    1,
                    cv2.LINE_AA,
                )
                success_color = get_success_debug_color(raw_success, success_ready)
                cv2.putText(
                    bgr,
                    success_line1,
                    (10, 68),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    success_color,
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    bgr,
                    success_line2,
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    success_color,
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow(PREVIEW_WINDOW_NAME, bgr)
                if not preview_window_moved:
                    cv2.moveWindow(PREVIEW_WINDOW_NAME, int(preview_window_x), int(preview_window_y))
                    preview_window_moved = True
                cv2.waitKey(1)
            except Exception:
                pass

        if raw_success and not success_ready:
            if not waiting_for_release_logged:
                print("[info] 目标已到位，等待关闭吸盘并完全释放后再结束采集")
                waiting_for_release_logged = True
        else:
            waiting_for_release_logged = False

        # Also break if we complete the task
        if task_completion_hold_count == 0:
            break

        # state machine to check for having a success for 10 consecutive timesteps
        if success_ready:
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1  # latched state, decrement count
            else:
                task_completion_hold_count = 10  # reset count on first success timestep
        else:
            task_completion_hold_count = -1  # null the counter if there's no success

    print(count)
    if saving and record_cameras:
        try:
            payload = {}
            for cam in record_cameras:
                frames = camera_frames.get(cam, [])
                if len(frames) > 0:
                    payload[cam] = np.array(frames, dtype=np.uint8)
            if payload:
                np.savez_compressed(
                    os.path.join(env.ep_directory, "camera_obs.npz"),
                    **payload,
                )
        except Exception as e:
            print(f"[warn] 相机观测保存失败：{e}")

    if saving and not finalize_data_collection_episode(env):
        raise RuntimeError("无法定位 DataCollectionWrapper，当前回合无法安全写入 HDF5")

    # cleanup for end of data collection episodes
    if not saving:
        remove_directory.append(env.ep_directory.split("/")[-1])
    try:
        cv2.destroyWindow(PREVIEW_WINDOW_NAME)
    except Exception:
        pass
    return saving


def sanitize_for_path(text):
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")
    return sanitized or "demo"


def ensure_unique_path(path):
    if not os.path.exists(path):
        return path

    root, ext = os.path.splitext(path)
    suffix = 2
    while True:
        candidate = f"{root}_{suffix:03d}{ext}"
        if not os.path.exists(candidate):
            return candidate
        suffix += 1


def build_episode_hdf5_path(output_dir, problem_info, episode_dir):
    domain_name = sanitize_for_path(problem_info.get("domain_name", "domain"))
    problem_name = sanitize_for_path(problem_info.get("problem_name", "task"))
    instruction = sanitize_for_path(problem_info.get("language_instruction", "demo"))
    source_episode = sanitize_for_path(os.path.basename(episode_dir))
    file_name = f"{domain_name}_ln_{problem_name}_{instruction}_{source_episode}.hdf5"
    return ensure_unique_path(os.path.join(output_dir, file_name))


def write_episode_to_hdf5(episode_dir, out_file, env_info, problem_info, args):
    state_paths = sorted(glob(os.path.join(episode_dir, "state_*.npz")))
    if not state_paths:
        return False

    states = []
    actions = []
    env_name = None

    for state_file in state_paths:
        dic = np.load(state_file, allow_pickle=True)
        env_name = str(dic["env"])
        states.extend(dic["states"])
        for ai in dic["action_infos"]:
            actions.append(ai["actions"])

    if len(states) <= 1 or len(actions) == 0:
        return False

    del states[-1]
    if len(states) != len(actions):
        raise ValueError(
            f"Episode {os.path.basename(episode_dir)} 中 states({len(states)}) 与 actions({len(actions)}) 数量不一致"
        )

    with open(args.bddl_file, "r", encoding="utf-8") as bddl_file:
        bddl_file_content = bddl_file.read()

    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    with h5py.File(out_file, "w") as hdf5_file:
        traj_grp = hdf5_file.create_group("trajectory")
        traj_grp.attrs["source_episode"] = os.path.basename(episode_dir)

        xml_path = os.path.join(episode_dir, "model.xml")
        with open(xml_path, "r", encoding="utf-8") as model_file:
            traj_grp.attrs["model_file"] = model_file.read()

        traj_grp.create_dataset("states", data=np.array(states))
        traj_grp.create_dataset("actions", data=np.array(actions))
        traj_grp.attrs["num_samples"] = len(actions)
        traj_grp.attrs["init_state"] = np.array(states[0])

        cam_obs_path = os.path.join(episode_dir, "camera_obs.npz")
        if os.path.exists(cam_obs_path):
            try:
                obs_data = np.load(cam_obs_path, allow_pickle=True)
                obs_grp = traj_grp.create_group("observations")
                action_len = len(actions)
                for cam_name in obs_data.files:
                    frames = obs_data[cam_name]
                    if frames.ndim != 4:
                        continue
                    if len(frames) != action_len:
                        n = min(len(frames), action_len)
                        frames = frames[:n]
                        print(
                            f"[warn] {os.path.basename(episode_dir)}:{cam_name} 帧数({len(obs_data[cam_name])})"
                            f" 与动作数({action_len})不一致，已截断到 {n}"
                        )
                    obs_grp.create_dataset(
                        f"{cam_name}_image", data=frames, compression="gzip"
                    )
            except Exception as e:
                print(f"[warn] 相机观测写入 HDF5 失败（{os.path.basename(episode_dir)}）：{e}")

        now = datetime.datetime.now()
        hdf5_file.attrs["file_structure"] = "single_trajectory"
        hdf5_file.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
        hdf5_file.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
        hdf5_file.attrs["repository_version"] = suite.__version__
        if env_name is not None:
            hdf5_file.attrs["env"] = env_name
        hdf5_file.attrs["env_info"] = env_info
        hdf5_file.attrs["problem_info"] = json.dumps(problem_info)
        hdf5_file.attrs["bddl_file_name"] = args.bddl_file
        hdf5_file.attrs["bddl_file_content"] = bddl_file_content

    return True


def gather_demonstrations_as_hdf5(
    directory,
    output_dir,
    env_info,
    problem_info,
    args,
    remove_directory=None,
    cleanup_processed=False,
):
    if remove_directory is None:
        remove_directory = []

    if not os.path.isdir(directory):
        print(f"[warn] 临时目录不存在，跳过 HDF5 汇总: {directory}")
        return []

    exported_files = []
    for ep_directory in sorted(os.listdir(directory)):
        if ep_directory in remove_directory:
            continue

        episode_dir = os.path.join(directory, ep_directory)
        if not os.path.isdir(episode_dir):
            continue

        out_file = build_episode_hdf5_path(output_dir, problem_info, episode_dir)
        wrote_file = write_episode_to_hdf5(
            episode_dir,
            out_file,
            env_info,
            problem_info,
            args,
        )
        if not wrote_file:
            continue

        exported_files.append(out_file)

        if cleanup_processed:
            shutil.rmtree(episode_dir, ignore_errors=True)

    return exported_files


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default="demonstration_data",
    )
    parser.add_argument(
        "--tmp-dir-root",
        type=str,
        default="",
        help="采集时 DataCollectionWrapper 的临时块目录根路径；为空时默认放到输出目录下的 _tmp_chunks",
    )
    parser.add_argument(
        "--robots",
        nargs="+",
        type=str,
        default="Panda",
        help="Which robot(s) to use in the env",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="single-arm-opposed",
        help="Specified environment configuration if necessary",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="right",
        help="Which arm to control (eg bimanual) 'right' or 'left'",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="agentview",
        help="Which camera to use for collecting demos",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="OSC_POSE",
        help="Choice of controller. Can be 'IK_POSE' or 'OSC_POSE'",
    )
    parser.add_argument("--device", type=str, default="spacemouse")
    parser.add_argument(
        "--pos-sensitivity",
        type=float,
        default=1.5,
        help="How much to scale position user inputs",
    )
    parser.add_argument(
        "--rot-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale rotation user inputs",
    )
    parser.add_argument(
        "--action-translation-scale",
        type=float,
        default=1.0,
        help="额外缩放键盘平移动作，建议在粗粒度键盘控制时设为 < 1",
    )
    parser.add_argument(
        "--action-rotation-scale",
        type=float,
        default=1.0,
        help="额外缩放键盘旋转动作，建议在粗粒度键盘控制时设为 < 1",
    )
    parser.add_argument(
        "--translation-deadzone",
        type=float,
        default=0.0,
        help="平移动作死区，小于该阈值的平移量置零",
    )
    parser.add_argument(
        "--rotation-deadzone",
        type=float,
        default=0.0,
        help="旋转动作死区，小于该阈值的旋转量置零",
    )
    parser.add_argument(
        "--max-translation-norm",
        type=float,
        default=0.0,
        help="平移动作最大范数；<= 0 表示不裁剪",
    )
    parser.add_argument(
        "--max-rotation-norm",
        type=float,
        default=0.0,
        help="旋转动作最大范数；<= 0 表示不裁剪",
    )
    parser.add_argument(
        "--num-demonstration",
        type=int,
        default=50,
        help="How much to scale rotation user inputs",
    )
    parser.add_argument("--bddl-file", type=str)
    parser.add_argument(
        "--record-cameras",
        type=str,
        default="",
        help="落盘保存的相机视角（逗号分隔）。为空则不保存图像（默认离线渲染流程）",
    )
    parser.add_argument(
        "--preview-window-x",
        type=int,
        default=120,
        help="腕部摄像头预览窗口左上角 X 坐标",
    )
    parser.add_argument(
        "--preview-window-y",
        type=int,
        default=80,
        help="腕部摄像头预览窗口左上角 Y 坐标",
    )
    parser.add_argument(
        "--preview-flip",
        type=str,
        default="none",
        choices=["none", "x", "y", "xy"],
        help="仅预览窗口图像翻转（不影响保存图像）：none/x/y/xy",
    )
    parser.add_argument(
        "--show-grip-cylinder",
        dest="show_grip_cylinder",
        action="store_true",
        default=True,
        help="录制时显示吸盘绿色柱（默认开启，仅显示辅助，不影响保存数据）",
    )
    parser.add_argument(
        "--hide-grip-cylinder",
        dest="show_grip_cylinder",
        action="store_false",
        help="录制时隐藏吸盘绿色柱",
    )
    parser.add_argument(
        "--invert-controls",
        type=str,
        default="none",
        choices=["none", "x", "y", "xy"],
        help="按键平移方向反转：none/x/y/xy（用于修正视角方向不一致）",
    )
    parser.add_argument(
        "--suction-normal-max-angle-deg",
        type=float,
        default=25.0,
        help="吸盘法向量最大夹角阈值（度）",
    )
    parser.add_argument(
        "--suction-display-proximity-threshold-m",
        type=float,
        default=0.05,
        help="状态窗口开始显示约束判断前，吸盘到最近可吸附物体表面的距离阈值（米）",
    )
    parser.add_argument(
        "--suction-effective-radius-ratio",
        type=float,
        default=0.7,
        help="吸盘有效支撑半径系数，相对 pad 半径",
    )
    parser.add_argument(
        "--suction-detach-grace-steps",
        type=int,
        default=2,
        help="吸盘脱附缓冲步数",
    )
    parser.add_argument(
        "--require-suction-release-for-success",
        dest="require_suction_release_for_success",
        action="store_true",
        default=True,
        help="只有在任务成功且吸盘已关闭并完全脱离工件后，才结束当前采集回合（默认开启）",
    )
    parser.add_argument(
        "--allow-success-while-attached",
        dest="require_suction_release_for_success",
        action="store_false",
        help="允许工件仍被吸住时按任务成功直接结束采集",
    )
    parser.add_argument(
        "--show-workpiece-center-marker",
        dest="show_workpiece_center_marker",
        action="store_true",
        default=True,
        help="采集窗口中显示工件中心点标记（默认开启，仅影响人工采集可视化）",
    )
    parser.add_argument(
        "--hide-workpiece-center-marker",
        dest="show_workpiece_center_marker",
        action="store_false",
        help="隐藏工件中心点辅助标记",
    )

    parser.add_argument("--vendor-id", type=int, default=9583)
    parser.add_argument("--product-id", type=int, default=50734)

    args = parser.parse_args()

    # Get controller config
    controller_config = load_controller_config(default_controller=args.controller)

    # Create argument configuration
    config = {
        "robots": args.robots,
        "controller_configs": controller_config,
    }

    assert os.path.exists(args.bddl_file)
    problem_info = BDDLUtils.get_problem_info(args.bddl_file)
    # Check if we're using a multi-armed environment and use env_configuration argument if so

    # Create environment
    record_cameras = [c.strip() for c in args.record_cameras.split(",") if c.strip()]

    # 始终添加腕部摄像头用于实时预览
    WRIST_CAM = "robot0_eye_in_hand"
    camera_names = [WRIST_CAM]
    for cam in record_cameras:
        if cam not in camera_names:
            camera_names.append(cam)
    enable_camera_obs = True

    problem_name = problem_info["problem_name"]
    domain_name = problem_info["domain_name"]
    language_instruction = problem_info["language_instruction"]
    if "TwoArm" in problem_name:
        config["env_configuration"] = args.config
    print(language_instruction)
    env_kwargs = dict(
        bddl_file_name=args.bddl_file,
        **config,
        has_renderer=True,
        has_offscreen_renderer=enable_camera_obs,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=enable_camera_obs,
        reward_shaping=True,
        control_freq=20,
    )
    if enable_camera_obs:
        env_kwargs.update(
            {
                "camera_names": camera_names,
                "camera_heights": 256,
                "camera_widths": 256,
            }
        )

    env = TASK_MAPPING[problem_name](**env_kwargs)

    # Wrap this with visualization wrapper
    env = VisualizationWrapper(env)

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)

    # wrap the environment with data collection wrapper
    tmp_root = args.tmp_dir_root.strip() if args.tmp_dir_root else ""
    if tmp_root:
        if not os.path.isabs(tmp_root):
            tmp_root = os.path.abspath(tmp_root)
    else:
        tmp_root = os.path.join(os.path.abspath(args.directory), "_tmp_chunks")

    tmp_directory = os.path.join(
        tmp_root,
        "{}_ln_{}".format(
            problem_name,
            language_instruction.replace(" ", "_").strip('""'),
        ),
        str(time.time()).replace(".", "_"),
    )
    os.makedirs(tmp_directory, exist_ok=True)
    print(f"[info] 采集临时目录: {tmp_directory}")

    env = DataCollectionWrapper(env, tmp_directory)
    env = SuctionStickyWrapper(
        env,
        normal_max_angle_deg=args.suction_normal_max_angle_deg,
        effective_radius_ratio=args.suction_effective_radius_ratio,
        detach_grace_steps=args.suction_detach_grace_steps,
    )

    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(
            pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity
        )
        register_keyboard_callbacks(env.viewer, device)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(
            args.vendor_id,
            args.product_id,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
    else:
        raise Exception(
            "Invalid device choice: choose either 'keyboard' or 'spacemouse'."
        )

    os.makedirs(args.directory, exist_ok=True)
    output_pattern = build_episode_hdf5_path(
        args.directory,
        problem_info,
        os.path.join(tmp_directory, "ep_<timestamp>"),
    )
    print(
        f"[info] 采集输出模式: 每个成功回合一个独立 HDF5，命名示例: {output_pattern}"
    )

    # collect demonstrations

    remove_directory = []
    i = 0
    while i < args.num_demonstration:
        print(i)
        saving = collect_human_trajectory(
            env,
            device,
            args.arm,
            args.config,
            problem_info,
            remove_directory,
            record_cameras=record_cameras,
            preview_window_x=args.preview_window_x,
            preview_window_y=args.preview_window_y,
            preview_flip=args.preview_flip,
            show_grip_cylinder=args.show_grip_cylinder,
            invert_controls=args.invert_controls,
            translation_scale=args.action_translation_scale,
            rotation_scale=args.action_rotation_scale,
            translation_deadzone=args.translation_deadzone,
            rotation_deadzone=args.rotation_deadzone,
            max_translation_norm=args.max_translation_norm,
            max_rotation_norm=args.max_rotation_norm,
            suction_display_proximity_threshold_m=args.suction_display_proximity_threshold_m,
            require_suction_release_for_success=args.require_suction_release_for_success,
            show_workpiece_center_marker=args.show_workpiece_center_marker,
        )
        if saving:
            print(remove_directory)
            exported_files = gather_demonstrations_as_hdf5(
                tmp_directory,
                args.directory,
                env_info,
                problem_info,
                args,
                remove_directory,
                cleanup_processed=True,
            )
            if not exported_files:
                raise RuntimeError("当前回合成功结束，但没有生成对应的 HDF5 文件")
            print(f"[info] 已保存: {exported_files[-1]}")
            i += 1
        else:
            print("[info] 本回合未保存。只有成功完成任务的轨迹会写入独立 hdf5 文件")

    exported_files = gather_demonstrations_as_hdf5(
        tmp_directory,
        args.directory,
        env_info,
        problem_info,
        args,
        remove_directory,
        cleanup_processed=True,
    )
    for exported_file in exported_files:
        print(f"[info] 收尾导出: {exported_file}")

    env.close()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
