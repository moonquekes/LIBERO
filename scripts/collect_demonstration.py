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
                cv2.imshow(PREVIEW_WINDOW_NAME, bgr)
                if not preview_window_moved:
                    cv2.moveWindow(PREVIEW_WINDOW_NAME, int(preview_window_x), int(preview_window_y))
                    preview_window_moved = True
                cv2.waitKey(1)
            except Exception:
                pass
        # Also break if we complete the task
        if task_completion_hold_count == 0:
            break

        # state machine to check for having a success for 10 consecutive timesteps
        if env._check_success():
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
    env = SuctionStickyWrapper(env)

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
