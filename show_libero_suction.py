"""
show_libero_suction.py
──────────────────────
在 LIBERO 数据集真实任务场景里可视化吸盘夹爪。

用法示例：
  conda activate vla-adapter
  cd /home/x/vla/libero

  # 交互式窗口——默认 libero_spatial 任务 0（720×720）
  python show_libero_suction.py

  # 指定任务套件和任务编号
  python show_libero_suction.py --suite libero_spatial --task 2

  # 自定义分辨率（如 1080）
  python show_libero_suction.py --resolution 1080

  # 多视角并排显示（agentview + 手腕相机 + 俯视）
  python show_libero_suction.py --cameras agentview,robot0_eye_in_hand,birdview

  # 列出所有可选任务
  python show_libero_suction.py --list

  # 离屏渲染，保存为 PNG（适合无显示服务器的情况）
  python show_libero_suction.py --offscreen --save_png suction_preview.png

  # 离屏多视角保存（每一步多列拼图）
  python show_libero_suction.py --offscreen --cameras agentview,robot0_eye_in_hand --save_video out.mp4

操作提示（交互窗口打开后）：
  按 q 键退出 | Ctrl-C → 退出
"""

import argparse
import os
import sys
import time

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# 路径注册
# ──────────────────────────────────────────────────────────────────────────────
LIBERO_ROOT = os.path.dirname(os.path.abspath(__file__))
if LIBERO_ROOT not in sys.path:
    sys.path.insert(0, LIBERO_ROOT)

# 触发 MountedSuctionPanda 注册到 REGISTERED_ROBOTS 和 ROBOT_CLASS_MAPPING
import libero.libero.envs.robots  # noqa: F401


# ──────────────────────────────────────────────────────────────────────────────
# 辅助：列出可用任务
# ──────────────────────────────────────────────────────────────────────────────
def list_tasks(suite_name: str):
    from libero.libero import benchmark

    bm_cls = benchmark.get_benchmark_dict().get(suite_name)
    if bm_cls is None:
        available = list(benchmark.get_benchmark_dict().keys())
        print(f"[错误] 找不到套件 '{suite_name}'，可用套件：{available}")
        return
    bm = bm_cls()
    print(f"\n套件 '{suite_name}'（共 {bm.n_tasks} 个任务）：")
    for i in range(bm.n_tasks):
        task = bm.get_task(i)
        print(f"  [{i:2d}] {task.language}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="在 LIBERO 真实任务场景中渲染吸盘夹爪"
    )
    parser.add_argument(
        "--suite",
        default="libero_spatial",
        choices=["libero_spatial", "libero_object", "libero_goal",
                 "libero_10", "libero_90", "libero_100"],
        help="LIBERO 任务套件 (默认: libero_spatial)",
    )
    parser.add_argument(
        "--task", type=int, default=0, help="任务编号 (默认: 0)"
    )
    parser.add_argument(
        "--bddl_file",
        default="",
        help="直接指定自定义 BDDL 文件路径。指定后将忽略 --suite/--task",
    )
    parser.add_argument(
        "--list", action="store_true", help="列出指定套件的所有任务后退出"
    )
    parser.add_argument(
        "--offscreen", action="store_true",
        help="离屏渲染模式（不弹窗口，配合 --save_png / --save_video 使用）"
    )
    parser.add_argument(
        "--native_viewer",
        action="store_true",
        help="启用 MuJoCo 原生交互窗口（可鼠标自由旋转/缩放视角）",
    )
    parser.add_argument(
        "--save_png", default="", help="离屏模式：将第一帧保存为 PNG 文件"
    )
    parser.add_argument(
        "--save_video", default="", help="离屏模式：录制 N 步并保存为 MP4（需要 opencv）"
    )
    parser.add_argument(
        "--steps", type=int, default=300,
        help="交互/录制步数 (默认: 300)"
    )
    parser.add_argument(
        "--camera", default="agentview",
        help="渲染相机，已被 --cameras 取代，保留向后兼容 (默认: agentview)"
    )
    parser.add_argument(
        "--cameras",
        default="agentview",
        help=(
            "逗号分隔的相机名称列表，支持多视角并排显示。\n"
            "可用名称：agentview, robot0_eye_in_hand, birdview, frontview, sideview\n"
            "示例：agentview,robot0_eye_in_hand,birdview  (默认: agentview)"
        ),
    )
    parser.add_argument(
        "--resolution", type=int, default=0,
        help="渲染分辨率（宽=高）。交互模式默认 720，离屏模式默认 512；0 表示使用默认值"
    )
    args = parser.parse_args()

    # 解析多视角相机列表
    camera_list = [c.strip() for c in args.cameras.split(",") if c.strip()]
    if not camera_list:
        camera_list = ["agentview"]

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv
    from libero.libero.envs.bddl_utils import get_problem_info
    from libero.libero.envs.suction_sticky_wrapper import SuctionStickyWrapper

    if args.list:
        list_tasks(args.suite)
        return

    # ── 获取 bddl 路径 ──────────────────────────────────────────────────────
    if args.bddl_file:
        bddl_file = os.path.abspath(args.bddl_file)
        if not os.path.exists(bddl_file):
            print(f"[错误] BDDL 文件不存在: {bddl_file}")
            sys.exit(1)
        info = get_problem_info(bddl_file)
        print(f"\n[LIBERO 吸盘可视化]")
        print("  模式  : 自定义 BDDL")
        print(f"  问题  : {info['problem_name']}")
        print(f"  指令  : {info['language_instruction']}")
        print(f"  bddl  : {bddl_file}")
    else:
        bm_dict = benchmark.get_benchmark_dict()
        if args.suite not in bm_dict:
            print(f"[错误] 套件 '{args.suite}' 不存在，可用：{list(bm_dict.keys())}")
            sys.exit(1)

        bm = bm_dict[args.suite]()
        if args.task >= bm.n_tasks:
            print(f"[错误] 任务 ID {args.task} 超出范围 0~{bm.n_tasks-1}")
            sys.exit(1)

        bddl_file = bm.get_task_bddl_file_path(args.task)
        task_info = bm.get_task(args.task)
        print(f"\n[LIBERO 吸盘可视化]")
        print(f"  套件  : {args.suite}")
        print(f"  任务  : [{args.task}] {task_info.language}")
        print(f"  bddl  : {bddl_file}")
    print(f"  机器人: SuctionPanda → MountedSuctionPanda (吸盘夹爪)")
    print(f"  相机  : {camera_list}")

    # ── 分辨率 ──────────────────────────────────────────────────────────────
    if args.resolution > 0:
        res_interactive = args.resolution
        res_offscreen = args.resolution
    else:
        res_interactive = 720   # 交互模式默认 720×720
        res_offscreen = 512     # 离屏模式默认 512×512
    print(f"  分辨率: {res_interactive if not args.offscreen else res_offscreen} px")
    print(f"  模式  : {'原生窗口' if args.native_viewer else ('离屏' if args.offscreen else 'CV2窗口')}\n")

    # ── 构建环境 ────────────────────────────────────────────────────────────
    # ── 交互模式：离屏渲染抓帧，用 cv2.imshow 实时并排显示（GLFW segfault 规避）
    # ── 离屏模式：同样离屏，多摄像头 + 保存 PNG/视频
    if args.native_viewer:
        env = ControlEnv(
            bddl_file_name=bddl_file,
            robots=["SuctionPanda"],
            has_renderer=True,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            render_camera=camera_list[0],
            control_freq=20,
            horizon=max(args.steps + 50, 1000),
            ignore_done=True,
            hard_reset=False,
        )
    elif not args.offscreen:
        env = ControlEnv(
            bddl_file_name=bddl_file,
            robots=["SuctionPanda"],
            has_renderer=False,
            has_offscreen_renderer=True,    # ← 离屏抓帧，再用 cv2.imshow 显示
            use_camera_obs=True,
            camera_names=camera_list,
            camera_heights=res_interactive,
            camera_widths=res_interactive,
            control_freq=20,
            horizon=10000,
            ignore_done=True,
            hard_reset=False,
        )
    else:
        env = ControlEnv(
            bddl_file_name=bddl_file,
            robots=["SuctionPanda"],
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=camera_list,
            camera_heights=res_offscreen,
            camera_widths=res_offscreen,
            control_freq=20,
            horizon=args.steps + 50,
            ignore_done=False,
        )

    env = SuctionStickyWrapper(env)
    obs = env.reset()
    print("[OK] 环境重置完成。")

    # action: OSC_POSE 6维位姿delta + 1维吸盘 = 7维
    null_action = np.zeros(7)

    # ──────────────────────────────────────────────────────────────────────
    # 模式 0：MuJoCo 原生窗口（支持鼠标自由视角）
    # ──────────────────────────────────────────────────────────────────────
    if args.native_viewer:
        print("操作提示：鼠标左键旋转，右键平移，滚轮缩放，Ctrl-C 退出。")
        step = 0
        try:
            while step < args.steps:
                suction = 1.0 if (step // 150) % 2 == 0 else -1.0
                act = null_action.copy()
                act[-1] = suction
                obs, reward, done, info = env.step(act.tolist())
                if hasattr(env, "render"):
                    env.render()
                elif hasattr(env, "env") and hasattr(env.env, "render"):
                    env.env.render()
                if done:
                    env.reset()
                step += 1
        except KeyboardInterrupt:
            print("\n已退出原生窗口模式。")
        finally:
            env.env.close()
        return

    # ──────────────────────────────────────────────────────────────────────
    # 模式 A：离屏抓帧 + cv2.imshow 实时显示（多视角并排，规避 GLFW segfault）
    # ──────────────────────────────────────────────────────────────────────
    if not args.offscreen:
        import cv2

        font_scale = max(0.5, res_interactive / 720 * 0.7)
        print(f"操作提示：按 q 键退出窗口  |  视角数：{len(camera_list)}\n")
        step = 0
        try:
            while True:
                suction = 1.0 if (step // 150) % 2 == 0 else -1.0
                act = null_action.copy()
                act[-1] = suction

                obs, reward, done, info = env.step(act.tolist())

                # 收集所有相机帧，上下翻转（robosuite 惯例），拼成横排
                panels = []
                for cam in camera_list:
                    key_name = f"{cam}_image"
                    frame = obs.get(key_name)
                    if frame is None:
                        continue
                    bgr = cv2.cvtColor(frame[::-1], cv2.COLOR_RGB2BGR)
                    # 相机名称标注
                    cv2.putText(bgr, cam, (8, int(26 * font_scale / 0.5)),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                (0, 220, 255), 2, cv2.LINE_AA)
                    panels.append(bgr)

                if panels:
                    canvas = np.hstack(panels)
                    label = f"step={step}  suction={'ON' if suction > 0 else 'OFF'}"
                    cv2.putText(canvas, label, (8, canvas.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.imshow("LIBERO - SuctionPanda  (q=quit)", canvas)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

                step += 1
        except KeyboardInterrupt:
            print("\n已退出。")
        finally:
            cv2.destroyAllWindows()
            env.env.close()
        return

    # ──────────────────────────────────────────────────────────────────────
    # 模式 B：离屏渲染
    # ──────────────────────────────────────────────────────────────────────
    # frames_dict: {cam_name: [frame, ...]}，每帧均已垂直翻转
    frames_dict = {cam: [] for cam in camera_list}
    print(f"[离屏] 录制 {args.steps} 步，视角：{camera_list} ...")

    for step in range(args.steps):
        suction = 1.0 if (step // 150) % 2 == 0 else -1.0
        act = null_action.copy()
        act[-1] = suction
        obs, reward, done, info = env.step(act.tolist())

        for cam in camera_list:
            frame = obs.get(f"{cam}_image")      # (H, W, 3) uint8, RGB
            if frame is not None:
                frames_dict[cam].append(frame[::-1])  # 垂直翻转（robosuite 惯例）

        if done:
            obs = env.reset()

    env.env.close()

    # 取各视角第一帧拼成横排，作为代表帧
    def make_mosaic(step_idx: int):
        """将各相机在 step_idx 的帧横向拼接，返回 RGB numpy 数组。"""
        panels = []
        for cam in camera_list:
            lst = frames_dict.get(cam, [])
            if lst and step_idx < len(lst):
                panels.append(lst[step_idx])
        return np.hstack(panels) if panels else None

    # 保存第一帧 PNG（多视角横拼）
    if args.save_png:
        mosaic = make_mosaic(0)
        if mosaic is not None:
            import cv2
            cv2.imwrite(args.save_png, mosaic[:, :, ::-1])  # RGB → BGR
            print(f"[离屏] 第一帧（{len(camera_list)} 视角拼图）已保存："
                  f"{os.path.abspath(args.save_png)}")

    # 保存视频（多视角横拼）
    if args.save_video:
        import cv2
        mosaic0 = make_mosaic(0)
        if mosaic0 is not None:
            h, w = mosaic0.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_path = os.path.abspath(args.save_video)
            vw = cv2.VideoWriter(out_path, fourcc, 20, (w, h))
            total = min(len(v) for v in frames_dict.values() if v)
            for i in range(total):
                mosaic = make_mosaic(i)
                if mosaic is not None:
                    vw.write(mosaic[:, :, ::-1])
            vw.release()
            print(f"[离屏] 视频（{len(camera_list)} 视角拼图）已保存："
                  f"{out_path}（{total} 帧）")

    if not args.save_png and not args.save_video:
        print("[离屏] 未指定 --save_png / --save_video，跳过保存。")
        # 用 matplotlib 显示多视角预览图
        try:
            import matplotlib.pyplot as plt
            n_cams = len(camera_list)
            sample_steps = min(4, min(len(v) for v in frames_dict.values() if v))
            step_indices = np.linspace(0, sample_steps - 1, sample_steps, dtype=int)
            fig, axes = plt.subplots(
                n_cams, sample_steps,
                figsize=(4 * sample_steps, 3 * n_cams),
                squeeze=False,
            )
            for r, cam in enumerate(camera_list):
                for c, idx in enumerate(step_indices):
                    lst = frames_dict.get(cam, [])
                    if idx < len(lst):
                        axes[r][c].imshow(lst[idx])
                    axes[r][c].set_title(f"{cam}\nstep {idx}", fontsize=8)
                    axes[r][c].axis("off")
            plt.suptitle(f"MountedSuctionPanda — {task_info.language[:60]}")
            plt.tight_layout()
            out_png = "suction_libero_preview.png"
            plt.savefig(out_png, dpi=120)
            print(f"[离屏] 预览图（{n_cams} 视角）已保存：{os.path.abspath(out_png)}")
            plt.show()
        except Exception as e:
            print(f"[提示] matplotlib 显示失败：{e}")
            print("       请加上 --save_png out.png 参数重试。")


if __name__ == "__main__":
    main()
