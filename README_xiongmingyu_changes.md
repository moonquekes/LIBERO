# xiongmingyu 本地 LIBERO 修改总结

本文档基于 2026-05-15 对 `/home/x/vla/libero` 的实际 git 历史整理。

统计范围：

```bash
origin/master..HEAD
```

除本文档提交外，当前 `master` 相比官方 `origin/master` 超前 11 个功能提交，作者均为：

```text
xmy <2081932959@qq.com>
```

整体结论：这批提交把原版 LIBERO 扩展成了面向吸盘机器人、钢板/料框任务、多零件分拣数据采集和离线转换的一套本地工作流。

## 提交列表

| 日期 | 提交 | 标题 | 主要含义 |
| --- | --- | --- | --- |
| 2026-03-10 | `595544d` | `suction` | 第一轮吸盘采集与转换适配，修改环境兼容性、对象注册、采集脚本和转换脚本 |
| 2026-03-10 | `119df5d` | `suction` | 新增吸盘机器人、吸盘 gripper、吸盘 wrapper、钢板进篮子任务、采集/转换/回放脚本和可视化工具 |
| 2026-03-10 | `9a71736` | `修改批量逻辑` | 调整批量转换、转换输出和 replay 相关逻辑 |
| 2026-03-13 | `0dc4d3c` | `update data collection scripts` | 强化人工采集、动作回放、noop 过滤和离线转换参数 |
| 2026-03-18 | `ce3c6b3` | `Add suction constraints and diagnostics workflow` | 增加吸盘约束诊断、接触角/半径约束、采集和回放中的诊断字段 |
| 2026-05-15 | `c1a9627` | `Add industrial steel plate assets` | 新增圆形/三角钢板、红/蓝/黄料框资产，并更新钢板资产 |
| 2026-05-15 | `eb9148a` | `Add multi-part sorting task definitions` | 新增三份多零件分拣 BDDL，并注册对应对象类型 |
| 2026-05-15 | `c43e05a` | `Avoid debugger on region placement failure` | 删除区域采样失败时的 `pdb.set_trace()`，避免批处理卡住 |
| 2026-05-15 | `1170766` | `Tighten suction collection success checks` | 采集端要求释放吸盘后才结束成功回合，转换端增加明显高抛投过滤 |
| 2026-05-15 | `207df75` | `Standardize multi-part sorting workflow paths` | 统一多零件分拣数据目录、脚本默认路径和命令文档 |
| 2026-05-15 | `cbdbfb0` | `Refresh multi-part sorting preview image` | 用一张多零件分拣总览图替换旧 steel plate/basket 预览图 |

## 当前修改主线

### 1. 吸盘机器人与环境适配

新增或修改的核心文件：

- `setup/suction_gripper.py`
- `setup/suction_gripper.xml`
- `libero/libero/envs/robots/suction_mounted_panda.py`
- `libero/libero/envs/robots/__init__.py`
- `libero/libero/envs/suction_sticky_wrapper.py`
- `libero/libero/envs/bddl_base_domain.py`
- `libero/libero/benchmark/__init__.py`

含义：

- 新增 `SuctionPanda` / `MountedSuctionPanda` 机器人入口。
- 将 Panda 原本的双指夹爪替换为吸盘 gripper。
- 用 `SuctionStickyWrapper` 在环境外层实现吸盘开关、吸附、释放和诊断。
- 兼容吸盘场景下 gripper 状态维度、PyTorch `torch.load` 等底层行为。

### 2. 自定义物体资产与任务

新增的吸盘任务资产包括：

- `libero/libero/assets/turbosquid_objects/steel_plate/steel_plate.xml`
- `libero/libero/assets/turbosquid_objects/steel_plate_large/steel_plate_large.xml`
- `libero/libero/assets/turbosquid_objects/basket_large/basket_large.xml`
- `libero/libero/assets/turbosquid_objects/steel_plate_round/steel_plate_round.xml`
- `libero/libero/assets/turbosquid_objects/steel_plate_triangle/steel_plate_triangle.xml`
- `libero/libero/assets/turbosquid_objects/steel_plate_triangle/steel_plate_triangle.obj`
- `libero/libero/assets/turbosquid_objects/red_bin/red_bin.xml`
- `libero/libero/assets/turbosquid_objects/blue_bin/blue_bin.xml`
- `libero/libero/assets/turbosquid_objects/yellow_bin/yellow_bin.xml`

已保留的自定义 BDDL：

- `libero/libero/bddl_files/custom/pick_up_the_steel_plate_and_place_it_in_the_basket.bddl`
- `libero/libero/bddl_files/custom/pick_up_the_steel_plate_large_and_place_it_in_the_basket_large.bddl`
- `libero/libero/bddl_files/custom/pick_up_the_rectangular_steel_plate_and_place_it_in_the_red_bin.bddl`
- `libero/libero/bddl_files/custom/pick_up_the_round_steel_plate_and_place_it_in_the_blue_bin.bddl`
- `libero/libero/bddl_files/custom/pick_up_the_triangular_steel_plate_and_place_it_in_the_yellow_bin.bddl`

多零件分拣的三份 BDDL 现在使用无 `shared_stable_v1` 后缀的正式文件名。三份任务共用同一套桌面布局，只改变目标物、目标料框和 `:goal`：

- 矩形钢板 -> 红框
- 圆形钢板 -> 蓝框
- 三角钢板 -> 黄框

### 3. 人工采集流程

主要文件：

- `scripts/collect_demonstration.py`
- `scripts/collect_only.sh`
- `scripts/suction_dataset_workflow.sh`

主要能力：

- 使用键盘或 SpaceMouse 采集吸盘任务。
- 支持腕部相机实时预览、相机录制、窗口位置、翻转、动作缩放和 deadzone。
- 显示吸盘状态、候选接触点、接触角、径向偏移等诊断信息。
- 显示工件中心点 marker，辅助对准吸盘。
- 默认只有任务成功且吸盘关闭、工件完全释放后，当前回合才会结束并保存。
- 默认数据根目录已切换为：

```bash
data/suction_dataset_multi_part_sorting
```

### 4. 离线转换、过滤和回放

主要文件：

- `scripts/create_dataset.py`
- `scripts/offline_convert.sh`
- `scripts/replay_converted_dataset_to_mp4.py`

主要能力：

- 从 raw HDF5 重放动作并渲染 `agentview` / `eye_in_hand`。
- 支持 noop 过滤、动作拆分、稳定步数、吸盘约束参数和相机分辨率配置。
- 在 converted HDF5 中写入吸盘诊断字段。
- 记录 `raw_goal_success` 和最终 `success`。
- 默认启用明显高抛投过滤：如果目标件释放高度或释放后峰值明显高于料框上沿，即使原始 goal 成功，也会把该条标为失败。
- 可用下面方式临时关闭高抛投过滤：

```bash
FILTER_OBVIOUS_THROWS=0 bash scripts/offline_convert.sh
```

### 5. 文档、预览和安装辅助

主要文件：

- `scripts/SH_SCRIPTS_TUTORIAL.md`
- `scripts/MULTI_PART_SORTING_COMMANDS.md`
- `setup/README.md`
- `setup/setup_suction_collection.sh`
- `setup/suction.sh`
- `show_libero_suction.py`
- `industrial_scene_previews/multi_part_sorting_scene_mosaic.png`

含义：

- `SH_SCRIPTS_TUTORIAL.md` 是通用吸盘数据脚本教程。
- `MULTI_PART_SORTING_COMMANDS.md` 是多零件分拣的专用命令清单。
- `show_libero_suction.py` 用于快速可视化自定义 BDDL，可以输出 PNG 或 MP4。
- 当前只保留一张多零件分拣预览图：

```bash
industrial_scene_previews/multi_part_sorting_scene_mosaic.png
```

## 已移除或不再作为主流程的内容

以下旧内容已经不再作为当前主流程保留：

- 旧的 `scripts/INDUSTRIAL_COLLECT_COMMANDS.md`
- 旧的 `scripts/INDUSTRIAL_SHARED_STABLE_V1_COMMANDS.md`
- 带 `shared_stable_v1` 后缀的 BDDL 文件名
- 3 个旧的 shifted steel plate/basket BDDL 变体
- 根目录下旧的 steel plate/basket 预览图：
  - `steel_plate_scene_check.png`
  - `steel_plate_large_basket_large_scene_check_v8.png`
  - `steel_plate_large_basket_large_scene_check_farther.png`

## 当前推荐入口

多零件分拣采集：

```bash
bash scripts/collect_only.sh
```

批量转换：

```bash
bash scripts/offline_convert.sh data/suction_dataset_multi_part_sorting/raw_hdf5
```

完整 workflow：

```bash
MODE=all bash scripts/suction_dataset_workflow.sh
```

多零件分拣命令参考：

```bash
scripts/MULTI_PART_SORTING_COMMANDS.md
```

## 当前 git 状态说明

截至本文档更新前，功能改动统计为：

- 39 个文件相对 `origin/master` 发生变化
- 4838 行新增
- 183 行删除
- `master` 相比 `origin/master` 超前 11 个功能提交

本文档本身是后续补充的说明文件，用于把上述实际 git 历史整理成可读说明。
