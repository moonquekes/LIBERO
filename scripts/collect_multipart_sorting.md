# 多零件按形状分拣 — 数据采集 SOP

## ⚙️ 采集前置环境（必读）

> 采集是**人工键盘遥操作 + 有头渲染窗口**，需要可用的 X display。**每次新开终端、或与评测共用终端时，先执行**：
>
> 
>
> 不设 DISPLAY 会在 import 阶段直接挂：failed to acquire X connection: Bad display name（与几何/BDDL 改动无关）。
> 建议**采集与评测各开一个终端**：评测要 MUJOCO_GL=egl 无头，采集要 DISPLAY=:0 有头，二者冲突。

## ⚙️ 采集前置环境（必读）

> 采集是**人工键盘遥操作 + 有头渲染窗口**，需要可用的 X display。**每次新开终端、或与评测共用终端时，先执行**：
>
> ```bash
> export DISPLAY=:0     # WSLg 的 X server，让 pynput 连上 + 渲染窗口弹出
> unset  MUJOCO_GL      # 若该终端跑过评测(MUJOCO_GL=egl 无头)，必须清掉，否则窗口出不来
> ```
>
> 不设 `DISPLAY` 会在 `import` 阶段直接挂：`failed to acquire X connection: Bad display name`（与几何/BDDL 改动无关）。
> 建议**采集与评测各开一个终端**：评测要 `MUJOCO_GL=egl` 无头，采集要 `DISPLAY=:0` 有头，二者冲突。

> **⚠️ Round 2 更新（2026-06-08）：三角放大 + 槽加宽 + 数据复用决定**
> - 三角钢板已放大 **1.15×**（内切圆 2.6→3.0cm，匹配圆形；commit `aa2e654`）。asset 与 BDDL 已改但**文件名不变 → 本 SOP 的采集命令照用，会自动用新几何+新槽**。
> - 工件槽**中心保持不变**，仅把槽宽 2cm→8cm（消除放大后的 spawn 重叠，实测 0%/2700）。新坐标见第 1 节。
> - **数据决定：旧 rect+round 全部复用（不重采），只重跑 `raw2xvla` 重打包；只有三角全删重采**——默认 25 + swap0 15 + swap1 15，用新大三角 + 贴到板面再吸。rect/round swap +10 增采暂缓。
> - 复用前提=**槽中心没动**，旧 demo 固定位置仍落在新评测分布内；若改动中心则旧数据作废，切勿移中心。
> - 注意：工件**朝向固定（yaw 不随机）**，只随机槽内位置。

> 目标：为 **X-VLA 微调** 采集"按形状把钢板放进对应颜色框"的多任务数据。
> 三任务**混合 co-train**，语言指令区分目标。共 **90 条** = 每任务 30 条（20 默认 + 5 换位A + 5 换位B）。
> 框/相机/背景/光照固定；**朝向固定(yaw不随机)**；只随机：待抓零件所在 slot（换位）+ slot 内位置（8cm 槽宽，实际散布 方块±0.5/圆±0.2/三角±1.4cm）。

## 0. 路径约定

```bash
LIBERO=/home/x/vla/libero
CUSTOM=$LIBERO/libero/libero/bddl_files/custom
DATA=$LIBERO/data/suction_dataset_multi_part_sorting
cd $LIBERO/scripts          # collect_only.sh 在此
```
采集环境为人工遥操作（`--device keyboard`），需要图形界面（WSLg）。conda env 默认 `vla-adapter`。

## 1. 默认布局 & slot 坐标（Round2: 中心不变, 槽宽 2→8cm）

| slot | region 名 | 默认零件 | x 范围 | y 范围 |
|---|---|---|---|---|
| slot_0 | rectangular_workpiece_slot | 矩形 | -0.305~-0.225 | -0.325~-0.245 |
| slot_1 | round_workpiece_slot | 圆形 | -0.185~-0.105 | -0.325~-0.245 |
| slot_2 | triangular_workpiece_slot | 三角 | -0.065~0.015 | -0.325~-0.245 |

## 2. 三任务 & 语言指令（换位时指令**不变**）

| 任务 | 目标框 | 语言指令 | 默认 BDDL |
|---|---|---|---|
| 矩形→红框 | red_bin | Pick the rectangular steel plate and place it gently in the red bin | `pick_up_the_rectangular_steel_plate_and_place_it_in_the_red_bin.bddl` |
| 圆形→蓝框 | blue_bin | Pick the round steel plate and place it gently in the blue bin | `pick_up_the_round_steel_plate_and_place_it_in_the_blue_bin.bddl` |
| 三角→黄框 | yellow_bin | Pick the triangular steel plate and place it gently in the yellow bin | `pick_up_the_triangular_steel_plate_and_place_it_in_the_yellow_bin.bddl` |

## 3. 换位布局（两两交换：目标零件与占位零件互换 slot，第三件不动）

3 套物理布局 P/Q/R，对应 6 个换位 BDDL（语言/目标不变，仅 `:init` 两行交换）。已渲染校验通过（见 `data/.../swap_previews/`）。

| 布局 | slot_0 | slot_1 | slot_2 | 服务的换位 BDDL |
|---|---|---|---|---|
| **P** | 圆 | 矩 | 三 | `..._red_bin_swap1`（矩in slot1）、`..._blue_bin_swap0`（圆in slot0）|
| **Q** | 三 | 圆 | 矩 | `..._red_bin_swap2`（矩in slot2）、`..._yellow_bin_swap0`（三in slot0）|
| **R** | 矩 | 三 | 圆 | `..._blue_bin_swap2`（圆in slot2）、`..._yellow_bin_swap1`（三in slot1）|

> swap 后缀数字 = **目标零件落到的 slot 编号**。

## 4. 90 条分配表（含已采 / 补采）

| 任务 | 布局 | 目标零件所在 slot | 目标数 | 已采 | **补采** | raw 子目录 |
|---|---|---|---|---|---|---|
| 矩形→红 | 默认 | slot_0 | 20 | 6 | **14** | `rectangular_red_bin` |
| 矩形→红 | P(swap1) | slot_1 | 5 | 0 | **5** | `rectangular_red_bin_swap1` |
| 矩形→红 | Q(swap2) | slot_2 | 5 | 0 | **5** | `rectangular_red_bin_swap2` |
| 圆形→蓝 | 默认 | slot_1 | 20 | 2 | **18** | `round_blue_bin` |
| 圆形→蓝 | P(swap0) | slot_0 | 5 | 0 | **5** | `round_blue_bin_swap0` |
| 圆形→蓝 | R(swap2) | slot_2 | 5 | 0 | **5** | `round_blue_bin_swap2` |
| 三角→黄 | 默认 | slot_2 | 20 | 6 | **14** | `triangular_yellow_bin` |
| 三角→黄 | Q(swap0) | slot_0 | 5 | 0 | **5** | `triangular_yellow_bin_swap0` |
| 三角→黄 | R(swap1) | slot_1 | 5 | 0 | **5** | `triangular_yellow_bin_swap1` |
| **合计** | | | **90** | **14** | **76** | |

> ⚠️ 换位与默认的语言指令相同 → raw 文件名会撞 → **必须用不同 COLLECT_DIR 子目录隔离**。

## 5. 采集命令（9 条；改 `BDDL_FILE` / `COLLECT_DIR` / `NUM_DEMO`）

```bash
# ============ 任务1：矩形 → 红框 ============
# 默认布局，补 14（已有6 → 共20）
BDDL_FILE=$CUSTOM/pick_up_the_rectangular_steel_plate_and_place_it_in_the_red_bin.bddl \
COLLECT_DIR=$DATA/raw_hdf5/rectangular_red_bin \
NUM_DEMO=14 bash collect_only.sh

# 换位 P（矩in slot1），5 条
BDDL_FILE=$CUSTOM/pick_up_the_rectangular_steel_plate_and_place_it_in_the_red_bin_swap1.bddl \
COLLECT_DIR=$DATA/raw_hdf5/rectangular_red_bin_swap1 \
NUM_DEMO=5 bash collect_only.sh

# 换位 Q（矩in slot2），5 条
BDDL_FILE=$CUSTOM/pick_up_the_rectangular_steel_plate_and_place_it_in_the_red_bin_swap2.bddl \
COLLECT_DIR=$DATA/raw_hdf5/rectangular_red_bin_swap2 \
NUM_DEMO=5 bash collect_only.sh

# ============ 任务2：圆形 → 蓝框 ============
# 默认布局，补 18（已有2 → 共20）
BDDL_FILE=$CUSTOM/pick_up_the_round_steel_plate_and_place_it_in_the_blue_bin.bddl \
COLLECT_DIR=$DATA/raw_hdf5/round_blue_bin \
NUM_DEMO=18 bash collect_only.sh

# 换位 P（圆in slot0），5 条
BDDL_FILE=$CUSTOM/pick_up_the_round_steel_plate_and_place_it_in_the_blue_bin_swap0.bddl \
COLLECT_DIR=$DATA/raw_hdf5/round_blue_bin_swap0 \
NUM_DEMO=5 bash collect_only.sh

# 换位 R（圆in slot2），5 条
BDDL_FILE=$CUSTOM/pick_up_the_round_steel_plate_and_place_it_in_the_blue_bin_swap2.bddl \
COLLECT_DIR=$DATA/raw_hdf5/round_blue_bin_swap2 \
NUM_DEMO=5 bash collect_only.sh

# ============ 任务3：三角 → 黄框 ============
# 默认布局，补 14（已有6 → 共20）
BDDL_FILE=$CUSTOM/pick_up_the_triangular_steel_plate_and_place_it_in_the_yellow_bin.bddl \
COLLECT_DIR=$DATA/raw_hdf5/triangular_yellow_bin \
NUM_DEMO=14 bash collect_only.sh

# 换位 Q（三in slot0），5 条
BDDL_FILE=$CUSTOM/pick_up_the_triangular_steel_plate_and_place_it_in_the_yellow_bin_swap0.bddl \
COLLECT_DIR=$DATA/raw_hdf5/triangular_yellow_bin_swap0 \
NUM_DEMO=5 bash collect_only.sh

# 换位 R（三in slot1），5 条
BDDL_FILE=$CUSTOM/pick_up_the_triangular_steel_plate_and_place_it_in_the_yellow_bin_swap1.bddl \
COLLECT_DIR=$DATA/raw_hdf5/triangular_yellow_bin_swap1 \
NUM_DEMO=5 bash collect_only.sh
```

## 6. 采集后：完整性检查 + 离线转换

```bash
# 条数检查
for d in $DATA/raw_hdf5/*/; do echo "$(find "$d" -name '*.hdf5' | wc -l)  $d"; done

# 离线渲染转换（对每个子目录批量转 converted_hdf5；可肉眼抽检）
MODE=convert bash suction_dataset_workflow.sh $DATA/raw_hdf5/rectangular_red_bin
# …其余 8 个子目录同理，或直接对整个 raw_hdf5 跑：
MODE=convert bash suction_dataset_workflow.sh $DATA/raw_hdf5

# 抽 2~3 条 replay 成 MP4 复核成功性
python replay_converted_dataset_to_mp4.py --dataset <converted.hdf5> --output-dir /tmp/replay_check
```

## 7. 采集前渲染校验（新建/改 BDDL 后必做）

```bash
cd $LIBERO
python show_libero_suction.py --bddl_file $CUSTOM/<某 bddl> \
  --offscreen --steps 2 --cameras agentview,robot0_eye_in_hand \
  --save_png /tmp/preview.png
```
确认：三件平放、相互分离不重叠、不碰框、机械臂可达。不过则调坐标重渲。

## 8. 评测方案

固定框/相机/背景/光照。两组各 30 次，综合成功率 = 60 次合计。评测用 X-VLA server 推理（`evaluation/libero/run_steel_plate_xvla.py` 风格，`--bddl-file` 指对应布局）。

**评测 A（默认主分布）**：用默认 BDDL，slot 内 ~2cm 扰动自然产生。每任务 10 次，共 30。
**评测 B（换位反捷径）**：用换位 BDDL，目标零件出现在非默认 slot。每任务 10 次，共 30：

| 任务 | 5次 | 5次 |
|---|---|---|
| 矩形→红 | P(swap1, slot1) | Q(swap2, slot2) |
| 圆形→蓝 | P(swap0, slot0) | R(swap2, slot2) |
| 三角→黄 | Q(swap0, slot0) | R(swap1, slot1) |

报告三个数：**默认成功率**（基本能力）、**换位成功率**（是否真按形状选择而非背 slot）、**综合成功率**。

**压缩版（时间紧）**：每任务 5 默认 + 3 换位A + 3 换位B = 11，三任务共 33。

## 9. 下游：打包成 X-VLA 训练格式

采集 + 转换完成后，用 `X-VLA/evaluation/libero/raw2xvla.py` 把 raw replay 成 X-VLA 训练 hdf5
（`abs_action_6d[T,10]` + `observations/{agentview,robot0_eye_in_hand}_image` + `language_instruction`），
90 条汇总进一个 `meta.json`（`dataset_name=libero` → domain_id=3）混合 co-train。
