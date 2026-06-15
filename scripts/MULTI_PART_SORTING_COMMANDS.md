# Multi-Part Sorting Commands

这组命令用于三零件分拣场景。三份 BDDL 使用完全相同的 `:regions` 和 `:init`，因此三次采集看到的是同一套桌面布局；区别只在任务语言、`obj_of_interest` 和 `:goal`。

- rectangular steel plate -> red bin
- round steel plate -> blue bin
- triangular steel plate -> yellow bin

布局原则：

- 红框和矩形件避开机械臂底座附近的近身下探区。
- 黄框不放得比旧黄色位置更远。
- 三个框尺寸不变，三个零件尺寸不变。
- 三个框采用紧凑三角布局，三个零件放在前方同一条拾取线上。

数据统一放在：

```bash
$PWD/data/suction_dataset_multi_part_sorting
```

## 🔴 补采 Round 2（诊断后定版，2026-06-07）—— 逐条执行清单

> 下方"数据方案（90 条总计）"是 **Round 1（已完成）**，仅作存档；本节是 Round 1 训练+评测后的**新增补采**，以本节为准。

90 条训完 + 一整轮免采集补救（过采样×3、推理高积分步 `flow_steps=40`、闭环 `replan=2`、抓取阶段加损失 `XVLA_GRASP_WEIGHT=4`）后，确认**两个场景只能靠补采**（round 抓取已靠过采样修好 0→100%，不用补）：

- **场景 A｜三角抓取精度**：默认 rect/round 各 6/6，**三角全 0/6 `no_contact`**（吸盘没贴到板）；四种免采集手段全无效 → 三角 demo 源精度不够（audit 有 3 条 throw 偏高）。**对策：整组三角（默认 + 2 个 swap）删旧重采、精抓。**
- **场景 B｜位置捷径**：swap 仅 ~17%，模型按指令去"记忆 slot"抓；过采样现有 swap（复制条目）反而更差 → 必须补**真实不同位姿**的非默认位置 demo（现每形状 20:5:5 偏默认槽）。**对策：rect/round 的 4 个 swap 各追加 10 条。**

**逐条清单（本次原计划共采 95 条；截至 2026-06-09，#1 已成功写出 18/25 条，下面命令按剩余量续采）：**

| # | 场景 | 采集目录后缀（raw_hdf5/ 下） | 现有 | 终态 | **本次采** | 旧数据 |
|---|---|---|---|---|---|---|
| 1 | A 三角·默认 | `triangular_yellow_bin` | 已重采 18/25 | 25 | **剩 7** | 留·追加 |
| 2 | A 三角·swap0 | `triangular_yellow_bin_swap0` | 5 | 15 | **15** | **先删** |
| 3 | A 三角·swap1 | `triangular_yellow_bin_swap1` | 5 | 15 | **15** | **先删** |
| 4 | B rect·swap1 | `rectangular_red_bin_swap1` | 5 | 15 | **10** | 留·追加 |
| 5 | B rect·swap2 | `rectangular_red_bin_swap2` | 5 | 15 | **10** | 留·追加 |
| 6 | B round·swap0 | `round_blue_bin_swap0` | 5 | 15 | **10** | 留·追加 |
| 7 | B round·swap2 | `round_blue_bin_swap2` | 5 | 15 | **10** | 留·追加 |

> #1 已完成删旧后的前 18 条，续采时不要再删目录；#2–#3 仍按三角**删旧重采**处理。采时**对准三角中心、下探到吸盘贴住板面再吸、轻放入框**。
> #4–#7 rect/round **直接追加**（collect_only.sh 时间戳命名，不覆盖）。9 个 BDDL 都已存在，**无需新建文件**。
> 可选增强：把槽位 region 从 ~2cm 加宽到 ~5cm 做连续位置随机化（改 BDDL `:init`，注意防碰撞）——非必须。

**采集命令（逐条跑，对应上表 #1–#7）：**

```bash
cd /home/x/vla/libero
source /home/x/miniforge3/etc/profile.d/conda.sh && conda activate vla-adapter
D=$PWD/data/suction_dataset_multi_part_sorting
C=$PWD/libero/libero/bddl_files/custom

# ============ 场景 A：三角重采/续采（精抓：贴到板面再吸、轻放）============
# #1 三角·默认 剩余 ×7（已采 18/25，不删）
# 已采 18 条后继续补采，不要删除已有目录。首次重采时才使用：
# rm -rf $D/raw_hdf5/triangular_yellow_bin $D/tmp_chunks/triangular_yellow_bin
# BDDL_FILE=$C/pick_up_the_triangular_steel_plate_and_place_it_in_the_yellow_bin.bddl \
# COLLECT_DIR=$D/raw_hdf5/triangular_yellow_bin \
# TMP_DIR_ROOT=$D/tmp_chunks/triangular_yellow_bin NUM_DEMO=7 bash scripts/collect_only.sh

# #2 三角·swap0 ×15
# rm -rf $D/raw_hdf5/triangular_yellow_bin_swap0 $D/tmp_chunks/triangular_yellow_bin_swap0
# BDDL_FILE=$C/pick_up_the_triangular_steel_plate_and_place_it_in_the_yellow_bin_swap0.bddl \
# COLLECT_DIR=$D/raw_hdf5/triangular_yellow_bin_swap0 \
# TMP_DIR_ROOT=$D/tmp_chunks/triangular_yellow_bin_swap0 NUM_DEMO=13 bash scripts/collect_only.sh

# #3 三角·swap1 ×15
# rm -rf $D/raw_hdf5/triangular_yellow_bin_swap1 $D/tmp_chunks/triangular_yellow_bin_swap1
# BDDL_FILE=$C/pick_up_the_triangular_steel_plate_and_place_it_in_the_yellow_bin_swap1.bddl \
# COLLECT_DIR=$D/raw_hdf5/triangular_yellow_bin_swap1 \
# TMP_DIR_ROOT=$D/tmp_chunks/triangular_yellow_bin_swap1 NUM_DEMO=15 bash scripts/collect_only.sh

# ============ 场景 B：rect/round 的 4 个 swap 各 +10（追加，不删）============
# #4 rect·swap1 +10
# BDDL_FILE=$C/pick_up_the_rectangular_steel_plate_and_place_it_in_the_red_bin_swap1.bddl \
# COLLECT_DIR=$D/raw_hdf5/rectangular_red_bin_swap1 \
# TMP_DIR_ROOT=$D/tmp_chunks/rectangular_red_bin_swap1 NUM_DEMO=10 bash scripts/collect_only.sh

# #5 rect·swap2 +10
# BDDL_FILE=$C/pick_up_the_rectangular_steel_plate_and_place_it_in_the_red_bin_swap2.bddl \
# COLLECT_DIR=$D/raw_hdf5/rectangular_red_bin_swap2 \
# TMP_DIR_ROOT=$D/tmp_chunks/rectangular_red_bin_swap2 NUM_DEMO=10 bash scripts/collect_only.sh

# #6 round·swap0 +10
# BDDL_FILE=$C/pick_up_the_round_steel_plate_and_place_it_in_the_blue_bin_swap0.bddl \
# COLLECT_DIR=$D/raw_hdf5/round_blue_bin_swap0 \
# TMP_DIR_ROOT=$D/tmp_chunks/round_blue_bin_swap0 NUM_DEMO=10 bash scripts/collect_only.sh

# #7 round·swap2 +10
# BDDL_FILE=$C/pick_up_the_round_steel_plate_and_place_it_in_the_blue_bin_swap2.bddl \
# COLLECT_DIR=$D/raw_hdf5/round_blue_bin_swap2 \
# TMP_DIR_ROOT=$D/tmp_chunks/round_blue_bin_swap2 NUM_DEMO=10 bash scripts/collect_only.sh
```

**采完之后**：① `audit_success.py` 抽查成功率 → ② `raw2xvla.py` 重新打包（递归全子目录，自动含新数据）→ ③ 传 h100 改 meta 前缀 → ④ 训练。**数据已均衡，用原始 meta 即可、不必再过采样**（若三角仍弱，再单独对 `triangular_*` 子目录 ×2）。终态约 155 条：rect 50 / round 50 / 三角 55。

## Preview

已保留一张当前布局预览：

```bash
industrial_scene_previews/multi_part_sorting_scene_mosaic.png
```

重新生成预览：

```bash
cd /home/x/vla/libero

source /home/x/miniforge3/etc/profile.d/conda.sh
conda activate vla-adapter
export MUJOCO_GL=egl

python show_libero_suction.py \
  --offscreen \
  --steps 1 \
  --resolution 512 \
  --cameras agentview,birdview,frontview,sideview \
  --bddl_file libero/libero/bddl_files/custom/pick_up_the_rectangular_steel_plate_and_place_it_in_the_red_bin.bddl \
  --save_png industrial_scene_previews/multi_part_sorting_scene_mosaic.png
```

## 数据方案（90 条总计）

| 任务 | 布局 | 目标件所在 slot | 目标 | 已采 | **补采** |
|---|---|---|---|---|---|
| 矩形→红 | 默认 | slot_0 | 20 | 6 | **14** |
| 矩形→红 | P(swap1) | slot_1 | 5 | 0 | **5** |
| 矩形→红 | Q(swap2) | slot_2 | 5 | 0 | **5** |
| 圆形→蓝 | 默认 | slot_1 | 20 | 2 | **18** |
| 圆形→蓝 | P(swap0) | slot_0 | 5 | 0 | **5** |
| 圆形→蓝 | R(swap2) | slot_2 | 5 | 0 | **5** |
| 三角→黄 | 默认 | slot_2 | 20 | 6 | **14** |
| 三角→黄 | Q(swap0) | slot_0 | 5 | 0 | **5** |
| 三角→黄 | R(swap1) | slot_1 | 5 | 0 | **5** |
| **合计** | | | **90** | **14** | **76** |

> 换位布局 P {slot0:圆,slot1:矩,slot2:三} / Q {slot0:三,slot1:圆,slot2:矩} / R {slot0:矩,slot1:三,slot2:圆}
> swap 后缀数字 = 目标零件落到的 slot 编号。换位与默认语言指令相同，子目录必须隔离。

## Smoke Test Collection

建议每个任务先采 1-2 条，确认红框下探不抽搐、腕部凸起不明显撞框，再扩大采集。

### Rectangular -> Red Bin（默认布局，补 14；已有 6 → 共 20）

<!-- ```bash
cd /home/x/vla/libero

NUM_DEMO=14 \
COLLECT_DIR="$PWD/data/suction_dataset_multi_part_sorting/raw_hdf5/rectangular_red_bin" \
TMP_DIR_ROOT="$PWD/data/suction_dataset_multi_part_sorting/tmp_chunks/rectangular_red_bin" \
BDDL_FILE="$PWD/libero/libero/bddl_files/custom/pick_up_the_rectangular_steel_plate_and_place_it_in_the_red_bin.bddl" \
bash scripts/collect_only.sh
``` -->

### Rectangular -> Red Bin（换位 P，矩在 slot_1，采 5）

<!-- ```bash
cd /home/x/vla/libero

NUM_DEMO=5 \
COLLECT_DIR="$PWD/data/suction_dataset_multi_part_sorting/raw_hdf5/rectangular_red_bin_swap1" \
TMP_DIR_ROOT="$PWD/data/suction_dataset_multi_part_sorting/tmp_chunks/rectangular_red_bin_swap1" \
BDDL_FILE="$PWD/libero/libero/bddl_files/custom/pick_up_the_rectangular_steel_plate_and_place_it_in_the_red_bin_swap1.bddl" \
bash scripts/collect_only.sh
``` -->

### Rectangular -> Red Bin（换位 Q，矩在 slot_2，采 5）

<!-- ```bash
cd /home/x/vla/libero

NUM_DEMO=5 \
COLLECT_DIR="$PWD/data/suction_dataset_multi_part_sorting/raw_hdf5/rectangular_red_bin_swap2" \
TMP_DIR_ROOT="$PWD/data/suction_dataset_multi_part_sorting/tmp_chunks/rectangular_red_bin_swap2" \
BDDL_FILE="$PWD/libero/libero/bddl_files/custom/pick_up_the_rectangular_steel_plate_and_place_it_in_the_red_bin_swap2.bddl" \
bash scripts/collect_only.sh
``` -->

### Round -> Blue Bin（默认布局，补 18；已有 2 → 共 20）
<!-- 
```bash
cd /home/x/vla/libero

NUM_DEMO=8 \
COLLECT_DIR="$PWD/data/suction_dataset_multi_part_sorting/raw_hdf5/round_blue_bin" \
TMP_DIR_ROOT="$PWD/data/suction_dataset_multi_part_sorting/tmp_chunks/round_blue_bin" \
BDDL_FILE="$PWD/libero/libero/bddl_files/custom/pick_up_the_round_steel_plate_and_place_it_in_the_blue_bin.bddl" \
bash scripts/collect_only.sh
``` -->

### Round -> Blue Bin（换位 P，圆在 slot_0，采 5）

<!-- ```bash
cd /home/x/vla/libero

NUM_DEMO=5 \
COLLECT_DIR="$PWD/data/suction_dataset_multi_part_sorting/raw_hdf5/round_blue_bin_swap0" \
TMP_DIR_ROOT="$PWD/data/suction_dataset_multi_part_sorting/tmp_chunks/round_blue_bin_swap0" \
BDDL_FILE="$PWD/libero/libero/bddl_files/custom/pick_up_the_round_steel_plate_and_place_it_in_the_blue_bin_swap0.bddl" \
bash scripts/collect_only.sh
``` -->

### Round -> Blue Bin（换位 R，圆在 slot_2，采 5）

<!-- ```bash
cd /home/x/vla/libero

NUM_DEMO=5 \
COLLECT_DIR="$PWD/data/suction_dataset_multi_part_sorting/raw_hdf5/round_blue_bin_swap2" \
TMP_DIR_ROOT="$PWD/data/suction_dataset_multi_part_sorting/tmp_chunks/round_blue_bin_swap2" \
BDDL_FILE="$PWD/libero/libero/bddl_files/custom/pick_up_the_round_steel_plate_and_place_it_in_the_blue_bin_swap2.bddl" \
bash scripts/collect_only.sh
``` -->

### Triangular -> Yellow Bin（默认布局，补 14；已有 6 → 共 20）

<!-- ```bash
cd /home/x/vla/libero

NUM_DEMO=14 \
COLLECT_DIR="$PWD/data/suction_dataset_multi_part_sorting/raw_hdf5/triangular_yellow_bin" \
TMP_DIR_ROOT="$PWD/data/suction_dataset_multi_part_sorting/tmp_chunks/triangular_yellow_bin" \
BDDL_FILE="$PWD/libero/libero/bddl_files/custom/pick_up_the_triangular_steel_plate_and_place_it_in_the_yellow_bin.bddl" \
bash scripts/collect_only.sh
``` -->

### Triangular -> Yellow Bin（换位 Q，三在 slot_0，采 5）

<!-- ```bash
cd /home/x/vla/libero

NUM_DEMO=5 \
COLLECT_DIR="$PWD/data/suction_dataset_multi_part_sorting/raw_hdf5/triangular_yellow_bin_swap0" \
TMP_DIR_ROOT="$PWD/data/suction_dataset_multi_part_sorting/tmp_chunks/triangular_yellow_bin_swap0" \
BDDL_FILE="$PWD/libero/libero/bddl_files/custom/pick_up_the_triangular_steel_plate_and_place_it_in_the_yellow_bin_swap0.bddl" \
bash scripts/collect_only.sh
``` -->

### Triangular -> Yellow Bin（换位 R，三在 slot_1，采 5）

<!-- ```bash
cd /home/x/vla/libero

NUM_DEMO=5 \
COLLECT_DIR="$PWD/data/suction_dataset_multi_part_sorting/raw_hdf5/triangular_yellow_bin_swap1" \
TMP_DIR_ROOT="$PWD/data/suction_dataset_multi_part_sorting/tmp_chunks/triangular_yellow_bin_swap1" \
BDDL_FILE="$PWD/libero/libero/bddl_files/custom/pick_up_the_triangular_steel_plate_and_place_it_in_the_yellow_bin_swap1.bddl" \
bash scripts/collect_only.sh
``` -->

## Batch Convert（质检用，非训练格式）

> ⚠️ 这步产出 `converted_hdf5`（robomimic 风格），**仅用于质检 / replay 复核**。
> 它不是 X-VLA 训练格式，X-VLA 微调数据由下面的 `raw2xvla.py` 直接从 raw 打包。
> 如果只想要训练数据，可跳过本步直接跑「Pack to X-VLA」。
>
> 🖥️ **可选加速（独显渲染）**：WSL 里 MuJoCo EGL 走 Mesa→D3D12，默认挑 Intel 核显(UHD 770)。
> 在整行命令前加 `USE_NVIDIA_RENDER=1` 让离线渲染落到 RTX 4090（offline_convert.sh 会
> `export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA`）。例：
> `USE_NVIDIA_RENDER=1 OUTPUT_DIR=... bash scripts/offline_convert.sh <dir>`。
> 仅影响渲染速度，不影响结果；评测无需开（评测瓶颈在远端推理，本地渲染极廉价）。

```bash
cd /home/x/vla/libero

# 可选：在 for 之外 export USE_NVIDIA_RENDER=1（让全部 task 渲染用 4090）
for task in rectangular_red_bin rectangular_red_bin_swap1 rectangular_red_bin_swap2 \
            round_blue_bin round_blue_bin_swap0 round_blue_bin_swap2 \
            triangular_yellow_bin triangular_yellow_bin_swap0 triangular_yellow_bin_swap1; do
  OUTPUT_DIR="$PWD/data/suction_dataset_multi_part_sorting/converted_hdf5/$task" \
  bash scripts/offline_convert.sh \
  "$PWD/data/suction_dataset_multi_part_sorting/raw_hdf5/$task"
done
```

转换后核对每个子目录条数（`set -e` 可能在某条失败时提前中断该 task）：

```bash
cd /home/x/vla/libero
C=data/suction_dataset_multi_part_sorting/converted_hdf5
for d in "$C"/*/; do printf "%4s  %s\n" "$(find "$d" -name '*.hdf5' | wc -l)" "$(basename "$d")"; done
```

## Pack to X-VLA（真正的训练格式，最后一步）

一条命令递归打包全部 90 条 raw → X-VLA `LiberoHandler` 可读的 hdf5（每条一个文件，
按子目录结构输出避免换位/默认撞名）+ 汇总 `xvla_meta.json`（90 条进同一个 meta，混合 co-train）。

```bash
cd /home/x/vla/libero

/home/x/miniforge3/envs/vla-adapter/bin/python3 scripts/raw2xvla.py \
  --raw-root  "$PWD/data/suction_dataset_multi_part_sorting/raw_hdf5" \
  --out-dir   "$PWD/data/suction_dataset_multi_part_sorting/xvla_hdf5" \
  --meta-out  "$PWD/data/suction_dataset_multi_part_sorting/xvla_meta.json" \
  --libero-root "$PWD"
# 先验证 1 条：追加 --limit 1
```

打包后核对 `meta.json` 的 datalist 应为 90：

```bash
cd /home/x/vla/libero
/home/x/miniforge3/envs/vla-adapter/bin/python3 -c "import json;d=json.load(open('data/suction_dataset_multi_part_sorting/xvla_meta.json'));print('datalist:',len(d['datalist']));print('dataset_name:',d['dataset_name'])"
```
