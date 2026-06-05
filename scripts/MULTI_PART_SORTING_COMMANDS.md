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

```bash
cd /home/x/vla/libero

NUM_DEMO=14 \
COLLECT_DIR="$PWD/data/suction_dataset_multi_part_sorting/raw_hdf5/rectangular_red_bin" \
TMP_DIR_ROOT="$PWD/data/suction_dataset_multi_part_sorting/tmp_chunks/rectangular_red_bin" \
BDDL_FILE="$PWD/libero/libero/bddl_files/custom/pick_up_the_rectangular_steel_plate_and_place_it_in_the_red_bin.bddl" \
bash scripts/collect_only.sh
```

### Rectangular -> Red Bin（换位 P，矩在 slot_1，采 5）

```bash
cd /home/x/vla/libero

NUM_DEMO=5 \
COLLECT_DIR="$PWD/data/suction_dataset_multi_part_sorting/raw_hdf5/rectangular_red_bin_swap1" \
TMP_DIR_ROOT="$PWD/data/suction_dataset_multi_part_sorting/tmp_chunks/rectangular_red_bin_swap1" \
BDDL_FILE="$PWD/libero/libero/bddl_files/custom/pick_up_the_rectangular_steel_plate_and_place_it_in_the_red_bin_swap1.bddl" \
bash scripts/collect_only.sh
```

### Rectangular -> Red Bin（换位 Q，矩在 slot_2，采 5）

```bash
cd /home/x/vla/libero

NUM_DEMO=5 \
COLLECT_DIR="$PWD/data/suction_dataset_multi_part_sorting/raw_hdf5/rectangular_red_bin_swap2" \
TMP_DIR_ROOT="$PWD/data/suction_dataset_multi_part_sorting/tmp_chunks/rectangular_red_bin_swap2" \
BDDL_FILE="$PWD/libero/libero/bddl_files/custom/pick_up_the_rectangular_steel_plate_and_place_it_in_the_red_bin_swap2.bddl" \
bash scripts/collect_only.sh
```

### Round -> Blue Bin（默认布局，补 18；已有 2 → 共 20）

```bash
cd /home/x/vla/libero

NUM_DEMO=18 \
COLLECT_DIR="$PWD/data/suction_dataset_multi_part_sorting/raw_hdf5/round_blue_bin" \
TMP_DIR_ROOT="$PWD/data/suction_dataset_multi_part_sorting/tmp_chunks/round_blue_bin" \
BDDL_FILE="$PWD/libero/libero/bddl_files/custom/pick_up_the_round_steel_plate_and_place_it_in_the_blue_bin.bddl" \
bash scripts/collect_only.sh
```

### Round -> Blue Bin（换位 P，圆在 slot_0，采 5）

```bash
cd /home/x/vla/libero

NUM_DEMO=5 \
COLLECT_DIR="$PWD/data/suction_dataset_multi_part_sorting/raw_hdf5/round_blue_bin_swap0" \
TMP_DIR_ROOT="$PWD/data/suction_dataset_multi_part_sorting/tmp_chunks/round_blue_bin_swap0" \
BDDL_FILE="$PWD/libero/libero/bddl_files/custom/pick_up_the_round_steel_plate_and_place_it_in_the_blue_bin_swap0.bddl" \
bash scripts/collect_only.sh
```

### Round -> Blue Bin（换位 R，圆在 slot_2，采 5）

```bash
cd /home/x/vla/libero

NUM_DEMO=5 \
COLLECT_DIR="$PWD/data/suction_dataset_multi_part_sorting/raw_hdf5/round_blue_bin_swap2" \
TMP_DIR_ROOT="$PWD/data/suction_dataset_multi_part_sorting/tmp_chunks/round_blue_bin_swap2" \
BDDL_FILE="$PWD/libero/libero/bddl_files/custom/pick_up_the_round_steel_plate_and_place_it_in_the_blue_bin_swap2.bddl" \
bash scripts/collect_only.sh
```

### Triangular -> Yellow Bin（默认布局，补 14；已有 6 → 共 20）

```bash
cd /home/x/vla/libero

NUM_DEMO=14 \
COLLECT_DIR="$PWD/data/suction_dataset_multi_part_sorting/raw_hdf5/triangular_yellow_bin" \
TMP_DIR_ROOT="$PWD/data/suction_dataset_multi_part_sorting/tmp_chunks/triangular_yellow_bin" \
BDDL_FILE="$PWD/libero/libero/bddl_files/custom/pick_up_the_triangular_steel_plate_and_place_it_in_the_yellow_bin.bddl" \
bash scripts/collect_only.sh
```

### Triangular -> Yellow Bin（换位 Q，三在 slot_0，采 5）

```bash
cd /home/x/vla/libero

NUM_DEMO=5 \
COLLECT_DIR="$PWD/data/suction_dataset_multi_part_sorting/raw_hdf5/triangular_yellow_bin_swap0" \
TMP_DIR_ROOT="$PWD/data/suction_dataset_multi_part_sorting/tmp_chunks/triangular_yellow_bin_swap0" \
BDDL_FILE="$PWD/libero/libero/bddl_files/custom/pick_up_the_triangular_steel_plate_and_place_it_in_the_yellow_bin_swap0.bddl" \
bash scripts/collect_only.sh
```

### Triangular -> Yellow Bin（换位 R，三在 slot_1，采 5）

```bash
cd /home/x/vla/libero

NUM_DEMO=5 \
COLLECT_DIR="$PWD/data/suction_dataset_multi_part_sorting/raw_hdf5/triangular_yellow_bin_swap1" \
TMP_DIR_ROOT="$PWD/data/suction_dataset_multi_part_sorting/tmp_chunks/triangular_yellow_bin_swap1" \
BDDL_FILE="$PWD/libero/libero/bddl_files/custom/pick_up_the_triangular_steel_plate_and_place_it_in_the_yellow_bin_swap1.bddl" \
bash scripts/collect_only.sh
```

## Batch Convert

```bash
cd /home/x/vla/libero

for task in rectangular_red_bin rectangular_red_bin_swap1 rectangular_red_bin_swap2 \
            round_blue_bin round_blue_bin_swap0 round_blue_bin_swap2 \
            triangular_yellow_bin triangular_yellow_bin_swap0 triangular_yellow_bin_swap1; do
  OUTPUT_DIR="$PWD/data/suction_dataset_multi_part_sorting/converted_hdf5/$task" \
  bash scripts/offline_convert.sh \
  "$PWD/data/suction_dataset_multi_part_sorting/raw_hdf5/$task"
done
```
