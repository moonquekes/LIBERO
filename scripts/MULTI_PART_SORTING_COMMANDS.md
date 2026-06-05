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

## Smoke Test Collection

建议每个任务先采 1-2 条，确认红框下探不抽搐、腕部凸起不明显撞框，再扩大采集。

### Rectangular -> Red Bin

```bash
cd /home/x/vla/libero

NUM_DEMO=2 \
COLLECT_DIR="$PWD/data/suction_dataset_multi_part_sorting/raw_hdf5/rectangular_red_bin" \
TMP_DIR_ROOT="$PWD/data/suction_dataset_multi_part_sorting/tmp_chunks/rectangular_red_bin" \
BDDL_FILE="$PWD/libero/libero/bddl_files/custom/pick_up_the_rectangular_steel_plate_and_place_it_in_the_red_bin.bddl" \
bash scripts/collect_only.sh
```

### Round -> Blue Bin

```bash
cd /home/x/vla/libero

NUM_DEMO=2 \
COLLECT_DIR="$PWD/data/suction_dataset_multi_part_sorting/raw_hdf5/round_blue_bin" \
TMP_DIR_ROOT="$PWD/data/suction_dataset_multi_part_sorting/tmp_chunks/round_blue_bin" \
BDDL_FILE="$PWD/libero/libero/bddl_files/custom/pick_up_the_round_steel_plate_and_place_it_in_the_blue_bin.bddl" \
bash scripts/collect_only.sh
```

### Triangular -> Yellow Bin

```bash
cd /home/x/vla/libero

NUM_DEMO=20 \
COLLECT_DIR="$PWD/data/suction_dataset_multi_part_sorting/raw_hdf5/triangular_yellow_bin" \
TMP_DIR_ROOT="$PWD/data/suction_dataset_multi_part_sorting/tmp_chunks/triangular_yellow_bin" \
BDDL_FILE="$PWD/libero/libero/bddl_files/custom/pick_up_the_triangular_steel_plate_and_place_it_in_the_yellow_bin.bddl" \
bash scripts/collect_only.sh
```

## Batch Convert

```bash
cd /home/x/vla/libero

for task in rectangular_red_bin round_blue_bin triangular_yellow_bin; do
  OUTPUT_DIR="$PWD/data/suction_dataset_multi_part_sorting/converted_hdf5/$task" \
  bash scripts/offline_convert.sh \
  "$PWD/data/suction_dataset_multi_part_sorting/raw_hdf5/$task"
done
```
