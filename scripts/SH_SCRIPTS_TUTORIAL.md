# LIBERO 吸盘数据脚本教程

本教程对应 [collect_only.sh](collect_only.sh)、[offline_convert.sh](offline_convert.sh)、[suction_dataset_workflow.sh](suction_dataset_workflow.sh)。

## 目标

这 3 个脚本覆盖三件事：

1. 在线采集原始数据
2. 离线渲染生成训练用数据集
3. 可选导出重播 MP4

推荐使用的新目录结构：

- 原始采集 hdf5： [../data/suction_dataset_multi_part_sorting/raw_hdf5](../data/suction_dataset_multi_part_sorting/raw_hdf5)
- 临时采集块： [../data/suction_dataset_multi_part_sorting/tmp_chunks](../data/suction_dataset_multi_part_sorting/tmp_chunks)
- 离线渲染 hdf5： [../data/suction_dataset_multi_part_sorting/converted_hdf5](../data/suction_dataset_multi_part_sorting/converted_hdf5)
- 重播 mp4： [../data/suction_dataset_multi_part_sorting/replay_mp4](../data/suction_dataset_multi_part_sorting/replay_mp4)

---

## 1. collect_only.sh

文件： [collect_only.sh](collect_only.sh)

### 作用

只负责**在线采集**。

它会：
- 打开人工控制采集
- 把最终原始数据保存成 `.hdf5`
- 把录制过程中的临时分块写到 `tmp_chunks`

### 默认输出

- raw hdf5 → [../data/suction_dataset_multi_part_sorting/raw_hdf5](../data/suction_dataset_multi_part_sorting/raw_hdf5)
- tmp chunks → [../data/suction_dataset_multi_part_sorting/tmp_chunks](../data/suction_dataset_multi_part_sorting/tmp_chunks)

### 什么时候用

适合你**单独采集**时使用。

### 示例

下面示例默认你已经先进入仓库根目录：

```bash
cd /path/to/libero
```

```bash
bash scripts/collect_only.sh
```

如果要改任务：

```bash
BDDL_FILE=libero/libero/bddl_files/custom/pick_up_the_rectangular_steel_plate_and_place_it_in_the_red_bin.bddl \
NUM_DEMO=10 \
bash scripts/collect_only.sh
```

多零件分拣当前内置了 3 个 custom 任务：

```bash
libero/libero/bddl_files/custom/pick_up_the_rectangular_steel_plate_and_place_it_in_the_red_bin.bddl
libero/libero/bddl_files/custom/pick_up_the_round_steel_plate_and_place_it_in_the_blue_bin.bddl
libero/libero/bddl_files/custom/pick_up_the_triangular_steel_plate_and_place_it_in_the_yellow_bin.bddl
```

例如切换到圆形钢板任务：

```bash
BDDL_FILE=libero/libero/bddl_files/custom/pick_up_the_round_steel_plate_and_place_it_in_the_blue_bin.bddl \
NUM_DEMO=10 \
bash scripts/collect_only.sh
```

如果你准备开始正式采多零件分拣，推荐直接和旧数据分目录存放。
下面这 3 条命令统一使用新的根目录：

```bash
$PWD/data/suction_dataset_multi_part_sorting
```

并且每个任务单独拆开 `raw_hdf5` 和 `tmp_chunks` 子目录，避免和之前的：

```bash
$PWD/data/suction_dataset
```

混在一起。

### 推荐采集命令 1：矩形钢板 -> 红框

```bash
COLLECT_DIR="$PWD/data/suction_dataset_multi_part_sorting/raw_hdf5/rectangular_red_bin" \
TMP_DIR_ROOT="$PWD/data/suction_dataset_multi_part_sorting/tmp_chunks/rectangular_red_bin" \
BDDL_FILE="$PWD/libero/libero/bddl_files/custom/pick_up_the_rectangular_steel_plate_and_place_it_in_the_red_bin.bddl" \
NUM_DEMO=20 \
bash scripts/collect_only.sh
```

### 推荐采集命令 2：圆形钢板 -> 蓝框

```bash
COLLECT_DIR="$PWD/data/suction_dataset_multi_part_sorting/raw_hdf5/round_blue_bin" \
TMP_DIR_ROOT="$PWD/data/suction_dataset_multi_part_sorting/tmp_chunks/round_blue_bin" \
BDDL_FILE="$PWD/libero/libero/bddl_files/custom/pick_up_the_round_steel_plate_and_place_it_in_the_blue_bin.bddl" \
NUM_DEMO=20 \
bash scripts/collect_only.sh
```

### 推荐采集命令 3：三角形钢板 -> 黄框

```bash
COLLECT_DIR="$PWD/data/suction_dataset_multi_part_sorting/raw_hdf5/triangular_yellow_bin" \
TMP_DIR_ROOT="$PWD/data/suction_dataset_multi_part_sorting/tmp_chunks/triangular_yellow_bin" \
BDDL_FILE="$PWD/libero/libero/bddl_files/custom/pick_up_the_triangular_steel_plate_and_place_it_in_the_yellow_bin.bddl" \
NUM_DEMO=20 \
bash scripts/collect_only.sh
```

这里用 `$PWD/...` 而不是相对路径，是因为 `collect_only.sh` 运行时会先切到 `scripts/` 目录；直接写绝对路径更稳，不会把工业数据误写到旧目录或脚本目录下面。

---

## 2. offline_convert.sh

文件： [offline_convert.sh](offline_convert.sh)

### 作用

只负责**离线渲染转换**。

它会：
- 读取一个原始 raw `.hdf5`
- 重放动作并渲染 `agentview` / `eye_in_hand`
- 输出 converted `.hdf5`
- 顺带打印数据统计
- 默认保留 `success_settle_steps`，并启用保守的明显高抛投过滤

### 默认输入输出

- 默认从 [../data/suction_dataset_multi_part_sorting/raw_hdf5](../data/suction_dataset_multi_part_sorting/raw_hdf5) 找最新 `.hdf5`
- 输出到 [../data/suction_dataset_multi_part_sorting/converted_hdf5](../data/suction_dataset_multi_part_sorting/converted_hdf5)

### 什么时候用

适合你**采集完一批之后批量或单独转换**。

### 示例

转换最新 raw：

```bash
bash scripts/offline_convert.sh
```

转换指定文件：

```bash
bash scripts/offline_convert.sh data/suction_dataset_multi_part_sorting/raw_hdf5/xxx.hdf5
```

转换某个目录里最新的文件：

```bash
bash scripts/offline_convert.sh data/suction_dataset_multi_part_sorting/raw_hdf5
```

如果你想临时关闭明显高抛投过滤：

```bash
FILTER_OBVIOUS_THROWS=0 bash scripts/offline_convert.sh
```

---

## 3. suction_dataset_workflow.sh

文件： [suction_dataset_workflow.sh](suction_dataset_workflow.sh)

### 作用

这是总控脚本，支持：

- `collect`：只采集
- `convert`：批量或单个转换
- `replay`：对 converted hdf5 导出 mp4
- `all`：采集 + 转换 + 可选 replay

### 默认目录

- raw → [../data/suction_dataset_multi_part_sorting/raw_hdf5](../data/suction_dataset_multi_part_sorting/raw_hdf5)
- tmp → [../data/suction_dataset_multi_part_sorting/tmp_chunks](../data/suction_dataset_multi_part_sorting/tmp_chunks)
- converted → [../data/suction_dataset_multi_part_sorting/converted_hdf5](../data/suction_dataset_multi_part_sorting/converted_hdf5)
- replay → [../data/suction_dataset_multi_part_sorting/replay_mp4](../data/suction_dataset_multi_part_sorting/replay_mp4)

### 常用模式

#### 只采集

```bash
MODE=collect bash scripts/suction_dataset_workflow.sh
```

#### 批量转换 raw 目录里的所有 hdf5

```bash
MODE=convert bash scripts/suction_dataset_workflow.sh
```

#### 批量转换并同时导出 mp4

```bash
MODE=convert GENERATE_REPLAY=1 bash scripts/suction_dataset_workflow.sh
```

#### 转换单个 raw 文件

```bash
MODE=convert bash scripts/suction_dataset_workflow.sh data/suction_dataset_multi_part_sorting/raw_hdf5/xxx.hdf5
```

#### 对单个 converted 文件导出 mp4

```bash
MODE=replay bash scripts/suction_dataset_workflow.sh data/suction_dataset_multi_part_sorting/converted_hdf5/xxx.hdf5
```

#### 一条龙执行

```bash
MODE=all GENERATE_REPLAY=1 bash scripts/suction_dataset_workflow.sh
```

---

## 什么是临时采集块

临时采集块是在线采集阶段 `DataCollectionWrapper` 写出的中间文件，典型内容包括：

- `ep_xxx/`
- `state_*.npz`
- `model.xml`

它们的作用是：
- 支持实时录制分块
- 最终聚合成 raw `.hdf5`
- 用于调试“录制动作是否能真实重播”

### 要不要保留？

通常**不需要长期保留**。

只有你想调试以下问题时才有必要：
- 某条轨迹为什么重播失败
- 某一步吸盘为什么没吸住
- 原始分块和最终 hdf5 是否一致

工作流里默认：
- `CLEAN_TMP=1` 时会自动清理本次采集 tmp

如果你想保留 tmp：

```bash
CLEAN_TMP=0 MODE=collect bash scripts/suction_dataset_workflow.sh
```

---

## 哪个 replay 脚本是 workflow 需要的？

### workflow 实际使用的是

- [replay_converted_dataset_to_mp4.py](replay_converted_dataset_to_mp4.py)

因为 workflow 的 replay 输入是**converted hdf5**。

### 不再是 workflow 必需的是

- 早期的 tmp 调试回放脚本

它是针对旧的 `tmp_chunks` 调试用回放，不属于现在主流程。

---

## 推荐工作方式

如果你经常：
- 先连续采集很多条
- 后面再统一转换
- mp4 只是偶尔导出

建议这样：

### 第一步：只采集

```bash
MODE=collect bash scripts/suction_dataset_workflow.sh
```

重复采集多次。

### 第二步：批量转换

```bash
MODE=convert bash scripts/suction_dataset_workflow.sh
```

### 第三步：需要时再导出 mp4

```bash
MODE=replay bash scripts/suction_dataset_workflow.sh data/suction_dataset_multi_part_sorting/converted_hdf5/xxx.hdf5
```

---

## 当前主流程最少只需要保留的脚本

- [collect_only.sh](collect_only.sh)
- [offline_convert.sh](offline_convert.sh)
- [suction_dataset_workflow.sh](suction_dataset_workflow.sh)
- [replay_converted_dataset_to_mp4.py](replay_converted_dataset_to_mp4.py)

如果只保留主流程，`tmp` 回放脚本可以删除。
