# LIBERO 吸盘数据脚本教程

本教程对应 [collect_only.sh](collect_only.sh)、[offline_convert.sh](offline_convert.sh)、[suction_dataset_workflow.sh](suction_dataset_workflow.sh)。

## 目标

这 3 个脚本覆盖三件事：

1. 在线采集原始数据
2. 离线渲染生成训练用数据集
3. 可选导出重播 MP4

推荐使用的新目录结构：

- 原始采集 hdf5： [../data/suction_dataset/raw_hdf5](../data/suction_dataset/raw_hdf5)
- 临时采集块： [../data/suction_dataset/tmp_chunks](../data/suction_dataset/tmp_chunks)
- 离线渲染 hdf5： [../data/suction_dataset/converted_hdf5](../data/suction_dataset/converted_hdf5)
- 重播 mp4： [../data/suction_dataset/replay_mp4](../data/suction_dataset/replay_mp4)

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

- raw hdf5 → [../data/suction_dataset/raw_hdf5](../data/suction_dataset/raw_hdf5)
- tmp chunks → [../data/suction_dataset/tmp_chunks](../data/suction_dataset/tmp_chunks)

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
BDDL_FILE=libero/libero/bddl_files/custom/pick_up_the_steel_plate_and_place_it_in_the_basket.bddl \
NUM_DEMO=10 \
bash scripts/collect_only.sh
```

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

### 默认输入输出

- 默认从 [../data/suction_dataset/raw_hdf5](../data/suction_dataset/raw_hdf5) 找最新 `.hdf5`
- 输出到 [../data/suction_dataset/converted_hdf5](../data/suction_dataset/converted_hdf5)

### 什么时候用

适合你**采集完一批之后批量或单独转换**。

### 示例

转换最新 raw：

```bash
bash scripts/offline_convert.sh
```

转换指定文件：

```bash
bash scripts/offline_convert.sh data/suction_dataset/raw_hdf5/xxx.hdf5
```

转换某个目录里最新的文件：

```bash
bash scripts/offline_convert.sh data/suction_dataset/raw_hdf5
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

- raw → [../data/suction_dataset/raw_hdf5](../data/suction_dataset/raw_hdf5)
- tmp → [../data/suction_dataset/tmp_chunks](../data/suction_dataset/tmp_chunks)
- converted → [../data/suction_dataset/converted_hdf5](../data/suction_dataset/converted_hdf5)
- replay → [../data/suction_dataset/replay_mp4](../data/suction_dataset/replay_mp4)

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
MODE=convert bash scripts/suction_dataset_workflow.sh data/suction_dataset/raw_hdf5/xxx.hdf5
```

#### 对单个 converted 文件导出 mp4

```bash
MODE=replay bash scripts/suction_dataset_workflow.sh data/suction_dataset/converted_hdf5/xxx.hdf5
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
MODE=replay bash scripts/suction_dataset_workflow.sh data/suction_dataset/converted_hdf5/xxx.hdf5
```

---

## 当前主流程最少只需要保留的脚本

- [collect_only.sh](collect_only.sh)
- [offline_convert.sh](offline_convert.sh)
- [suction_dataset_workflow.sh](suction_dataset_workflow.sh)
- [replay_converted_dataset_to_mp4.py](replay_converted_dataset_to_mp4.py)

如果只保留主流程，`tmp` 回放脚本可以删除。
