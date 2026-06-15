#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_LIBERO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ENV_NAME="${ENV_NAME:-vla-adapter}"
LIBERO_ROOT="${LIBERO_ROOT:-$DEFAULT_LIBERO_ROOT}"
SCRIPTS_DIR="${SCRIPTS_DIR:-$SCRIPT_DIR}"
COLLECT_DIR="${COLLECT_DIR:-$LIBERO_ROOT/data/suction_dataset_multi_part_sorting/raw_hdf5}"
OUTPUT_DIR="${OUTPUT_DIR:-$LIBERO_ROOT/data/suction_dataset_multi_part_sorting/converted_hdf5}"
NOOP_THRESHOLD="${NOOP_THRESHOLD:-1e-4}"
NOOP_KEEP_BEFORE_GRIPPER_CHANGE="${NOOP_KEEP_BEFORE_GRIPPER_CHANGE:-8}"
NOOP_KEEP_AFTER_GRIPPER_CHANGE="${NOOP_KEEP_AFTER_GRIPPER_CHANGE:-8}"
CAMERA_RES="${CAMERA_RES:-256}"
REPLAY_TRANSLATION_SCALE="${REPLAY_TRANSLATION_SCALE:-1.0}"
REPLAY_ROTATION_SCALE="${REPLAY_ROTATION_SCALE:-1.0}"
MAX_TRANSLATION_NORM="${MAX_TRANSLATION_NORM:-0.0}"
MAX_ROTATION_NORM="${MAX_ROTATION_NORM:-0.5}"
SPLIT_LARGE_ACTIONS="${SPLIT_LARGE_ACTIONS:-0}"
SUBSTEP_TRANSLATION_NORM="${SUBSTEP_TRANSLATION_NORM:-0.25}"
SUBSTEP_ROTATION_NORM="${SUBSTEP_ROTATION_NORM:-0.15}"
MAX_ACTION_SUBSTEPS="${MAX_ACTION_SUBSTEPS:-16}"
SUCCESS_SETTLE_STEPS="${SUCCESS_SETTLE_STEPS:-40}"
SUCTION_NORMAL_MAX_ANGLE_DEG="${SUCTION_NORMAL_MAX_ANGLE_DEG:-10.0}"
SUCTION_EFFECTIVE_RADIUS_RATIO="${SUCTION_EFFECTIVE_RADIUS_RATIO:-0.7}"
SUCTION_DETACH_GRACE_STEPS="${SUCTION_DETACH_GRACE_STEPS:-2}"
FILTER_OBVIOUS_THROWS="${FILTER_OBVIOUS_THROWS:-1}"
THROW_RELEASE_HEIGHT_MARGIN_M="${THROW_RELEASE_HEIGHT_MARGIN_M:-0.06}"
THROW_PEAK_HEIGHT_MARGIN_M="${THROW_PEAK_HEIGHT_MARGIN_M:-0.10}"
OVERWRITE_EXISTING="${OVERWRITE_EXISTING:-0}"
SHOW_DATASET_INFO="${SHOW_DATASET_INFO:-0}"

# 可选：离线渲染用独显(RTX 4090)而非 WSL 默认的 Intel 核显。
# WSL 里 MuJoCo EGL 走 Mesa->D3D12，默认挑第一个适配器=核显(Intel UHD 770)；
# 置 USE_NVIDIA_RENDER=1 会让 Mesa d3d12 选 NVIDIA 适配器，渲染落到 4090，离线渲染更快。
# 用法：USE_NVIDIA_RENDER=1 bash offline_convert.sh <dir_or_file>
USE_NVIDIA_RENDER="${USE_NVIDIA_RENDER:-0}"
if [[ "$USE_NVIDIA_RENDER" == "1" ]]; then
  export MESA_D3D12_DEFAULT_ADAPTER_NAME="${MESA_D3D12_DEFAULT_ADAPTER_NAME:-NVIDIA}"
  echo "[convert] 渲染使用独显: MESA_D3D12_DEFAULT_ADAPTER_NAME=$MESA_D3D12_DEFAULT_ADAPTER_NAME (期望 RTX 4090)"
fi

NEWLY_CONVERTED_COUNT=0
NEWLY_SUCCESS_COUNT=0

cd "$SCRIPTS_DIR"

convert_one() {
  local demo_file="$1"
  local progress_label="${2:-}"
  local source_stem output_file create_log_file output_dataset success_summary

  if [[ ! -f "$demo_file" ]]; then
    echo "[convert] 跳过不存在文件: $demo_file"
    return 1
  fi

  source_stem="$(basename "$demo_file" .hdf5)"
  output_file="$OUTPUT_DIR/$source_stem.hdf5"

  if [[ -f "$output_file" && "$OVERWRITE_EXISTING" != "1" ]]; then
    echo "[convert] ${progress_label}已存在输出，跳过: $(basename "$demo_file")"
    return 0
  fi

  mkdir -p "$OUTPUT_DIR"

  if [[ -f "$output_file" && "$OVERWRITE_EXISTING" == "1" ]]; then
    rm -f "$output_file"
  fi

  echo "[convert] ${progress_label}处理: $(basename "$demo_file")"
  create_log_file=$(mktemp)
  if ! conda run -n "$ENV_NAME" python create_dataset.py \
    --demo-file "$demo_file" \
    --dataset-path "$output_file" \
    --use-camera-obs \
    --camera-resolution "$CAMERA_RES" \
    --filter-noop \
    --noop-threshold "$NOOP_THRESHOLD" \
    --noop-keep-before-gripper-change "$NOOP_KEEP_BEFORE_GRIPPER_CHANGE" \
    --noop-keep-after-gripper-change "$NOOP_KEEP_AFTER_GRIPPER_CHANGE" \
    --replay-translation-scale "$REPLAY_TRANSLATION_SCALE" \
    --replay-rotation-scale "$REPLAY_ROTATION_SCALE" \
    --max-translation-norm "$MAX_TRANSLATION_NORM" \
    --max-rotation-norm "$MAX_ROTATION_NORM" \
    --substep-translation-norm "$SUBSTEP_TRANSLATION_NORM" \
    --substep-rotation-norm "$SUBSTEP_ROTATION_NORM" \
    --max-action-substeps "$MAX_ACTION_SUBSTEPS" \
    --success-settle-steps "$SUCCESS_SETTLE_STEPS" \
    --suction-normal-max-angle-deg "$SUCTION_NORMAL_MAX_ANGLE_DEG" \
    --suction-effective-radius-ratio "$SUCTION_EFFECTIVE_RADIUS_RATIO" \
    --suction-detach-grace-steps "$SUCTION_DETACH_GRACE_STEPS" \
    --throw-release-height-margin "$THROW_RELEASE_HEIGHT_MARGIN_M" \
    --throw-peak-height-margin "$THROW_PEAK_HEIGHT_MARGIN_M" \
    --quiet \
    $( [[ "$SPLIT_LARGE_ACTIONS" == "1" ]] && printf '%s' '--split-large-actions' ) \
    $( [[ "$FILTER_OBVIOUS_THROWS" != "1" ]] && printf '%s' '--disable-throw-filter' ) >"$create_log_file" 2>&1; then
    echo "[convert] ${progress_label}失败: $(basename "$demo_file")"
    tail -n 20 "$create_log_file" || true
    rm -f "$create_log_file"
    return 1
  fi

  output_dataset="$output_file"

  if [[ -z "$output_dataset" || ! -f "$output_dataset" ]]; then
    echo "[convert] 未找到转换结果: $output_dataset"
    tail -n 20 "$create_log_file" || true
    rm -f "$create_log_file"
    return 1
  fi

  success_summary=$(conda run -n "$ENV_NAME" python -c '
import sys
import h5py

dataset_path = sys.argv[1]
with h5py.File(dataset_path, "r") as f:
    grp = f["data"]
    demos = list(grp.keys())
    total = len(demos)
    success = sum(int(grp[demo_name].attrs.get("success", 0)) for demo_name in demos)
    rate = float(success) / float(total) if total > 0 else 0.0
    file_success = "success" if success == total and total > 0 else "failed"
    print(f"result={file_success} success={success}/{total} ({rate:.1%})")
' "$output_dataset")
  rm -f "$create_log_file"

  NEWLY_CONVERTED_COUNT=$((NEWLY_CONVERTED_COUNT + 1))
  if [[ "$success_summary" == result=success* ]]; then
    NEWLY_SUCCESS_COUNT=$((NEWLY_SUCCESS_COUNT + 1))
  fi

  echo "[convert] ${progress_label}完成: $(basename "$output_dataset")  ${success_summary}"

  if [[ "$SHOW_DATASET_INFO" == "1" ]]; then
    if ! conda run -n "$ENV_NAME" python get_dataset_info.py --dataset "$output_dataset"; then
      echo "[convert] 警告: get_dataset_info.py 校验未通过（常见于动作范围不在 [-1, 1]），但转换文件已生成。"
    fi
  fi
}

convert_batch() {
  local search_dir="$1"
  local demo_file
  local -a demo_files=()
  local total index progress_label

  echo "[convert] 目录输入: $search_dir"
  mapfile -t demo_files < <(find "$search_dir" -type f -name "*.hdf5" ! -path "*/tmp/*" | sort)
  total="${#demo_files[@]}"
  for ((index=0; index<total; index++)); do
    demo_file="${demo_files[$index]}"
    [[ -z "$demo_file" ]] && continue
    progress_label="[$((index + 1))/$total] "
    convert_one "$demo_file" "$progress_label"
  done
}

DEMO_FILE="${1:-}"
if [[ -z "$DEMO_FILE" ]]; then
  DEMO_FILE=$(find "$COLLECT_DIR" -type f -name "*.hdf5" ! -path "*/tmp/*" -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)
  if [[ -z "$DEMO_FILE" || ! -f "$DEMO_FILE" ]]; then
    echo "[convert] 未找到可转换的 .hdf5 文件，请传入路径或检查 COLLECT_DIR=$COLLECT_DIR"
    exit 1
  fi
  convert_one "$DEMO_FILE"
elif [[ -d "$DEMO_FILE" ]]; then
  convert_batch "$DEMO_FILE"
elif [[ -f "$DEMO_FILE" ]]; then
  convert_one "$DEMO_FILE"
else
  echo "[convert] 输入路径无效: $DEMO_FILE"
  exit 1
fi

if [[ "$NEWLY_CONVERTED_COUNT" -gt 0 ]]; then
  summary_rate=$(conda run -n "$ENV_NAME" python -c '
import sys
success = int(sys.argv[1])
total = int(sys.argv[2])
rate = float(success) / float(total) if total > 0 else 0.0
print(f"{success}/{total} success ({rate:.1%})")
' "$NEWLY_SUCCESS_COUNT" "$NEWLY_CONVERTED_COUNT")
  echo "[convert] summary: newly converted ${summary_rate}"
else
  echo "[convert] summary: no new files converted"
fi
