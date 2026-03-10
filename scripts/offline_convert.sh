#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_LIBERO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ENV_NAME="${ENV_NAME:-vla-adapter}"
LIBERO_ROOT="${LIBERO_ROOT:-$DEFAULT_LIBERO_ROOT}"
SCRIPTS_DIR="${SCRIPTS_DIR:-$SCRIPT_DIR}"
COLLECT_DIR="${COLLECT_DIR:-$LIBERO_ROOT/data/suction_dataset/raw_hdf5}"
OUTPUT_DIR="${OUTPUT_DIR:-$LIBERO_ROOT/data/suction_dataset/converted_hdf5}"
NOOP_THRESHOLD="${NOOP_THRESHOLD:-1e-4}"
CAMERA_RES="${CAMERA_RES:-256}"
REPLAY_TRANSLATION_SCALE="${REPLAY_TRANSLATION_SCALE:-1.0}"
REPLAY_ROTATION_SCALE="${REPLAY_ROTATION_SCALE:-1.0}"
MAX_TRANSLATION_NORM="${MAX_TRANSLATION_NORM:-0.0}"
MAX_ROTATION_NORM="${MAX_ROTATION_NORM:-0.5}"
SPLIT_LARGE_ACTIONS="${SPLIT_LARGE_ACTIONS:-0}"
SUBSTEP_TRANSLATION_NORM="${SUBSTEP_TRANSLATION_NORM:-0.25}"
SUBSTEP_ROTATION_NORM="${SUBSTEP_ROTATION_NORM:-0.15}"
MAX_ACTION_SUBSTEPS="${MAX_ACTION_SUBSTEPS:-16}"
OVERWRITE_EXISTING="${OVERWRITE_EXISTING:-0}"

cd "$SCRIPTS_DIR"

convert_one() {
  local demo_file="$1"
  local source_stem output_subdir create_log_file output_dataset

  if [[ ! -f "$demo_file" ]]; then
    echo "[convert] 跳过不存在文件: $demo_file"
    return 1
  fi

  source_stem="$(basename "$demo_file" .hdf5)"
  output_subdir="$OUTPUT_DIR/$source_stem"

  if [[ -d "$output_subdir" && "$OVERWRITE_EXISTING" != "1" ]]; then
    if find "$output_subdir" -maxdepth 1 -type f -name "*.hdf5" | grep -q .; then
      echo "[convert] 已存在输出，跳过: $demo_file"
      find "$output_subdir" -maxdepth 1 -type f -name "*.hdf5" | sort
      return 0
    fi
  fi

  mkdir -p "$output_subdir"

  echo "[convert] 输入: $demo_file"
  create_log_file=$(mktemp)
  conda run -n "$ENV_NAME" python create_dataset.py \
    --demo-file "$demo_file" \
    --dataset-path "$output_subdir" \
    --use-camera-obs \
    --camera-resolution "$CAMERA_RES" \
    --filter-noop \
    --noop-threshold "$NOOP_THRESHOLD" \
    --replay-translation-scale "$REPLAY_TRANSLATION_SCALE" \
    --replay-rotation-scale "$REPLAY_ROTATION_SCALE" \
    --max-translation-norm "$MAX_TRANSLATION_NORM" \
    --max-rotation-norm "$MAX_ROTATION_NORM" \
    --substep-translation-norm "$SUBSTEP_TRANSLATION_NORM" \
    --substep-rotation-norm "$SUBSTEP_ROTATION_NORM" \
    --max-action-substeps "$MAX_ACTION_SUBSTEPS" \
    $( [[ "$SPLIT_LARGE_ACTIONS" == "1" ]] && printf '%s' '--split-large-actions' ) | tee "$create_log_file"

  output_dataset=$(awk '/The created dataset is saved in the following path:/{getline; print; exit}' "$create_log_file" | tr -d '[:space:]')
  rm -f "$create_log_file"

  if [[ -z "$output_dataset" || ! -f "$output_dataset" ]]; then
    echo "[convert] 未找到转换结果: $output_dataset"
    return 1
  fi

  echo "[convert] 数据统计"
  echo "[convert] 输出: $output_dataset"
  if ! conda run -n "$ENV_NAME" python get_dataset_info.py --dataset "$output_dataset"; then
    echo "[convert] 警告: get_dataset_info.py 校验未通过（常见于动作范围不在 [-1, 1]），但转换文件已生成。"
  fi
}

convert_batch() {
  local search_dir="$1"
  local demo_file

  echo "[convert] 目录输入: $search_dir"
  while IFS= read -r demo_file; do
    [[ -z "$demo_file" ]] && continue
    convert_one "$demo_file"
  done < <(find "$search_dir" -type f -name "*.hdf5" ! -path "*/tmp/*" | sort)
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
