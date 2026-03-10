#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_LIBERO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ENV_NAME="${ENV_NAME:-vla-adapter}"
LIBERO_ROOT="${LIBERO_ROOT:-$DEFAULT_LIBERO_ROOT}"
SCRIPTS_DIR="${SCRIPTS_DIR:-$SCRIPT_DIR}"
BDDL_FILE="${BDDL_FILE:-$LIBERO_ROOT/libero/libero/bddl_files/custom/pick_up_the_steel_plate_and_place_it_in_the_basket.bddl}"
COLLECT_DIR="${COLLECT_DIR:-$LIBERO_ROOT/data/suction_dataset/raw_hdf5}"
TMP_DIR_ROOT="${TMP_DIR_ROOT:-$LIBERO_ROOT/data/suction_dataset/tmp_chunks}"
NUM_DEMO="${NUM_DEMO:-20}"
INVERT_CONTROLS="${INVERT_CONTROLS:-none}"
PREVIEW_FLIP="${PREVIEW_FLIP:-xy}"
PREVIEW_WINDOW_X="${PREVIEW_WINDOW_X:-120}"
PREVIEW_WINDOW_Y="${PREVIEW_WINDOW_Y:-80}"
POS_SENSITIVITY="${POS_SENSITIVITY:-0.2}"
ROT_SENSITIVITY="${ROT_SENSITIVITY:-0.4}"
ACTION_TRANSLATION_SCALE="${ACTION_TRANSLATION_SCALE:-0.2}"
ACTION_ROTATION_SCALE="${ACTION_ROTATION_SCALE:-0.4}"
TRANSLATION_DEADZONE="${TRANSLATION_DEADZONE:-0.0}"
ROTATION_DEADZONE="${ROTATION_DEADZONE:-0.0}"
MAX_TRANSLATION_NORM="${MAX_TRANSLATION_NORM:-0.3}"
MAX_ROTATION_NORM="${MAX_ROTATION_NORM:-0.0}"

cd "$SCRIPTS_DIR"
mkdir -p "$COLLECT_DIR" "$TMP_DIR_ROOT"

echo "[collect] 采集 states/actions，并在每个成功回合后写入一个独立 HDF5 文件（默认不存图像）"
conda run -n "$ENV_NAME" python collect_demonstration.py \
  --device keyboard \
  --robots SuctionPanda \
  --camera agentview \
  --pos-sensitivity "$POS_SENSITIVITY" \
  --rot-sensitivity "$ROT_SENSITIVITY" \
  --action-translation-scale "$ACTION_TRANSLATION_SCALE" \
  --action-rotation-scale "$ACTION_ROTATION_SCALE" \
  --translation-deadzone "$TRANSLATION_DEADZONE" \
  --rotation-deadzone "$ROTATION_DEADZONE" \
  --max-translation-norm "$MAX_TRANSLATION_NORM" \
  --max-rotation-norm "$MAX_ROTATION_NORM" \
  --invert-controls "$INVERT_CONTROLS" \
  --preview-flip "$PREVIEW_FLIP" \
  --preview-window-x "$PREVIEW_WINDOW_X" \
  --preview-window-y "$PREVIEW_WINDOW_Y" \
  --show-grip-cylinder \
  --num-demonstration "$NUM_DEMO" \
  --directory "$COLLECT_DIR" \
  --tmp-dir-root "$TMP_DIR_ROOT" \
  --bddl-file "$BDDL_FILE"

echo "[collect] 完成。输出目录: $COLLECT_DIR"
