#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_LIBERO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LIBERO_ROOT="${LIBERO_ROOT:-$DEFAULT_LIBERO_ROOT}"
WORKFLOW_ROOT="${WORKFLOW_ROOT:-$LIBERO_ROOT/data/suction_dataset}"
RAW_DIR="${RAW_DIR:-$WORKFLOW_ROOT/raw_hdf5}"
TMP_DIR_ROOT="${TMP_DIR_ROOT:-$WORKFLOW_ROOT/tmp_chunks}"
CONVERTED_DIR="${CONVERTED_DIR:-$WORKFLOW_ROOT/converted_hdf5}"
REPLAY_DIR="${REPLAY_DIR:-$WORKFLOW_ROOT/replay_mp4}"
CLEAN_TMP="${CLEAN_TMP:-1}"
MODE="${MODE:-all}"
GENERATE_REPLAY="${GENERATE_REPLAY:-0}"
TARGET_PATH="${1:-}"

mkdir -p "$RAW_DIR" "$TMP_DIR_ROOT" "$CONVERTED_DIR" "$REPLAY_DIR"

collect_one() {
	echo "[workflow] collect: collect_only.sh"
	local collect_log_file
	collect_log_file=$(mktemp)
	COLLECT_DIR="$RAW_DIR" TMP_DIR_ROOT="$TMP_DIR_ROOT" bash "$SCRIPT_DIR/collect_only.sh" | tee "$collect_log_file"

	RAW_HDF5=$(find "$RAW_DIR" -type f -name "*.hdf5" -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)
	TMP_RUN_DIR=$(awk -F': ' '/\[info\] 采集临时目录:/{print $2; exit}' "$collect_log_file" | tr -d '[:space:]')
	rm -f "$collect_log_file"

	if [[ -z "$RAW_HDF5" || ! -f "$RAW_HDF5" ]]; then
		echo "[workflow] 未找到采集输出 hdf5"
		exit 1
	fi

	echo "[workflow] raw_hdf5: $RAW_HDF5"

	if [[ "$CLEAN_TMP" == "1" && -n "${TMP_RUN_DIR:-}" && -d "$TMP_RUN_DIR" ]]; then
		rm -rf "$TMP_RUN_DIR"
		echo "[workflow] 已清理临时目录: $TMP_RUN_DIR"
	fi
}

convert_one() {
	local raw_hdf5="$1"
	local convert_log_file converted_hdf5
	echo "[workflow] convert: $raw_hdf5"
	convert_log_file=$(mktemp)
	OUTPUT_DIR="$CONVERTED_DIR" bash "$SCRIPT_DIR/offline_convert.sh" "$raw_hdf5" | tee "$convert_log_file"
	converted_hdf5=$(awk -F': ' '/\[convert\] 输出:/{print $2; exit}' "$convert_log_file" | tr -d '[:space:]')
	rm -f "$convert_log_file"

	if [[ -z "$converted_hdf5" || ! -f "$converted_hdf5" ]]; then
		echo "[workflow] 未找到离线渲染输出 hdf5"
		exit 1
	fi

	echo "$converted_hdf5"
}

replay_one() {
	local converted_hdf5="$1"
	local dataset_stem replay_out_dir
	dataset_stem="$(basename "$converted_hdf5" .hdf5)"
	replay_out_dir="$REPLAY_DIR/$dataset_stem"
	echo "[workflow] replay: $converted_hdf5"
	conda run -n "${ENV_NAME:-vla-adapter}" python "$SCRIPT_DIR/replay_converted_dataset_to_mp4.py" \
		--dataset "$converted_hdf5" \
		--output-dir "$replay_out_dir"
	echo "$replay_out_dir"
}

convert_batch() {
	local search_root="${1:-$RAW_DIR}"
	local raw_hdf5 converted_hdf5 replay_out_dir
	while IFS= read -r raw_hdf5; do
		[[ -z "$raw_hdf5" ]] && continue
		converted_hdf5="$(convert_one "$raw_hdf5")"
		if [[ "$GENERATE_REPLAY" == "1" ]]; then
			replay_out_dir="$(replay_one "$converted_hdf5")"
			echo "[workflow] replay_dir: $replay_out_dir"
		fi
		echo "[workflow] converted: $converted_hdf5"
	done < <(find "$search_root" -type f -name "*.hdf5" | sort)
}

case "$MODE" in
	collect)
		collect_one
		;;
	convert)
		if [[ -n "$TARGET_PATH" ]]; then
			if [[ -d "$TARGET_PATH" ]]; then
				convert_batch "$TARGET_PATH"
			else
				CONVERTED_HDF5="$(convert_one "$TARGET_PATH")"
				echo "[workflow] converted: $CONVERTED_HDF5"
			fi
		else
			convert_batch "$RAW_DIR"
		fi
		;;
	replay)
		if [[ -z "$TARGET_PATH" ]]; then
			echo "[workflow] replay 模式需要传入 converted hdf5 文件路径"
			exit 1
		fi
		REPLAY_OUT_DIR="$(replay_one "$TARGET_PATH")"
		echo "[workflow] replay_dir: $REPLAY_OUT_DIR"
		;;
	all)
		collect_one
		CONVERTED_HDF5="$(convert_one "$RAW_HDF5")"
		echo "[workflow] converted: $CONVERTED_HDF5"
		if [[ "$GENERATE_REPLAY" == "1" ]]; then
			REPLAY_OUT_DIR="$(replay_one "$CONVERTED_HDF5")"
			echo "[workflow] replay_dir: $REPLAY_OUT_DIR"
		fi
		;;
	*)
		echo "[workflow] 未知 MODE=$MODE，支持: collect / convert / replay / all"
		exit 1
		;;
esac

echo "[workflow] done"
