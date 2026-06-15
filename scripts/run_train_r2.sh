#!/bin/bash
# Round-2：150条分拣数据 LoRA 微调（在 run_train_h100.sh 基础上改 OUTPUT_DIR 和 iters）
# 用法：bash run_train_r2.sh ~/data/sorting/sorting_meta_r2.json
#
# 150条 × ~360窗口 ≈ 5.4万窗口；batch=16 → 3375 it/epoch；iters=5000 ≈ 1.5 epoch
# H100 实测 ~0.25 s/it → 约 21 分钟
# 必须 accelerate launch --mixed_precision bf16，纯 python 跑 fp32 会 OOM

set -e

META="${1:?请传入 meta json 路径，如: bash run_train_r2.sh ~/data/sorting/sorting_meta_r2.json}"
MODEL_PATH="$HOME/data/X-VLA-Libero"
OUTPUT_DIR="$HOME/data/X-VLA-sorting-ckpt-r2"
XVLA_DIR="$HOME/data/X-VLA"

cd "$XVLA_DIR"

PYTHONPATH="$XVLA_DIR:$PYTHONPATH" \
~/miniconda3/envs/lingbot-vla/bin/accelerate launch \
  --num_processes 1 \
  --mixed_precision bf16 \
  peft_train.py \
  --models           "$MODEL_PATH" \
  --train_metas_path "$META" \
  --output_dir       "$OUTPUT_DIR" \
  --batch_size       16 \
  --iters            5000 \
  --freeze_steps     200 \
  --warmup_steps     300 \
  --learning_rate    2e-4 \
  --save_interval    500 \
  --log_interval     10 \
  --use_cosine_decay
