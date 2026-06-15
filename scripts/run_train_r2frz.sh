#!/bin/bash
# r2frz 实验：全程冻结 VLM + transformer 核心（freeze_steps=iters），只训 soft_prompts + action_heads
# 动机：r2bal 后 D 布局 rect/round 仍 0/8 全奔三角；VLM 此前按全额 2e-4 被改写，
#       怀疑语言 grounding 被冲掉。soft-prompt 是 X-VLA 设计的域适配通道，保骨干试一把。
# 注意：freeze 阶段 lr 恒定无 cosine 衰减（peft_train.py 的调度只在解冻后生效），靠 save_interval 选优。
# 用法：bash run_train_r2frz.sh ~/data/sorting/sorting_meta_r2_bal.json

set -e

META="${1:?请传入 meta json 路径}"
MODEL_PATH="$HOME/data/X-VLA-Libero"
OUTPUT_DIR="$HOME/data/X-VLA-sorting-ckpt-r2frz"
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
  --freeze_steps     5000 \
  --warmup_steps     300 \
  --learning_rate    2e-4 \
  --save_interval    500 \
  --log_interval     10 \
  --use_cosine_decay
