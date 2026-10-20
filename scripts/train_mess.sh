#!/usr/bin/env bash
# MESS training, paper setting: 8 GPUs x 8 img = total batch 64, 512 crops,
# 200K iters, AdamW lr 1e-4 -> 1e-6 cosine, lambda_con = 1.
set -e
cd "$(dirname "$0")/.."

DATA_ROOT=${DATA_ROOT:-/local/mnt/workspace/ruodcui/data/MESS_RAW}
NGPU=${NGPU:-8}

# 1) build manifest once (RAW renders are assumed to exist already)
[ -f "$DATA_ROOT/train.jsonl" ] || python tools/build_mess_manifest.py --data_root "$DATA_ROOT" --out "$DATA_ROOT/train.jsonl"

# 2) train
accelerate launch --multi_gpu --num_processes "$NGPU" --mixed_precision bf16 src/train_mess.py \
  --train_manifest "$DATA_ROOT/train.jsonl" \
  --output_dir output/mess \
  --crop_size 512 \
  --train_batch_size 8 --total_batch_size 64 \
  --max_train_steps 200000 \
  --learning_rate 1e-4 --min_learning_rate 1e-6 \
  --lambda_con 1.0 \
  --mixed_precision bf16 --allow_tf32 \
  --enable_xformers_memory_efficient_attention --gradient_checkpointing \
  --dataloader_num_workers 8 \
  --checkpointing_steps 5000 --viz_steps 2000 "$@"
