#!/usr/bin/env bash
set -euo pipefail

EXP_DIR=~/workspace/xtuner-lab/experiments/001_qwen3_06b_sft
MODEL_DIR=~/workspace/xtuner-lab/models/Qwen3-0.6B

cd ~/workspace/xtuner
export HF_ENDPOINT=https://hf-mirror.com
export XTUNER_USE_FA3=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

~/miniconda3/envs/xtuner/bin/torchrun --nproc-per-node 1 xtuner/v1/train/cli/sft.py \
  --model-cfg "$EXP_DIR/config/e1_model_cfg.py" \
  --load-from "$MODEL_DIR" \
  --chat_template qwen3 \
  --dataset "$EXP_DIR/data/train.jsonl" \
  --total-step 200 \
  --work-dir "$EXP_DIR/work_dir"
