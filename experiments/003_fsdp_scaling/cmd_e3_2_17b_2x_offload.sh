#!/usr/bin/env bash
set -euo pipefail

LAB=~/workspace/xtuner-lab
EXP_DIR=$LAB/experiments/003_fsdp_scaling
CFG_DIR=$LAB/experiments/002_qwen3_17b_4b_sft/config

cd ~/workspace/xtuner
export HF_ENDPOINT=https://hf-mirror.com
export XTUNER_USE_FA3=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

~/miniconda3/envs/xtuner/bin/torchrun --nproc-per-node 2 xtuner/v1/train/cli/sft.py \
  --model-cfg "$CFG_DIR/e2_model_17b.py" \
  --load-from "$LAB/models/Qwen3-1.7B" \
  --chat_template qwen3 \
  --dataset "$LAB/data/train.jsonl" \
  --total-step 100 \
  --pack-max-length 4096 \
  --fsdp-config.cpu-offload \
  --work-dir "$EXP_DIR/work_dir"
