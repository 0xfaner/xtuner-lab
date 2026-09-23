#!/usr/bin/env bash
set -euo pipefail

cd ~/workspace/xtuner
export HF_ENDPOINT=https://hf-mirror.com
export XTUNER_USE_FA3=0

~/miniconda3/envs/xtuner/bin/torchrun xtuner/v1/train/cli/sft.py \
  --model-cfg ~/workspace/xtuner-lab/experiments/000_tiny_sft/sft_qwen3_tiny_nocompile.py \
  --chat_template qwen3 \
  --dataset tests/resource/openai_sft.jsonl \
  --work-dir ~/workspace/xtuner-lab/experiments/000_tiny_sft/work_dir \
  --total-step 100
