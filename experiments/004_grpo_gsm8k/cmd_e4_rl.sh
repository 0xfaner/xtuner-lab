#!/usr/bin/env bash
set -euo pipefail

LAB=~/workspace/xtuner-lab
EXP_DIR=$LAB/experiments/004_grpo_gsm8k

export PATH=$HOME/miniconda3/envs/xtuner/bin:$PATH
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate xtuner

cd ~/workspace/xtuner
export HF_ENDPOINT=https://hf-mirror.com
export XTUNER_USE_FA3=0
export CUDA_VISIBLE_DEVICES=0

export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
export TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-12}
export ENABLE_EVALUATE=${ENABLE_EVALUATE:-1}
export WORK_DIR=$EXP_DIR/work_dir
export XTUNER_CUDA_ALLOC_CONF=expandable_segments:False

bash "$EXP_DIR/run_rl_lab.sh" \
  "$EXP_DIR/config/e4_grpo_06b.py" \
  lmdeploy \
  "$LAB/models/Qwen3-0.6B" \
  "$LAB/data/gsm8k_train.jsonl" \
  "$LAB/data/gsm8k_val.jsonl"
