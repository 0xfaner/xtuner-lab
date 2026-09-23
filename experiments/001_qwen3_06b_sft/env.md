# E1 环境记录（2026-09-18）

- 机器：GuGu-Workstation（RTX 4080 SUPER 16GB）
- 驱动：580.178.04（CUDA 13.0）
- conda 环境：xtuner（Python 3.11.16）
- 仓库：~/workspace/xtuner（commit `6c06f96ef`）；torch 2.9.1+cu128 / transformers 5.14.1
- 模型：Qwen/Qwen3-0.6B（hf-mirror 下载）→ `~/workspace/xtuner-lab/models/Qwen3-0.6B`
- 数据：llamafactory/alpaca_gpt4_zh（原始 42,677 条）→ 切 2000 train / 200 val（seed 42）
- 网络：HF_ENDPOINT=https://hf-mirror.com；训练命令见 `cmd.sh`
