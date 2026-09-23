#!/usr/bin/env bash
set -uo pipefail

echo "== 1. nvidia-smi（真实卡型 / 驱动 / 每卡显存）=="
nvidia-smi
echo

echo "== 2. 设备清单与链路拓扑 =="
nvidia-smi -L || true
nvidia-smi topo -m || true
echo

echo "== 3. torch / CUDA / bf16（conda 环境 xtuner）=="
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate xtuner 2>/dev/null || true
python - <<'PY' || true
import torch
print("torch:", torch.__version__, "| cuda:", torch.version.cuda, "| available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
print("bf16 supported:", torch.cuda.is_bf16_supported())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"[{i}] {p.name} | {p.total_memory / 2**30:.0f} GB")
PY
echo
