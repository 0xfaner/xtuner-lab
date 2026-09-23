#!/usr/bin/env bash
set -euo pipefail

LAB=~/workspace/xtuner-lab
export HF_ENDPOINT=https://hf-mirror.com
export PATH=$HOME/miniconda3/envs/xtuner/bin:$PATH
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate xtuner
ts() { date +%H:%M:%S; }
trap 'ec=$?; echo; echo "[$(ts)] !! 脚本中止：行 $LINENO（退出码 $ec）" >&2; echo "!! 重跑本脚本会从断点继续" >&2' ERR

echo "[$(ts)] == 0. 机器现状 =="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
df -h /dev/shm | tail -1
df -h /root/autodl-tmp | tail -1
echo

echo "[$(ts)] == 1. lmdeploy 推理引擎（要求 >=0.10.2） =="
if python -c "import lmdeploy; from packaging.version import Version; assert Version(lmdeploy.__version__) >= Version('0.10.2')" 2>/dev/null; then
  echo "（已就位，跳过安装）"
else
  pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple >/dev/null 2>&1 || true
  pip install "lmdeploy>=0.10.2"
fi
python - <<'EOF'
import lmdeploy, torch
print("lmdeploy:", lmdeploy.__version__, "| torch:", torch.__version__)
assert torch.__version__.startswith("2.9.1"), "!! torch 被改了，需重装 2.9.1（见 cloud_setup.sh）"
EOF
echo

echo "[$(ts)] == 2. Qwen3-0.6B 模型（ModelScope） =="
export MODELSCOPE_CACHE=/root/autodl-tmp/.mscache
if [ -f "$LAB/models/Qwen3-0.6B/config.json" ] && ls "$LAB/models/Qwen3-0.6B"/*.safetensors >/dev/null 2>&1; then
  echo "（已存在，跳过）"
else
  modelscope download --model "Qwen/Qwen3-0.6B" --local_dir "$LAB/models/Qwen3-0.6B"
fi
ls "$LAB/models/Qwen3-0.6B" | head -8
echo

echo "[$(ts)] == 3. 数据检查 =="
wc -l "$LAB/data/gsm8k_train.jsonl" "$LAB/data/gsm8k_val.jsonl"
python - <<'EOF'
import json
row = json.loads(open("/root/workspace/xtuner-lab/data/gsm8k_train.jsonl").readline())
assert row["reward_model"]["ground_truth"] is not None
print("数据首行字段 OK:", sorted(row.keys()))
EOF
echo

echo "[$(ts)] == 4. 预检（导入级 + 模型配置） =="
python - <<'EOF'
import ray, lmdeploy
print("ray:", ray.__version__, "| lmdeploy:", lmdeploy.__version__)
from xtuner.v1.model import get_model_config_from_hf
from pathlib import Path
cfg = get_model_config_from_hf(Path.home() / "workspace/xtuner-lab/models/Qwen3-0.6B")
print("0.6B model cfg:", type(cfg).__name__, "| tie(原始):", cfg.tie_word_embeddings)
EOF
echo "[$(ts)] == e4_prep 完成 ✅ =="
