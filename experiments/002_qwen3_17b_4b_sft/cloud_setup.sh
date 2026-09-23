#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT=https://hf-mirror.com
XTUNER_COMMIT=6c06f96ef0d28efb0781a90ed661973fd9938859
LAB=~/workspace/xtuner-lab
LAB_DATA=/root/autodl-tmp/xtuner-lab
ts() { date +%H:%M:%S; }

trap 'ec=$?; echo; echo "[$(ts)] !! 脚本中止：行 $LINENO（退出码 $ec）；失败命令：$BASH_COMMAND" >&2; echo "!! 重跑本脚本会从断点继续（已完成步骤自动跳过）" >&2' ERR

echo "[$(ts)] == 0. 机器检查 =="
nvidia-smi -L
df -h / | tail -1
df -h /dev/shm | tail -1
echo

echo "[$(ts)] == 0.5 实验目录安置到数据盘（幂等兜底） =="
mkdir -p "$(dirname "$LAB")" /root/autodl-tmp
if [ ! -L "$LAB" ]; then
  if [ -d "$LAB" ]; then
    rsync -a "$LAB/" "$LAB_DATA/" 2>/dev/null || cp -a "$LAB/." "$LAB_DATA/"
    rm -rf "$LAB"
  fi
  mkdir -p "$LAB_DATA"
  ln -s "$LAB_DATA" "$LAB"
fi
ls -ld "$LAB"
rm -f "$LAB/.setup_done"
echo

echo "[$(ts)] == 1. conda 环境 =="
if [ -d "$HOME/miniconda3" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "（镜像里没有 conda，下载安装 Miniconda）"
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
if conda env list | grep -qw xtuner; then
  echo "（xtuner 环境已存在，跳过创建）"
else
  conda create -y -n xtuner python=3.11
fi
conda activate xtuner
python -V | grep -q "3.11" || { echo "!! conda activate 后 Python 不是 3.11（环境异常）"; exit 1; }
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple >/dev/null 2>&1 || true
python -V
echo

echo "[$(ts)] == 2. torch 2.9.1+cu128（对齐官方 CI；清华镜像 → 官方源回退） =="
if python -c "import torch; assert torch.__version__.startswith('2.9.1')" 2>/dev/null; then
  echo "（torch 2.9.1 已就位，跳过安装）"
else
  pip install -q filelock sympy networkx jinja2 fsspec typing-extensions setuptools || true
  pip install torch==2.9.1 torchvision --index-url https://mirrors.tuna.tsinghua.edu.cn/pytorch-wheels/cu128 \
    || { echo "（清华镜像失败，回退官方源，速度会慢）"; pip install torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu128; }
fi
python - <<'EOF'
import torch
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(), "| 可见卡数:", torch.cuda.device_count())
assert torch.cuda.is_available(), "CUDA 不可用！检查驱动或安装"
EOF
echo

echo "[$(ts)] == 3. xtuner（commit $XTUNER_COMMIT） =="
mkdir -p ~/workspace && cd ~/workspace
if [ ! -d xtuner ]; then
  if [ -f /etc/network_turbo ]; then source /etc/network_turbo; else echo "!! 没有学术加速脚本，GitHub 直连大概率失败"; fi
  ok=0
  for i in 1 2 3; do
    echo "（clone 第 $i 次尝试）"
    if git clone https://github.com/InternLM/xtuner.git; then ok=1; break; fi
    rm -rf xtuner
  done
  unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY 2>/dev/null || true
  if [ "$ok" != 1 ]; then
    echo "!! GitHub clone 三次失败。备选：本机 rsync 上传本地仓库"
    exit 1
  fi
else
  echo "（仓库已存在，跳过 clone）"
fi
cd xtuner
git -c advice.detachedHead=false checkout "$XTUNER_COMMIT"
git log --oneline -1
pip install -e '.[rl]'
pip install -q more_itertools nvidia-ml-py
if ! python -c "from mmengine.utils.package_utils import is_installed; is_installed('numpy')" 2>/dev/null; then
  echo "（mmengine 修订版不兼容 → 从 PyPI 重装）"
  pip install -q -U --force-reinstall --no-deps -i https://pypi.org/simple "mmengine==0.11.0rc2"
  python -c "from mmengine.utils.package_utils import is_installed; is_installed('numpy')" \
    || { echo "!! mmengine 修复失败（PyPI 不可达？）——备选：手工替换 package_utils.py"; exit 1; }
fi
python -c "import xtuner, torch; print('xtuner:', xtuner.__file__, '| torch:', torch.__version__)"
echo

echo "[$(ts)] == 4. 模型下载（ModelScope 优先；已完整则跳过） =="
export MODELSCOPE_CACHE=/root/autodl-tmp/.mscache
pip install -q -U modelscope "huggingface_hub[cli]" hf_transfer
mkdir -p "$LAB/models"
for M in ${XTUNER_MODELS:-"Qwen3-1.7B Qwen3-4B"}; do
  if [ -f "$LAB/models/$M/config.json" ] && ls "$LAB/models/$M"/*.safetensors >/dev/null 2>&1; then
    echo "$M：已存在，跳过（若怀疑不完整：rm -rf 该模型目录后重跑）"
    continue
  fi
  echo "$M：开始下载 ..."
  modelscope download --model "Qwen/$M" --local_dir "$LAB/models/$M" \
    || HF_HUB_ENABLE_HF_TRANSFER=1 hf download "Qwen/$M" --local-dir "$LAB/models/$M"
done
du -sh "$LAB/models"/* 2>/dev/null || true
echo

echo "[$(ts)] == 5. 数据检查 =="
if ls "$LAB/data/train.jsonl" "$LAB/data/val.jsonl" "$LAB/data/gsm8k_train.jsonl" >/dev/null 2>&1; then
  echo "数据就位（SFT 训练/验证 + GSM8K）"
else
  echo "⚠️ 数据不全：回到本机执行 bash autodl.sh push"
fi
echo
echo "=============================================="
echo "[$(ts)] == setup 完成 ✅（总用时 $((SECONDS / 60)) 分钟） =="
echo "== 下一步（本机）：bash autodl.sh run e4_prep =="
echo "=============================================="
touch "$LAB/.setup_done"
