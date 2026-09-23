#!/usr/bin/env bash
set -uo pipefail
LAB=/root/workspace/xtuner-lab
EXP=$LAB/experiments/002_qwen3_17b_4b_sft
LOG=$EXP/log_setup_part2.txt

{
echo "===== 重配开始 $(date) ====="

echo "== 1. 停掉慢任务 =="
pkill -f cloud_setup.sh 2>/dev/null
pkill -f "pip install torch" 2>/dev/null
pkill -f "tee log_setup" 2>/dev/null
sleep 2
ps -eo pid,cmd | grep -E "cloud_setup|pip install|tee log_setup" | grep -v grep || echo "(已清空)"

echo "== 2. xtuner-lab 搬到数据盘 + 软链接 =="
if [ ! -L "$LAB" ]; then
  if [ -e /root/autodl-tmp/xtuner-lab ]; then
    echo "数据盘已有 xtuner-lab（保留数据盘版）"; rm -rf "$LAB"
  else
    mv "$LAB" /root/autodl-tmp/xtuner-lab
  fi
  ln -s /root/autodl-tmp/xtuner-lab "$LAB"
fi
ls -ld "$LAB"; du -sh /root/autodl-tmp/xtuner-lab

echo "== 3. 依赖预装 =="
source /root/miniconda3/etc/profile.d/conda.sh
conda activate xtuner
pip install -q sympy mpmath networkx pillow jinja2 MarkupSafe fsspec filelock typing-extensions numpy packaging modelscope hf_transfer more_itertools nvidia-ml-py
echo "预装完成"

echo "== 4. 补 torch（依赖已被上一步满足，主件走 pip 缓存） =="
if ! python -c "import torch" 2>/dev/null; then
  pip install torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu128
fi
python -c "import torch; print('torch =', torch.__version__, '| cuda =', torch.cuda.is_available())"

echo "== 5. 启动模型下载（ModelScope 优先，后台）=="
mkdir -p "$LAB/models"
cat > /root/fetch_models.sh <<'EOS'
#!/usr/bin/env bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate xtuner
export MODELSCOPE_CACHE=/root/autodl-tmp/.mscache
LAB=/root/workspace/xtuner-lab
for M in Qwen3-1.7B Qwen3-4B; do
  echo "=== $M start $(date) ==="
  modelscope download --model Qwen/$M --local_dir $LAB/models/$M || {
    echo "modelscope 失败，回退 hf_transfer"
    HF_HUB_ENABLE_HF_TRANSFER=1 HF_ENDPOINT=https://hf-mirror.com hf download Qwen/$M --local-dir $LAB/models/$M
  }
  echo "=== $M done $(date) ==="
  du -sh $LAB/models/$M
done
echo ALL_MODELS_DONE
EOS
nohup bash /root/fetch_models.sh > "$EXP/log_models.txt" 2>&1 &
echo "models 下载已后台启动 pid=$!"

echo "== 6. 补 xtuner 安装 =="
cd /root/workspace/xtuner
pip install -q -e '.[rl]'
python -c "import xtuner, torch; print('xtuner OK ->', xtuner.__file__)"

echo "== 7. 模型下载速度采样 =="
a=$(du -sb "$LAB/models" 2>/dev/null | cut -f1); sleep 15; b=$(du -sb "$LAB/models" 2>/dev/null | cut -f1)
echo "15 秒增长 $((b-a)) 字节 => 约 $(( (b-a)/15/1048576 )) MB/s"
tail -3 "$EXP/log_models.txt" 2>/dev/null

echo "===== 重配结束 $(date) ====="
} 2>&1 | tee "$LOG"
