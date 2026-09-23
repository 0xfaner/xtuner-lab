"""E2 · Qwen3-4B 模型配置：从 HF config 推导 + 一处修正（compile）。"""
import os

from xtuner.v1.model import get_model_config_from_hf

MODEL_PATH = os.environ.get(
    "E2_4B_PATH",
    os.path.expanduser("~/workspace/xtuner-lab/models/Qwen3-4B"),
)

model = get_model_config_from_hf(MODEL_PATH)
model.compile_cfg = False
