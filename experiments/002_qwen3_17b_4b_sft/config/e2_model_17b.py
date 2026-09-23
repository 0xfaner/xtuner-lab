"""E2 · Qwen3-1.7B 模型配置：从 HF config 推导 + 两处修正（compile、tie_word_embeddings）。"""
import os

from xtuner.v1.model import get_model_config_from_hf

MODEL_PATH = os.environ.get(
    "E2_17B_PATH",
    os.path.expanduser("~/workspace/xtuner-lab/models/Qwen3-1.7B"),
)

model = get_model_config_from_hf(MODEL_PATH)
model.tie_word_embeddings = False
model.compile_cfg = False
