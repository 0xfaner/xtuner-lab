"""E1 模型配置：从 HF config 自动推导，修正 compile 与 tie_word_embeddings 两处。"""
import os

from xtuner.v1.model import get_model_config_from_hf

MODEL_PATH = os.environ.get(
    "QWEN3_0P6B_PATH",
    os.path.expanduser("~/workspace/xtuner-lab/models/Qwen3-0.6B"),
)

model = get_model_config_from_hf(MODEL_PATH)
model.tie_word_embeddings = False
model.compile_cfg = False
