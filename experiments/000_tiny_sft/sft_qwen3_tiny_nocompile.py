"""E0 校验配置：官方 tiny 配置的等价版本（仅关闭 compile）。"""
from xtuner.v1.model import Qwen3Dense8BConfig

model = Qwen3Dense8BConfig(num_hidden_layers=2, hidden_size=512)
model.compile_cfg = False
