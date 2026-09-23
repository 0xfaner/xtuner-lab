# E0 环境记录（2026-09-17）

- 机器：GuGu-Workstation（RTX 4080 SUPER 16GB）
- 驱动：580.178.04（CUDA 13.0）
- conda 环境：xtuner（Python 3.11.16）
- 仓库：~/workspace/xtuner（github.com/InternLM/xtuner），commit `6c06f96ef`
- torch：2.9.1+cu128（对齐官方 CI 版本；曾装 2.14.0+cu130 触发 DTensor bug，见 notes.md）；triton 3.5.1；torchvision 0.24.1+cu128
- transformers：5.14.1
- xtuner：0.2.0 editable（`pip install -e .`，指向 ~/workspace/xtuner）
- 安装备注：清华 conda / PyPI 镜像均 403，改用默认源成功；依赖均为预编译 wheel；RL 扩展 `.[rl]`（ray 2.58.0、checkpoint-engine 0.4.2、mooncake-transfer-engine 0.3.13）+ 补装 more_itertools、nvidia-ml-py；torch 用官方源 download.pytorch.org/whl/cu128
- 网络：HF 走 hf-mirror.com；旧环境 `xtuner-env`（xtuner 0.2.0 PyPI 旧版栈）已清除
