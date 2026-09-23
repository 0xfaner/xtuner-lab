## 实验 000：官方 tiny SFT 校验（E0）

- **目的 / 假设**：在 4080 本机跑通 XTuner V1 官方安装校验（13 步玩具模型），验证环境可用。
- **实际命令**：见 `cmd.sh`（官方命令的等价版：仅改用本地配置关闭 compile，其余参数与官方一致）。
- **关键数字**：13 步全部完成；loss 11.91 → 9.08；max_memory 5.35GB / reserved 7.43GB；训练耗时 7.1s；checkpoint（HF + DCP）已保存至 `work_dir/20260917173327/`。
- **踩坑与解决（4 次失败，全部留档）**：
  1. `ModuleNotFoundError: No module named 'ray'` —— 官方 `pip install -e .` 不含 RL 依赖，但 SFT 路径也会 import RL utils。解决：`pip install -e '.[rl]'`（ray 2.58.0 / checkpoint-engine 0.4.2 / mooncake-transfer-engine 0.3.13）。
  2. `ModuleNotFoundError: No module named 'more_itertools'` —— pyproject 未声明（连带的 `nvidia-ml-py` 一并补装）。
  3. `AttributeError: Compiling Error! Cannot locate the function: ...per_block_quant_gemm.per_block_quant_torch` —— `triton_kernels/__init__.py` 中同名符号（CustomOpDef）覆盖子模块属性，`pydoc.locate` 无法解析；`DEFAULT_FLOAT8_CFG` 中该条目必挂。解决（官方先例）：模型配置 `compile_cfg=False`（官方 `sft_glm5p2.py` 默认同样关闭 compile）。**官方文档原命令在 HEAD 6c06f96 上无法直接通过。**
  4. `RuntimeError: '>=' not supported between instances of 'torch.dtype' and 'int'`（torch 2.14 DTensor `_foreach_norm` 分片传播 bug）——官方 CI / 镜像默认 torch 2.9.1，pip 自由解析会装到 2.14。解决：`pip install torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu128`。
- **结论 / 未解问题**：
  - 环境跑通：xtuner（editable，`~/workspace/xtuner`）+ torch 2.9.1+cu128 + transformers 5.14.1；4080 单卡全流程（训练 → HF/DCP 保存）正常。
  - 工程观察：官方 quickstart 假设用户使用其镜像（torch 2.9.1 / lmdeploy 0.17 等固定组合），"干净 pip 环境"并不能一把过；问题 1、2 属依赖声明缺失，问题 3 属 HEAD 上的真实 bug。
  - 待确认：是否将问题 1–3 整理成 issue 上报。
- **日志清单**：`log.txt`（成功）；`log_attempt1_fail_ray_missing.txt`、`log_attempt2_fail_more_itertools.txt`、`log_attempt3_fail_compile_locate.txt`、`log_attempt4_fail_foreach_norm_torch214.txt`（失败现场）。

## 附加实验：步数扫描（09-17 晚）

- 动机：实测"步数加大后 loss 如何变化"。
- 结果：同一玩具配置（3 层 / 512）分别跑 13 / 100 / 1000 步：

| 步数 | 末步 reduced_llm_loss | 耗时 | 记录目录（work_dir/） |
|---|---|---|---|
| 13 | 9.0783 | 4.9s | 20260917184020 |
| 100 | 6.3637 | 19.1s | 20260917184438 |
| 1000 | 0.0062 | 164.6s | 20260917184518 |

- 解读：1000 步 ≈ 同样 13 条打包样本被重复约 77 遍 → loss→0.006 属**记忆 / 过拟合**现象（困惑度 e^0.006 ≈ 1.006，模型几乎逐字背下训练样本）；训练 loss 低 ≠ 泛化能力。
- 复现：`cmd.sh` 现为 1000 步版本；13 步版本 = 去掉 `--total-step` 参数。

## 对照实验：层数 3→2（09-17 晚）

- 变量：`num_hidden_layers` 3→2（其余不变；同数据、同 100 步）。
- 结果对照（均 100 步）：

| 指标 | 3 层（v1） | 2 层（v2） |
|---|---|---|
| 参数量 | 227.94M | 203.82M（−24.1M ≈ 恰好一层） |
| 末步 reduced_llm_loss | 6.3637 | 6.4069 |
| 耗时 | 19.1s | 16.9s |
| 峰值显存 | 5.35 GB | 5.08 GB |

- 观察：深度减少带来参数/耗时/显存的同步下降，但 loss 差异仅约 0.04（可忽略）——在玩具设置（13 条数据、100 步）下"深度"的影响几乎不显形；想观察深度的真实影响需要真模型/真数据。
- 备注：18:52 有一次 1000 步的 v2 试跑被中途停止（日志保留、无产物），正式结果以 18:58 的 100 步运行为准。
- 记录目录：`work_dir/20260917185759`；脚本 `cmd_v2.sh`。
