# xtuner-lab

2026-09-17 ~ 09-21，在一台 4080S 工作站和 AutoDL 云实例上从零跑通 XTuner v1（`InternLM/xtuner`，commit `6c06f96`）的全过程记录：SFT → FSDP scaling → GRPO → 训推一致性。命令脚本、配置、patch、原始日志与实验笔记都在这里。

- 本机：RTX 4080 SUPER 16G（单卡）
- 云端：2×vGPU-32G（AutoDL；实为 RTX 4080 SUPER 切片，PHB 互联、无 NVLink）

## 实验索引

| # | 实验 | 环境 | 一句话结果 |
|---|---|---|---|
| 000 | 官方 tiny SFT 校验 | 本机单卡 | 跑通；官方安装命令在 HEAD 上要修 4 处（依赖声明 ×2、compile locate bug、torch 版本），4 个失败现场全部留档 |
| 001 | Qwen3-0.6B 全参 SFT（Alpaca-GPT4-zh，2000 条） | 本机单卡 | 验证集 loss 先降后升：2.19 →（≈1 遍）1.81 →（≈3 遍）2.33——过拟合 + 早停的证据链 |
| 002 | Qwen3-1.7B / 4B SFT（显存账本实验） | 云端单卡 | 1.7B 裸跑差 ~1GiB OOM；offload 后 1.7B/4B 都跑通，但 tgs 降到 294（1.7B）；4B 单卡必 offload（55m41s/100 步） |
| 003 | FSDP2 单卡 vs 双卡 scaling | 云端双卡 | 双卡让 1.7B 裸跑装下（2684 tgs/卡，比单卡+offload 快 ~6×）；4B 双卡裸跑仍差 ~0.45G |
| 004 | GRPO on GSM8K（0.6B 共卡 + lmdeploy）＋ 1.7B 拉伸档 ＋ E5 训推 mismatch | 云端单卡 | 12 步：eval 0.28 → 0.38；1.7B 经 3 个本地 patch 后改 swap 路线跑通；mismatch：kl 1e-4~1e-3、ppl_ratio 偏离 <0.13% |

每个目录里：`cmd_*.sh`（实际运行命令）、`config/`（模型/训练配置）、`log_*.txt`（原始日志，不删不改，含失败现场）、`notes.md`（结果、数字、坑与结论）。004 另有 `patches/`（3 个本地补丁）与 `trajectory_samples.md`（rollout 样例）。

## 环境与复现前提

- XTuner：clone 到 `~/workspace/xtuner`，checkout `6c06f96`，`pip install -e '.[rl]'`
- torch 2.9.1+cu128（更新的 2.14 会触发 DTensor `_foreach_norm` bug，见 `experiments/000_tiny_sft/notes.md`）
- conda 环境 `xtuner`（Python 3.11）；torchrun 用该环境绝对路径调用（`~/miniconda3/envs/xtuner/bin/torchrun`）
- 模型放 `~/workspace/xtuner-lab/models/`：Qwen3-0.6B / 1.7B / 4B
- 数据（不随仓库发布）放 `~/workspace/xtuner-lab/data/`：
  - SFT（001–003）：alpaca-gpt4-zh 原始 json → `python experiments/001_qwen3_06b_sft/scripts/prepare_data.py`（2000/200，seed 42）
  - RL（004）：`gsm8k_train.jsonl` / `gsm8k_val.jsonl`（verl 格式，含 `reward_model.ground_truth`）
- 云端流程：`bash autodl.sh connect <host> <port>` → `push` → `setup` → `run <任务>` → `pull`（云端 bootstrap 是 002 的 `cloud_setup.sh`）

脚本里的路径按上面的目录约定写；在本机对应位置放好模型/数据后，进入对应实验目录 `bash cmd_*.sh` 即可。
