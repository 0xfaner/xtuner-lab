# 实验 004：GRPO on GSM8K（E4）+ 1.7B 拉伸档 + 训推 mismatch（E5）

- **目的 / 假设**：0.6B 单卡（vGPU-32G）跑通 GRPO（lmdeploy 引擎与训练共卡）；观察 reward、显存与时间分布；E5 在同一套代码上做纯测量（`ONLY_CALC_MISMATCH_RATIO=1`，不更新权重）：量一量同一权重下 rollout 引擎与训练引擎两条数值路径的差。
- **实际命令**：`cmd_e4_prep.sh`（装 lmdeploy + 下模型）→ `cmd_e4_rl.sh`（主实验，12 步）→ `cmd_e4_mismatch.sh`（E5，3 步纯测量）→ 拉伸档 `cmd_e4_rl_17b_offload.sh`（1.7B，3 步）。

## E4 主实验（work_dir/20260921144842）

- 12/12 步全过；**eval 0.28 → 0.38**（100 题口径）；每步 reward_mean 0.05 ~ 0.50（末步 0.40）。
- GRPO 代理损失 −0.01 ~ 0.10（不是交叉熵，量纲与 SFT 的 loss 不同——均值天然贴近 0、可正可负）；grad_norm 0.37 ~ 1.76。
- 峰值显存 18.86 GiB / reserved 24.17（首步 13.23）；整场 **14m17s**。
- 时间分布（12 步均值）：**生成 ≈ 80% / 训练 ≈ 16% / 权重同步 ≈ 4%**（生成 39.6s、训练 7.7s、同步 2.0s 每步）；权重同步首步 3.79s，此后稳定 1.80 ~ 2.36s。
- 吞吐（step 12 口径）：rollout 822.3 tgs / training 3825.3 tgs / e2e 619.4 tgs。
- 回答长度：mean 791 / min 275 / max 1024；step 12 的 40 条里 15 条撞 1024 截断（37.5%）。
- 共卡显存跷跷板（外部采样，每 60s 一条）：生成段引擎驻留 ≈15.4G + 训练侧 0.3~1.5G；训练瞬间引擎 0.4G、训练侧 25.1~25.7G。

### 预测 vs 实际（跑前写的预测）

| 预测 | 实际 | 差在哪 |
|---|---|---|
| loss 2.0~3.0 → 1.5~2.0 | 每步 −0.01 ~ 0.10 | 两个 loss 不是一回事：GRPO 的 loss 是策略梯度代理目标 |
| 训练后小幅提升 | 0.28 → 0.38（+0.10） | 方向对了 |
| 生成 30% / 训练 60% / 同步 10% | 生成 ≈80% / 训练 ≈16% / 同步 ≈4% | 差最多的一项：生成远大于预期 |
| 整场 20min 以内 | 14m17s | ✓ |
| 峰值显存 15~20G | 训练段 18.86 GiB | ✓ |

### 踩坑与解决

1. **权重同步 HTTP 500（`pidfd_getfd: Operation not permitted`）**：`run_rl.sh` 给 lmdeploy 分支默认开 `expandable_segments`；该分配器下跨进程共享张量要走 fd 路径（`pidfd_getfd`），而 AutoDL 容器 seccomp 拒绝该 syscall。处置：`run_rl_lab.sh` 允许用 `XTUNER_CUDA_ALLOC_CONF` 覆盖为 `False`（回到经典 CUDA-IPC 句柄）。最小复现在 `scripts/share_test.py`（A 默认 / B expandable / C 新旧变量名，三组环境对照）。
2. **1.7B 拉伸档：全量 cpu_offload 路线连撞四关**（见下节）。

## 1.7B 拉伸档（最终走 swap 路线）

单卡 32G 装不下 1.7B 的"参数 + 梯度 + 优化器状态"（≈24GB 三件套），先试全量 `cpu_offload`，前后 6 枪失败：

- 梯度裁剪：范数标量在 CPU，通信组是 NCCL（只收 GPU 张量）→ 补丁①；
- 权重收集：FSDP 分片在 CPU，all_gather 同样上 NCCL → 补丁②；
- 反向 OOM：训练侧 ~15.6G + 优化器 onload ~13.6G + 引擎让位 0.4G ≈ 29.6/31.5G，分配器缓存 8.0G 全碎、凑不出 1.16G 整块 → 补丁③（训练步入口 `empty_cache`）；
- 优化器设备混用：普通 AdamW 与 SwapAdamW 两套实现都不覆盖"param/grad 在 CPU"的情形 → 判定该组合无解。

换官方设计组合：**param/grad 留 GPU、优化器状态由 SwapAdamW 常驻 CPU 分块换入**（`FSDP_CPU_OFFLOAD=0` + `SWAP_OPTIMIZER=1`），第 7 枪一次跑通：3/3 步、4m29s、每步 40~45s（生成 27~29 / 训练 10~15 / 同步 2.6）、max 19.80 GiB。（`rewards/mean` 三步全 0——10 样本/步的冷启动，样本太小，不作结论。）

三个本地补丁在 `patches/`（未提交上游；上游 main 同款代码无修复）：

- `dtensor_cpu_offload_gradnorm.patch`（修复型）：`cal_total_norm` 里 all_reduce 前把标量挪到 mesh 设备、归约后挪回；
- `loadspec_cpu_offload_allgather.patch`（修复型）：`_foreach_all_gather_save_shards` 里 all_gather 前把 CPU 分片挪到计算设备、收集后挪回；
- `worker_cpu_offload_emptycache.patch`（诊断型）：训练步入口清分配器缓存；swap 路线下不触发，仅作排障脚手架保留。

## E5 训推 mismatch 纯测量（3 步）

| 来源 | step | mismatch_kl | ppl_ratio | logprob_abs_diff |
|---|---|---|---|---|
| E4（训练中） | 1–12（范围） | 1.2e-4 ~ 9.3e-4 | 1.0001 ~ 1.0009 | 0.0139 ~ 0.0177 |
| E5（纯测量） | 1 | 0.001296 | 1.001297 | 0.017918 |
| E5（纯测量） | 2 | 0.000231 | 1.000231 | 0.015245 |
| E5（纯测量） | 3 | 0.000849 | 1.000849 | 0.014743 |

- 量级稳定：kl 1e-4 ~ 1e-3、逐 token logprob 差 0.014 ~ 0.018、ppl_ratio 离 1 不超过 0.13%；数值上 ppl_ratio − 1 ≈ kl。
- 纯测量 3 步用时 39.1 / 36.1 / 27.9s（其中生成 33.1 / 32.8 / 26.1s），全场 3m47s。

## 运行记录

- 主实验 work_dir：`20260921144842`（另有早期失败现场 `20260921143556`）；E5：`work_dir_mismatch/20260921151940`；拉伸档：`work_dir_17b/`（7 次尝试全部留档）。这些目录未随仓库发布。
- 轨迹样例（step 12 的 40 条：16 对 / 24 错，含三条完整样例）：`trajectory_samples.md`。
- 失败现场日志：`log_e4_rl_attempt1_wsync500.txt`、`log_e4_rl_17b_attempt1~6*.txt`；成功日志：`log_e4_rl.txt`、`log_e4_rl_17b.txt`、`log_e4_mismatch.txt`。
