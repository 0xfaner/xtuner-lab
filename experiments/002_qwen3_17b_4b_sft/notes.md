## 实验 002：Qwen3-1.7B / 4B 云上 SFT（E2）

- **目的 / 假设**：云上单卡（vGPU-32G = RTX 4080 SUPER 魔改切片，每卡 32760 MiB）做显存账本实验——16 B/参数（权重 2 + 梯度 2 + 优化器两矩 8 + fp32 主副本 4）→ 1.7B≈27GB / 4B≈64GB；验证"裸跑 OOM 边界"与"offload 的代价值不值"。
- **实际命令**：R1 `cmd_r1_17b.sh` → R2 `cmd_r2_17b_offload.sh` → R3 `cmd_r3_17b_offload_nopack.sh` → R4 `cmd_r4_4b_offload.sh`；R5 已砍（信息与 R1 同质）；R6（A800 顺路场）未做。模型配置 `config/e2_model_17b.py`、`config/e2_model_4b.py`（两处修正：关 compile、`tie_word_embeddings=False`）。
- **关键数字**（原始日志 `log_*.txt`；日志里 max/reserved 标签为 GB、实为 GiB；tgs/单步为末行·第 100 步口径）：
  - R1 ✗ OOM（32s）：死在第一次优化器更新（adam `state["exp_avg_sq"] = zeros_like`）——Tried 1.16 GiB、剩 1.13 GiB；单笔差 ~30MB 是最后一根稻草（其后仍欠同尺寸更新临时量，结构缺口 ≥1GiB 量级）。崩时 30.33/31.48 GiB，日志无 Step 输出。
  - R2 ✓ 100/100，24m34s：max 4.57 / reserved 6.77；tgs 294.4。
  - R3 ✓ 100/100，25m37s（no-pack）：max 3.98 / 6.18；tgs 13.9；消费 13,938 tokens（≈R2 的 1/29，R2 为 404,680）。末尾 DCP 存档因磁盘满中断（HF 存档先成功）。
  - R4 ✓ 100/100，55m41s（4B）：max 5.28 / 8.21；tgs 110.9。同"磁盘满存档中断"（HF 成功）。
  - 权重：R2 `work_dir/20260920164728/hf-100`、R3 `20260920172200/hf-100`（≈4G）、R4 `20260920180100/hf-100`（≈8G，2 分片）；R1 无（`20260920164105` 失败保留）。
  - val（200 条 held-out；assistant-only token 加权交叉熵）：1.7B 基座 2.1898 → 100 步 1.6553；4B 基座 2.0788 → 100 步 1.6671。
- **踩坑与解决**：
  1. 磁盘写爆 ×2：全量 DCP（1.7B≈23G / 4B 截断 24G）塞爆盘 → 两次 `clean-dcp` + 数据盘扩容 50→250G（19:23 重启生效）。
  2. R3/R4 存档报 `unexpected pos ...`（inline_container）——根因 ENOSPC，非代码问题；HF 权重完好。
  3. 基础设施坑（pip 源 / ModelScope / 隧道 / commit 笔误）见 `cloud_setup.sh`、`autodl.sh`、`snapshot_before_reboot.txt`。
- **结论 / 未解问题**：
  - 1.7B 单卡裸跑贴爆（缺 ≥1GiB 量级）；offload 让 1.7B/4B 均可在单卡跑完，但时间税显著（tgs 294 vs 不 offload 场景 2684/卡，见 003）。
  - 4B 单卡账本 ≈64GB——必 offload；offload 后 55m41s。
  - 未解（顺路场）：R6——"不缺显存时该不该 offload"的大卡对照（并入后续 A800 场；未做则列未做项）。
- **运行记录**：work_dir 时间戳：R1 `20260920164105`（失败保留）、R2 `20260920164728`、R3 `20260920172200`、R4 `20260920180100`。
