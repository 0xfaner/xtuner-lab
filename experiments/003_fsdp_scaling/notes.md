## 实验 003：FSDP 1 vs 2 卡 scaling（E3）

- **目的 / 假设**：同一台 2×vGPU-32G 上对照"单卡+offload（002 的 R2/R4）"与"双卡 FSDP2（本实验）"：装得下吗？缩放多少？策略 = FSDP2（PyTorch `fully_shard`；数据并行 + 参数/梯度/优化器状态分片；非 TP）。
- **实际命令**：`cmd_e3_1_17b_2x.sh`（1.7B 裸跑）→ `cmd_e3_3_4b_2x.sh`（4B 裸跑）→ `cmd_e3_2_17b_2x_offload.sh` → `cmd_e3_4_4b_2x_offload.sh`；模型配置复用 002（`torchrun --nproc-per-node 2`）。
- **关键数字**（原始日志 `log_e3_*.txt`；tgs/单步为末行·第 100 步口径；"GB"标签实为 GiB）：
  - e3_1 ✓ 100/100，4m04.204s：max 16.61 / reserved 21.40；tgs 2684.2/卡。
  - e3_2 ✓ 100/100，25m12.318s（双卡+offload）：max 5.15 / 6.75；每步均摊 15.12s ≈ R2 的 14.74s；全局消费 809,590 tokens（≈2×，两 rank 各一条数据流）。
  - e3_3 ✗ OOM（32.5s；4B 裸跑）：两 rank 均死在 `_single_tensor_adam` 更新临时量——Tried 742 MiB、剩 277.6 MiB（64GB 账本÷2 贴爆，差 ~0.45G）。
  - e3_4 ✓ 100/100，60m31.000s（4B+offload）：max 6.01 / 8.29；tgs 180.6。
- **踩坑与解决**：无。e3_2/e3_4 日志里的"报错关键字"均为无害警告（flash-attn 缺失→flex_attention 回退、bitsandbytes 提示、OMP/FutureWarning）。
- **结论 / 未解问题**：
  - 双卡分片让 1.7B 裸跑装下（reserved 21.40 < 31.48）且比"单卡+offload"快 ~6×（4m04 vs 24m34，含 offload 税 + 卡数效应）；4B 裸跑仍差 ~0.45G → offload 必要；4B+offload 双卡跑通（60m31s）。
  - offload 的时间税（同为双卡，纯对照）：e3_2 15.12s/步 vs e3_1 2.44s/步 = 6.2×；offload 场景加卡：单步延迟不变（瓶颈=CPU↔GPU 搬运），系统吞吐 ×2（每卡一条数据流）。
  - 未解：单卡 1.7B 无 offload 的每步基线（R1 未跑通，无法直接测）——e3_1 的"2.44s/步"是该场景唯一锚点。
- **运行记录**：work_dir 时间戳：e3_1 `20260920195539`、e3_3 `20260920200018`（失败保留）、e3_2 `20260920200413`、e3_4 `20260920203312`。
