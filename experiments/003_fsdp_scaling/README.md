# 003 · E3 实验现场：FSDP 1 vs 2 卡 scaling

- 与 E2（`../002_qwen3_17b_4b_sft/`）**同一场 session**（2×vGPU-32GB）：先跑完 002 的 R1–R5（固定单卡），再在这里跑 E3（双卡），最后一起关机。
- 模型配置复用 002 的（实验对象相同）：`../002_qwen3_17b_4b_sft/config/e2_model_17b.py`、`e2_model_4b.py`
- 运行顺序：`cmd_e3_1_17b_2x.sh`（主对照）→ `cmd_e3_3_4b_2x.sh`（边界点）→ 选做 `cmd_e3_2_*`、`cmd_e3_4_*`
- 日志：`log_e3_*.txt`（原始输出，不删不改）；环境记录：`env.md`；实验笔记：`notes.md`
- 上传/收日志：本机 `bash ../autodl.sh push` / `bash ../autodl.sh pull`（一次会带上本目录；用法：`bash ../autodl.sh`）
