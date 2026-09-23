# E2 云环境记录（主 session 09-20 跑毕；R6 未启用）

- 主 session：AutoDL · 重庆A区
- R6 场（1×A800-80G，北京B区）：不独立开场，顺路（E4 切换时并场）
- GPU：主 = vGPU-32GB ×2（实锤 RTX 4080 SUPER 32G 魔改切片；驱动 595.71.05；PHB 互联、无 NVLink；明细见 log_check.txt）；R6 = A800-80G（未启用）
- 系统 / 镜像：Miniconda conda3 / Python 3.10（ubuntu22.04）CUDA 11.8；实例规格 32 vCPU / 124GB 内存 / 系统盘 30G + 数据盘 50G（可扩容）
- conda 环境：xtuner（Python 3.11）；torch 2.9.1+cu128（脚本对齐）
- 仓库：~/workspace/xtuner，commit `6c06f96ef0d28efb0781a90ed661973fd9938859`
- 模型：Qwen3-1.7B（约 4G）/ Qwen3-4B（约 8G+），ModelScope 下载（实测 ~6-9MB/s）
- 磁盘布局：实验目录已迁至数据盘 /root/autodl-tmp/xtuner-lab，~/workspace/xtuner-lab 为软链接
- 数据：train.jsonl / val.jsonl（E1 的 2000/200，scp 上传）
- 备注：双卡链路留档 `nvidia-smi topo -m`
