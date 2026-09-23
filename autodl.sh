#!/usr/bin/env bash
set -euo pipefail

LAB=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENVF="$LAB/.autodl.env"
REMOTE=/root/workspace/xtuner-lab
REMOTE_DATA=/root/autodl-tmp/xtuner-lab
EXP002=$REMOTE/experiments/002_qwen3_17b_4b_sft
EXP003=$REMOTE/experiments/003_fsdp_scaling
EXP004=$REMOTE/experiments/004_grpo_gsm8k

[ -f "$ENVF" ] || echo "autodl" > "$ENVF"
CUR=$(head -1 "$ENVF" 2>/dev/null || true)
[ -n "${CUR:-}" ] || { CUR=autodl; echo "$CUR" > "$ENVF"; }

SSHOPT=(-o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)
RSYNC_SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

rsh() { ssh -q "${SSHOPT[@]}" "$CUR-tun" "$@"; }

need_target() {
  grep -q "# >>> autodl.sh: $CUR >>>" ~/.ssh/config 2>/dev/null \
    || { echo "!! 还没绑定实例 [$CUR]：请执行  bash autodl.sh connect <host> <port>"; exit 1; }
}

usage() {
  cat <<'EOF'
用法：bash autodl.sh <命令> [参数]

  connect <host> <port> [名字]   绑定实例（写 ~/.ssh/config，记为当前；重启后重跑）
  use <名字>                     切换当前实例
  status                         状态：GPU / 关键进程 / 磁盘 / 最新日志
  sh                             交互式登录当前实例
  push                           上传 data + 002 + 003 + 004（rsync 增量）
  pull                           拉回所有 log_*.txt（rsync 增量）
  pull-work                      拉回 work_dir 小文件（不含权重）
  setup [模型列表]               云端初始化（可重复跑/断点续跑；默认下 1.7B+4B，setup "Qwen3-1.7B" 只下 1.7B）
  check                          上机自检（cmd_check.sh，结果自动拉回）
  run <任务>                     后台开跑 cmd_<任务>.sh（精确名优先，无则前缀匹配；如 run r1 / run e4_rl）
  tail <任务>                    跟随任务日志（Ctrl-C 只是离开，云端继续）
  clean-dcp                      删除 DCP 存档（checkpoints/；只用于断点续训，保留 hf-*）
  exec "<命令>"                  云端执行一条命令（引号包起来，例：exec "nvidia-smi"）
EOF
}

cmd_connect() {
  local host=${1:-} port=${2:-} name=${3:-autodl}
  [ -n "$host" ] && [ -n "$port" ] || { echo "用法: bash autodl.sh connect <host> <port> [名字]"; exit 1; }
  mkdir -p ~/.ssh; touch ~/.ssh/config; chmod 600 ~/.ssh/config
  local tmp; tmp=$(mktemp)
  awk -v n="$name" '
    index($0, "# >>> autodl.sh: " n " >>>") == 1 {skip=1; next}
    index($0, "# <<< autodl.sh: " n " <<<") == 1 {skip=0; next}
    !skip {print}
  ' ~/.ssh/config > "$tmp"
  printf '\n' >> "$tmp"
  cat >> "$tmp" <<EOF
# >>> autodl.sh: $name >>>（此块由 autodl.sh 维护，勿手改）
Host $name
  HostName $host
  Port $port
  User root
  ServerAliveInterval 30
  ServerAliveCountMax 6
Host $name-tun
  HostName autodl-via-tunnel
  User root
  ProxyCommand ssh -o BatchMode=yes -W 127.0.0.1:22 $name
  StrictHostKeyChecking no
  UserKnownHostsFile /dev/null
  ServerAliveInterval 30
  ServerAliveCountMax 6
# <<< autodl.sh: $name <<<
EOF
  mv "$tmp" ~/.ssh/config; chmod 600 ~/.ssh/config
  echo "$name" > "$ENVF"; CUR=$name
  echo "== 测试连接 [$name → $host:$port] =="
  if rsh 'hostname; nvidia-smi -L 2>/dev/null | head -3'; then
    echo "== OK：实例已就绪（当前机器 = $name）=="
  else
    echo "!! 连接失败：核对 host/port 是否抄对；若报权限错误，先在 AutoDL 控制台把本机公钥装到实例"
    exit 1
  fi
}

cmd_use() {
  local name=${1:-}
  [ -n "$name" ] || { echo "用法: bash autodl.sh use <名字>"; exit 1; }
  grep -q "# >>> autodl.sh: $name >>>" ~/.ssh/config 2>/dev/null || { echo "!! 配置里没有 [$name]，先 connect"; exit 1; }
  echo "$name" > "$ENVF"
  echo "== 当前机器切到 [$name] =="
}

cmd_status() {
  need_target
  echo "== 当前机器：$CUR =="
  rsh '
    date "+%F %T"
    echo "--- GPU ---"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || nvidia-smi -L
    echo "--- 关键进程 ---"
    ps -eo pid,etime,pcpu,pmem,cmd --sort=-pcpu | grep -E "[t]orchrun|[p]ip install|[m]odelscope|[c]md_[a-z0-9_]+\.sh" | head -6
    echo "--- 磁盘 ---"
    df -h /root/autodl-tmp | tail -1
    du -sh /root/autodl-tmp/xtuner-lab/models/* 2>/dev/null || true
    echo "--- 最新日志 ---"
    f=$(ls -t /root/workspace/xtuner-lab/experiments/*/log_*.txt 2>/dev/null | head -1 || true)
    if [ -n "$f" ]; then echo "($f)"; tail -n 6 "$f" | tr "\r" "\n" | tail -5; else echo "(暂无日志)"; fi
  '
}

cmd_push() {
  need_target
  echo "== 云端目录准备（fresh 机器上 /root/workspace 可能不存在） =="
  rsh "mkdir -p $REMOTE/experiments"
  echo "== 上传 data（增量） =="
  rsync -az --info=progress2 --exclude __pycache__ -e "$RSYNC_SSH" \
    "$LAB/data" "$CUR-tun:$REMOTE/"
  echo "== 上传 002 + 003 + 004（增量；不含 work_dir / log） =="
  rsync -az --info=progress2 --exclude work_dir --exclude work_dir_mismatch --exclude work_dir_17b \
    --exclude __pycache__ --exclude 'log_*.txt' \
    -e "$RSYNC_SSH" \
    "$LAB/experiments/002_qwen3_17b_4b_sft" "$LAB/experiments/003_fsdp_scaling" "$LAB/experiments/004_grpo_gsm8k" \
    "$CUR-tun:$REMOTE/experiments/"
  echo "== 上传完成 =="
}

cmd_pull() {
  need_target
  echo "== 拉回 log_*.txt（002 / 003 / 004 / 根目录） =="
  rsync -az --info=progress2 --include='*/' --include='log_*.txt' --exclude='*' -e "$RSYNC_SSH" \
    "$CUR-tun:$EXP002/" "$LAB/experiments/002_qwen3_17b_4b_sft/" || echo "(002 无日志)"
  rsync -az --info=progress2 --include='*/' --include='log_*.txt' --exclude='*' -e "$RSYNC_SSH" \
    "$CUR-tun:$EXP003/" "$LAB/experiments/003_fsdp_scaling/" || echo "(003 无日志)"
  rsync -az --info=progress2 --include='*/' --include='log_*.txt' --exclude='*' -e "$RSYNC_SSH" \
    "$CUR-tun:$EXP004/" "$LAB/experiments/004_grpo_gsm8k/" || echo "(004 无日志)"
  rsync -az --info=progress2 --include='*/' --include='log_*.txt' --exclude='*' -e "$RSYNC_SSH" \
    "$CUR-tun:$REMOTE/" "$LAB/" || true
  echo "== 完成 =="
  ls -lht "$LAB/experiments/"*/log_*.txt 2>/dev/null | head -10 || true
}

cmd_pull_work() {
  need_target
  echo "== 拉回 work_dir 小文件（不含权重） =="
  for exp in 002_qwen3_17b_4b_sft 003_fsdp_scaling 004_grpo_gsm8k; do
    rsync -az --info=progress2 \
      --include='*/' --include='*.txt' --include='*.log' --include='*.json' --include='*.jsonl' \
      --exclude='*.pt' --exclude='*.pth' --exclude='*.safetensors' --exclude='*.bin' --exclude='*' \
      -e "$RSYNC_SSH" "$CUR-tun:$REMOTE/experiments/$exp/" "$LAB/experiments/$exp/" \
      && echo "  [$exp] OK" || echo "  [$exp] 跳过"
  done
}

cmd_setup() {
  need_target
  local models=${1:-}
  if rsh "pgrep -f '[c]loud_setup.sh' >/dev/null 2>&1"; then
    echo "== 云端 setup 已在运行 =="
  else
    echo "== 启动云端 setup（nohup 后台${models:+；模型列表：$models}） =="
    rsh "mkdir -p /root/autodl-tmp /root/workspace; if [ ! -L $REMOTE ]; then if [ -d $REMOTE ]; then if rsync -a $REMOTE/ $REMOTE_DATA/ 2>/dev/null || cp -a $REMOTE/. $REMOTE_DATA/; then rm -rf $REMOTE; else echo '!! 合并到数据盘失败，未删除原目录'; exit 1; fi; fi; mkdir -p $REMOTE_DATA; ln -s $REMOTE_DATA $REMOTE; fi; cd $EXP002 || exit 9; nohup ${models:+env XTUNER_MODELS='$models' }bash cloud_setup.sh > log_setup.txt 2>&1 < /dev/null & echo 已启动"
  fi
  echo "== 跟随日志（Ctrl-C 只是离开，云端继续跑） =="
  rsh "tail -n 40 -f $EXP002/log_setup.txt"
}

cmd_check() {
  need_target
  rsh "cd $EXP002 && bash cmd_check.sh 2>&1 | tee log_check.txt"
  rsync -az -e "$RSYNC_SSH" "$CUR-tun:$EXP002/log_check.txt" "$LAB/experiments/002_qwen3_17b_4b_sft/" \
    && echo "== log_check.txt 已拉回本机 =="
}

cmd_run() {
  local task=${1:-}
  [ -n "$task" ] || { echo "用法: bash autodl.sh run <任务名>（如 r1 / e3_1 / e4_rl）"; exit 1; }
  need_target
  local files n
  files=$(rsh "ls $REMOTE/experiments/*/cmd_${task}.sh 2>/dev/null || true" || true)
  if [ -z "$files" ]; then
    files=$(rsh "ls $REMOTE/experiments/*/cmd_${task}*.sh 2>/dev/null || true" || true)
  fi
  n=$(printf '%s\n' "$files" | sed '/^[[:space:]]*$/d' | wc -l)
  if [ "$n" -eq 0 ]; then
    echo "!! 云端找不到 cmd_${task}.sh 或 cmd_${task}*.sh —— 先 bash autodl.sh push？"
    echo "   云端现有任务：" ; rsh "ls $REMOTE/experiments/*/cmd_*.sh 2>/dev/null | sed 's|.*/cmd_||'" || true
    exit 1
  fi
  if [ "$n" -gt 1 ]; then
    echo "!! 匹配到多个，请写具体一点："
    printf '%s\n' "$files"
    exit 1
  fi
  local script base dir
  script=$files; base=$(basename "$script"); dir=$(dirname "$script")
  if rsh "pgrep -f '[c]md_${task}' >/dev/null 2>&1"; then
    echo "!! 好像已有 [$task] 在跑：bash autodl.sh tail $task"; exit 1
  fi
  rsh "cd $dir || exit 9; nohup bash -c 'time bash $base' > log_${task}.txt 2>&1 < /dev/null & echo 已开跑"
  echo "== 任务 [$task] 云端后台运行中 =="
  echo "   跟日志: bash autodl.sh tail $task   （Ctrl-C 只是离开，云端继续）"
  echo "   状态:   bash autodl.sh status"
}

cmd_tail() {
  local task=${1:-}
  [ -n "$task" ] || { echo "用法: bash autodl.sh tail <任务名>"; exit 1; }
  need_target
  local log
  log=$(rsh "ls -t $REMOTE/experiments/*/log_${task}.txt 2>/dev/null | head -1 || true" || true)
  if [ -z "$log" ]; then
    log=$(rsh "ls -t $REMOTE/experiments/*/log_${task}*.txt 2>/dev/null | head -1 || true" || true)
  fi
  [ -n "$log" ] || { echo "!! 找不到 log_${task}.txt 或 log_${task}*.txt"; exit 1; }
  echo "== tail -f $log（Ctrl-C 离开） =="
  rsh "tail -n 50 -f $log"
}

cmd_clean_dcp() {
  need_target
  echo "== 待删除的 DCP 存档（只留 hf-*） =="
  rsh "du -sh $REMOTE/experiments/*/work_dir/*/checkpoints 2>/dev/null || true"
  rsh "rm -rf $REMOTE/experiments/*/work_dir/*/checkpoints && echo 已清理"
  echo "== 清理后磁盘 =="
  rsh "df -h /root/autodl-tmp | tail -1"
}

cmd_exec() {
  local cmd=${1:-}
  [ -n "$cmd" ] || { echo "用法: bash autodl.sh exec \"<云端命令>\""; exit 1; }
  need_target
  rsh "$cmd"
}

case "${1:-help}" in
  connect)   shift; cmd_connect "$@" ;;
  use)       shift; cmd_use "$@" ;;
  status)    cmd_status ;;
  sh)        need_target; exec ssh "${SSHOPT[@]}" -t "$CUR" ;;
  push)      cmd_push ;;
  pull)      cmd_pull ;;
  pull-work) cmd_pull_work ;;
  setup)     shift; cmd_setup "$@" ;;
  check)     cmd_check ;;
  run)       shift; cmd_run "$@" ;;
  tail)      shift; cmd_tail "$@" ;;
  clean-dcp) cmd_clean_dcp ;;
  exec)      shift; cmd_exec "$@" ;;
  *)         usage ;;
esac
