#!/bin/bash
# 把 P5 的实验串成一条接力队列（每人同时只能运行 1 个作业，所以只能排队）。
#
#   bash scripts/rangpur/submit_p5.sh            # 全部
#   bash scripts/rangpur/submit_p5.sh ladder     # 只跑缩放三点
#   bash scripts/rangpur/submit_p5.sh ablation   # 只跑种子方差 + 6 组消融
#   bash scripts/rangpur/submit_p5.sh --dry-run  # 只打印计划，不提交
#
# 每个实验申请多长时间、分几个作业接力，由 scripts/rangpur/p5_plan.py 按**实测**的每步耗时算出
# （一开始按假设的 MFU 估时，作业在 303/360 步被截断，所以不再手写时长）。
#
# 已经训完的实验（run 目录里有 final.pt）直接跳过；训完但还没压缩、没做最终评测的，
# 会排一个短作业：Trainer 发现已训完，只补最终评测并压缩存档，几分钟结束。
#
# 共享集群礼仪（docs/09-rangpur.md §六）：排这么多作业会占用较长时间，避开 demo 和作业截止周；
# 别人急用时用 scancel 让出来，训练能从 checkpoint 续上。
set -euo pipefail

GROUP="all"
DRY=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY=1 ;;
    all|ladder|ablation) GROUP="$arg" ;;
    *) echo "用法: submit_p5.sh [all|ladder|ablation] [--dry-run]"; exit 1 ;;
  esac
done

cd "$HOME/myTransformer"
mkdir -p logs
source "$HOME/mt-venv/bin/activate"

prev=""
while read -r config job_min train_min jobs run_dir; do
  name=$(basename "$config" .yaml)
  case "$GROUP" in
    ladder)   [[ "$name" == ladder_* ]] || continue ;;
    ablation) [[ "$name" == ladder_* ]] && continue ;;
  esac
  if [ -f "$run_dir/final.pt" ]; then
    echo "跳过 $name（已训完：$run_dir/final.pt）"
    continue
  fi
  # 有未压缩的 checkpoint 时照常排队：训完了就只补评测、压缩，没训完就续跑，都由 Trainer 判断
  for _ in $(seq 1 "$jobs"); do
    args=(--parsable --time="$(printf '%02d:%02d:00' $((job_min / 60)) $((job_min % 60)))"
          --export="ALL,CONFIG=$config,MAX_MINUTES=$train_min")
    [ -n "$prev" ] && args+=(--dependency="afterany:$prev")
    if [ "$DRY" = 1 ]; then
      echo "[dry-run] sbatch ${args[*]} scripts/rangpur/train.sbatch"
      prev="DRY"
    else
      prev=$(sbatch "${args[@]}" scripts/rangpur/train.sbatch)
      echo "已提交 $prev  $name（申请 ${job_min} 分钟，训练预算 ${train_min} 分钟）"
    fi
  done
done < <(python scripts/rangpur/p5_plan.py --lines)

echo
python scripts/rangpur/p5_plan.py | tail -1
[ "$DRY" = 1 ] || squeue --me --format="%.10i %.30j %.10T %.10M %.10l %R"
