#!/bin/bash
# 把 P5 的全部实验串成一条接力队列（每人同时只能运行 1 个作业，所以只能排队）。
#
#   bash scripts/rangpur/submit_p5.sh            # 全部 12 个实验，约 14 小时 A100
#   bash scripts/rangpur/submit_p5.sh ladder     # 只跑缩放三点（约 5 小时）
#   bash scripts/rangpur/submit_p5.sh ablation   # 只跑种子方差 + 6 组消融（约 9 小时）
#
# 顺序是**先短后长**：小实验先出结果，能尽早发现配置错误；4.4 小时的 ladder_s3 放最后。
# 每个实验一个作业，用 afterany 串起来；ladder_s3 用两个作业接力（单作业 4 小时上限内存档退出、续跑）。
#
# 共享集群礼仪（docs/09-rangpur.md §六）：一次排这么多作业会占用较长时间，
# 避开 demo 和作业截止周；别人急用时用 scancel 让出来，训练能从 checkpoint 续上。
set -euo pipefail

GROUP="${1:-all}"
cd "$HOME/myTransformer"
mkdir -p logs

# 配置:小时数:作业数
SHORT=(
  "configs/train/p5/ladder_s1.yaml:0.5:1"
  "configs/train/p5/ladder_s2.yaml:1.5:1"
)
ABLATION=(
  "configs/train/p5/base_seed0.yaml:1.5:1"
  "configs/train/p5/base_seed1.yaml:1.5:1"
  "configs/train/p5/base_seed2.yaml:1.5:1"
  "configs/train/p5/a1_muon.yaml:1.5:1"
  "configs/train/p5/a2_gelu.yaml:1.5:1"
  "configs/train/p5/a3_mha.yaml:1.5:1"
  "configs/train/p5/a4_mix_v2.yaml:1.5:1"
  "configs/train/p5/a5_cosine.yaml:1.5:1"
  "configs/train/p5/a6_qknorm_no_zloss.yaml:1.5:1"
)
BIG=("configs/train/p5/ladder_s3.yaml:4:2")

case "$GROUP" in
  all)      QUEUE=("${SHORT[@]}" "${ABLATION[@]}" "${BIG[@]}") ;;
  ladder)   QUEUE=("${SHORT[@]}" "${BIG[@]}") ;;
  ablation) QUEUE=("${ABLATION[@]}") ;;
  *) echo "用法: submit_p5.sh [all|ladder|ablation]"; exit 1 ;;
esac

prev=""
for item in "${QUEUE[@]}"; do
  IFS=: read -r config hours jobs <<< "$item"
  [ -f "$config" ] || { echo "找不到配置 $config"; exit 1; }
  # 作业时长留 15 分钟给启动、编译和最后存档；MAX_MINUTES 到点主动存档退出
  minutes=$(python -c "print(int($hours * 60))")
  budget=$((minutes - 15))
  for _ in $(seq 1 "$jobs"); do
    args=(--parsable --time="$(printf '%02d:%02d:00' $((minutes / 60)) $((minutes % 60)))"
          --export="ALL,CONFIG=$config,MAX_MINUTES=$budget")
    [ -n "$prev" ] && args+=(--dependency="afterany:$prev")
    prev=$(sbatch "${args[@]}" scripts/rangpur/train.sbatch)
    echo "已提交 $prev  $config（${hours}h）"
  done
done
echo
squeue --me --format="%.10i %.30j %.10T %.10M %.10l %R"
