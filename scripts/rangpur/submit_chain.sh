#!/bin/bash
# 把同一个训练配置提交成 N 个首尾相接的作业。
#
#   bash scripts/rangpur/submit_chain.sh CONFIG [作业数]
#   SET="seed=1 run_dir=runs/xxx_seed1" bash scripts/rangpur/submit_chain.sh CONFIG 2
#
# 每人同时只能运行 1 个作业，所以串起来就是接力：一个作业到 225 分钟存档退出，
# 下一个启动后从 checkpoint 续跑。
#
# 用 afterany 而不是 afterok：前一个不管是正常结束、到时被杀还是出错，都接着跑——
# 续跑逻辑本来就能处理这些情况。训练提前完成时，后面排着的作业启动后发现"已训完"就立即退出。
# 如果某个作业报错（比如续跑自检失败），后面的也会很快失败：用 scancel 取消剩下的，查完再重交。
set -euo pipefail

CONFIG="${1:?用法: submit_chain.sh CONFIG [作业数]}"
N="${2:-1}"

cd "$HOME/myTransformer"
[ -f "$CONFIG" ] || { echo "找不到配置文件 $CONFIG"; exit 1; }
mkdir -p logs  # Slurm 不会自动创建 --output 所在的目录；目录不存在时作业会直接失败，且没有任何日志

EXPORT="ALL,CONFIG=$CONFIG"
if [ -n "${SET:-}" ]; then EXPORT="$EXPORT,SET=$SET"; fi

prev=$(sbatch --parsable --export="$EXPORT" scripts/rangpur/train.sbatch)
echo "已提交 $prev（1/$N）"
for i in $(seq 2 "$N"); do
  prev=$(sbatch --parsable --dependency="afterany:$prev" --export="$EXPORT" scripts/rangpur/train.sbatch)
  echo "已提交 $prev（$i/$N，等前一个结束后开始）"
done
squeue --me
