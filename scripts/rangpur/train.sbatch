#!/bin/bash
#SBATCH --job-name=mt-train
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.out
# 不加 --mem（见 docs/09-rangpur.md §二）
#
# 一般用 submit_chain.sh 提交，它会把多个作业串成接力。单独提交：
#   sbatch --export=ALL,CONFIG=configs/train/xxx.yaml scripts/rangpur/train.sbatch
# 环境变量：
#   CONFIG       训练配置（必需）
#   SET          空格分隔的配置覆盖，例如 "seed=1 run_dir=runs/xxx_seed1"
#   MAX_MINUTES  训练多少分钟后主动存档退出，默认 225
set -eo pipefail
: "${CONFIG:?需要 CONFIG=configs/train/xxx.yaml}"

echo "=== Job $SLURM_JOB_ID on $(hostname), $(date)"
echo "CONFIG=$CONFIG ${SET:+SET=\"$SET\"}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

source "$HOME/mt-venv/bin/activate"
export HF_HOME="${TMPDIR:-/tmp}/hf" PIP_NO_CACHE_DIR=1
export PYTHONUNBUFFERED=1  # 实时写日志，tail -f 才看得到最新进度
export TORCHINDUCTOR_CACHE_DIR="$HOME/myTransformer/.cache/inductor"
export WANDB_DIR="${TMPDIR:-/tmp}/wandb"
cd "$HOME/myTransformer"
echo "git $(git rev-parse --short HEAD)"

SET_ARGS=()
for kv in ${SET:-}; do SET_ARGS+=(--set "$kv"); done

# 两道保险：
#   1. --max-minutes：训练 225 分钟后主动存档退出，给启动、编译、存档留 15 分钟
#   2. --signal=B:USR1@600：万一没按时停，被杀前 10 分钟 Slurm 发 USR1
# B: 表示信号发给"批处理 shell"。exec 让 python 直接取代这个 shell，信号才能送到 python；
# 不用 exec 的话，信号只到 bash，python 收不到，照样会在到时被强杀。
exec python -m mytransformer.train.pretrain --config "$CONFIG" "${SET_ARGS[@]}" --max-minutes "${MAX_MINUTES:-225}"
