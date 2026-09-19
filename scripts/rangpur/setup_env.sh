#!/bin/bash
# 一次性环境配置。在 CPU 交互作业里运行，不要在登录节点上装包：
#
#   srun --partition=cpu --time=00:30:00 --pty bash
#   bash ~/myTransformer/scripts/rangpur/setup_env.sh
#   exit
#
# 在课程的 torch 环境上叠一个 venv（~/mt-venv）：
#   --system-site-packages 让它直接用课程环境里的 torch，不重复安装（只占几百 MB）；
#   本项目的依赖只装进 mt-venv，不改动后续作业还要用的 miniconda3/envs/torch。
# 收尾时 rm -rf ~/mt-venv 即可，见 docs/09-rangpur.md §七。
#
# 不用 set -u：conda 的 activate 脚本会引用未定义的变量。
set -eo pipefail

REPO="$HOME/myTransformer"
VENV="$HOME/mt-venv"

# 装包属于"大规模安装"，课程指南 §5 要求放在作业里，不能在登录节点上做
if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "请先进入 CPU 作业再运行：srun --partition=cpu --time=00:30:00 --pty bash"
  exit 1
fi

[ -d "$REPO/.git" ] || { echo "先把仓库克隆到 $REPO（见 docs/09-rangpur.md §3.4）"; exit 1; }

source "$HOME/miniconda3/bin/activate"
conda activate torch
echo "课程环境：$(python --version)，torch $(python -c 'import torch; print(torch.__version__)')"

if [ ! -d "$VENV" ]; then
  python -m venv --system-site-packages "$VENV"
  echo "已创建 $VENV"
fi
source "$VENV/bin/activate"

# dev：pytest + transformers（HF 数值对齐测试要用）。data / train 两组依赖到 P2 再装。
pip install --no-cache-dir -e "$REPO[dev]"

python - <<'PY'
import sys
import torch
import mytransformer
print("python  ", sys.executable)
print("torch   ", torch.__version__, "（来自", torch.__file__.split("/site-packages")[0], "）")
print("项目包  ", mytransformer.__file__)
PY
echo "mt-venv 占用：$(du -sh "$VENV" | cut -f1)"
df -h "$HOME" | tail -1
