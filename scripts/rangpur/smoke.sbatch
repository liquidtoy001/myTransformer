#!/bin/bash
#SBATCH --job-name=mt-smoke
#SBATCH --partition=a100-test
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.out
# 不加 --mem：实测加了就排不上（与课程指南不一致，见 docs/09-rangpur.md §二）
#
# 在 A100 上把整条链路验证一遍。提交方式（在 ~/myTransformer 下）：
#   mkdir -p logs && sbatch scripts/rangpur/smoke.sbatch
#
#   [1/4] 全部测试在 A100 上仍然通过（包括只能在 GPU 上跑的分块交叉熵显存测试）
#   [2/4] 准备合成数据（与本地冒烟相同，理论下界 ln4 ≈ 1.39）
#   [3/4] 打开 torch.compile 训练，中途给它发一个真实的 SIGUSR1，确认它存档退出
#         —— Windows 上既没有 SIGUSR1 也没有 triton，这两件事只能在这里验证
#   [4/4] 同一条命令再启动：应当续跑、通过自检、训完
set -eo pipefail

echo "=== Job $SLURM_JOB_ID on $(hostname), $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "TMPDIR=${TMPDIR:-（未设置）}"
if [ -n "$TMPDIR" ]; then df -h "$TMPDIR" | tail -1; fi
df -h "$HOME" | tail -1

source "$HOME/mt-venv/bin/activate"
export HF_HOME="${TMPDIR:-/tmp}/hf" PIP_NO_CACHE_DIR=1
export PYTHONUNBUFFERED=1  # 输出写到文件时默认会缓存，tail -f 看到的日志会一段一段延迟出现
export TORCHINDUCTOR_CACHE_DIR="$HOME/myTransformer/.cache/inductor"
cd "$HOME/myTransformer"
echo "git $(git rev-parse --short HEAD)"
python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.version.cuda, '|', torch.cuda.get_device_name(0))"

echo; echo "=== [1/4] 测试"
python -m pytest tests/ -q

echo; echo "=== [2/4] 合成数据"
if [ -f data/synthetic/val_000.bin ]; then echo "已存在，跳过"; else python scripts/make_synthetic_data.py --out data/synthetic; fi

RUN=runs/smoke_s1_rangpur
rm -rf "$RUN"
ARGS=(--config configs/train/smoke_s1.yaml --set compile=true --set "run_dir=$RUN")

echo; echo "=== [3/4] 训练开始后发 SIGUSR1：应当做完当前这一步、存档、退出"
python -m mytransformer.train.pretrain "${ARGS[@]}" &
PID=$!
# 等到第一条训练记录出现再发信号：信号处理函数要进了 fit() 才装上，
# 在那之前收到 SIGUSR1，进程会按默认行为直接被杀掉，什么都不会存。
for _ in $(seq 1 150); do
  grep -q '"type": "train"' "$RUN/metrics.jsonl" 2>/dev/null && break
  sleep 2
done
sleep 10
kill -USR1 "$PID" 2>/dev/null || echo "（发信号时训练已经结束了）"
wait "$PID" || echo "第一段退出码 $?"

# 明确检查，不能只看"最后训完了"：信号要是来得太早，进程会被直接杀掉、什么都不存，
# 第 4 步就会从头训练、照样训完——冒烟看起来通过了，续跑却根本没被验证到。
FAIL=0
if grep -q "停下（signal:SIGUSR1）" "logs/smoke_${SLURM_JOB_ID}.out"; then
  echo "✓ 收到 SIGUSR1 后存档退出"
else
  echo "!! 第 3 步没有因 SIGUSR1 存档退出"; FAIL=1
fi

echo; echo "=== [4/4] 同一条命令再启动：应当续跑、自检通过、训完"
PART4="${TMPDIR:-/tmp}/smoke_part4.log"
python -m mytransformer.train.pretrain "${ARGS[@]}" 2>&1 | tee "$PART4"
if grep -q "^续跑：" "$PART4"; then
  echo "✓ 从 checkpoint 续跑，自检通过"
else
  echo "!! 第 4 步是从头训练的，没有续跑"; FAIL=1
fi

echo; echo "=== 结果"
ls -la "$RUN"
python - "$RUN" <<'PY'
import json
import sys

recs = [json.loads(line) for line in open(f"{sys.argv[1]}/metrics.jsonl", encoding="utf-8")]
tr = [r for r in recs if r["type"] == "train"]
ev = [r for r in recs if r["type"] == "eval"]
print(f"最终 val_loss {ev[-1]['val_loss']:.4f}（理论下界约 1.390；本地 RTX 4070 Ti SUPER 上是 1.421）")
steady = tr[len(tr) // 2 :]  # 后半段：跳过编译和预热
print(f"后半段平均 {sum(r['tok_s'] for r in steady) / len(steady):,.0f} tok/s，"
      f"MFU {sum(r.get('mfu', 0) for r in steady) / len(steady):.1%}，"
      f"显存峰值 {max(r.get('mem_gib', 0) for r in tr):.1f} GiB")
PY
if [ "$FAIL" -ne 0 ]; then echo "=== 冒烟失败 $(date)"; exit 1; fi
echo "=== 冒烟通过 $(date)"
