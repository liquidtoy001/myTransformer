#!/bin/bash
# 租到多卡机器后的第一件事：多卡验收。不通过就不开始正式训练。
#
#   bash scripts/cloud/ddp_check.sh
#
# 前提：仓库、虚拟环境已就绪，data/p5/mix/ 和 tokenizer/mt32k.json 已上传。
# 本机（Windows、无 NCCL）只能用 CPU 验证多卡逻辑（tests/test_ddp.py）；
# 真实 GPU + NCCL + torch.compile 的组合只能在这里验证。
# 预计 15–25 分钟（含每次启动的编译），按 8×H100 的价格约几美元。
set -euo pipefail

CFG=configs/train/p6_check/ddp_250m.yaml
OUT=runs/ddp_check
N=$(nvidia-smi -L | wc -l)
echo "检测到 $N 张卡：$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
[ "$N" -ge 2 ] || { echo "至少要 2 张卡"; exit 1; }

run() {  # run 名字 卡数 [其他参数...]
  local name=$1 nproc=$2; shift 2
  torchrun --standalone --nproc_per_node "$nproc" -m mytransformer.train.pretrain \
    --config "$CFG" --set run_dir="$OUT/$name" "$@"
}

summary() {  # 打印 step >= 20 的吞吐中位数（前面的步含编译和预热）
  python - "$1" <<'EOF'
import json, statistics, sys
recs = [json.loads(l) for l in open(f"{sys.argv[1]}/metrics.jsonl", encoding="utf-8")]
tr = [r for r in recs if r["type"] == "train" and r["step"] >= 20]
tok = statistics.median(r["tok_s"] for r in tr)
mfu = statistics.median(r.get("mfu", 0) for r in tr)
print(f"{sys.argv[1]}：{tok:,.0f} tokens/s，MFU {mfu:.1%}，峰值显存 {max(r.get('mem_gib', 0) for r in tr):.1f} GiB")
print(f"  按这个速度，P6 的 9.44B tokens 需要 {9.44e9 / tok / 3600:.2f} 小时（不含评测和存档）")
EOF
}

echo; echo "== 1/3 一致性：1 卡 vs 2 卡，同样的 global batch，各 30 步"
run g1 1 --set max_steps=30
run g2 2 --set max_steps=30
# bf16 + 非确定性 kernel，做不到逐位相同；梯度没同步或数据切错时，差距会远超 2%
python scripts/compare_runs.py "$OUT/g1" "$OUT/g2" --rtol 0.02

echo; echo "== 2/3 吞吐：$N 卡 60 步"
run "g$N" "$N"
summary "$OUT/g$N"

echo; echo "== 3/3 中断续跑：$N 卡训到第 20 步后给各 worker 发 SIGUSR1，存档退出后再续跑到 60 步"
run resume "$N" & TR=$!
until python -c "
import json, sys
try:
    steps = [json.loads(l)['step'] for l in open('$OUT/resume/metrics.jsonl') if '\"train\"' in l]
except FileNotFoundError:
    sys.exit(1)
sys.exit(0 if steps and max(steps) >= 20 else 1)"; do sleep 5; done
# 只发给 torchrun 的子进程（各 rank）。torchrun 本身收到 USR1 会直接退出，不能发给它
pkill -USR1 -P "$TR"
wait "$TR"
ls "$OUT/resume"
run resume "$N"
python scripts/compare_runs.py "$OUT/g$N" "$OUT/resume" --rtol 0.02

echo; echo "全部通过。把上面的吞吐填进 docs/10-p6-preparation.md，再决定正式训练的配置和预算。"
