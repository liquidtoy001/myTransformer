"""在本机 GPU 上实测各规模模型的训练吞吐与 MFU。

规格表上的 TFLOPS 推不出真实训练速度，所以这里全部实测：

1. 用大矩阵乘测出本卡 bf16 的**可达峰值**，而不是用厂商标称值
2. 自动探测每个配置能塞下的最大 micro_batch
3. 跑真实的 前向+反向+优化器 步，测 tokens/s
4. 外推到不同 token 预算的耗时

**每个测试点跑在独立子进程里。** 因为一旦触发 OOM，cuBLAS 句柄会被污染，
同进程内后续所有 matmul 都会报 CUBLAS_STATUS_EXECUTION_FAILED，
无法靠 try/except 恢复。

用法:
    python scripts/benchmark.py
    python scripts/benchmark.py --configs s3 250m --seq 2048
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# 直接读仓库里的配置文件，保证测的就是要训的模型。
ROOT = Path(__file__).resolve().parent.parent
CONFIGS: dict[str, str] = {
    "s1": "configs/model/ladder_s1.yaml",
    "s2": "configs/model/ladder_s2.yaml",
    "s3": "configs/model/ladder_s3.yaml",
    "250m": "configs/model/250m.yaml",
    "500m": "configs/model/500m.yaml",
    "1b": "configs/model/1b.yaml",
}
DEFAULT = ["s1", "s2", "s3", "250m"]

BUDGETS = [("1B", 1e9), ("3B", 3e9), ("10B", 1e10), ("25B", 2.5e10)]


# ---------------------------------------------------------------- 子进程侧


def run_worker(name: str, seq: int, batch: int, ckpt: bool) -> None:
    """在子进程里跑一个测试点，结果以 JSON 打到 stdout。OOM 就非零退出。"""
    import torch

    from mytransformer.model import ModelConfig, Transformer

    cfg = ModelConfig.from_yaml(ROOT / CONFIGS[name])
    cfg.max_seq_len = seq
    counts = cfg.count_params()
    device = "cuda"

    model = Transformer(cfg).to(device)
    if ckpt:
        model.gradient_checkpointing = True
    opt = torch.optim.AdamW(model.param_groups(0.1), lr=1e-4, betas=(0.9, 0.95), fused=True)

    def step() -> None:
        ids = torch.randint(0, cfg.vocab_size, (batch, seq), device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(ids, targets=ids)["loss"]
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    for _ in range(3):  # warmup
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    iters = 8
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters

    # 公式见 ModelConfig.flops_per_token：6N + lm_head + 注意力打分
    fpt = cfg.flops_per_token(seq)
    tok_s = batch * seq / dt

    print(
        "RESULT"
        + json.dumps(
            {
                "name": name,
                "batch": batch,
                "params": counts["total"],
                "step_s": dt,
                "tok_s": tok_s,
                "achieved": fpt * tok_s,
                "mem_gb": torch.cuda.max_memory_allocated() / 1024**3,
            }
        )
    )


def run_peak() -> None:
    import torch

    n = 8192
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    iters = 20
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    p = torch.cuda.get_device_properties(0)
    print(
        "RESULT"
        + json.dumps(
            {
                "peak": 2 * n**3 * iters / dt,
                "device": p.name,
                "mem": p.total_memory / 1024**3,
                "sm": f"{p.major}{p.minor}",
                "torch": torch.__version__,
            }
        )
    )


# ---------------------------------------------------------------- 父进程侧


def call(args: list[str]) -> dict | None:
    proc = subprocess.run(
        [sys.executable, __file__, *args], capture_output=True, text=True, encoding="utf-8"
    )
    for line in (proc.stdout or "").splitlines():
        if line.startswith("RESULT"):
            return json.loads(line[len("RESULT") :])
    return None


def fmt_time(sec: float) -> str:
    if sec < 3600:
        return f"{sec / 60:.0f} 分钟"
    if sec < 86400:
        return f"{sec / 3600:.1f} 小时"
    return f"{sec / 86400:.1f} 天"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=DEFAULT, choices=list(CONFIGS))
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--ckpt", action="store_true", help="开梯度检查点（省显存、慢约 30%%）")
    # 子进程内部用
    ap.add_argument("--_worker")
    ap.add_argument("--_batch", type=int)
    ap.add_argument("--_peak", action="store_true")
    args = ap.parse_args()

    if args._peak:
        return run_peak()
    if args._worker:
        return run_worker(args._worker, args.seq, args._batch, args.ckpt)

    info = call(["--_peak"])
    assert info, "峰值测试失败，检查 CUDA 是否可用"
    peak = info["peak"]
    print(f"设备: {info['device']}  {info['mem']:.1f} GB  sm_{info['sm']}  torch {info['torch']}")
    print(f"实测 bf16 可达峰值: {peak / 1e12:.1f} TFLOP/s  (8192³ 矩阵乘)")
    print(f"序列长度 {args.seq}，梯度检查点 {'开' if args.ckpt else '关'}\n")

    rows = []
    for name in args.configs:
        best = None
        for b in (1, 2, 4, 8, 16, 32):
            extra = ["--ckpt"] if args.ckpt else []
            r = call(["--_worker", name, "--_batch", str(b), "--seq", str(args.seq), *extra])
            if r is None:
                break  # OOM，上一个 batch 就是上限
            best = r
        if best is None:
            print(f"{name:>5}  micro_batch=1 都放不下，跳过")
            continue
        best["mfu"] = best["achieved"] / peak
        rows.append(best)
        print(
            f"{name:>5}  {best['params'] / 1e6:6.1f}M 参数  "
            f"micro_batch={best['batch']:<3} "
            f"{best['tok_s']:>8,.0f} tok/s  "
            f"{best['achieved'] / 1e12:5.1f} TFLOP/s  "
            f"MFU {best['mfu']:5.1%}  "
            f"显存 {best['mem_gb']:.1f} GB"
        )

    if not rows:
        return
    print("\n本卡上各 token 预算的实测耗时:")
    header = f"{'模型':>6} " + " ".join(f"{b[0]:>12}" for b in BUDGETS)
    print(header)
    print("-" * (len(BUDGETS) * 13 + 7))
    for r in rows:
        cells = " ".join(f"{fmt_time(b[1] / r['tok_s']):>12}" for b in BUDGETS)
        print(f"{r['name']:>6} {cells}")


if __name__ == "__main__":
    main()
