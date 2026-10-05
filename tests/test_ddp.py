"""多卡（DDP）：用 torchrun 在 CPU 上起 2 个进程（gloo 后端），检查多卡逻辑。

真正的多卡 GPU 只有 P6 租机时才有，所以能在本机验证的都在这里验证：

1. 2 个进程训练的结果 ≈ 1 个进程（同样的 global batch）——梯度平均、数据切分、验证集分摊都对
2. 只有一个 rank 收到停止请求时，两个进程在同一步一起存档退出，不会互相等死
3. 多卡中断后续跑 = 不中断
4. 多卡存的 checkpoint 能用单卡接着训（租的机器换了卡数也能续）
5. 训完压缩为 final.pt 时不卡住

每个用例启动一次 torchrun 要几秒（两个进程各自 import torch），整个文件约一分钟。
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch
import yaml

from mytransformer.data.shards import write_shard
from mytransformer.train import checkpoint as ck

VOCAB, SEQ = 128, 16

# 每个进程跑的脚本：可以让指定 rank 在指定 step 请求停止，用来模拟"信号只到了一个进程"
RUNNER = textwrap.dedent("""
    import sys
    from mytransformer.train import distributed
    from mytransformer.train.config import TrainConfig
    from mytransformer.train.trainer import Trainer

    cfg = TrainConfig.from_yaml(sys.argv[1])
    stop_step, stop_rank = int(sys.argv[2]), int(sys.argv[3])

    def hook(t):
        if t.step == stop_step and t.dist.rank == stop_rank:
            t.request_stop("test")

    t = Trainer(cfg, device="cpu", on_step=hook, log=lambda _: None)
    reason = t.fit()
    print(f"RESULT rank={t.dist.rank} reason={reason} step={t.step}", flush=True)
    distributed.shutdown()
""")


@pytest.fixture
def env(tmp_path):
    (tmp_path / "tiny.yaml").write_text(yaml.safe_dump(dict(
        vocab_size=VOCAB, d_model=32, n_layers=2, n_heads=2, n_kv_heads=1,
        head_dim=16, ffn_hidden=64, max_seq_len=SEQ,
    )))
    rng = np.random.default_rng(0)
    for i in range(3):
        write_shard(tmp_path / "data" / f"train_{i:03d}.bin", rng.integers(0, VOCAB, 4000))
    write_shard(tmp_path / "data" / "val_000.bin", rng.integers(0, VOCAB, 4000))
    (tmp_path / "runner.py").write_text(RUNNER, encoding="utf-8")
    return tmp_path


def write_cfg(env, run, **kw):
    cfg = dict(
        model=str(env / "tiny.yaml"),
        train_data=str(env / "data" / "train_*.bin"),
        val_data=str(env / "data" / "val_*.bin"),
        run_dir=str(env / run),
        seq_len=SEQ, micro_batch_size=4,
        global_batch_tokens=4 * SEQ * 4,   # 单进程累积 4 次；2 个进程各累积 2 次
        max_steps=10, lr=1e-2, warmup_steps=2,
        dtype="float32", ce_chunk=20, z_loss=1e-4,
        log_every=1, eval_every=4, eval_batches=3,  # 3 个验证 batch：rank 0 算 2 个、rank 1 算 1 个
        ckpt_every=100, ckpt_keep=2,
    )
    cfg.update(kw)
    path = env / f"{run}.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def launch(env, cfg_path, nproc, stop_step=0, stop_rank=0) -> list[str]:
    """nproc=1 直接用 python 启动（走单卡路径）；否则用 torchrun。返回各 rank 的 RESULT 行。"""
    cmd = [sys.executable, str(env / "runner.py"), str(cfg_path), str(stop_step), str(stop_rank)]
    base = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1"}
    for k in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
        base.pop(k, None)
    if nproc > 1:
        # 自己设 torchrun 会设的那几个环境变量，各起一个进程。不直接用 torchrun：Windows 版 PyTorch
        # 没编译 libuv，而 torchrun 的启动代理不理会 USE_LIBUV=0。训练器读环境变量的方式（env://）完全一样，
        # Linux 租机上照常用 torchrun
        base.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(free_port()), WORLD_SIZE=str(nproc), USE_LIBUV="0")
    procs = [
        subprocess.Popen(cmd, env={**base, **({"RANK": str(r), "LOCAL_RANK": str(r)} if nproc > 1 else {})},
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for r in range(nproc)
    ]
    lines = []
    for p in procs:
        try:
            out, err = p.communicate(timeout=300)  # 进程间互相等死时在这里超时，而不是让测试永远挂着
        except subprocess.TimeoutExpired:
            for q in procs:
                q.kill()
            raise
        assert p.returncode == 0, out[-3000:] + err[-3000:]
        lines += [line for line in out.splitlines() if line.startswith("RESULT")]
    return sorted(lines)


def records(env, run, kind="train"):
    lines = (env / run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    return [r for r in map(json.loads, lines) if r["type"] == kind]


def weights(env, run):
    path = ck.latest(env / run) or ck.final_path(env / run)
    return ck.load(path)["model"]


def assert_same_run(env, a, b, rtol):
    la, lb = records(env, a), records(env, b)
    assert [r["step"] for r in la] == [r["step"] for r in lb]
    np.testing.assert_allclose([r["loss"] for r in la], [r["loss"] for r in lb], rtol=rtol)
    np.testing.assert_allclose([r["grad_norm"] for r in la], [r["grad_norm"] for r in lb], rtol=rtol)
    va, vb = records(env, a, "eval"), records(env, b, "eval")
    assert [r["step"] for r in va] == [r["step"] for r in vb]
    np.testing.assert_allclose([r["val_loss"] for r in va], [r["val_loss"] for r in vb], rtol=rtol)
    wa, wb = weights(env, a), weights(env, b)
    for k in wa:
        torch.testing.assert_close(wa[k], wb[k], rtol=rtol, atol=1e-6, msg=k)


def test_two_ranks_match_one_process(env):
    """同样的 global batch：2 卡各累积 2 次 ≈ 1 卡累积 4 次。

    数据切分保证每一步两种方式读到的是同一批 token（只是分给谁算不同），
    所以差别只来自浮点求和顺序。梯度没取平均、数据切错、验证集分摊算错，都会远超这个容差。"""
    launch(env, write_cfg(env, "one"), 1)
    cfg = write_cfg(env, "two")
    out = launch(env, cfg, 2)
    assert out == ["RESULT rank=0 reason=done step=10", "RESULT rank=1 reason=done step=10"]
    assert_same_run(env, "one", "two", rtol=1e-4)
    global_batch = yaml.safe_load(cfg.read_text(encoding="utf-8"))["global_batch_tokens"]
    assert records(env, "two")[-1]["tokens"] == 10 * global_batch  # tokens 按全体计，不是每卡


def test_stop_on_one_rank_stops_all_then_resume_is_identical(env):
    """停止请求只发给 rank 1：两个进程在同一步存档退出（rank 0 的原因记为 peer）。
    再启动一次接着训完，结果与不中断的 2 卡训练逐位相同。"""
    launch(env, write_cfg(env, "straight"), 2)

    cfg = write_cfg(env, "resumed")
    out = launch(env, cfg, 2, stop_step=5, stop_rank=1)
    assert out == ["RESULT rank=0 reason=peer step=5", "RESULT rank=1 reason=test step=5"]
    assert ck.latest(env / "resumed") == ck.path_for(env / "resumed", 5)

    out = launch(env, cfg, 2)
    assert out == ["RESULT rank=0 reason=done step=10", "RESULT rank=1 reason=done step=10"]
    assert_same_run(env, "straight", "resumed", rtol=0)


def test_checkpoint_from_two_ranks_resumes_on_one(env):
    """2 卡存的 checkpoint 换成 1 卡续跑：加载器状态是共享的 offset，与卡数无关，
    续跑自检（模型输出 + 下一个 batch）通过，后续结果与一直用 1 卡训练一致。"""
    launch(env, write_cfg(env, "one"), 1)

    cfg = write_cfg(env, "switched")
    launch(env, cfg, 2, stop_step=6, stop_rank=0)
    out = launch(env, cfg, 1)
    assert out == ["RESULT rank=0 reason=done step=10"]
    assert_same_run(env, "one", "switched", rtol=1e-4)


def test_final_weights_only_under_ddp(env):
    """训完压缩：主进程写 final.pt、删完整存档，其他进程在 barrier 上等，不卡死；重跑是空操作。"""
    cfg = write_cfg(env, "final", final_weights_only=True, max_steps=6)
    launch(env, cfg, 2)
    assert ck.final_path(env / "final") is not None and ck.latest(env / "final") is None
    assert records(env, "final", "eval")[-1]["step"] == 6  # 6 不是 eval_every 的倍数，补了最终评测

    out = launch(env, cfg, 2)
    assert out == ["RESULT rank=0 reason=done step=6", "RESULT rank=1 reason=done step=6"]
    assert len(records(env, "final", "eval")) == 2  # 第 4 步一次 + 最终一次，重跑没有多记
