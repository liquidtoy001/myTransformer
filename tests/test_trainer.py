"""训练循环：续跑等价、续跑自检、各种退出方式、梯度累积等价、分叉、NaN 保护。

全部在 CPU + fp32 上用极小的模型跑，几秒钟完成。CPU fp32 的计算是确定性的，
所以"续跑"与"不中断"可以要求逐位相同，而不只是近似相等。
"""

from __future__ import annotations

import json
import os
import signal

import numpy as np
import pytest
import torch
import yaml

from mytransformer.data.shards import write_shard
from mytransformer.train import checkpoint as ck
from mytransformer.train.config import TrainConfig
from mytransformer.train.trainer import ResumeError, Trainer

VOCAB, SEQ = 128, 16


@pytest.fixture
def env(tmp_path):
    """一个极小的模型配置 + 几个随机数据分片。"""
    model_yaml = tmp_path / "tiny.yaml"
    model_yaml.write_text(yaml.safe_dump(dict(
        vocab_size=VOCAB, d_model=32, n_layers=2, n_heads=2, n_kv_heads=1,
        head_dim=16, ffn_hidden=64, max_seq_len=SEQ,
    )))
    rng = np.random.default_rng(0)
    for i in range(3):
        write_shard(tmp_path / "data" / f"train_{i:03d}.bin", rng.integers(0, VOCAB, 2000))
    write_shard(tmp_path / "data" / "val_000.bin", rng.integers(0, VOCAB, 2000))
    return tmp_path


def make_cfg(env, run="run", **kw) -> TrainConfig:
    base = dict(
        model=str(env / "tiny.yaml"),
        train_data=str(env / "data" / "train_*.bin"),
        val_data=str(env / "data" / "val_*.bin"),
        run_dir=str(env / run),
        seq_len=SEQ, micro_batch_size=4, global_batch_tokens=4 * SEQ * 2,  # 梯度累积 2 步
        max_steps=12, lr=1e-2, warmup_steps=2,
        dtype="float32", ce_chunk=20, z_loss=1e-4,
        log_every=1, eval_every=4, eval_batches=2, ckpt_every=4, ckpt_keep=2,
    )
    base.update(kw)
    return TrainConfig(**base)


def train(cfg, **kw) -> str:
    return Trainer(cfg, device="cpu", log=lambda _: None, **kw).fit()


def stop_at(step):
    return lambda t: t.request_stop("test") if t.step == step else None


def records(env, run="run", kind="train"):
    lines = (env / run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    return [r for r in map(json.loads, lines) if r["type"] == kind]


def final_weights(env, run="run"):
    return ck.load(ck.latest(env / run))["model"]


# ---------------------------------------------------------------- 续跑


def test_resume_is_bitwise_identical_to_uninterrupted(env):
    """核心保证：中途存档退出再续跑，每一步的 loss 和最终权重都与不中断时逐位相同。"""
    assert train(make_cfg(env, "straight")) == "done"

    assert train(make_cfg(env, "resumed"), on_step=stop_at(5)) == "test"
    assert ck.latest(env / "resumed").name == "ckpt_00000005.pt"
    assert train(make_cfg(env, "resumed")) == "done"

    a = {r["step"]: r["loss"] for r in records(env, "straight")}
    b = {r["step"]: r["loss"] for r in records(env, "resumed")}
    assert sorted(b) == list(range(1, 13)), "续跑后每一步都应恰好记录一次"
    assert a == b
    wa, wb = final_weights(env, "straight"), final_weights(env, "resumed")
    assert all(torch.equal(wa[k], wb[k]) for k in wa)


@pytest.mark.parametrize("delta, caught_by", [
    (0.01, "模型输出"),    # loss 变 2e-5、均方根变 1e-2：两项都超过容差
    (0.001, "logit_rms"),  # loss 只变 2e-6，低于 1e-5 的容差——只有均方根（变 1e-3）抓得到
])
def test_resume_detects_changed_weights(env, delta, caught_by):
    """训练初期模型接近均匀分布，loss 对权重的整体偏移很迟钝，所以自检还要比 logits 均方根。"""
    train(make_cfg(env), on_step=stop_at(4))
    path = ck.latest(env / "run")
    state = torch.load(path, weights_only=True)
    state["model"]["norm.weight"] += delta
    torch.save(state, path)
    with pytest.raises(ResumeError, match=caught_by):
        train(make_cfg(env))


def test_resume_detects_changed_data(env):
    train(make_cfg(env), on_step=stop_at(4))
    rng = np.random.default_rng(99)
    for i in range(3):  # 分片内容被换掉（比如重新分词了），大小不变
        write_shard(env / "data" / f"train_{i:03d}.bin", rng.integers(0, VOCAB, 2000))
    with pytest.raises(ResumeError, match="数据流"):
        train(make_cfg(env))


def test_finished_run_is_a_noop(env):
    train(make_cfg(env))
    n = len(records(env))
    assert train(make_cfg(env)) == "done"
    assert len(records(env)) == n, "已训完的 run 再启动不应多跑任何一步"


# ---------------------------------------------------------------- 退出方式


def test_time_budget_saves_and_exits(env):
    assert train(make_cfg(env), max_minutes=1e-9) == "time"
    assert ck.latest(env / "run").name == "ckpt_00000001.pt"
    assert train(make_cfg(env)) == "done"


def test_signal_saves_and_exits(env):
    """SLURM 的 USR1/TERM 只设标志，当前这一步做完再存档。直接调处理函数模拟信号到达。"""
    reason = train(make_cfg(env), on_step=lambda t: t._on_signal(signal.SIGTERM, None) if t.step == 3 else None)
    assert reason == "signal:SIGTERM"
    assert ck.latest(env / "run").name == "ckpt_00000003.pt"


POSIX_SIGNALS = pytest.mark.skipif(
    not hasattr(signal, "SIGUSR1"), reason="需要真实的 POSIX 信号：在 Linux（Rangpur 冒烟第 1 步）上运行，Windows 自动跳过"
)


def kill_self_at(step):
    return lambda t: os.kill(os.getpid(), signal.SIGUSR1) if t.step == step else None


@POSIX_SIGNALS
def test_real_sigusr1_when_blocked_at_start(env):
    """H1：进程启动时 SIGUSR1 就处于屏蔽状态。训练循环必须自己解除屏蔽，真实信号才能生效。"""
    before = signal.pthread_sigmask(signal.SIG_BLOCK, [signal.SIGUSR1])
    try:
        reason = train(make_cfg(env), on_step=kill_self_at(3))
    finally:
        # 修复失效时信号会一直挂起；这里一解除屏蔽它就会被递送，而此时处理函数已恢复成默认行为，
        # 会把整个 pytest 进程杀掉。先设成忽略——挂起的信号会被丢弃——再恢复掩码和处理函数。
        prev = signal.signal(signal.SIGUSR1, signal.SIG_IGN)
        signal.pthread_sigmask(signal.SIG_SETMASK, before)
        signal.signal(signal.SIGUSR1, prev)
    assert reason == "signal:SIGUSR1"
    assert ck.latest(env / "run").name == "ckpt_00000003.pt"


@POSIX_SIGNALS
def test_real_sigusr1_after_handler_replaced_in_first_step(env):
    """H2：第一步期间（torch.compile 编译时）有库把处理函数换掉了。第一步做完后必须重新装上。"""
    trainer = Trainer(make_cfg(env), device="cpu", log=lambda _: None, on_step=kill_self_at(3))
    real_fwd = trainer.fwd

    def clobbering_fwd(*args, **kw):
        if trainer.step == 0:
            signal.signal(signal.SIGUSR1, lambda *_: None)  # 模拟某个库在编译时换掉了处理函数
        return real_fwd(*args, **kw)

    trainer.fwd = clobbering_fwd
    assert trainer.fit() == "signal:SIGUSR1"


def test_nonfinite_grad_skips_update(env):
    """某一步梯度变成 NaN：跳过这一步的更新，权重保持不变，训练继续。"""
    snapshots = {}
    trainer = Trainer(make_cfg(env), device="cpu", log=lambda _: None,
                      on_step=lambda t: snapshots.__setitem__(t.step, {k: v.clone() for k, v in t.model.state_dict().items()}))
    real_fwd = trainer.fwd

    def poisoned(*args, **kw):
        out = real_fwd(*args, **kw)
        if trainer.step == 2:  # 第 3 步（step 从 0 计）
            out["loss"] = out["loss"] * float("nan")
        return out

    trainer.fwd = poisoned
    assert trainer.fit() == "done"
    assert all(torch.equal(snapshots[2][k], snapshots[3][k]) for k in snapshots[2]), "NaN 那一步不应更新权重"
    assert not all(torch.equal(snapshots[3][k], snapshots[4][k]) for k in snapshots[3]), "之后应恢复正常更新"
    assert all(torch.isfinite(v).all() for v in snapshots[12].values())


# ---------------------------------------------------------------- 全局 batch 与分叉


def test_grad_accum_matches_bigger_micro_batch(env):
    """跨硬件纪律第 1 条：全局 batch 不变时，micro 4 × 累积 2 与 micro 8 × 累积 1 数学上等价。

    不是逐位相同（浮点加法顺序不同），但必须在舍入误差以内。
    """
    train(make_cfg(env, "accum", micro_batch_size=4))
    train(make_cfg(env, "big", micro_batch_size=8))
    a = [r["loss"] for r in records(env, "accum")]
    b = [r["loss"] for r in records(env, "big")]
    assert a == pytest.approx(b, rel=1e-5)


def test_fork_continues_from_another_run(env):
    """缩放律分叉：从主训练的 checkpoint 起步，换一条从分叉点开始衰减的调度。"""
    train(make_cfg(env, "main"), on_step=stop_at(8))
    fork = make_cfg(env, "fork", init_from=str(ck.latest(env / "main")), max_steps=10, decay_start=8)
    assert train(fork) == "done"

    recs = records(env, "fork")
    assert [r["step"] for r in recs] == [9, 10], "分叉从第 8 步接着走"
    assert [r["lr"] for r in recs] == pytest.approx([1e-2, 5e-3]), "从分叉点开始线性衰减"
    assert ck.latest(env / "main").name == "ckpt_00000008.pt", "分叉不能改动主训练的目录"


# ---------------------------------------------------------------- 配置


def test_config_rejects_unknown_keys(env, tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump({"model": "x", "train_data": "y", "warmup_step": 10}))
    with pytest.raises(ValueError, match="warmup_step"):
        TrainConfig.from_yaml(p)


@pytest.mark.parametrize("path", sorted(__import__("glob").glob("configs/train/*.yaml")))
def test_repo_train_configs_are_valid(path):
    """仓库里每个训练配置都必须能加载、batch 能整除、引用的模型配置存在。

    pretrain_250m.yaml 曾经是在训练循环写好之前按文档手写的，字段对不上，根本加载不了——
    这个测试保证这种情况在跑测试时就暴露，而不是等到租了 8×H100 才发现。
    """
    from pathlib import Path

    cfg = TrainConfig.from_yaml(path)
    assert Path(cfg.model).is_file(), f"{path} 引用的模型配置 {cfg.model} 不存在"
    assert cfg.grad_accum_steps() >= 1
    from mytransformer.model import ModelConfig
    assert cfg.seq_len <= ModelConfig.from_yaml(cfg.model).max_seq_len


def test_config_rejects_indivisible_batch(env):
    with pytest.raises(ValueError, match="整除"):
        make_cfg(env, global_batch_tokens=100).grad_accum_steps()


def test_cli_overrides_parse_types():
    from mytransformer.train.pretrain import parse_overrides

    got = parse_overrides(["compile=true", "lr=1e-3", "max_steps=50", "betas=[0.9, 0.99]", "run_dir=runs/x"])
    assert got == {"compile": True, "lr": 0.001, "max_steps": 50, "betas": [0.9, 0.99], "run_dir": "runs/x"}
    with pytest.raises(ValueError, match="键=值"):
        parse_overrides(["compile"])


def test_cli_override_typo_is_rejected(env, tmp_path):
    """--set 的字段名拼错，和 yaml 里拼错一样直接报错。"""
    from mytransformer.train.pretrain import main

    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(make_cfg(env).to_dict()))
    with pytest.raises(ValueError, match="complie"):
        main(["--config", str(p), "--set", "complie=true"])


# ---------------------------------------------------------------- 数据核对与分子集验证


def test_val_loss_reported_per_file(env):
    """val_*.bin 有多个时，每个文件（子集）单独报 loss，val_loss 是它们的等权平均。"""
    rng = np.random.default_rng(1)
    write_shard(env / "data" / "val_zh.bin", rng.integers(0, VOCAB, 2000))
    train(make_cfg(env, max_steps=4))
    rec = records(env, kind="eval")[-1]
    assert set(rec["val"]) == {"000", "zh"}
    assert rec["val_loss"] == pytest.approx(sum(rec["val"].values()) / 2)


def test_refuses_shards_from_another_tokenizer(env):
    """分片目录的 meta.json 记录的分词器指纹和配置的分词器不一致：训练前就报错，而不是训出乱码。"""
    from pathlib import Path

    mt32k = Path(__file__).resolve().parents[1] / "tokenizer" / "mt32k.json"
    shards = sorted((env / "data").glob("*.bin"))
    n = {p.name: p.stat().st_size // 2 for p in shards}
    meta = {"tokenizer": "0000000000000000",
            "train": {"shards": [{"file": k, "tokens": v} for k, v in n.items() if k.startswith("train")]},
            "val": {"000": {"file": "val_000.bin", "tokens": n["val_000.bin"]}}}
    (env / "data" / "meta.json").write_text(json.dumps(meta))
    big = env / "big.yaml"  # 词表放得下 mt32k 的模型，这样报错只能来自指纹
    big.write_text((env / "tiny.yaml").read_text().replace(f"vocab_size: {VOCAB}", "vocab_size: 32768"))
    with pytest.raises(ValueError, match="分片是用分词器 0000000000000000 切的"):
        Trainer(make_cfg(env, model=str(big), tokenizer=str(mt32k)), device="cpu", log=lambda _: None)


# ---------------------------------------------------------------- 训完之后：最终评测、压缩存档


def test_final_eval_happens_even_when_steps_not_multiple_of_eval_every(env):
    """max_steps=10、eval_every=4：以前只在 4、8 步评测，第 10 步（训完时）的 loss 没有记录。
    WSD 的衰减集中在最后，这恰恰是缩放律要用的那个数。"""
    train(make_cfg(env, max_steps=10, eval_every=4))
    assert [r["step"] for r in records(env, kind="eval")] == [4, 8, 10]


def test_rerun_of_finished_run_adds_missing_final_eval(env):
    """已经训完、但当时没做最终评测的 run（P5 的 ladder_s1/s2 就是这样）：重新提交一次，补上即可，不重训。"""
    cfg = make_cfg(env, max_steps=10, eval_every=4)
    train(cfg)
    lines = (env / "run" / "metrics.jsonl").read_text("utf-8").splitlines()
    kept = [l for l in lines if not (json.loads(l)["type"] == "eval" and json.loads(l)["step"] == 10)]
    (env / "run" / "metrics.jsonl").write_text("\n".join(kept) + "\n", "utf-8")   # 模拟旧版代码的产物
    n_train = len(records(env))

    assert train(cfg) == "done"
    assert [r["step"] for r in records(env, kind="eval")] == [4, 8, 10]
    assert len(records(env)) == n_train  # 没有重新训练


def test_final_weights_only_compacts_and_rerun_is_noop(env):
    cfg = make_cfg(env, max_steps=8, final_weights_only=True)
    train(cfg)
    files = sorted(p.name for p in (env / "run").iterdir() if p.suffix == ".pt")
    assert files == ["final.pt"]
    state = ck.load(env / "run" / "final.pt")
    assert set(state) == {"model", "step", "tokens_seen", "meta"} and state["step"] == 8

    # 再提交一次：认出已训完，不从头训练，也不报错
    n_train = len(records(env))
    assert train(cfg) == "done"
    assert len(records(env)) == n_train
    final = ck.load(env / "run" / "final.pt")["model"]
    assert all(torch.equal(final[k], state["model"][k]) for k in final)


def test_interrupted_run_is_not_compacted(env):
    """只有训完才压缩。中途停下的 run 必须保留完整 checkpoint，否则没法续跑。"""
    train(make_cfg(env, max_steps=12, final_weights_only=True), on_step=stop_at(6))
    names = {p.name for p in (env / "run").iterdir()}
    assert "final.pt" not in names and any(n.startswith("ckpt_") for n in names)
