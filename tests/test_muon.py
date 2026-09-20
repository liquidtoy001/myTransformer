"""Muon 优化器（消融 A1）：正交化真的把奇异值推向 1，训练能收敛，和 AdamW 一样能续跑。"""

from __future__ import annotations

import torch
import yaml

from mytransformer.model import ModelConfig, Transformer
from mytransformer.train.muon import Muon, orthogonalize, split_params


def test_orthogonalize_flattens_singular_values():
    """造一个奇异值相差 1000 倍的矩阵，正交化后差距缩到 5 倍以内。

    注意 Newton-Schulz 5 步**不是精确正交化**：奇异值落在 1 附近（这里实测 0.34–1.20），
    不会都等于 1。Muon 要的就是"别让少数方向主导"，够用即可，多迭代几步才更接近 1。
    """
    torch.manual_seed(0)
    u, _ = torch.linalg.qr(torch.randn(64, 64))
    v, _ = torch.linalg.qr(torch.randn(32, 32))
    g = u[:, :32] @ torch.diag(torch.logspace(0, -3, 32)) @ v.T
    before = torch.linalg.svdvals(g)
    assert before.max() / before.min() > 500

    sv = torch.linalg.svdvals(orthogonalize(g).float())
    assert sv.max() / sv.min() < 5 and sv.max() < 1.3, sv
    sv10 = torch.linalg.svdvals(orthogonalize(g, steps=10).float())
    assert sv10.min() > sv.min()  # 迭代更多步更接近 1


def test_orthogonalize_keeps_direction():
    """正交化不应该把矩阵变成别的东西：和原矩阵的方向仍然高度一致。"""
    torch.manual_seed(0)
    g = torch.randn(32, 16)
    out = orthogonalize(g).float()
    cos = torch.sum(out * g) / (out.norm() * g.norm())
    assert cos > 0.8


def test_orthogonalize_handles_tall_and_wide():
    """高矩阵和宽矩阵都要能处理（实现里对高矩阵先转置）。"""
    torch.manual_seed(0)
    for shape in [(64, 16), (16, 64)]:
        sv = torch.linalg.svdvals(orthogonalize(torch.randn(*shape)).float())
        assert 0.6 < sv.min() and sv.max() < 1.3, (shape, sv)


def test_muon_minimizes_a_reachable_quadratic():
    """最小化 ||WX − W*X||²：最优解可达（目标由另一个矩阵生成），loss 应降到接近 0。"""
    torch.manual_seed(0)
    x = torch.randn(16, 64)
    target = torch.randn(8, 16) @ x
    w = torch.zeros(8, 16, requires_grad=True)
    opt = Muon([w], lr=0.1)
    losses = []
    for _ in range(200):
        loss = ((w @ x) - target).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.01 * losses[0], (losses[0], losses[-1])


def test_muon_overfits_a_single_batch():
    """和 tests/test_overfit.py 同样的检验，换成 Muon：能把一个 batch 背下来，说明更新方向是对的。"""
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=128, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2,
                      head_dim=16, ffn_hidden=128, max_seq_len=32)
    model = Transformer(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 24))
    muon_params, adamw_params = split_params(model)
    opts = [Muon(muon_params, lr=0.02), torch.optim.AdamW(adamw_params, lr=3e-3, betas=(0.9, 0.95))]
    for _ in range(300):
        loss = model(ids, targets=ids)["loss"]
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for o in opts:
            o.step()
    assert loss.item() < 0.1, f"300 步后 loss 仍为 {loss.item():.4f}"


def test_split_params_keeps_embeddings_in_adamw():
    cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2,
                      head_dim=8, ffn_hidden=64, max_seq_len=32)
    model = Transformer(cfg)
    muon, adamw = split_params(model)
    assert all(p.ndim == 2 for p in muon)
    ids = {id(p) for p in muon}
    assert id(model.embed_tokens.weight) not in ids      # 每行是一个词向量，正交化没有意义
    assert all(p.ndim == 1 for p in adamw if id(p) != id(model.embed_tokens.weight))
    assert len(muon) + len(adamw) == len(list(model.parameters()))


def test_muon_resumes_bitwise(tmp_path):
    """整条训练链路走 Muon：中断续跑后的权重与不中断逐位相同（Muon 的动量也要存进 checkpoint）。"""
    import numpy as np

    from mytransformer.data.shards import write_shard
    from mytransformer.train import checkpoint as ck
    from mytransformer.train.config import TrainConfig
    from mytransformer.train.trainer import Trainer

    vocab, seq = 64, 16
    (tmp_path / "m.yaml").write_text(yaml.safe_dump(dict(
        vocab_size=vocab, d_model=32, n_layers=2, n_heads=2, n_kv_heads=1,
        head_dim=16, ffn_hidden=64, max_seq_len=seq)))
    rng = np.random.default_rng(0)
    for i in range(2):
        write_shard(tmp_path / "data" / f"train_{i}.bin", rng.integers(0, vocab, 2000))

    def cfg(run: str, steps: int) -> TrainConfig:
        return TrainConfig(
            model=str(tmp_path / "m.yaml"), train_data=str(tmp_path / "data" / "train_*.bin"),
            run_dir=str(tmp_path / run), seq_len=seq, micro_batch_size=4,
            global_batch_tokens=4 * seq, max_steps=steps, lr=1e-2, warmup_steps=1,
            dtype="float32", ce_chunk=0, optimizer="muon", muon_lr=0.05,
            log_every=1, ckpt_every=4, ckpt_keep=3,
        )

    Trainer(cfg("full", 8), device="cpu", log=lambda _: None).fit()
    Trainer(cfg("split", 4), device="cpu", log=lambda _: None).fit()       # 先训一半
    Trainer(cfg("split", 8), device="cpu", log=lambda _: None).fit()       # 续跑到底
    a = ck.load(ck.latest(tmp_path / "full"))["model"]
    b = ck.load(ck.latest(tmp_path / "split"))["model"]
    for k in a:
        assert torch.equal(a[k], b[k]), k
