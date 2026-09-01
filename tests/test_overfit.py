"""单 batch 过拟合测试。

模型能不能把一个 batch 背下来，是训练循环有没有写对的最基本判据。
loss 降不到 0.1 以下，说明梯度流、优化器或 loss 计算里有问题。
"""

from __future__ import annotations

import torch

from mytransformer.model import ModelConfig, Transformer


def test_overfit_single_batch():
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=128,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        ffn_hidden=128,
        max_seq_len=32,
    )
    model = Transformer(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 24))

    opt = torch.optim.AdamW(model.param_groups(0.0), lr=3e-3, betas=(0.9, 0.95))
    first = None
    for step in range(300):
        loss = model(ids, targets=ids)["loss"]
        if first is None:
            first = loss.item()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    assert first > 4.0, f"初始 loss {first:.3f} 应接近 ln(vocab)={torch.tensor(128.0).log():.3f}"
    assert loss.item() < 0.1, f"300 步后 loss 仍为 {loss.item():.4f}，训练循环有问题"
