"""loss 计算：分块交叉熵、z-loss、shift 语义。"""

from __future__ import annotations

import pytest
import torch

from mytransformer.model import ModelConfig, Transformer


def small_model() -> Transformer:
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=512, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2,
        head_dim=16, ffn_hidden=128, max_seq_len=64,
    )
    return Transformer(cfg).double()


def loss_and_grads(model, ids, targets, **kw):
    model.zero_grad(set_to_none=True)
    loss = model(ids, targets, **kw)["loss"]
    loss.backward()
    return loss.detach(), {n: p.grad.clone() for n, p in model.named_parameters()}


@pytest.mark.parametrize("shift", [True, False])
@pytest.mark.parametrize("z_loss", [0.0, 1e-4])
@pytest.mark.parametrize("chunk", [7, 64, 1000])  # 不整除、整除、比总长还大
def test_chunked_ce_matches_plain(shift, z_loss, chunk):
    """分块 + 激活重算只改变显存，不能改变 loss 和梯度。"""
    model = small_model()
    ids = torch.randint(0, 512, (3, 40))
    tgt = torch.randint(0, 512, (3, 40))
    tgt[0, 5] = -100  # 被屏蔽的位置也要一致处理

    l_plain, g_plain = loss_and_grads(model, ids, tgt, shift=shift, z_loss=z_loss)
    l_chunk, g_chunk = loss_and_grads(model, ids, tgt, shift=shift, z_loss=z_loss, ce_chunk=chunk)

    assert torch.allclose(l_plain, l_chunk, atol=1e-12), (l_plain, l_chunk)
    for n in g_plain:
        assert torch.allclose(g_plain[n], g_chunk[n], atol=1e-12), n


def test_chunked_mode_does_not_return_logits():
    model = small_model()
    ids = torch.randint(0, 512, (2, 16))
    out = model(ids, ids, ce_chunk=8)
    assert out["logits"] is None and out["loss"] is not None


def test_shift_false_uses_every_position():
    """shift=False 时 targets 已是下一个 token：等价于 shift=True 喂一个多一位的序列。"""
    model = small_model()
    seq = torch.randint(0, 512, (2, 33))
    x, y = seq[:, :-1], seq[:, 1:]
    a = model(x, y, shift=False)["loss"]
    # shift=True 看的是 seq[:, :-1] 预测 seq[:, 1:]，输入同样是前 32 个 token
    logits = model(x)["logits"]
    b = torch.nn.functional.cross_entropy(logits.reshape(-1, 512), y.reshape(-1))
    assert torch.allclose(a, b, atol=1e-12)


def test_z_loss_penalizes_logit_drift():
    model = small_model()
    ids = torch.randint(0, 512, (2, 16))
    base = model(ids, ids, z_loss=0.0)["loss"]
    with_z = model(ids, ids, z_loss=1e-2)["loss"]
    assert with_z > base


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA 才能测显存")
def test_chunked_ce_reduces_peak_memory():
    """词表大、序列长时，分块必须显著降低峰值显存——这是它存在的唯一理由。"""
    cfg = ModelConfig(
        vocab_size=32768, d_model=256, n_layers=2, n_heads=4, n_kv_heads=2,
        head_dim=64, ffn_hidden=704, max_seq_len=2048,
    )
    torch.manual_seed(0)
    model = Transformer(cfg).cuda()
    ids = torch.randint(0, cfg.vocab_size, (4, 2048), device="cuda")

    def peak(**kw) -> float:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(ids, ids, shift=False, **kw)["loss"]
        loss.backward()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 2**30

    plain, chunked = peak(), peak(ce_chunk=1024)
    # 4×2048×32768 的 logits：bf16 0.5 GiB、fp32 1 GiB，加上 softmax 中间量与梯度
    assert chunked < plain * 0.5, f"分块 {chunked:.2f} GiB vs 不分块 {plain:.2f} GiB"
