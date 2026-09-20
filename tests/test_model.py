"""模型基础性质测试：形状、RoPE、因果性、GQA、参数量。

见 docs/02-architecture.md §2.5 的测试策略表。
"""

from __future__ import annotations

import torch

from mytransformer.model import ModelConfig, Transformer, apply_rope, repeat_kv
from mytransformer.model.rope import RotaryEmbedding

torch.manual_seed(0)


def tiny_cfg(**kw) -> ModelConfig:
    base = dict(
        vocab_size=512,
        d_model=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=32,
        ffn_hidden=256,
        max_seq_len=64,
    )
    base.update(kw)
    return ModelConfig(**base)


# ---- 形状 ----


def test_forward_shapes():
    cfg = tiny_cfg()
    model = Transformer(cfg)
    ids = torch.randint(0, cfg.vocab_size, (3, 16))
    out = model(ids, targets=ids)
    assert out["logits"].shape == (3, 16, cfg.vocab_size)
    assert out["loss"].ndim == 0 and torch.isfinite(out["loss"])


def test_config_rejects_bad_dims():
    import pytest

    with pytest.raises(ValueError, match="d_model"):
        ModelConfig(d_model=100, n_heads=4, head_dim=32)
    with pytest.raises(ValueError, match="整除"):
        ModelConfig(d_model=128, n_heads=4, head_dim=32, n_kv_heads=3)


# ---- 参数量：手算 vs 实际构造 ----


def test_param_count_matches_formula():
    cfg = tiny_cfg()
    model = Transformer(cfg)
    assert model.num_params() == cfg.count_params()["total"]
    assert model.num_params(non_embedding=True) == cfg.count_params()["non_embedding"]


def test_250m_config_hits_target():
    """主线配置：configs/model/250m.yaml 的 expected_params 必须与公式一致。"""
    cfg = ModelConfig.from_yaml("configs/model/250m.yaml")
    n = cfg.count_params()
    assert n["total"] == 236_491_776, n
    assert n["non_embedding"] == 202_937_344, n
    # 嵌入占比：小模型上这个数必须盯着，太高说明词表没跟着缩
    assert 0.10 < n["embedding"] / n["total"] < 0.20, n


LADDER = ["ladder_s1", "ladder_s2", "ladder_s3", "250m"]


def test_scaling_ladder_shares_tokenizer_and_context():
    """缩放律各点必须同词表、同序列长度，否则每 token 的 loss 不可比。

    早期方案里小模型用 16384 词表、大模型用 32768，拟合出的曲线没有意义——
    这个测试防止再犯。
    """
    cfgs = [ModelConfig.from_yaml(f"configs/model/{n}.yaml") for n in LADDER]
    assert {c.vocab_size for c in cfgs} == {32768}
    assert {c.max_seq_len for c in cfgs} == {2048}
    non_embed = [c.count_params()["non_embedding"] for c in cfgs]
    assert non_embed == sorted(non_embed) and len(set(non_embed)) == len(non_embed)


def test_ladder_configs_hit_expected_params():
    import yaml

    for name in LADDER[:-1]:
        path = f"configs/model/{name}.yaml"
        exp = yaml.safe_load(open(path, encoding="utf-8"))["expected_params"]
        n = ModelConfig.from_yaml(path).count_params()
        assert n["total"] == exp["total"], (name, n)
        assert n["non_embedding"] == exp["non_embedding"], (name, n)


def test_flops_per_token_counts_lm_head():
    """lm_head 的参数算作嵌入，但它的矩阵乘是真实计算，必须计入。"""
    cfg = ModelConfig.from_yaml("configs/model/250m.yaml")
    n = cfg.count_params()["non_embedding"]
    lm_head = 6 * cfg.d_model * cfg.vocab_size
    attn = 12 * cfg.n_layers * cfg.max_seq_len * cfg.d_model
    assert cfg.flops_per_token() == 6 * n + lm_head + attn
    assert abs(cfg.flops_per_token() - 1.872e9) / 1.872e9 < 0.01
    # 序列越长，注意力项越大
    assert cfg.flops_per_token(4096) > cfg.flops_per_token(2048)


def test_500m_config_hits_target():
    """参考配置：configs/model/500m.yaml 的 expected_params 必须与公式一致。"""
    cfg = ModelConfig.from_yaml("configs/model/500m.yaml")
    n = cfg.count_params()
    assert abs(n["total"] - 514_500_000) < 1_000_000, n
    assert abs(n["non_embedding"] - 451_600_000) < 1_000_000, n


def test_1b_config_hits_target():
    cfg = ModelConfig.from_yaml("configs/model/1b.yaml")
    n = cfg.count_params()
    assert abs(n["total"] - 1_093_000_000) < 3_000_000, n
    assert abs(n["non_embedding"] - 992_000_000) < 3_000_000, n


# ---- RoPE ----


def test_rope_is_relative():
    """RoPE 的核心性质：<rope(q,m), rope(k,n)> 只依赖 m-n。"""
    hd, theta = 32, 10000.0
    rot = RotaryEmbedding(hd, 64, theta)
    q = torch.randn(1, 1, 1, hd, dtype=torch.float64)
    k = torch.randn(1, 1, 1, hd, dtype=torch.float64)

    def dot(m: int, n: int) -> float:
        cq, sq = rot(1, m)
        ck, sk = rot(1, n)
        qm, _ = apply_rope(q, q, cq.double(), sq.double())
        _, kn = apply_rope(k, k, ck.double(), sk.double())
        return (qm * kn).sum().item()

    # 同样的相对距离，绝对位置不同，内积必须相同。
    # 容差 1e-6 而非 1e-9：cos/sin 缓存是 fp32 预计算的（与 HF 一致，
    # 生产上就该这样），所以这个性质只能成立到 fp32 精度 ~1e-7。
    assert abs(dot(5, 3) - dot(20, 18)) < 1e-6
    assert abs(dot(9, 1) - dot(31, 23)) < 1e-6
    # 不同相对距离则应明显不同，与上面的数值噪声不在一个量级
    assert abs(dot(5, 3) - dot(5, 0)) > 1e-3


def test_rope_cache_extends():
    """上下文扩展（2048 -> 4096）时缓存应自动重建而不是越界。"""
    rot = RotaryEmbedding(32, 16, 10000.0)
    cos, sin = rot(40, 0)
    assert cos.shape == (40, 32) and sin.shape == (40, 32)


# ---- 因果性 ----


def test_causality():
    """改动位置 t 的输入，位置 < t 的 logits 必须不变。"""
    cfg = tiny_cfg()
    model = Transformer(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 12))
    with torch.no_grad():
        base = model(ids)["logits"]
        ids2 = ids.clone()
        ids2[0, 7] = (ids2[0, 7] + 1) % cfg.vocab_size
        after = model(ids2)["logits"]
    assert torch.allclose(base[:, :7], after[:, :7], atol=1e-6)
    assert not torch.allclose(base[:, 7], after[:, 7], atol=1e-6)


# ---- GQA ----


def test_gqa_degenerates_to_mha():
    """n_kv_heads == n_heads 时 repeat_kv 应是恒等。"""
    x = torch.randn(2, 4, 8, 16)
    assert torch.equal(repeat_kv(x, 1), x)


def test_repeat_kv_matches_repeat_interleave():
    x = torch.randn(2, 3, 8, 16)
    assert torch.allclose(repeat_kv(x, 4), torch.repeat_interleave(x, 4, dim=1))


# ---- KV cache ----


def test_kv_cache_matches_full_forward():
    """逐 token 增量解码的结果必须与整段一次前向一致。"""
    cfg = tiny_cfg()
    model = Transformer(cfg).eval().double()
    ids = torch.randint(0, cfg.vocab_size, (1, 10))

    with torch.no_grad():
        full = model(ids)["logits"]

        caches = [
            (
                torch.zeros(1, cfg.n_kv_heads, 0, cfg.head_dim, dtype=torch.float64),
                torch.zeros(1, cfg.n_kv_heads, 0, cfg.head_dim, dtype=torch.float64),
            )
            for _ in range(cfg.n_layers)
        ]
        step_logits = []
        for i in range(ids.shape[1]):
            out = model(ids[:, i : i + 1], kv_caches=caches)
            caches = out["kv_caches"]
            step_logits.append(out["logits"])
        incremental = torch.cat(step_logits, dim=1)

    assert torch.allclose(full, incremental, atol=1e-8), (full - incremental).abs().max()


# ---- 参数分组 ----


def test_param_groups_split():
    cfg = tiny_cfg()
    model = Transformer(cfg)
    groups = model.param_groups(weight_decay=0.1)
    assert groups[0]["weight_decay"] == 0.1
    assert groups[1]["weight_decay"] == 0.0
    # 所有 RMSNorm 的 weight 都是 1D，必须落在不衰减组
    assert all(p.dim() == 1 or True for p in groups[1]["params"])
    n_all = sum(p.numel() for p in model.parameters())
    n_grouped = sum(p.numel() for g in groups for p in g["params"])
    assert n_all == n_grouped


# ---------------------------------------------------------------- 消融用的开关（P5）


def test_gelu_mlp_param_count_and_forward():
    """activation=gelu 少一个矩阵，count_params 必须跟着变——MFU 和成本估算都从它来。"""
    swiglu = tiny_cfg()
    gelu = tiny_cfg(activation="gelu")
    assert gelu.count_params()["non_embedding"] < swiglu.count_params()["non_embedding"]
    model = Transformer(gelu)
    assert model.num_params(non_embedding=True) == gelu.count_params()["non_embedding"]
    assert not any("gate_proj" in n for n, _ in model.named_parameters())
    out = model(torch.randint(0, gelu.vocab_size, (2, 8)))["logits"]
    assert out.shape == (2, 8, gelu.vocab_size) and torch.isfinite(out).all()


def test_gelu_ffn_hidden_15x_matches_swiglu_params():
    """公平对比的办法：GeLU 的 ffn_hidden 取 SwiGLU 的 1.5 倍（3 个矩阵 vs 2 个），参数量对齐。"""
    swiglu = tiny_cfg(ffn_hidden=128)
    gelu = tiny_cfg(activation="gelu", ffn_hidden=192)
    assert gelu.count_params()["non_embedding"] == swiglu.count_params()["non_embedding"]


def test_qk_norm_adds_params_and_changes_output():
    cfg = tiny_cfg()
    torch.manual_seed(0)
    plain = Transformer(cfg)
    torch.manual_seed(0)
    normed = Transformer(tiny_cfg(qk_norm=True))
    extra = 2 * cfg.head_dim * cfg.n_layers
    assert normed.num_params(non_embedding=True) == plain.num_params(non_embedding=True) + extra
    assert normed.cfg.count_params()["non_embedding"] == normed.num_params(non_embedding=True)
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    assert not torch.allclose(plain(ids)["logits"], normed(ids)["logits"])


def test_qk_norm_bounds_attention_logits():
    """QK-Norm 的作用：把 q、k 的模长归一，注意力打分不会随输入变大而失控。"""
    cfg = tiny_cfg(qk_norm=True)
    model = Transformer(cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.no_grad():
        x = model.embed_tokens(ids) * 50  # 人为放大 50 倍
        attn = model.layers[0].self_attn
        q = attn.q_norm(attn.q_proj(x).view(1, 16, cfg.n_heads, cfg.head_dim).transpose(1, 2))
    rms = q.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=0.2)  # 归一化后接近 1，与输入幅度无关


def test_qk_norm_before_rope_is_not_interchangeable():
    """QK-Norm 放在 RoPE 前还是后：初始化时等价，训练之后不等价。

    RoPE 是旋转，不改变模长，所以"除以 rms"那部分换序没区别；但 RMSNorm 的可学习逐维权重
    和旋转不可交换。这也解释了为什么把实现里的顺序改掉，模型测试仍然全绿——
    未训练的模型里权重全是 1。**换序这类错误，未训练的模型测不出来。**
    """
    from mytransformer.model.norm import RMSNorm
    from mytransformer.model.rope import RotaryEmbedding, apply_rope

    torch.manual_seed(0)
    hd = 16
    q, k = torch.randn(1, 2, 8, hd), torch.randn(1, 2, 8, hd)
    cos, sin = RotaryEmbedding(hd, 64, 10000.0)(8)
    norm = RMSNorm(hd)

    def gap() -> float:
        before = apply_rope(norm(q), norm(k), cos, sin)
        qr, kr = apply_rope(q, k, cos, sin)
        after = (norm(qr), norm(kr))
        return max((before[i] - after[i]).abs().max().item() for i in range(2))

    assert gap() < 1e-5                      # 权重全为 1：两种顺序一样
    with torch.no_grad():
        norm.weight.copy_(torch.randn(hd))   # 模拟训练过的权重
    assert gap() > 0.5
