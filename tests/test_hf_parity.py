"""与 HuggingFace LlamaForCausalLM 的数值对齐测试。

**这是 P3 的生死线。** 这个测试不过，说明架构实现有错，后面训练全是白干。

因为权重命名刻意对齐了 HF Llama，这里可以直接把 state_dict 搬过去，
不需要写转换层——唯一的差别是 HF 在所有非 lm_head 的键上多一层 "model." 前缀。
"""

from __future__ import annotations

import pytest
import torch

from mytransformer.model import ModelConfig, Transformer

transformers = pytest.importorskip("transformers")

torch.manual_seed(0)

CFG = ModelConfig(
    vocab_size=512,
    d_model=128,
    n_layers=3,
    n_heads=4,
    n_kv_heads=2,
    head_dim=32,
    ffn_hidden=256,
    max_seq_len=64,
    rope_theta=10000.0,
    norm_eps=1e-5,
    tie_embeddings=False,  # 先分开测，绑定单独测
)


def build_pair(cfg: ModelConfig):
    """构造我们的模型和 HF 模型，权重完全相同。"""
    from transformers import LlamaConfig, LlamaForCausalLM

    mine = Transformer(cfg).double().eval()

    hf_cfg = LlamaConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.d_model,
        intermediate_size=cfg.ffn_hidden,
        num_hidden_layers=cfg.n_layers,
        num_attention_heads=cfg.n_heads,
        num_key_value_heads=cfg.n_kv_heads,
        head_dim=cfg.head_dim,
        max_position_embeddings=cfg.max_seq_len,
        rms_norm_eps=cfg.norm_eps,
        rope_theta=cfg.rope_theta,
        tie_word_embeddings=cfg.tie_embeddings,
        attention_bias=cfg.attention_bias,
        mlp_bias=False,
        hidden_act="silu",
        attn_implementation="eager",
    )
    hf = LlamaForCausalLM(hf_cfg).double().eval()

    # 键名映射：除 lm_head 外全部加 "model." 前缀
    sd = {
        (k if k.startswith("lm_head") else f"model.{k}"): v
        for k, v in mine.state_dict().items()
    }
    missing, unexpected = hf.load_state_dict(sd, strict=False)
    assert not unexpected, f"我们的模型有 HF 不认识的权重: {unexpected}"
    # HF 只允许缺 rotary 的 inv_freq 这类可重算 buffer
    assert all("inv_freq" in k or "rotary" in k for k in missing), f"HF 缺少权重: {missing}"
    return mine, hf


@pytest.mark.parametrize("tie", [False, True])
def test_logits_match_hf(tie: bool):
    cfg = ModelConfig(**{**CFG.__dict__, "tie_embeddings": tie})
    mine, hf = build_pair(cfg)

    ids = torch.randint(0, cfg.vocab_size, (2, 17))
    with torch.no_grad():
        a = mine(ids)["logits"]
        b = hf(ids).logits

    max_err = (a - b).abs().max().item()
    assert max_err < 1e-4, f"logits 最大误差 {max_err:.3e} 超过 1e-4"


def test_per_layer_hidden_states_match():
    """逐层对齐：某一层开始发散能立刻定位到是哪个模块写错了。"""
    cfg = CFG
    mine, hf = build_pair(cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, 13))

    with torch.no_grad():
        hf_out = hf(ids, output_hidden_states=True)
        hf_states = hf_out.hidden_states  # (embed, layer1, ..., layerN)

        # 注意 HF 的语义：hidden_states 里存的是每层的**输入**（第 0 项即
        # embedding 输出），循环结束后再追加一项 norm(最后一层输出)。
        # 所以最后一项是过了 final norm 的，必须照样处理才有可比性。
        cos, sin = mine.rotary(ids.shape[1], 0)
        x = mine.embed_tokens(ids)
        mine_states = [x]
        for block in mine.layers:
            x, _ = block(x, cos.double(), sin.double())
            mine_states.append(x)
        mine_states[-1] = mine.norm(mine_states[-1])

    errs = []
    for i, (a, b) in enumerate(zip(mine_states, hf_states, strict=True)):
        err = (a - b).abs().max().item()
        errs.append(err)
        label = "embedding" if i == 0 else (f"第 {i} 层后 + final norm" if i == len(mine_states) - 1 else f"第 {i} 层后")
        assert err < 1e-5, f"{label} 输出发散，最大误差 {err:.3e}（各层误差 {errs}）"


def test_loss_matches_hf():
    cfg = CFG
    mine, hf = build_pair(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 20))
    with torch.no_grad():
        my_loss = mine(ids, targets=ids)["loss"]
        hf_loss = hf(ids, labels=ids).loss
    assert abs(my_loss.item() - hf_loss.item()) < 1e-6, (my_loss.item(), hf_loss.item())
