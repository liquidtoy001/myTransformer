"""P5 实验配置：消融必须只改一处，缩放三点必须可比。

配置写错是最贵的错误之一——跑完 1 小时才发现两组差了两个变量，结论就作废了。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mytransformer.model import ModelConfig
from mytransformer.train.config import TrainConfig

P5 = Path(__file__).resolve().parents[1] / "configs" / "train" / "p5"
BASE = P5 / "_base_s2.yaml"

# 每组消融允许和底座不同的字段（run_dir 一定不同，不列）
ALLOWED = {
    "a1_muon.yaml": {"optimizer", "muon_lr"},
    "a2_gelu.yaml": {"model"},
    "a3_mha.yaml": {"model"},
    "a4_mix_v2.yaml": {"train_data", "val_data"},
    "a5_cosine.yaml": {"schedule"},
    "a6_qknorm_no_zloss.yaml": {"model", "z_loss"},
    "base_seed0.yaml": set(),
    "base_seed1.yaml": {"seed"},
    "base_seed2.yaml": {"seed"},
}


def raw(path: Path) -> dict:
    return yaml.safe_load(path.read_text("utf-8"))


@pytest.mark.parametrize("name", sorted(ALLOWED))
def test_ablation_changes_exactly_one_thing(name):
    base, other = raw(BASE), raw(P5 / name)
    diff = {k for k in base | other if base.get(k) != other.get(k)} - {"run_dir"}
    assert diff == ALLOWED[name], f"{name} 和底座的差异是 {diff}，期望 {ALLOWED[name]}"


def test_every_p5_config_is_loadable():
    for f in sorted(P5.glob("*.yaml")):
        cfg = TrainConfig.from_yaml(f)
        assert cfg.tokenizer and cfg.val_data, f
        cfg.grad_accum_steps()  # global_batch_tokens 必须能被整除


def test_gelu_and_qknorm_variants_keep_parameter_count():
    """A2 必须参数量相等（比的是激活函数）；A6 只多出几个缩放参数。
    A3 的 MHA 天然多 14% 参数——那正是 GQA 省下的，见配置里的说明。"""
    base = ModelConfig.from_yaml("configs/model/ladder_s2.yaml").count_params()["non_embedding"]
    gelu = ModelConfig.from_yaml("configs/model/s2_gelu.yaml").count_params()["non_embedding"]
    qk = ModelConfig.from_yaml("configs/model/s2_qknorm.yaml").count_params()["non_embedding"]
    assert gelu == base
    assert 0 < qk - base < 0.001 * base


def test_ladder_points_are_comparable_and_chinchilla():
    """三个点共用分词器、序列长度、调度；每个点的 tokens ≈ 20 × 非嵌入参数（Chinchilla）。"""
    cfgs = [TrainConfig.from_yaml(P5 / f"ladder_s{i}.yaml") for i in (1, 2, 3)]
    assert len({(c.tokenizer, c.seq_len, c.schedule, c.global_batch_tokens) for c in cfgs}) == 1
    sizes = []
    for c in cfgs:
        m = ModelConfig.from_yaml(c.model)
        assert m.vocab_size == 32768 and m.max_seq_len == 2048
        n = m.count_params()["non_embedding"]
        sizes.append(n)
        assert abs(c.max_steps * c.global_batch_tokens / n - 20) < 0.2, c.model
    assert sizes == sorted(sizes) and sizes[-1] / sizes[0] > 5  # 至少跨一个数量级的一半，拟合才有意义


def test_ablations_and_seeds_train_on_the_same_token_budget():
    """消融之间比的是 loss，训练量必须一样，否则比的是谁训得久。"""
    budgets = {f.name: TrainConfig.from_yaml(f).max_steps * TrainConfig.from_yaml(f).global_batch_tokens
               for f in P5.glob("*.yaml") if not f.name.startswith("ladder")}
    assert len(set(budgets.values())) == 1, budgets
