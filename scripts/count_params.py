"""核对配置的参数量与显存占用。用法: python scripts/count_params.py configs/model/500m.yaml"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from mytransformer.model import ModelConfig


def main(paths: list[str]) -> int:
    rc = 0
    for p in paths:
        cfg = ModelConfig.from_yaml(p)
        n = cfg.count_params()
        raw = yaml.safe_load(Path(p).read_text(encoding="utf-8"))
        print(f"\n=== {p} ===")
        print(f"  层数/隐藏维      {cfg.n_layers} / {cfg.d_model}")
        print(f"  注意力           {cfg.n_heads} Q / {cfg.n_kv_heads} KV, head_dim={cfg.head_dim}")
        print(f"  单层参数         {n['per_layer']:>15,}")
        print(f"  非嵌入参数       {n['non_embedding']:>15,}")
        print(f"  嵌入参数         {n['embedding']:>15,}  ({'绑定' if cfg.tie_embeddings else '独立'})")
        print(f"  总参数           {n['total']:>15,}")
        print(f"  bf16 权重        {n['bf16_bytes'] / 1024**3:>15.2f} GB")
        # 训练显存：bf16 权重2 + bf16 梯度2 + fp32 主权重4 + Adam m,v 8 = 16 B/参数
        print(f"  训练状态(AdamW)  {n['total'] * 16 / 1024**3:>15.2f} GB  (不含激活值)")

        exp = raw.get("expected_params")
        if exp:
            for key, got in (("total", n["total"]), ("non_embedding", n["non_embedding"])):
                if key in exp:
                    diff = abs(got - exp[key]) / exp[key]
                    ok = diff < 0.01
                    rc |= 0 if ok else 1
                    print(f"  [{'OK' if ok else '不符'}] {key}: 实际 {got:,} vs 预期 {exp[key]:,} (偏差 {diff:.2%})")
    return rc


if __name__ == "__main__":
    args = sys.argv[1:] or ["configs/model/500m.yaml", "configs/model/1b.yaml"]
    raise SystemExit(main(args))
