# 架构设计

分两部分：**模型架构**（数学上是什么）和**代码架构**（工程上怎么组织）。

---

## 一、模型架构

### 1.1 总体选型

Decoder-only，LLaMA 系当代标配。每一项选择的理由：

| 组件 | 选择 | 替代方案 | 为什么这么选 |
|---|---|---|---|
| Norm 位置 | **Pre-norm** | Post-norm | Post-norm 在深层需要精细 warmup，pre-norm 训练稳定性显著更好 |
| Norm 类型 | **RMSNorm** | LayerNorm | 去掉均值中心化，少一次 reduce，速度快 ~10%，效果无损 |
| 位置编码 | **RoPE**（base=10000） | 绝对/ALiBi | 相对位置、可外推、生态最成熟；退火期改 base=500000 扩到 4096 |
| 激活 | **SwiGLU** | GeLU | 同等参数下困惑度更低（GLU Variants 论文），代价是 FFN 变三个矩阵 |
| 注意力 | **GQA**（20 Q / 5 KV） | MHA / MQA | KV cache 缩小 4 倍，推理显存和吞吐大幅改善，质量损失可忽略 |
| 内核 | **FlashAttention-2**（`F.scaled_dot_product_attention`） | 朴素实现 | 显存 O(N) 而非 O(N²)，长序列必需 |
| 嵌入 | **权重绑定**（input embedding = lm_head） | 独立 | 省 63M 参数；小模型上绑定通常更好 |
| Bias | **全部去掉** | 带 bias | 现代 LLM 通用做法，省参数且更稳定 |
| Dropout | **0.0** | 0.1 | 预训练数据量远大于参数量，不会过拟合；dropout 只会拖慢收敛 |

**刻意不用的东西（以及原因）**：MoE（工程复杂度对学习目标不划算）、MLA（DeepSeek 的多头潜在注意力，可作为进阶消融）、并行 Attention+FFN（收益不稳定）、QK-Norm（可作为 loss spike 的备选补救手段，见 [06](06-training-recipe.md)）。

### 1.2 超参数（500M，主线）

```yaml
# configs/model/500m.yaml
vocab_size:      49152      # 48 × 1024，对 tensor core 友好
d_model:         1280
n_layers:        26
n_heads:         20
n_kv_heads:      5          # GQA group size = 4
head_dim:        64         # = d_model / n_heads
ffn_hidden:      3456       # ≈ (8/3)·d_model，向上取整到 256 的倍数
max_seq_len:     2048       # 退火期 → 4096
rope_theta:      10000.0    # 退火期 → 500000.0
norm_eps:        1e-5
tie_embeddings:  true
attention_bias:  false
dropout:         0.0
```

### 1.3 参数量推导（务必自己算一遍）

**单层**：

| 部分 | 公式 | 数值 |
|---|---|---|
| `W_q` | d × (n_heads · head_dim) = 1280×1280 | 1.638 M |
| `W_k` | d × (n_kv · head_dim) = 1280×320 | 0.410 M |
| `W_v` | 同上 | 0.410 M |
| `W_o` | 1280×1280 | 1.638 M |
| SwiGLU `W_gate/W_up/W_down` | 3 × 1280 × 3456 | 13.271 M |
| 2× RMSNorm | 2 × 1280 | 0.003 M |
| **单层合计** | | **17.370 M** |

**整模型**：

```
26 层         : 26 × 17.370 M = 451.6 M   ← 非嵌入参数（scaling law 用这个）
嵌入(绑定)     : 49152 × 1280  =  62.9 M
最终 RMSNorm  :                   0.001 M
------------------------------------------
总计                          ≈ 514.5 M
bf16 权重体积                  ≈ 1.03 GB
```

**训练显存**（AdamW，混合精度）：

```
bf16 权重     : 514M × 2  = 1.03 GB
bf16 梯度     : 514M × 2  = 1.03 GB
fp32 主权重   : 514M × 4  = 2.06 GB
Adam m, v     : 514M × 8  = 4.11 GB
--------------------------------------
优化器状态小计            ≈ 8.2 GB
+ 激活值(bs=8, seq=2048, 开梯度检查点) ≈ 4–6 GB
--------------------------------------
单卡峰值                  ≈ 14 GB  → 24GB 的 4090 可单卡训练
```

结论：**500M 不需要 FSDP/ZeRO，单卡 DDP 就够**。仍然要实现 FSDP 路径——因为这是要学的东西，而且退火期扩到 4096 上下文后激活值会翻倍。

### 1.4 1B 变体（可选升级）

架构完全相同，只改宽度和深度。**代码一行不用动，换个 yaml 就行**——这也是把超参全部收进 config 的意义。

```yaml
# configs/model/1b.yaml
vocab_size:      49152      # 与 500M 共用同一个分词器
d_model:         2048
n_layers:        22
n_heads:         16
n_kv_heads:      4          # GQA group size = 4，与 500M 一致
head_dim:        128        # 注意：从 64 变成 128
ffn_hidden:      5632       # ≈ (8/3)·2048，取整到 256 的倍数
# 其余（rope/norm/tie/bias/dropout/init）与 500M 完全相同
```

参数量：

| 部分 | 公式 | 数值 |
|---|---|---|
| `W_q` / `W_o` | 2048×2048 各一 | 8.389 M |
| `W_k` / `W_v` | 2048×512 各一 | 2.097 M |
| SwiGLU ×3 | 3 × 2048 × 5632 | 34.604 M |
| **单层** | | **45.09 M** |

```
22 层         : 22 × 45.09 M = 992.0 M   ← 非嵌入参数
嵌入(绑定)     : 49152 × 2048 = 100.7 M
------------------------------------------
总计                        ≈ 1.093 B
bf16 权重体积                ≈ 2.19 GB
```

**训练显存**（这是 1B 与 500M 的真正分水岭）：

```
权重+梯度+fp32主权重+Adam 状态 : 1.093B × 16 = 17.5 GB
+ 激活值(seq 2048, 开梯度检查点)            ≈  8 GB
--------------------------------------------------
单卡峰值                                  ≈ 26 GB
```

**超过 24GB 的 4090**。三个选项，按推荐顺序：

1. **ZeRO-2 / FSDP `SHARD_GRAD_OP`**（8 卡）— 优化器状态和梯度切分后每卡降到 ~5 GB，最干净，也是本项目想学的东西
2. **8-bit AdamW**（`bitsandbytes`）— Adam 状态从 8 字节/参数降到 2，单卡峰值降到 ~19 GB，能塞进 24G 卡
3. **换 A100 40G / H100 80G** — 花钱解决

### 1.5 前向流程

```
input_ids (B, T)
   └─ embedding                       → (B, T, 1280)
      └─ × 26 个 TransformerBlock:
           h = x + Attn(RMSNorm(x))          # 残差直连，不缩放
           h = h + SwiGLU(RMSNorm(h))
      └─ RMSNorm
      └─ lm_head (= embedding.weight.T)  → (B, T, 49152)
      └─ cross_entropy(shift by 1)
```

Attention 内部：
```
q,k,v = Wq·x, Wk·x, Wv·x
q,k   = apply_rope(q, k, pos)                    # 只对 q,k 施加，不动 v
k,v   = repeat_kv(k, v, n_rep=4)                 # GQA：5 组 → 20 头
out   = flash_attn(q, k, v, causal=True)
out   = Wo · out
```

### 1.6 初始化

- 所有线性层：`normal(0, 0.02)`
- 残差分支的输出投影（`W_o`、`W_down`）：额外缩放 `1/sqrt(2·n_layers)`（GPT-2 做法，防止残差流方差随深度爆炸）
- RMSNorm 的 `weight`：全 1
- Embedding：`normal(0, 0.02)`

---

## 二、代码架构

### 2.1 分层原则

```
配置层 (configs/*.yaml)          ← 唯一的超参真源，入 git
   ↓
纯函数/纯模块层 (model/)          ← 无 IO、无全局状态、可单测
   ↓
编排层 (train/, data/, eval/)     ← 有 IO 和分布式状态
   ↓
入口层 (scripts/, __main__)       ← 只做参数解析和调度
```

**硬规则**：`src/mytransformer/model/` 下的任何文件都不许 import `torch.distributed`、不许读环境变量、不许打日志。模型只是一个函数。这样它才能被单测和被数值对齐。

### 2.2 模块职责

| 模块 | 文件 | 职责 | 关键接口 |
|---|---|---|---|
| `tokenizer/` | `train_bpe.py`<br>`tokenizer.py`<br>`analyze.py` | 训练 BPE、编解码、fertility 分析 | `Tokenizer.encode/decode` |
| `data/` | `download.py`<br>`filter.py`<br>`dedup.py`<br>`tokenize_shard.py`<br>`loader.py` | 原始文本 → 定长 uint16 分片 → 采样器 | `ShardDataset`，输出 `.bin` + `.idx` |
| `model/` | `config.py`<br>`rope.py`<br>`norm.py`<br>`attention.py`<br>`mlp.py`<br>`block.py`<br>`transformer.py` | 纯模型 | `Transformer(cfg).forward(ids, targets=None)` |
| `train/` | `optim.py`<br>`schedule.py`<br>`checkpoint.py`<br>`dist.py`<br>`trainer.py`<br>`pretrain.py` | 训练循环、AdamW/Muon、WSD 调度、断点续训 | `Trainer.fit()` |
| `posttrain/` | `sft.py`<br>`dpo.py`<br>`chat_template.py` | 指令微调与偏好对齐 | 复用 `Trainer` |
| `eval/` | `perplexity.py`<br>`harness_adapter.py` | 内置困惑度 + lm-eval-harness 桥接 | |
| `infer/` | `kv_cache.py`<br>`generate.py`<br>`chat.py` | KV cache、采样、CLI | `generate(model, prompt, ...)` |

### 2.3 数据格式约定

分词后的语料统一落成 **flat uint16 数组分片**（nanoGPT 惯例），因为词表 49152 < 65536：

```
data/tokens/fineweb_edu/shard_00000.bin   # 纯 uint16，无 header
data/tokens/fineweb_edu/meta.json         # {"n_tokens": ..., "tokenizer_hash": ..., "shards": [...]}
```

- 文档之间用 `<|endoftext|>` (id=0) 分隔，训练时**跨文档拼接**成定长序列（不做 padding，零浪费）。
- 采样时按 mixture 权重从各子集抽 shard，权重写在 `configs/data/mixture_v1.yaml`。
- 每个 shard 200M tokens（约 400MB），方便并行处理和断点。

### 2.4 checkpoint 内容

除了权重，必须存下这些，否则复现不了：

```python
{
  "model": state_dict,
  "optimizer": optim_state,
  "step": int,
  "tokens_seen": int,
  "rng_states": {...},          # torch / numpy / python，保证续训后数据顺序一致
  "data_loader_state": {...},   # 读到哪个 shard 的哪个 offset
  "config": {...},              # 完整 yaml 快照
  "git_commit": "abc1234",      # 代码版本
  "wandb_run_id": "...",
}
```

### 2.5 测试策略（这是从零写 LLM 最容易被跳过、也最要命的部分）

| 测试 | 内容 | 通过标准 |
|---|---|---|
| `test_shapes.py` | 各模块输入输出形状 | 全部匹配 |
| `test_rope.py` | RoPE 旋转的相对位置性质：`⟨rope(q,m), rope(k,n)⟩` 只依赖 `m-n` | 误差 < 1e-5 |
| `test_causal.py` | 改动位置 `t` 的输入，位置 `< t` 的 logits 不变 | 逐位相等 |
| `test_gqa.py` | GQA 在 `n_kv=n_heads` 时退化为 MHA，与 MHA 实现等价 | 误差 < 1e-5 |
| `test_hf_parity.py` | **与 HuggingFace `LlamaForCausalLM` 权重互转后逐层输出对齐** | 最大绝对误差 < 1e-4 |
| `test_kv_cache.py` | 增量解码结果 == 全序列一次前向的对应位置 | 误差 < 1e-4 |
| `test_tokenizer.py` | 随机 UTF-8 文本编解码往返 | 完全一致 |
| `test_loader.py` | 断点续训后数据流与不中断时逐 batch 相同 | 完全一致 |
| `test_overfit.py` | 在 1 个 batch 上训 200 步 | loss → < 0.1 |

`test_hf_parity` 和 `test_overfit` 是两道生死线。前者保证架构没写错，后者保证训练循环没写错。**在这两个测试通过之前，不要花任何一分钱租 GPU。**

### 2.6 日志与可观测性

每步记录，全部推到 W&B：

- `loss`（train / val）、`perplexity`
- `grad_norm`（裁剪**前**的值——这是发现 loss spike 前兆的核心指标）
- `lr`、`tokens_seen`、`step_time`
- **`MFU`**（模型算力利用率）= `6·N·B·T / (step_time · GPU数 · 峰值FLOPS)`
- 每 1000 步：各层权重与梯度的 RMS 直方图（发现死层/爆炸层）
- 每 2000 步：固定 prompt 的生成样本（比 loss 更直观地看出模型在学什么）
