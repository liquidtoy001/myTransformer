# 训练配方

---

## 一、预训练超参（500M）

```yaml
# configs/train/pretrain_500m.yaml
# ---- 数据 ----
seq_len:              2048
global_batch_tokens:  1048576        # 1M tokens/step
micro_batch_size:     16             # 每卡；8 卡 × 16 × 2048 = 262144
grad_accum_steps:     4              # 262144 × 4 = 1048576 ✓
total_tokens:         51_000_000_000
max_steps:            48_640         # 51e9 / 1.048e6

# ---- 优化器 ----
optimizer:            adamw_fused
lr:                   6.0e-4         # 峰值
betas:                [0.9, 0.95]
eps:                  1.0e-8
weight_decay:         0.1            # 只作用于 2D 权重，norm/bias/embedding 不衰减
grad_clip:            1.0

# ---- 学习率日程（WSD） ----
schedule:             wsd
warmup_steps:         2000           # ≈ 4% 
stable_lr_until:      43_776         # 90%
decay_to:             0.0            # 最后 10% 线性衰减到 0（退火期）

# ---- 精度与性能 ----
dtype:                bfloat16
compile:              true
grad_checkpointing:   false          # 500M 显存够，关掉省 30% 时间
sdpa_backend:         flash

# ---- 稳定性 ----
z_loss:               1.0e-4         # 抑制 logits 漂移，PaLM 做法
init_std:             0.02
residual_init_scale:  1/sqrt(2*26)   # 残差输出投影额外缩放

# ---- 日志与存档 ----
log_every:            10
eval_every:           500
sample_every:         2000
ckpt_every:           2000
ckpt_keep:            5              # 加上每 10000 步的永久保留
```

**为什么是这些值**：

- **LR 6e-4** — 500M 规模的常规区间是 3e-4 ~ 1e-3。经验规律：LR 大致随 `1/sqrt(d_model)` 缩放。可在 P5 用 120M 模型扫 {3e-4, 6e-4, 1e-3} 确认。
- **β₂=0.95 而非 0.999** — LLM 预训练的标准做法，二阶矩窗口更短，对分布变化响应更快，更稳。
- **全局 batch 1M tokens** — 太小则梯度噪声大、GPU 打不满；太大则每步收益递减。500M 规模 0.5M–2M 都合理。
- **WSD 而非 cosine** — 稳定段的 LR 恒定，好处是：① 可以随时在任意点分叉做退火实验，不用重训；② 中途想延长训练不用重新规划日程。cosine 一旦定了总步数就锁死了。
- **z-loss** — `1e-4 · log²(Z)`，Z 是 softmax 分母。防止 logits 整体漂移导致 bf16 下溢出，几乎零成本的保险。

---

## 二、参数分组

```python
decay_params   = [p for n,p in model.named_parameters() if p.dim() >= 2]
nodecay_params = [p for n,p in model.named_parameters() if p.dim() <  2]
# norm 的 weight 是 1D → 不衰减
# 嵌入是 2D，但通常也不衰减，需显式排除
```

---

## 三、并行策略

| 规模 | 策略 |
|---|---|
| 单卡冒烟 | 无 |
| **500M / 8 卡（本项目）** | **DDP + 梯度累积**。500M 权重+优化器仅 8.2GB，不需要切分 |
| 若扩到 2B+ | FSDP（`ShardingStrategy.FULL_SHARD`）或 ZeRO-2 |
| 若上多机 | FSDP + `HYBRID_SHARD`（机内切分、机间复制） |

即便用不上，也**实现一条 FSDP 路径**并在 P5 测一次——这是要学的核心技能之一。

DDP 要点：
- `gradient_as_bucket_view=True` 省显存
- 梯度累积时前 N-1 步用 `no_sync()`，避免无谓的 all-reduce
- `broadcast_buffers=False`（没有 buffer 需要同步）

---

## 四、退火 / 中训（midtrain）

预训练最后 10%（约 5.1B tokens）：

1. 数据切换到高质量混合（见 [05-data.md](05-data.md)）
2. LR 从 6e-4 线性衰减到 0
3. **同时**把 `rope_theta` 从 10000 提到 500000，`max_seq_len` 从 2048 扩到 4096
4. 全局 batch 保持不变（序列变长则 micro-batch 减半）

RoPE base 扩展加长上下文是当前主流做法（Llama 3、Qwen 都这么干），几乎零成本。**注意：base 一改，之前的 checkpoint 就不能直接续了**，所以这个切换点要作为一个明确的阶段边界，checkpoint 单独命名。

---

## 五、后训练

### SFT
```yaml
lr:            2.0e-5          # 比预训练小 30 倍
schedule:      cosine
warmup_ratio:  0.03
epochs:        3
batch_tokens:  131072
loss_mask:     assistant_only  # 只在回复 token 上算 loss，prompt 部分屏蔽
packing:       true            # 多条样本拼进一条序列，但要用 block-diagonal mask 隔开
```

Chat template（写死在 tokenizer config 里，推理必须一致）：
```
<|system|>{system}<|endoftext|>
<|user|>{query}<|endoftext|>
<|assistant|>{response}<|endoftext|>
```

### DPO
```yaml
lr:      5.0e-7
beta:    0.1
epochs:  1
```
监控 `reward_margin`（应稳定上升）和 `KL(policy||ref)`（不应爆炸）。500M 规模上 DPO 收益有限，主要是**为了学流程**——如实在报告里写清楚。

---

## 六、故障处置手册

| 症状 | 可能原因 | 处置 |
|---|---|---|
| **loss 突然尖峰后不回落** | 坏数据 batch / LR 过高 / bf16 溢出 | 回滚到上一 checkpoint，**跳过接下来 200 个 batch**，降 LR 20% 继续。多数情况这样就过去了 |
| **loss 尖峰后自己回落** | 正常现象 | 记录位置，不用管，但要写进报告 |
| **grad_norm 持续上升** | 训练要炸的前兆 | 提前降 LR，或加 QK-Norm |
| **loss = NaN** | 溢出 / 除零 | 检查是否有全 padding 的序列、attention mask 全 `-inf` 的行 |
| **loss 卡住不降** | LR 太小 / warmup 太长 / 数据有问题 | 先看生成样本，再看是不是某个子集权重给错了 |
| **loss 降得异常快** | 数据泄漏或重复 | 查去重，查 val 是否混进了训练集 |
| **中文 loss 不降但英文降** | 分词器中文压缩率差 / 中文数据占比太低 | 回到 P1 |
| **MFU 突然掉** | 数据读取瓶颈 / 其它进程抢卡 | `nvidia-smi dmon`、看 dataloader 等待时间 |
| **多卡 loss 不一致** | 随机种子/数据切分错误 | 每卡应读不同数据，但模型初始化必须相同 |
| **实例被抢占** | 竞价实例常态 | 自动重启脚本 + 从对象存储恢复最新 checkpoint |

**最重要的一条**：`grad_norm`（裁剪前）是最灵敏的先行指标。loss 还没动的时候它可能已经在爬了。把它放在监控面板最显眼的位置。

---

## 七、开跑前检查清单

- [ ] `tests/` 全绿，尤其 `test_hf_parity` 和 `test_overfit`
- [ ] 冒烟实验的 loss 曲线正常，MFU ≥ 35%
- [ ] checkpoint 存/取往返测试通过（存了立刻读回来，loss 应完全一致）
- [ ] checkpoint 自动同步到对象存储的脚本已验证
- [ ] W&B 项目已建，alert 已配到手机
- [ ] val 集确认不在训练分片里（写个断言脚本）
- [ ] 数据总量 ≥ 计划 token 数 × 1.05
- [ ] 用 1% 数据跑 30 分钟，实际吞吐与估算误差 < 20%
- [ ] 算过一遍这次要花多少钱，且能接受
