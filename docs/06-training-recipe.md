# 训练配方

---

## 一、预训练超参（250M 主线）

```yaml
# configs/train/pretrain_250m.yaml（与仓库中的文件一致；tests/test_trainer.py 保证它能被加载）
model: configs/model/250m.yaml
train_data: data/tokens/main/train_*.bin    # P2 产出，上传到对象存储，H100 机器开机后拉取到这里
val_data: data/tokens/main/val_*.bin
run_dir: runs/pretrain_250m

seq_len: 2048
global_batch_tokens: 524288     # 0.5M tokens/step，与硬件无关的常量
micro_batch_size: 16            # 8 卡 × 16 × 2048 × 累积 2 = 524288
max_steps: 18000                # = 9.437B tokens = 39.9 tokens/param

lr: 7.0e-4                      # 6e-4 × sqrt(1280/1024) ≈ 6.7e-4，取整；P5 用 ladder_s2 扫描确认
betas: [0.9, 0.95]
eps: 1.0e-8
weight_decay: 0.1
grad_clip: 1.0

schedule: wsd
warmup_steps: 720               # 4%
decay_start: 16200              # 90%；最后 10%（0.94B tokens）线性衰减到 0
min_lr_ratio: 0.0

dtype: bfloat16
compile: true
ce_chunk: 4096
z_loss: 1.0e-4

seed: 0
log_every: 10
eval_every: 250
eval_batches: 16
ckpt_every: 1000                # 竞价实例随时被抢占
ckpt_keep: 3
ckpt_keep_every: 1000           # 每个 checkpoint 都永久保留：step 9000 是缩放律分叉点，
                                # 其余用于"能力随训练量增长"的中间评测。共 18 份，同步到对象存储
peak_tflops: 989                # H100 SXM bf16 稠密
```

**为什么是这些值**：

- **LR 7e-4** — 经验规律 LR 大致随 `1/sqrt(d_model)` 缩放。d=1024 比 500M 方案的 d=1280 小，所以从 6e-4 提到 `6e-4 × sqrt(1280/1024) ≈ 6.7e-4`，取整 7e-4。P5 用 `ladder_s2` 扫 {5e-4, 7e-4, 1e-3} 确认。
- **β₂=0.95 而非 0.999** — LLM 预训练标准做法，二阶矩窗口更短，对分布变化响应更快。
- **全局 batch 0.5M tokens** — 比 500M 方案的 1M 小一半。模型越小，临界 batch 越小；给太大只是浪费。
- **WSD 而非 cosine** — ① 稳定段 LR 恒定，可在任意点分叉做退火，缩放律的第 4 个点就是这么白拿的：`configs/train/pretrain_250m_fork9000.yaml` 从 step 9000 起步、只改了 `decay_start` 和 `max_steps`（见 [04-compute.md](04-compute.md) §五）；② 中途想延长训练不用重新规划日程。cosine 一旦定了总步数就锁死。
- **`ckpt_keep_every: 1000`** — 分叉必须有 step 9000 的 checkpoint，只保留最近 3 份的话它在 step 12000 就被删了。干脆每一份都留着，顺便用于中间评测。
- **z-loss** — `1e-4 · log²(Z)`，防止 logits 整体漂移导致 bf16 下溢出，几乎零成本的保险。
- **ckpt_every 1000（而非 2000）** — 用竞价实例，抢占是常态；2.5 小时的训练存 18 次不算多。

### 分块交叉熵（chunked_ce）

**这是 16GB 消费级卡上跑得动跑不动的分界线。**

朴素实现会一次性materialize 完整 logits：`batch × seq × vocab`。micro_batch=16、seq=2048、vocab=32768 时：

```
bf16 logits           16×2048×32768×2 = 2.00 GiB
.float() 拷贝                          = 4.00 GiB
log_softmax 中间量                     ≈ 4.00 GiB
------------------------------------------------
仅 loss 一项就                          ≈ 10 GiB
```

而 250M 模型本身的训练状态才 3.5 GB。**loss 的显存是模型的三倍。**

做法：把序列切成若干块，逐块算 `lm_head` + `cross_entropy` 并累加，**每块都做激活重算**（`torch.utils.checkpoint`），前向只保存该块的输入 `h`，反向时再算一遍这一块的 logits。

**只切块、不重算是不够的。** 实测（vocab 32768，4×2048）：

| 方式 | 峰值显存 |
|---|---|
| 不分块 | 3.32 GiB |
| 只切块 | 1.57 GiB |
| 切块 + 激活重算 | **0.72 GiB** |

只切块能省掉那几个同时存在的完整大张量，但每块的 `log_softmax` 输出都要留到反向，加起来仍是一整份 fp32 logits（1.00 GiB）。

代价：反向时多算一遍 lm_head 的前向，每 token 多 `2·d·V` FLOPs，对 250M 约 **+3.6%** 计算量。已在 `Transformer.forward(..., ce_chunk=...)` 实现，见 `docs/learn/p4-1-chunked-loss.md`。

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
| P4 冒烟 / P5 小实验（单卡） | 无 |
| **250M / 8 卡（P6 本项目）** | **DDP + 梯度累积**。250M 训练状态仅 3.5GB，远不需要切分 |
| 若扩到 1B+ | FSDP（`SHARD_GRAD_OP` / `FULL_SHARD`）或 ZeRO-2 |
| 若上多机 | FSDP + `HYBRID_SHARD`（机内切分、机间复制） |

即便用不上，也**实现一条 FSDP 路径**并在 P5 测一次——这是要学的核心技能之一。P6 选 8 卡而非更便宜的单卡，为的就是把真实的多卡工程走一遍。

DDP 要点：
- `gradient_as_bucket_view=True` 省显存
- 梯度累积时前 N-1 步用 `no_sync()`，避免无谓的 all-reduce
- `broadcast_buffers=False`（没有 buffer 需要同步）

---

## 四、退火 / 中训（midtrain）

预训练最后 10%（1800 步，约 0.94B tokens）：

1. 数据切换到高质量混合（见 [05-data.md](05-data.md)）
2. LR 从 7e-4 线性衰减到 0
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

## 六、跨硬件实验纪律

本项目跨三种硬件：本地 RTX 4070 Ti SUPER（P0–P4 调试）、Rangpur A100 40G（P4 冒烟、P5、P7、P8）、8×H100 竞价（P6）。**换硬件本身不会让结果出错，但会让结果不可比**，而且分两类：

| 指标类型 | 跨硬件可比吗 | 例子 |
|---|---|---|
| **性能指标** | **完全不可比** | MFU、step_time、吞吐、耗时、成本 |
| **质量指标** | **基本可比**，但要控变量 | val loss、困惑度、下游任务分 |

`loss vs token 数`是硬件无关的数学性质。而 MFU 是硬件特性，本地测到 40% 不代表 H100 上有 40%——H100 算力是本地卡的 13 倍但带宽只有 5 倍，**相对更"算力过剩"**，RMSNorm / softmax / 优化器更新这些带宽瓶颈的开销占比更大，小模型上 MFU 反而更低。

噪声量级，从小到大：

1. **硬件数值差异**（kernel 实现、归约顺序不同）：bf16 下 logits 相对差异 ~1e-3，对 val loss 影响 < 0.005
2. **不同 seed**：小模型 val loss 波动 **0.01–0.02**
3. **全局 batch 不一致**：这不是噪声，是**真实的实验混淆**

**硬件引起的差异比 seed 差异还小。** 所以"跨卡"本身不是问题，问题是跨卡时往往同时改了别的东西（batch、精度、编译选项），把混淆变量误当成硬件问题。

### 五条硬性纪律

1. **全局 batch 是硬件无关的常量，用梯度累积吸收显存差异。**
   本地 `micro=2 × accum=128`，Rangpur A100 `micro=16 × accum=16`，8×H100 `micro=16 × accum=2`，全局都是 524288 tokens。数学上等价。**不遵守这条，后面所有对比作废**——尤其是缩放律：前三个点在 A100 上、第 4 个点在 H100 上，全局 batch 不一致就没法放进同一条曲线。

2. **一整组消融必须在同一硬件上跑完。** A1–A6 六组和 seed 方差全在 Rangpur A100 上，不能一半本地一半 Rangpur。缩放律的前三个点同理——任何系统性偏差都会污染拟合出的 α。唯一的例外是第 4 个点（P6 分叉，H100），报告里单独注明。

3. **本地 GPU 只验证正确性，不测性能。** P4 在本地跑，看的是 loss 曲线形状、断点续训一致性、有无 NaN。A100 的性能在 P5 第一个作业时校准；P6 开跑前的核对必须用 H100 自己的基线。

4. **先测 seed 方差，再看消融结果。** 用同一配置跑 2–3 个不同 seed，得到 val loss 的标准差。**任何小于这个标准差的消融差异都不能下结论。** 这一步几乎没人做，但它决定了消融表里哪些行是真结论、哪些是噪声。

5. **报告里逐实验标注硬件。** OLMo、SmolLM2 的技术报告都这么做。

### 跨设备一致性测试

`tests/test_cross_device.py`：同一份 checkpoint + 同一个 batch，CPU fp32 跑一遍、GPU bf16 跑一遍，比 loss。差异应在 1e-2 以内（bf16 精度决定）。

**如果差很多，说明有算子在不同后端走了不同路径**——这类 bug 只在换硬件时暴露，非常难查。现在就加上，顺带覆盖以后换到 H100 的情况。

---

## 七、故障处置手册

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

## 八、开跑前检查清单

- [ ] `tests/` 全绿，尤其 `test_hf_parity` 和 `test_overfit`
- [ ] 冒烟实验的 loss 曲线正常（本地跑，**只验证正确性，不看 MFU**）
- [ ] checkpoint 存/取往返测试通过（存了立刻读回来，loss 应完全一致）
- [ ] checkpoint 自动同步到对象存储的脚本已验证
- [ ] W&B 项目已建，alert 已配到手机
- [ ] val 集确认不在训练分片里（写个断言脚本）
- [ ] 数据总量 ≥ 9.44B × 1.1 ≈ 10.4B tokens
- [ ] 在**目标硬件上**用 1% 数据跑 10 分钟，实测 MFU 并重算耗时与费用
- [ ] 算过一遍这次要花多少钱（预期 $30 / 45 AUD），且能接受
- [ ] 竞价实例的自动恢复脚本已验证（存 → 杀进程 → 重启 → loss 无跳变）
