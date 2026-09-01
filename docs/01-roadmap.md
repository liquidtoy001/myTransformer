# 路线规划

约 30 个工作日（业余时间可拉长到 2–3 个月），分 9 个阶段。**每个阶段有明确验收标准，不达标不进下一阶段。**

总算力预算：**$400–600**（其中正式预训练约 $300）。

---

## 阶段总览

| 阶段 | 天数 | 名称 | 产出 | 算力 |
|---|---|---|---|---|
| P0 | D1–2 | 站在巨人肩上 | 跑通 2 个参考项目 | ¥20（单卡） |
| P1 | D3–4 | 分词器 | 49152 BPE + fertility 报告 | CPU |
| P2 | D5–9 | 数据管线 | 55B tokens 的 `.bin` 分片 | CPU 大盘 |
| P3 | D10–13 | 模型实现 | 通过 HF 数值对齐 | 单卡 |
| P4 | D14–17 | 训练框架 | 50M 冒烟模型跑通 | 单卡 ~$10 |
| P5 | D18–21 | 缩放律与消融 | 4 点缩放曲线 + 消融表 | ~$60 |
| P6 | D22–24 | 正式预训练 | 500M base 模型 | **~$300** |
| P7 | D25–27 | 退火 + 后训练 | chat 模型 | ~$60 |
| P8 | D28–30 | 评测与报告 | 技术报告 + 模型卡 | ~$20 |
| P9 | D31–34 | 1B 升级（可选） | 1B 模型 + 第 5 个缩放点 | ~$260 |

---

## P0 · 站在巨人肩上（D1–2）

**别一上来自己写。** 先把别人的跑通，建立"正常的 loss 曲线长什么样"的直觉。

- [ ] 单卡跑完 `jingyaogong/minimind` 的 pretrain + SFT（约 2 小时，¥5），感受全流程
- [ ] 跑 `karpathy/nanochat` 的 `speedrun.sh`（有预算就 8×H100 4 小时；没有就只读代码 + 单卡跑缩小版）
- [ ] 看完 Karpathy 的 "Let's reproduce GPT-2 (124M)" 视频，逐行跟敲一遍
- [ ] 开好 W&B 项目、租号、配好 SSH/rsync/tmux 工作流

**验收**：能说清 nanochat 从 tokenizer 到 web chat 的每一个脚本在干什么。

---

## P1 · 分词器（D3–4）

- [ ] 从数据混合中抽 10GB 样本训 BPE，vocab=49152，byte-level fallback
- [ ] 特殊 token：`<|endoftext|>`(0)、`<|user|>`、`<|assistant|>`、`<|system|>`、`<|tool|>` 预留
- [ ] **fertility 分析**：中/英/代码/数学各测每词平均 token 数，和 Qwen2.5、Llama-3、GPT-4o 的分词器对比
- [ ] 编解码往返测试通过（含 emoji、生僻字、不完整 UTF-8）

**验收**：中文 fertility ≤ 1.7 tokens/字，英文 ≤ 1.35 tokens/word，写出 `reports/tokenizer.md`。

> 中文压缩率直接决定你的**有效 token 预算**。同样 50B tokens，分词器差 20% 就等于白扔 10B tokens 的算力。这是最容易被忽视、投入产出比最高的一步。

---

## P2 · 数据管线（D5–9）

详见 [05-data.md](05-data.md)。

- [ ] 下载各子集（FineWeb-Edu / DCLM / 中文语料 / 代码 / 数学 / 百科）
- [ ] 质量过滤：Gopher 规则 + 语言识别 + 困惑度过滤
- [ ] 去重：文档级 MinHash-LSH（Jaccard 0.8）+ 段落级精确去重
- [ ] **从所有语料中剔除评测集**（decontamination）——n-gram 匹配，否则评测分数是假的
- [ ] 分词打包成 uint16 shard，写 `meta.json`
- [ ] 留出 val set（每个子集各 2M tokens，**训练集中必须删掉**）

**验收**：≥55B tokens 可用（50B 训练 + 冗余），随机抽 200 条人工过目，`reports/data.md` 写清每一步过滤掉了多少。

---

## P3 · 模型实现（D10–13）

按 [02-architecture.md](02-architecture.md) 的模块划分逐个实现，**每写一个模块立刻写它的测试**。

顺序：`config` → `rope` → `norm` → `attention` → `mlp` → `block` → `transformer`

- [ ] `tests/` 全绿
- [ ] `test_hf_parity.py`：与 HuggingFace `LlamaForCausalLM` 逐层对齐，误差 < 1e-4
- [ ] 参数量脚本输出 514.5M，与手算一致

**验收**：数值对齐通过。**这一关不过，后面全是白干。**

---

## P4 · 训练框架 + 冒烟（D14–17）

- [ ] AdamW（fused）、WSD 学习率、梯度裁剪、梯度累积
- [ ] bf16 混合精度 + `torch.compile`
- [ ] 梯度检查点（可开关）
- [ ] checkpoint 保存/恢复（含 dataloader 状态和 RNG）
- [ ] DDP 单机多卡；FSDP 路径预留
- [ ] W&B 日志：loss / grad_norm / lr / MFU / 生成样本
- [ ] **冒烟实验**：50M 模型 × 1B tokens，单卡约 3 小时

**验收**：
- `test_overfit`（单 batch 过拟合到 loss<0.1）通过
- 断点续训后 loss 曲线无跳变
- 50M 模型 val loss 落在 3.2–3.6 区间（合理范围），生成的英文是通顺句子
- **MFU ≥ 35%**（低于这个说明有性能问题，先查 dataloader 和 `torch.compile`）

---

## P5 · 缩放律与消融（D18–21）

这是把"跑通"变成"做研究"的一步，也是报告里最有含金量的部分。

**缩放曲线**：训 4 个模型 —— 30M / 60M / 120M / 250M，各按 Chinchilla 20× tokens，拟合 `L(N) = L∞ + A·N^(-α)`，外推预测 500M 的 loss。正式训完后回来看预测准不准。

**消融实验**（每个用 120M × 3B tokens，约 40 分钟/次）：

| # | 对照 | 想回答的问题 |
|---|---|---|
| A1 | AdamW vs Muon | 新优化器在小规模真的更快吗 |
| A2 | SwiGLU vs GeLU | 值不值多那一个矩阵 |
| A3 | GQA(5) vs MHA(20) | 质量损失有多大 |
| A4 | 数据配比 v1 vs v2（中文 20% vs 35%） | 中英配比怎么选 |
| A5 | WSD vs cosine | 退火策略 |
| A6 | 有无 z-loss / QK-Norm | 稳定性代价 |

**验收**：`reports/ablation.md` 一张表，每行有 val loss、下游任务分、训练时间，并写明**你据此做了什么决定**。

---

## P6 · 正式预训练（D22–24）

- [ ] 租 8×H100，先用 1% 数据跑 30 分钟确认吞吐与 loss 正常
- [ ] 正式开跑：51B tokens，全局 batch 1M tokens，约 50k 步，**约 14 小时**
- [ ] 每 2000 步存 checkpoint 到对象存储（机器随时可能被抢占）
- [ ] 挂着监控：grad_norm 突增、loss spike、NaN 自动告警 + 自动回滚到上一 checkpoint

**验收**：训练完成，val loss ≈ 2.5–2.8（依数据而定），HellaSwag ≳ 40%，无未解释的 loss spike。

**风险预案**：见 [06-training-recipe.md](06-training-recipe.md) 的"故障处置"一节。

---

## P7 · 退火 + 后训练（D25–27）

- [ ] **退火（midtrain）**：最后 10% 的 token 换成高质量混合（教科书、数学、代码、合成 QA），LR 线性衰减到 0，同时把 RoPE base 提到 500000、上下文扩到 4096
- [ ] **SFT**：整理 20–50 万条指令数据，定义 chat template，训 2–3 epoch
- [ ] **DPO**：偏好数据对齐，观察 reward margin 和 KL
- [ ] 实现 KV cache + 采样，做一个能对话的 CLI/WebUI

**验收**：模型能进行多轮对话、遵循基本指令格式。**同时诚实记录它做不到什么**（复杂推理、长程一致性、事实准确性）——这部分对报告同样重要。

---

## P8 · 评测与报告（D28–30）

- [ ] 接 `lm-evaluation-harness`，跑 HellaSwag / PIQA / ARC-e / ARC-c / WinoGrande / LAMBADA / C-Eval / CMMLU
- [ ] 与 Pythia-410M、SmolLM2-360M、Qwen2.5-0.5B 同条件对照
- [ ] 画 loss 曲线、缩放律拟合图、MFU 曲线
- [ ] 按 [08-report-template.md](08-report-template.md) 写技术报告
- [ ] 传 HuggingFace Hub：权重 + 中间 checkpoint + 模型卡 + 训练日志

**验收**：一份别人照着能复现的报告。

---

## P9 · 1B 升级（可选，D31–34）

**只在 P8 完全交付之后再做。** 用完全相同的代码，换成 `configs/model/1b.yaml`：

- [ ] 打开 ZeRO-2 / FSDP `SHARD_GRAD_OP`（1B 单卡 26GB 装不下，见 [04-compute.md](04-compute.md)）
- [ ] 先用 1% 数据验证切分后的 loss 与 DDP 路径一致
- [ ] 训 25B tokens，8×H100 约 13 小时 ≈ $260
- [ ] 同一套评测跑一遍，与 500M 并排对照
- [ ] 把这个点加进缩放律曲线（此时有 5 个规模点），检验之前的外推

**为什么放在最后而不是替换主线**：同预算下 1B 只能喂到 ~23 tokens/param，而 500M 能喂到 100 —— 后者实际更好用。而且 1B 不能在单卡上调试，所有 bug 都得在多卡环境里查，这个成本比多花的钱大得多。等流水线在 500M 上验证过了再跑 1B，风险低一个量级。

**验收**：1B 在同条件下各项指标均优于 500M（若不是，说明数据量不足或有配置问题，这本身就是报告里值得写的发现）。

---

## 关键决策点

| 时点 | 决策 | 如果答案是"否" |
|---|---|---|
| P3 结束 | HF 数值对齐通过了吗 | 停下来 debug，不租卡 |
| P4 结束 | MFU ≥ 35% 吗 | 先优化性能，否则 P6 多烧一倍钱 |
| P5 结束 | 缩放曲线外推的 500M loss 合理吗 | 曲线异常说明有系统性 bug |
| P6 开始 30 分钟 | 吞吐和 loss 与冒烟实验一致吗 | 立刻停机，别烧 14 小时 |

---

## 预算表

| 项目 | 明细 | 成本 |
|---|---|---|
| P0 参考项目 | 单卡 4090 ~10h | ¥20 |
| P2 数据处理 | 大内存 CPU 机 ~40h | ¥100 |
| P4 冒烟 | 单卡 H100 ~5h | $12 |
| P5 缩放+消融 | 8×H100 ~3h | $60 |
| P6 正式预训练 | 8×H100 ~15h（含重试余量） | $300 |
| P7 退火+SFT+DPO | 8×H100 ~3h | $60 |
| P8 评测 | 单卡 ~8h | $20 |
| 对象存储 | ~2TB·月 | $40 |
| **合计（P0–P8，主线 500M）** | | **≈ $500** |
| P9 1B 升级（可选） | 8×H100 ~14h | +$260 |

省钱做法：用竞价实例（便宜 50–70%，但要做好抢占恢复）；申请 Google TPU Research Cloud 免费额度；只训 Chinchilla 最优的 10B tokens（P6 降到 $60，模型会明显更弱但流程完全一样）。
