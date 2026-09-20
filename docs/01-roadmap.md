# 路线规划

约 30 个工作日（业余时间可拉长到 2–3 个月），分 9 个阶段。**每个阶段有明确验收标准，不达标不进下一阶段。**

总算力预算：**约 66 AUD（$43 USD）**，含 30% 重试余量，明细见 [04-compute.md](04-compute.md) §五。

主线模型 **250M**（236.5M 参数，vocab 32768，9.44B tokens）。规模由 100 AUD 预算倒推：500M 光主训练就要 103 AUD，装不下。降到 250M 后缩放律和消融全都保得住，而那才是学习价值最高的部分。

**硬件分三处**：

| 硬件 | 负责 | 成本 |
|---|---|---|
| 本地 RTX 4070 Ti SUPER | P0–P4 开发调试 | 0 |
| 学院 Rangpur A100 40G（见 [09-rangpur.md](09-rangpur.md)） | P4 冒烟、P5 全部、P7、P8、P6 的数据准备 | 0（约 17 GPU 小时） |
| 云上 8×H100 竞价 | **仅 P6 正式训练**（2.5 小时） | ~66 AUD |

跨硬件的实验纪律见 [06-training-recipe.md](06-training-recipe.md) §六。

---

## 阶段总览

| 阶段 | 天数 | 名称 | 产出 | 硬件 | 成本 |
|---|---|---|---|---|---|
| P0 | D1–2 | 站在巨人肩上 | 跑通 2 个参考项目 | 本地 | 0 |
| P1 | D3–4 | 分词器 | 32768 BPE + fertility 报告 | 本地 CPU | 0 |
| P2 | D5–8 | 数据管线 | P5 数据 2B tokens + P6 数据 10.4B tokens | Rangpur `cpu` | 0 |
| P3 | D9–12 | 模型实现 | 通过 HF 数值对齐 | 本地 | 0 ✅ |
| P4 | D13–16 | 训练框架 | 冒烟 + 分块交叉熵 + 接力续跑 | 本地 → Rangpur `a100-test` | 0 |
| P5 | D17–20 | 缩放律与消融 | 3 点缩放曲线 + 消融表 | Rangpur A100 | 0 |
| P6 | D21–22 | 正式预训练 | **250M base 模型** | 8×H100 竞价 | **~66 AUD** |
| P7 | D23–25 | 退火 + 后训练 | chat 模型 | Rangpur A100 | 0 |
| P8 | D26–30 | 评测与报告 | 技术报告 + 模型卡 | Rangpur A100 | 0 |

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

- [x] 从数据混合中抽 1 GB 样本训 BPE，**vocab=32768**，字节级（P1-2；原计划 5 GB，1 GB 在本机 6 分钟训完，见报告）
- [x] 特殊 token：`<|endoftext|>`(0)、`<|system|>`、`<|user|>`、`<|assistant|>`、`<|tool|>`、`<|pad|>`、10 个 `<|reserved_i|>`；文本里的字面 `<|endoftext|>` 不识别为特殊 token（P1-1）
- [x] 预切分：GPT-4 风格正则，数字一位一切（P1-1）
- [x] **fertility 分析**：中/英/代码/数学，和 Qwen2.5、Llama-3、GPT-4o、DeepSeek-V3、SmolLM2、GPT-2 对比；词表大小扫描（16k–64k）；数字切分消融
- [x] 编解码往返测试通过（含 emoji、生僻字、不完整 UTF-8）（P1-1，用小语料现场训练）

**验收**：~~中文 ≤ 1.7 tokens/字，英文 ≤ 1.35 tokens/word~~ → 按实测修改为 **zh_web ≤ 0.75 tokens/字，en_web ≤ 1.50 tokens/word，混合 bytes/token ≥ 3.85**，写出 `reports/tokenizer.md`。✅ 实测 0.697 / 1.447 / 3.894。

> 原标准的英文一条在 32k 中英共享词表下做不到（64k 也只到 1.37），中文一条又太松（纯英文分词器的水平）。依据见 [reports/tokenizer.md](../reports/tokenizer.md) §5。

> 中文压缩率直接决定你的**有效 token 预算**。同样 9.44B tokens，分词器差 20% 就等于白扔 1.9B tokens 的算力（≈ 8 AUD）。这是最容易被忽视、投入产出比最高的一步。
>
> **词表定 32768 而不是 49152**：小模型上嵌入层会挤占计算参数。250M 配 32768 时嵌入占 14.2%，配 49152 会到 19.9%。见 [02-architecture.md](02-architecture.md) §1.2。

---

## P2 · 数据管线（D5–8）

详见 [05-data.md](05-data.md)。

- [x] 下载各子集（FineWeb-Edu / DCLM / chinese-fineweb-edu-v2 / github-code-clean / FineMath / 维基）：11.1 GB 文本、214 万篇（P2-5）
- [x] 质量过滤：Gopher 规则（照 datatrove 实现）+ 中文汉字占比 + 代码 StarCoder 规则，在真实文档上校准（P2-1）。来源已做过语言识别和质量分类，不再重复
- [x] 去重：文档级 MinHash-LSH（Jaccard 0.8，256 个哈希 / 32 段，暴力枚举校准）+ 段落级精确去重（代码不做）（P2-2）
- [x] **从所有语料中剔除评测集**（decontamination）：11 个评测集的 13-gram（汉字算半个词），排除 220 个通用片段（P2-3）。局限：C-Eval 27%、CMMLU 38% 的题太短没进索引
- [x] 分词打包成 uint16 shard，写 `meta.json`（分词器指纹 + 每个分片的 token 数，训练前核对）；按配比在打包时混合（P2-4）
- [x] 留出 val set（每个子集各 1M tokens，**训练集中必须删掉**）；训练时每个子集单独报 val loss（P2-4）

**验收**：
- [x] P5 数据 **2.000B tokens**（20 个分片，3.8 GB）在本机产出，真实分片冒烟训练通过（`configs/train/smoke_p5.yaml`）→ 传 Rangpur 见 [09-rangpur.md](09-rangpur.md) §把 P5 数据传上去
- [x] 随机抽 200 条供人工过目（`data/review_sample.txt`）
- [x] `reports/data.md` 写清每一步过滤掉了多少
- [ ] P6 数据 ≥10.4B tokens 上传到对象存储 —— **推迟到 P6**：同一条流水线换配置即可（`total_tokens` 调大），现在做只会占着本机磁盘和云存储费用

> **在哪做**：Rangpur `cpu` 分区的 batch 作业。原始数据下载到计算节点本地盘 `$TMPDIR`（约 197 GB，作业结束即失效），分词后 P5 数据写回 home（4 GB），P6 数据逐分片上传对象存储（19 GB，home 放不下）。每个作业只处理若干分片并留 `.done` 标记。模板见 [09-rangpur.md](09-rangpur.md) §4.4。

> 省钱提示：FineWeb-Edu、CCI3-HQ 这些数据集本身已经做过质量过滤和去重，**正式训练直接用**。自己的清洗/去重流水线在 1% 子集上跑一遍就够学到东西，不必也不该在全量上重跑一遍——那既费钱又不会让数据更好。

---

## P3 · 模型实现（D9–12）✅ 已完成

按 [02-architecture.md](02-architecture.md) 的模块划分逐个实现，**每写一个模块立刻写它的测试**。

顺序：`config` → `rope` → `norm` → `attention` → `mlp` → `block` → `transformer`

- [x] `tests/` 全绿（17 项）
- [x] `test_hf_parity.py`：与 HuggingFace `LlamaForCausalLM` 逐层对齐，**实测 logits 最大误差 8.9e-8**（阈值 1e-4）
- [x] 参数量脚本输出 236,491,776，与手算一致（`python scripts/count_params.py`）

**验收**：✅ 已通过（2026-09-02）。数值对齐 8.9e-8，比阈值好 3 个数量级。

---

## P4 · 训练框架 + 冒烟（D13–16）

- [x] AdamW（CUDA 上 fused）、WSD 学习率、梯度裁剪、梯度累积
- [x] bf16 混合精度
- [ ] `torch.compile`：开关已接入；Windows 没有 triton，待在 Rangpur（Linux）上验证
- [x] checkpoint 原子写入、续跑（含加载器状态、优化器状态、RNG）
- [x] **分块交叉熵**（见 [06-training-recipe.md](06-training-recipe.md) §一）：实测峰值 3.32 → 0.72 GiB
- [x] **接力续跑**：启动即从最新 checkpoint 恢复；收到 SLURM `USR1`/`TERM` 或到 `--max-minutes` 时存档退出
- [x] **续跑自检**：固定小 batch 上比对 loss 与 logits 均方根，并比对下一个 batch
- [x] NaN 保护：梯度非有限时跳过更新
- [x] 日志：`metrics.jsonl`（loss / val_loss / lr / grad_norm / tok/s / MFU / 显存）
- [x] **本地冒烟**：合成马尔可夫数据（理论下界 ln4），`ladder_s1` × 19.7M tokens，中途存档再续跑，val loss 1.421（见 [learn/p4-5](learn/p4-5-trainer.md) §3）
- [ ] DDP 单机多卡（P6 要用）；FSDP 路径预留
- [ ] W&B：目前只写 jsonl
- [ ] **跨设备一致性测试**（CPU fp32 vs GPU bf16）
- [x] `scripts/rangpur/`：`setup_env.sh`、`smoke.sbatch`、`train.sbatch`、`submit_chain.sh`（`data.sbatch` 随 P2）
- [x] 在 `a100-test` 上跑冒烟：作业 600202 通过（`torch.compile`、真实 `SIGUSR1` 中途存档、续跑、A100 节点 `$TMPDIR` 均验证）。第一次（600172）暴露了 compile 会在 C 层替换信号处理函数，已修复
- [ ] 真实文本冒烟：等 P1 分词器、P2 数据就绪后，`ladder_s1` × 0.3B tokens（本机已用 1000 万 token 的迷你数据跑通链路：`configs/train/smoke_realtext.yaml`，P2-4）
- 梯度检查点：250M 在 A100/H100 上显存充足，暂不实现

**验收**：
- `test_overfit`（单 batch 过拟合到 loss<0.1）通过
- 断点续训后 loss 曲线无跳变
- 手动杀掉一次作业再续跑，loss 曲线无跳变（这就是 P5 和 P6 每天要做的事）
- `ladder_s1` 的 val loss 落在合理区间，生成的英文是通顺句子
- **本地只验证正确性，不看 MFU**——MFU 是硬件特性：A100 的在 P5 第一个作业时测，H100 的在 P6 开跑前测

---

## P5 · 缩放律与消融（D17–20，Rangpur A100，免费）

这是把"跑通"变成"做研究"的一步，也是报告里最有含金量的部分。全部实验在 Rangpur 上跑，合计约 11 小时纯训练，按 4 小时一个作业接力（见 [09-rangpur.md](09-rangpur.md) §五）。

**所有实验共用同一份数据**（2B tokens，4 GB）和同一个分词器。

- [x] **P5-1 准备**（本机）：Muon 优化器、GeLU / QK-Norm 开关、12 个实验配置（测试守住"只改一处"）、提交脚本、分析脚本（缩放律拟合用已知答案验证过）

**先测 seed 方差**：`ladder_s2` 跑 3 个不同 seed（共 2.0 h），得到 val loss 的标准差。**任何小于这个标准差的消融差异都不能下结论**——这一步几乎没人做，但它决定了消融表里哪些行是真结论。

**缩放曲线**：`ladder_s1` / `ladder_s2` / `ladder_s3`，各按 Chinchilla 20× tokens（共 5.1 h）。拟合 `L(N) = L∞ + A·N^(-α)`，**N 用非嵌入参数**，**外推预测 250M 的 loss**。

- 四个点**统一词表 32768、序列长度 2048**，否则每 token 的 loss 不可比（`test_scaling_ladder_shares_tokenizer_and_context` 守着这条）
- 第 4 个点（250M）在 P6 通过 WSD 分叉得到，约 $1.5 而非重训的 $15——正式训完后回来看外推准不准

**消融实验**（每个用 `ladder_s2` × 0.79B tokens，约 40 分钟，6 组共 4.0 h）：

| # | 对照 | 想回答的问题 |
|---|---|---|
| A1 | AdamW vs Muon | 新优化器在小规模真的更快吗 |
| A2 | SwiGLU vs GeLU | 值不值多那一个矩阵 |
| A3 | GQA vs MHA | 质量损失有多大 |
| A4 | 数据配比 v1 vs v2（中文 20% vs 35%） | 中英配比怎么选 |
| A5 | WSD vs cosine | 退火策略 |
| A6 | 有无 z-loss / QK-Norm | 稳定性代价 |

**验收**：`reports/ablation.md` 一张表，每行有 val loss、下游任务分、训练时间、**与 seed 方差的比较**，并写明你据此做了什么决定。

进阶：挑 2 组效应最大的在 `ladder_s3` 上复跑，看效应是放大还是消失。

---

## P6 · 正式预训练（D21–22，8×H100 竞价，~66 AUD）

- [ ] 租 8×H100 竞价，从对象存储拉取 P2 准备好的数据分片
- [ ] 先用 1% 数据跑 10 分钟：**在 H100 上实测 MFU 并重算耗时与费用**，与估算差 >20% 就停机排查（Rangpur 上测的 MFU 推不出 H100 的）
- [ ] 正式开跑：**9.44B tokens**，全局 batch 524288，18000 步，**约 2.5 小时 ≈ $30 / 45 AUD**
- [ ] step 9000 触发**缩放律分叉**：另起一支跑 900 步衰减到 0，产出第 4 个缩放律数据点
- [ ] 每 1000 步存 checkpoint 到对象存储（竞价随时被抢占）——与 Rangpur 接力续跑是同一套代码，P5 已经反复验证过
- [ ] 挂着监控：grad_norm 突增、loss spike、NaN 自动告警 + 自动回滚

> **为什么不在 Rangpur 上跑**：约 47 GPU 小时，在全班共用的 10 块 A100 上太显眼，而且没有助教的明确许可。
>
> **为什么用 8 卡而不是更便宜的单卡**：单卡 H100 要 16.5 小时、约 55 AUD，比 8 卡便宜。选 8 卡是为了把**真实的多卡 DDP 工程**走一遍——Rangpur 每节点只有 1 块 GPU，学不到这一课。

**验收**：训练完成，val loss ≈ 2.8–3.1（依数据而定），HellaSwag ≳ 33%，无未解释的 loss spike，第 4 个缩放律点拿到。

**风险预案**：见 [06-training-recipe.md](06-training-recipe.md) 的"故障处置"一节。

---

## P7 · 退火 + 后训练（D23–25，Rangpur A100）

- [ ] **退火（midtrain）**：最后 10% 的 token（1800 步，约 0.94B）换成高质量混合（教科书、数学、代码、合成 QA），LR 线性衰减到 0，同时把 RoPE base 提到 500000、上下文扩到 4096
- [ ] **SFT**：整理 20–50 万条指令数据，定义 chat template，训 2–3 epoch
- [ ] **DPO**：偏好数据对齐，观察 reward margin 和 KL
- [ ] 实现 KV cache + 采样，做一个能对话的 CLI/WebUI

**验收**：模型能进行多轮对话、遵循基本指令格式。**同时诚实记录它做不到什么**（复杂推理、长程一致性、事实准确性）——这部分对报告同样重要。

---

## P8 · 评测与报告（D26–30，Rangpur A100）

- [ ] 接 `lm-evaluation-harness`，跑 HellaSwag / PIQA / ARC-e / ARC-c / WinoGrande / LAMBADA / C-Eval / CMMLU
- [ ] 与 GPT-2 124M、Pythia-160M、Pythia-410M、SmolLM2-360M、Qwen2.5-0.5B 同条件对照
- [ ] 画 loss 曲线、缩放律拟合图、MFU 曲线
- [ ] 按 [08-report-template.md](08-report-template.md) 写技术报告
- [ ] 传 HuggingFace Hub：权重 + 中间 checkpoint + 模型卡 + 训练日志

**验收**：一份别人照着能复现的报告。

---

## P9 · 放大（可选，预算宽裕时）

**只在 P8 完全交付之后再考虑。** 用完全相同的代码，换 `configs/model/500m.yaml` 或 `1b.yaml`：

| 目标 | tokens | 8×H100 竞价 | AUD |
|---|---|---|---|
| 500M | 10.3B (20×) | 5.6 小时 | 103 |
| 500M | 20.6B (40×) | 11.3 小时 | 206 |

（按 `flops_per_token()` 与 8 卡 MFU 25% 估算；1B 需要先上 FSDP，届时另算。）

1B 的额外门槛：单卡训练状态 17.5 GB，必须上 ZeRO-2/FSDP 或 8-bit optimizer，**不能在单卡上完整调试**。

每放大一档就给缩放律曲线加一个点，报告的含金量会明显不同。

---

## 关键决策点

| 时点 | 决策 | 如果答案是"否" |
|---|---|---|
| P3 结束 | HF 数值对齐通过了吗 | 停下来 debug，不租卡 |
| P4 结束 | 分块交叉熵做了吗、断点续训 loss 无跳变吗 | 先修，竞价实例上会反复用到 |
| P5 结束 | 消融差异大于 seed 方差吗 | 小于方差的差异不能下结论 |
| P5 结束 | 缩放曲线外推的 250M loss 合理吗 | 曲线异常说明有系统性 bug |
| P4 结束 | 在 `a100-test` 上杀掉再续跑，loss 无跳变吗 | 不修好不进 P5，P5 要接力很多次 |
| P6 开始 10 分钟 | H100 上实测 MFU 重算的费用还在预算内吗 | 立刻停机排查 |

---

## 预算表

按 1 USD = 1.52 AUD。耗时按 `ModelConfig.flops_per_token()`（含 lm_head 与注意力）计算。

| 项目 | 硬件 | 纯训练时间 | USD | AUD |
|---|---|---|---|---|
| P0–P3 | 本地 | — | $0 | 0 |
| P4 冒烟 | 本地 + Rangpur `a100-test` | 6 分钟 | $0 | 0 |
| P5 缩放律 S1 / S2 / S3 | Rangpur A100 | 5.1 h | $0 | 0 |
| P5 seed 方差 + 6 组消融 | Rangpur A100 | 6.0 h | $0 | 0 |
| **P6 正式预训练 250M** | **8×H100 竞价** | **2.5 h** | **$29.77** | **45.3** |
| P6 WSD 分叉 | 8×H100 竞价 | 7 分钟 | $1.49 | 2.3 |
| P7 SFT + DPO | Rangpur A100 | 1.4 h | $0 | 0 |
| P8 评测 + 对照模型 | Rangpur A100 | ~2 h | $0 | 0 |
| 对象存储（P6 数据 19 GB，1 个月） | — | — | $2.00 | 3.0 |
| **小计** | | | **$33.26** | **50.6** |
| +30% 重试/抢占余量 | | | $9.98 | 15.2 |
| **合计** | | | **$43.24** | **65.7** |

100 AUD 预算下留出约 34 AUD 缓冲。Rangpur 上合计约 17 GPU 小时。

### 更省 / 更贵

| 方案 | 花费 | 说明 |
|---|---|---|
| P6 降到 20× tokens | ≈ 32 AUD | 模型弱一档；它本身就是第 4 个缩放律点 |
| P6 改用单卡 H100 | ≈ 55 AUD | 学不到多卡 DDP |
| ★ **推荐：P6 在 8×H100，40× tokens** | **≈ 66 AUD** | |
| P6 也放 Rangpur | 0 | 约 47 GPU 小时，共享资源上太显眼 |

**关键认知：算力买的是模型质量，不是学习收益。** P1–P5 全部在本地和 Rangpur 上免费完成，那才是项目主体；P6 只是把一条已验证的流水线放大跑一次。
