# 参考资源

带评价的清单。**★** = 强烈推荐先看。

---

## 一、代码仓库

### 端到端全流程（分词器 → 预训练 → SFT → RL → 推理）

| 仓库 | 规模 | 评价 |
|---|---|---|
| ★ **`karpathy/nanochat`** | ~560M | 目前最好的"全流程最小实现"。`speedrun.sh` 在 8×H100 上约 4 小时、约 $100 跑出能对话的模型，全程约 8000 行可读代码，没有 `Trainer` 黑箱。**与本项目目标最接近，是首要参考。** |
| ★ **`jingyaogong/minimind`** | 26M–1B | 中文社区最活跃的同类项目。单张 3090 约 2 小时跑完，覆盖 Pretrain / SFT / LoRA / DPO / 蒸馏 / MoE，中文注释详尽。**适合 P0 先跑通一遍建立直觉。** |
| `rasbt/LLMs-from-scratch` | GPT-2 小 | Sebastian Raschka 书的配套代码，逐层手写，教学性最强，但规模不够 |

### 预训练本身

| 仓库 | 评价 |
|---|---|
| ★ **`karpathy/build-nanogpt`** | 配套 4 小时视频"Let's reproduce GPT-2 (124M)"，从空文件逐行写起。**跟着敲一遍，收益极高。** |
| `karpathy/nanoGPT` | 经典基线，代码最干净，适合抄结构 |
| ★ **`KellerJordan/modded-nanogpt`** | GPT-2 复现速度竞赛仓库，Muon 优化器的来源。**读它的 record 历史能学到大量现代工程 trick**（每条记录都写了改了什么、快了多少） |
| `pytorch/torchtitan` | PyTorch 官方分布式预训练参考（FSDP2 / TP / PP / CP）。上多机时看这个 |
| `Lightning-AI/litgpt` | 工程化程度高，内置 TinyLlama 复现配方 |
| `karpathy/llm.c` | 纯 C/CUDA 实现 GPT-2，想搞懂底层算子和显存布局时看 |
| `allenai/OLMo` | 全透明训练框架，数据/日志/中间 checkpoint 全开 |

### 工具

- `EleutherAI/lm-evaluation-harness` — 评测事实标准，必用
- `huggingface/datatrove` — 数据清洗去重流水线（FineWeb 就是用它做的）
- `huggingface/tokenizers` — Rust BPE，训分词器用它
- `mosaicml/streaming` 或自己写 shard loader — 大规模数据流式读取

---

## 二、论文与课程

### 课程

- ★ **Stanford CS336 — Language Modeling from Scratch**
  作业就是从零写 BPE、Transformer、优化器、分布式、缩放律、数据处理、对齐。讲义与视频公开。**目前最系统的"全流程"教材，本项目的路线基本对齐它的作业顺序。**

### 全透明技术报告（写自己报告时的模板）

- ★ **OLMo / OLMo 2** (AI2) — 数据（Dolma）、代码、中间 checkpoint、训练日志全开。**报告结构直接照抄它。**
- **Pythia** (EleutherAI) — 70M–12B 一整套模型 + 154 个中间 checkpoint，为研究可复现性而设计
- ★ **SmolLM2 / SmolLM3** (HuggingFace) — 135M–1.7B，**规模与本项目最接近**，数据配比和多阶段退火策略讲得最细
- **HuggingFace Smol Training Playbook**（2025）— SmolLM3 的完整训练手记，包含失败实验

### 核心论文

| 主题 | 论文 | 要点 |
|---|---|---|
| 算力分配 | **Chinchilla** (Hoffmann et al. 2022) | 计算最优 ≈ 20 tokens/param；但推理成本导向下应 over-train 到 100× |
| 分布式 | **HuggingFace Ultra-Scale Playbook** (2025) | 并行策略、显存计算、MFU 调优的百科全书 |
| 小模型上限 | **TinyStories** (Eldan & Li 2023) | 数据质量 > 参数量 |
| 位置编码 | **RoFormer / RoPE** (Su et al.) | 旋转位置编码 |
| 激活 | **GLU Variants Improve Transformer** (Shazeer 2020) | SwiGLU 的来源 |
| 注意力 | **GQA** (Ainslie et al. 2023) | 分组查询注意力 |
| 内核 | **FlashAttention-2** (Dao 2023) | IO-aware 注意力 |
| Norm | **RMSNorm** (Zhang & Sennrich 2019) | 去中心化的 LayerNorm |
| 训练稳定 | **PaLM** (Chowdhery et al.) | z-loss、loss spike 的处置方式 |
| LR 日程 | **MiniCPM** / WSD | warmup-stable-decay，退火期的价值 |
| 优化器 | **Muon** (Jordan et al. 2024) | 隐藏层用正交化更新，实测显著加速 |
| 数据 | **FineWeb** (Penedo et al. 2024) | 网页数据清洗与消融的教科书 |
| 数据 | **DataComp-LM (DCLM)** (2024) | 数据筛选的系统性对照实验 |
| 后训练 | **InstructGPT** / **DPO** (Rafailov et al.) | SFT 与偏好对齐 |

---

## 三、数据集

| 数据集 | 语言 | 规模 | 说明 |
|---|---|---|---|
| `HuggingFaceFW/fineweb-edu` | 英 | 1.3T | 教育质量分类器筛过，小模型首选 |
| `mlfoundations/dclm-baseline-1.0` | 英 | 3.8T | DCLM 基线，质量高 |
| `opencsg/chinese-fineweb-edu-v2` | 中 | ~180B | 中文教育质量语料 |
| `BAAI/CCI3-HQ` | 中 | ~500B | 智源高质量中文语料 |
| `opendatalab/WanJuan`（书生·万卷） | 中 | 大 | 多模态，取文本部分 |
| `bigcode/the-stack-v2` / StarCoder2 数据 | 代码 | 大 | 取常用语言子集 |
| `HuggingFaceTB/finemath` | 数学 | ~50B | 数学网页 |
| `open-web-math/open-web-math` | 数学 | 15B | LaTeX 保留良好 |
| Wikipedia zh+en | 中英 | ~30B | 高质量百科 |

SFT 数据：`HuggingFaceTB/smoltalk`、`teknium/OpenHermes-2.5`、`BAAI/Infinity-Instruct`（中文）、`shibing624/sharegpt_gpt4`
偏好数据：`HuggingFaceH4/ultrafeedback_binarized`、`argilla/dpo-mix-7k`

---

## 四、算力平台

见 [04-compute.md](04-compute.md) 的详细对比与报价。
