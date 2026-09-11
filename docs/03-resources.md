# 参考资源

带评价的清单。**★** = 强烈推荐先看。

---

## 一、代码仓库

### 端到端全流程（分词器 → 预训练 → SFT → RL → 推理）

| 仓库 | 规模 | 评价 |
|---|---|---|
| ★ **[`karpathy/nanochat`](https://github.com/karpathy/nanochat)** | ~560M | 目前最好的"全流程最小实现"。`speedrun.sh` 在 8×H100 上约 4 小时、约 $100 跑出能对话的模型，全程约 8000 行可读代码，没有 `Trainer` 黑箱。**与本项目目标最接近，是首要参考。** |
| ★ **[`jingyaogong/minimind`](https://github.com/jingyaogong/minimind)** | 26M–1B | 中文社区最活跃的同类项目。单张 3090 约 2 小时跑完，覆盖 Pretrain / SFT / LoRA / DPO / 蒸馏 / MoE，中文注释详尽。**适合 P0 先跑通一遍建立直觉。** |
| [`rasbt/LLMs-from-scratch`](https://github.com/rasbt/LLMs-from-scratch) | GPT-2 小 | Sebastian Raschka 书的配套代码，逐层手写，教学性最强，但规模不够 |

### 预训练本身

| 仓库 | 评价 |
|---|---|
| ★ **[`karpathy/build-nanogpt`](https://github.com/karpathy/build-nanogpt)** | 配套 4 小时视频"Let's reproduce GPT-2 (124M)"，从空文件逐行写起。**跟着敲一遍，收益极高。** |
| [`karpathy/nanoGPT`](https://github.com/karpathy/nanoGPT) | 经典基线，代码最干净，适合抄结构 |
| ★ **[`KellerJordan/modded-nanogpt`](https://github.com/KellerJordan/modded-nanogpt)** | GPT-2 复现速度竞赛仓库，Muon 优化器的来源。**读它的 record 历史能学到大量现代工程 trick**（每条记录都写了改了什么、快了多少） |
| [`pytorch/torchtitan`](https://github.com/pytorch/torchtitan) | PyTorch 官方分布式预训练参考（FSDP2 / TP / PP / CP）。上多机时看这个 |
| [`Lightning-AI/litgpt`](https://github.com/Lightning-AI/litgpt) | 工程化程度高，内置 TinyLlama 复现配方 |
| [`karpathy/llm.c`](https://github.com/karpathy/llm.c) | 纯 C/CUDA 实现 GPT-2，想搞懂底层算子和显存布局时看 |
| [`allenai/OLMo`](https://github.com/allenai/OLMo) | 全透明训练框架，数据/日志/中间 checkpoint 全开 |

### 工具

- [`EleutherAI/lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness) — 评测事实标准，必用
- [`huggingface/datatrove`](https://github.com/huggingface/datatrove) — 数据清洗去重流水线（FineWeb 就是用它做的）
- [`huggingface/tokenizers`](https://github.com/huggingface/tokenizers) — Rust BPE，训分词器用它
- [`mosaicml/streaming`](https://github.com/mosaicml/streaming) 或自己写 shard loader — 大规模数据流式读取

---

## 二、论文与课程

### 课程

- ★ **Stanford CS336 — Language Modeling from Scratch**
  - 课程主页与作业：<https://stanford-cs336.github.io/spring2025/>（另有 <https://cs336.stanford.edu/>）
  - 讲座录像：YouTube 搜 "Stanford CS336"，[Spring 2025 第一讲](https://www.youtube.com/watch?v=SQ3fZ1sAqXI)、[Spring 2026 第一讲](https://www.youtube.com/watch?v=JuoVZkPBiKk)

  作业就是从零写 BPE、Transformer、优化器、分布式、缩放律、数据处理、对齐。**目前最系统的"全流程"教材，本项目的路线基本对齐它的作业顺序。**

### 全透明技术报告（写自己报告时的模板）

- ★ **OLMo / OLMo 2** (AI2) — [arXiv:2402.00838](https://arxiv.org/abs/2402.00838)（OLMo 1）、代码 [allenai/OLMo](https://github.com/allenai/OLMo)、模型 <https://huggingface.co/allenai>。数据（Dolma）、代码、中间 checkpoint、训练日志全开。**报告结构直接照抄它。**
- **Pythia** (EleutherAI) — [arXiv:2304.01373](https://arxiv.org/abs/2304.01373)、[github.com/EleutherAI/pythia](https://github.com/EleutherAI/pythia)。70M–12B 一整套模型 + 154 个中间 checkpoint，为研究可复现性而设计
- ★ **SmolLM2** (HuggingFace) — [arXiv:2502.02737](https://arxiv.org/abs/2502.02737)、[模型集合](https://huggingface.co/collections/HuggingFaceTB/smollm2-6723884218bcda64b34d7db9)。135M–1.7B，**规模与本项目最接近**，数据配比和多阶段退火策略讲得最细
- **HuggingFace Smol Training Playbook / Ultra-Scale Playbook** — 在 <https://huggingface.co/spaces/HuggingFaceTB> 与 <https://huggingface.co/spaces/nanotron> 下找。SmolLM3 的完整训练手记 + 分布式训练百科，包含失败实验

### 核心论文

| 主题 | 论文 | 链接 | 要点 |
|---|---|---|---|
| 起点 | Attention Is All You Need | [1706.03762](https://arxiv.org/abs/1706.03762) | Transformer 原始论文 |
| 算力分配 | **Chinchilla** | [2203.15556](https://arxiv.org/abs/2203.15556) | 计算最优 ≈ 20 tokens/param；但推理成本导向下应 over-train |
| 小模型上限 | **TinyStories** | [2305.07759](https://arxiv.org/abs/2305.07759) | 数据质量 > 参数量；小模型配简单数据也能连贯 |
| 位置编码 | **RoFormer / RoPE** | [2104.09864](https://arxiv.org/abs/2104.09864) | 旋转位置编码 |
| 激活 | **GLU Variants Improve Transformer** | [2002.05202](https://arxiv.org/abs/2002.05202) | SwiGLU 的来源 |
| 注意力 | **GQA** | [2305.13245](https://arxiv.org/abs/2305.13245) | 分组查询注意力 |
| 内核 | **FlashAttention** / **-2** | [2205.14135](https://arxiv.org/abs/2205.14135) / [2307.08691](https://arxiv.org/abs/2307.08691) | IO-aware 注意力 |
| Norm | **RMSNorm** | [1910.07467](https://arxiv.org/abs/1910.07467) | 去中心化的 LayerNorm |
| 训练稳定 | **PaLM** | [2204.02311](https://arxiv.org/abs/2204.02311) | z-loss、loss spike 的处置方式 |
| LR 日程 | **MiniCPM** (WSD) | [2404.06395](https://arxiv.org/abs/2404.06395) | warmup-stable-decay，退火期的价值 |
| 优化器 | **Muon** (Keller Jordan) | <https://kellerjordan.github.io/posts/muon/> + [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) | 隐藏层用正交化更新（博客而非论文） |
| 数据 | **FineWeb** | [2406.17557](https://arxiv.org/abs/2406.17557) | 网页数据清洗与消融的教科书 |
| 数据 | **DataComp-LM (DCLM)** | [2406.11794](https://arxiv.org/abs/2406.11794) | 数据筛选的系统性对照实验 |
| 后训练 | **InstructGPT** | [2203.02155](https://arxiv.org/abs/2203.02155) | SFT + RLHF |
| 后训练 | **DPO** | [2305.18290](https://arxiv.org/abs/2305.18290) | 免 RL 的偏好对齐 |

> arXiv 编号除 SmolLM2（已核对）外均凭记忆填写。点开确认标题对得上再读；对不上就按标题搜。

---

## 三、数据集

| 数据集 | 语言 | 规模 | 说明 |
|---|---|---|---|
| [`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | 英 | 1.3T | 教育质量分类器筛过，小模型首选 |
| [`mlfoundations/dclm-baseline-1.0`](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) | 英 | 3.8T | DCLM 基线，质量高 |
| [`opencsg/chinese-fineweb-edu-v2`](https://huggingface.co/datasets/opencsg/chinese-fineweb-edu-v2) | 中 | ~180B | 中文教育质量语料 |
| [`BAAI/CCI3-HQ`](https://huggingface.co/datasets/BAAI/CCI3-HQ) | 中 | ~500B | 智源高质量中文语料 |
| [`opendatalab/WanJuan`](https://opendatalab.com/OpenDataLab/WanJuan1_dot_0)（书生·万卷） | 中 | 大 | 多模态，取文本部分 |
| [`bigcode/the-stack-v2`](https://huggingface.co/datasets/bigcode/the-stack-v2) / StarCoder2 数据 | 代码 | 大 | 取常用语言子集 |
| [`HuggingFaceTB/finemath`](https://huggingface.co/datasets/HuggingFaceTB/finemath) | 数学 | ~50B | 数学网页 |
| [`open-web-math/open-web-math`](https://huggingface.co/datasets/open-web-math/open-web-math) | 数学 | 15B | LaTeX 保留良好 |
| [Wikipedia zh+en](https://huggingface.co/datasets/wikimedia/wikipedia) | 中英 | ~30B | 高质量百科 |

SFT 数据：[`HuggingFaceTB/smoltalk`](https://huggingface.co/datasets/HuggingFaceTB/smoltalk)、[`teknium/OpenHermes-2.5`](https://huggingface.co/datasets/teknium/OpenHermes-2.5)、[`BAAI/Infinity-Instruct`](https://huggingface.co/datasets/BAAI/Infinity-Instruct)（中文）、[`shibing624/sharegpt_gpt4`](https://huggingface.co/datasets/shibing624/sharegpt_gpt4)
偏好数据：[`HuggingFaceH4/ultrafeedback_binarized`](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)、[`argilla/dpo-mix-7k`](https://huggingface.co/datasets/argilla/dpo-mix-7k)

---

## 四、算力平台

见 [04-compute.md](04-compute.md) 的详细对比与报价。
