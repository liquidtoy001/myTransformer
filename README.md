# myTransformer

从零实现并训练一个 **decoder-only Transformer LLM**（主线 250M），覆盖从分词器到对话推理的**全流程**：

```
数据采集/清洗  →  BPE 分词器  →  模型实现  →  预训练  →  退火(midtrain)
      →  SFT  →  偏好对齐(DPO)  →  评测  →  推理部署  →  技术报告
```

不使用 `Trainer` 这类黑箱封装，训练主循环、优化器调度、分布式策略、KV cache 全部自己写，目标是**把每一个数字的来源都讲清楚**。

---

## 目标模型规格

| 项 | ★ **250M（主线）** |
|---|---|
| 参数量 | **236.5M**（非嵌入 202.9M，嵌入占 14.2%） |
| 架构 | LLaMA 风格：RMSNorm(pre-norm) + RoPE + SwiGLU + GQA |
| 层数 / 隐藏维 | 18 / 1024 |
| 注意力头 | 16 Q / 4 KV（GQA，head_dim=64） |
| FFN 中间维 | 2816（SwiGLU） |
| 词表 | 32768（自训 BPE，中英+代码） |
| 上下文 | 2048 预训练 → 4096 退火期扩展 |
| 训练 token | **9.44B**（≈40 tokens/param，适度 over-train） |
| bf16 权重 | 0.44 GB；AdamW 训练状态 3.5 GB |
| 训练成本 | 8×H100 竞价 **2.2 小时 ≈ $27 USD** |

规模是**由预算倒推**出来的，不是拍脑袋：100 AUD 的总预算里，500M 光主训练就要 152 AUD（见 [docs/04-compute.md](docs/04-compute.md) §五）。降到 250M 后，缩放律和消融实验全都保得住——而那才是学习价值最高的部分。

**词表也跟着缩了**（49152 → 32768）。这是小模型必须做的配平：词表不缩的话，嵌入层会吃掉过大比例的参数，真正参与计算的部分反而变少。

更大的两档配置保留在 `configs/model/{500m,1b}.yaml`，以后预算宽裕可直接切换——代码一行不用改。

---

## 预算

**全项目 76 AUD（$50 USD）**，已含 30% 重试余量。

| 阶段 | 机时 | AUD |
|---|---|---|
| P4 冒烟 30M | 1 分 | 0.1 |
| P5 缩放律 30M / 60M / 120M（单卡） | 1.5 h | 3.4 |
| P5 第 4 点（从 P6 分叉，见下） | 7 分 | 2.1 |
| P5 消融 6 组 @ 60M（单卡） | 57 分 | 2.2 |
| **P6 正式训练 250M（8×H100）** | **2.2 h** | **40.5** |
| P7 SFT + DPO | | 3.0 |
| P8 评测 + 4 个对照模型 | | 4.6 |
| 对象存储 1 个月 | | 3.0 |
| 小计 | | 58.8 |
| +30% 重试/抢占余量 | | 17.6 |
| **合计** | | **76.4 AUD** |

三个把成本压下来的做法：

1. **缩放律第 4 个点用 WSD 分叉免费拿。** P6 训练到 4.73B tokens（=20× Chinchilla 点）时分叉出一支做短衰减到 0，得到与前三点可比的数据点——$1.4 而不是重训一次的 $11。这正是选 WSD 而非 cosine 的实际收益。
2. **消融在 60M 上跑**，6 组共 57 分钟。挑 2 组在 120M 上复跑，看效应是放大还是消失。
3. **单卡 / 8 卡分工**：小实验用单卡 H100 竞价（$1.5/h，简单便宜）；只有 P6 用 8 卡，时间短且把 DDP 学到。

**P0–P4 的全部调试在本地 GPU 上做，零成本。** 调试是短时的，不需要长时间占用机器。

---

## 文档导航

| 文档 | 内容 | 什么时候读 |
|---|---|---|
| [01-roadmap.md](docs/01-roadmap.md) | **30 天分阶段路线图**、里程碑与验收标准 | 开工前先读这个 |
| [02-architecture.md](docs/02-architecture.md) | 模型架构设计 + 代码架构 + 参数量推导 | 写第一行代码前 |
| [03-resources.md](docs/03-resources.md) | 参考仓库、论文、课程清单（附评价） | 全程查阅 |
| [04-compute.md](docs/04-compute.md) | 算力选型、FLOPs 估算、时间与成本表 | 掏钱之前 |
| [05-data.md](docs/05-data.md) | 数据来源、配比、清洗去重、分词打包 | 阶段 P2 |
| [06-training-recipe.md](docs/06-training-recipe.md) | 超参、LR 日程、并行策略、故障处置 | 阶段 P4–P6 |
| [07-evaluation.md](docs/07-evaluation.md) | 评测任务选择与基线对照 | 阶段 P8 |
| [08-report-template.md](docs/08-report-template.md) | 技术报告骨架（照 OLMo/SmolLM2） | 阶段 P8 |

---

## 规划中的仓库结构

```
myTransformer/
├── configs/                    # YAML 实验配置（一个实验一个文件，入 git）
│   ├── model/250m.yaml         # 主线
│   ├── model/500m.yaml         # 参考（预算宽裕时）
│   ├── model/1b.yaml           # 参考
│   ├── data/mixture_v1.yaml
│   └── train/pretrain_250m.yaml
├── src/mytransformer/
│   ├── tokenizer/              # BPE 训练 / 封装 / fertility 分析
│   ├── data/                   # 下载 → 过滤 → 去重 → 分词 → .bin 分片
│   ├── model/                  # config / rope / norm / attention / mlp / transformer
│   ├── train/                  # trainer / optim / lr_schedule / checkpoint / dist
│   ├── posttrain/              # sft / dpo
│   ├── eval/                   # 内置评测 + lm-eval-harness 适配
│   └── infer/                  # kv_cache / generate / chat CLI
├── scripts/                    # speedrun.sh、launch_multinode.sh、download_data.sh
├── tests/                      # 数值对齐测试、shape 测试、tokenizer 往返测试
├── reports/                    # 实验记录、消融表、最终技术报告
└── docs/                       # 见上表
```

---

## 快速开始（占位，随实现更新）

```bash
# 1. 环境
uv venv && uv pip install -e ".[dev]"

# 2. 冒烟测试：30M 模型 / 0.3B tokens / 单卡
bash scripts/smoke.sh

# 3. 正式预训练
torchrun --nproc_per_node=8 -m mytransformer.train.pretrain --config configs/train/pretrain_250m.yaml
```

---

## 原则

1. **先跑通再跑大** — 任何配置先在 30M/0.3B token 上端到端跑一遍，再上 8 卡。
2. **每个实验可复现** — config 入 git，commit hash 写进 checkpoint 元数据。
3. **数值对齐优先** — 模型实现完成后先与 HuggingFace `LlamaForCausalLM` 逐层对齐（误差 < 1e-4），再谈训练。
4. **不报无意义的分数** — 250M 规模在 MMLU/GSM8K 上接近随机，如实说明而不是粉饰。
5. **报告是产物之一** — loss 曲线、失败实验、loss spike 的处置过程都要记录。

## License

MIT
