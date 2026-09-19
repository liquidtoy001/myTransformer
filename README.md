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
| 训练成本 | 8×H100 竞价 **2.5 小时 ≈ $30 USD** |

规模是**由预算倒推**出来的，不是拍脑袋：100 AUD 的总预算里，500M 光主训练就要 103 AUD（见 [docs/04-compute.md](docs/04-compute.md) §二）。降到 250M 后，缩放律和消融实验全都保得住——而那才是学习价值最高的部分。

**词表也跟着缩了**（49152 → 32768）。这是小模型必须做的配平：词表不缩的话，嵌入层会吃掉过大比例的参数，真正参与计算的部分反而变少。

更大的两档配置保留在 `configs/model/{500m,1b}.yaml`，以后预算宽裕可直接切换——代码一行不用改。

---

## 预算与硬件

**全项目约 66 AUD（$43 USD）**，已含 30% 重试余量。硬件分三处：

| 硬件 | 负责 | 成本 |
|---|---|---|
| 本地 RTX 4070 Ti SUPER | P0–P4 开发调试 | 0 |
| 学院 Rangpur A100 40G（[说明](docs/09-rangpur.md)） | P4 冒烟、P5 缩放律与消融、P7 后训练、P8 评测、数据准备 | 0（约 17 GPU 小时） |
| 云上 8×H100 竞价 | **仅 P6 正式训练**（2.5 小时） | ~66 AUD |

几个关键取舍：

1. **缩放律第 4 个点用 WSD 分叉拿。** P6 训到 20 tokens/param 时分叉出一支做短衰减，得到与小模型可比的 250M 数据点，约 $1.5 而不是重训的 $15。这是选 WSD 而非 cosine 的实际收益。
2. **P6 不放 Rangpur**：约 47 GPU 小时在全班共用的 10 块 A100 上太显眼；而且 Rangpur 每节点只有 1 块 GPU，学不到多卡 DDP。
3. **P6 的数据在 Rangpur `cpu` 分区准备**，逐分片上传对象存储，不在 $12/小时的 GPU 机器上做 CPU 活。

**算力买的是模型质量，不是学习收益。** 分词器、数据管线、架构实现、数值对齐、训练框架、缩放律、消融——项目主体全部免费完成。P6 只是把一条已验证的流水线放大跑一次。

明细见 [docs/04-compute.md](docs/04-compute.md) §五。

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
| [09-rangpur.md](docs/09-rangpur.md) | Rangpur 集群：提交规则、存储预算、作业模板 | 第一次上 Rangpur 前 |
| [learn/](docs/learn/README.md) | **学习笔记**：每个部分怎么读、怎么验证、怎么自己从零搭，附"故意改错"练习 | 每完成一部分 |

---

## 规划中的仓库结构

```
myTransformer/
├── configs/                    # YAML 实验配置（一个实验一个文件，入 git）
│   ├── model/250m.yaml         # 主线
│   ├── model/ladder_s{1,2,3}.yaml  # 缩放律阶梯（与主线同词表、同序列长度）
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
├── scripts/                    # count_params.py、benchmark.py；rangpur/ 下放 sbatch 模板（P4）
├── tests/                      # 数值对齐测试、shape 测试、tokenizer 往返测试
├── reports/                    # 实验记录、消融表、最终技术报告
└── docs/                       # 见上表
```

---

## 快速开始（占位，随实现更新）

```bash
# 1. 环境
uv venv && uv pip install -e ".[dev]"

# 2. 冒烟测试：ladder_s1 / 0.3B tokens / 单卡
bash scripts/smoke.sh

# 3. 正式预训练
torchrun --nproc_per_node=8 -m mytransformer.train.pretrain --config configs/train/pretrain_250m.yaml
```

---

## 原则

1. **先跑通再跑大** — 任何配置先在 `ladder_s1` / 0.3B token 上端到端跑一遍，再上 Rangpur 或 8 卡。
2. **每个实验可复现** — config 入 git，commit hash 写进 checkpoint 元数据。
3. **数值对齐优先** — 模型实现完成后先与 HuggingFace `LlamaForCausalLM` 逐层对齐（误差 < 1e-4），再谈训练。
4. **不报无意义的分数** — 250M 规模在 MMLU/GSM8K 上接近随机，如实说明而不是粉饰。
5. **报告是产物之一** — loss 曲线、失败实验、loss spike 的处置过程都要记录。

## License

MIT
