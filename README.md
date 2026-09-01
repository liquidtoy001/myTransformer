# myTransformer

从零实现并训练一个 **decoder-only Transformer LLM**（主线 500M，可选升级到 1B），覆盖从分词器到对话推理的**全流程**：

```
数据采集/清洗  →  BPE 分词器  →  模型实现  →  预训练  →  退火(midtrain)
      →  SFT  →  偏好对齐(DPO)  →  评测  →  推理部署  →  技术报告
```

不使用 `Trainer` 这类黑箱封装，训练主循环、优化器调度、分布式策略、KV cache 全部自己写，目标是**把每一个数字的来源都讲清楚**。

---

## 目标模型规格

两档配置，架构完全相同，只有宽度/深度不同。**主线跑 500M，代码和流程一字不改就能切到 1B。**

| 项 | ★ **500M（主线）** | **1B（可选升级）** |
|---|---|---|
| 参数量 | **514M**（非嵌入 452M） | **1.09B**（非嵌入 992M） |
| 层数 / 隐藏维 | 26 / 1280 | 22 / 2048 |
| 注意力头 | 20 Q / 5 KV，head_dim 64 | 16 Q / 4 KV，head_dim 128 |
| FFN 中间维 | 3456 | 5632 |
| 训练 token | 51B（≈100 tok/param） | 25B（≈23 tok/param） |
| 8×H100 耗时 | **~14 小时 ≈ $300** | **~13 小时 ≈ $260** |
| 单卡训练显存 | ~14 GB（4090 可单卡跑） | ~26 GB（**需 ZeRO/FSDP 或 40G+ 卡**） |

共同部分：LLaMA 风格（RMSNorm pre-norm + RoPE + SwiGLU + GQA）、词表 49152 自训 BPE（中英+代码）、上下文 2048 预训练 → 4096 退火扩展、嵌入权重绑定、无 bias、无 dropout。

**为什么主线选 500M 而不是 1B**（同样约 $300 的预算下）：

1. **能 over-train 到 100 tokens/param**。1B 在同预算下只够 ~23 tok/param（勉强到 Chinchilla 最优），而 500M 能喂 4 倍于最优的数据。SmolLM2、Qwen2.5 这些实际好用的小模型走的都是 over-train 路线——**推理成本导向下，Chinchilla 最优并不是最优**。
2. **单卡装得下**。500M 训练显存 14GB，一张 4090 就能完整跑通预训练；1B 一定要上 ZeRO-2/FSDP 或 8-bit optimizer。这对调试阶段的迭代速度影响很大。
3. **迭代快**。P5 的缩放律和 6 组消融实验全都能在小规模上快速跑完。

1B 的价值在于：**它逼你真正用上分布式切分**。所以路线图安排是先把 500M 完整跑通拿到报告，有余力再用同一套代码跑 1B 作为第二个数据点——那时缩放律曲线上就有 5 个点了，报告的含金量会明显不同。

完整参数量推导与两档配置见 [docs/02-architecture.md](docs/02-architecture.md)。

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
│   ├── model/500m.yaml         # 主线
│   ├── model/1b.yaml           # 可选升级
│   ├── data/mixture_v1.yaml
│   └── train/pretrain_500m.yaml
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

# 2. 冒烟测试：50M 模型 / 1B tokens / 单卡
bash scripts/smoke.sh

# 3. 正式预训练
torchrun --nproc_per_node=8 -m mytransformer.train.pretrain --config configs/train/pretrain_500m.yaml
```

---

## 原则

1. **先跑通再跑大** — 任何配置先在 50M/1B token 上端到端跑一遍，再上 8 卡。
2. **每个实验可复现** — config 入 git，commit hash 写进 checkpoint 元数据。
3. **数值对齐优先** — 模型实现完成后先与 HuggingFace `LlamaForCausalLM` 逐层对齐（误差 < 1e-4），再谈训练。
4. **不报无意义的分数** — 500M 规模在 MMLU/GSM8K 上接近随机，如实说明而不是粉饰。
5. **报告是产物之一** — loss 曲线、失败实验、loss spike 的处置过程都要记录。

## License

MIT
