# myTransformer

从零实现并训练一个 **~500M 参数（bf16 约 1.0 GB）的 decoder-only Transformer LLM**，覆盖从分词器到对话推理的**全流程**：

```
数据采集/清洗  →  BPE 分词器  →  模型实现  →  预训练  →  退火(midtrain)
      →  SFT  →  偏好对齐(DPO)  →  评测  →  推理部署  →  技术报告
```

不使用 `Trainer` 这类黑箱封装，训练主循环、优化器调度、分布式策略、KV cache 全部自己写，目标是**把每一个数字的来源都讲清楚**。

---

## 目标模型规格

| 项 | 值 |
|---|---|
| 参数量 | **514M**（非嵌入 452M） |
| bf16 权重体积 | **1.03 GB** |
| 架构 | LLaMA 风格：RMSNorm(pre-norm) + RoPE + SwiGLU + GQA |
| 层数 / 隐藏维 | 26 / 1280 |
| 注意力头 | 20 query heads / 5 KV heads（GQA，head_dim=64） |
| FFN 中间维 | 3456（SwiGLU） |
| 词表 | 49152（自训 BPE，中英+代码） |
| 上下文 | 2048 预训练 → 4096 退火期扩展 |
| 训练 token | 51B（≈100 tokens/param，刻意 over-train） |
| 预算 | 8×H100 约 14 小时 ≈ **$300** |

完整推导见 [docs/02-architecture.md](docs/02-architecture.md)。

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
│   ├── model/500m.yaml
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
