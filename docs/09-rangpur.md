# Rangpur 集群使用说明

本项目 P4 冒烟、P5 全部实验、P7、P8 都在学院的 Rangpur 集群（A100 40G）上跑；P6 正式训练租云上 8×H100（见 [04-compute.md](04-compute.md) §五）。这份文档记录 Rangpur 的硬约束、存储方案和作业模板。

以下事实来自实际使用（GAN 训练）、终端实测（2026-09-19）和课程指南 *COMP3710 Getting Started on Rangpur*。文中学号一律写作 `sXXXXXXX`。

---

## 一、硬约束

| 约束 | 实际情况 | 对本项目的影响 |
|---|---|---|
| 调度器 | SLURM（`sbatch` / `srun` / `squeue`） | 用 batch 作业，不用交互会话跑训练 |
| 每人同时运行的作业数 | **1 个**；其余排队显示 `QOSMaxJobsPerUserLimit`，前一个结束后自动开始 | 多个作业天然串行接力 |
| 每个节点的 GPU | **1 块**（`gpu:a100:1`） | **Rangpur 上做不了多卡 DDP**，放到云上 P6 学 |
| `comp3710` 分区单作业时限 | 12 小时 | 实际申请 **4 小时左右**（见 §五） |
| home 配额 | **16 GB**，NFS，登录节点和所有计算节点共享；实测已用 8.6 GB，**只剩 7.5 GB** | 存储是最紧的约束，见 §三 |
| 计算节点本地盘 | 作业内 `$TMPDIR=/scratch/<作业号>`，约 **197 GB**，只属于当前作业 | 当作业内工作区，**不能长期存数据** |
| login0 的 `/scratch` | 登录节点**自己的本地盘**（`/dev/sdb1`，80% 已用，所有人共用） | **不要用**，A100 作业看不到 |
| A100 总数 | 全班共用 **10 块** | 注意占用，见 §六 |

---

## 二、提交规则

| 规则 | 不遵守的后果 | 来源 |
|---|---|---|
| **GPU 作业必须加 `--account=comp3710`** | 一直卡在 `PartitionConfig` | 实测 |
| **不要加 `--mem`** | 作业一直排不上 | 实测 ⚠️ |
| 下载数据用 **`cpu` 分区的 batch 作业**，不在 login 节点上下载 | 占用共享的登录节点 | 实测 + 指南 §5 |
| 正式训练前先在 **`a100-test`** 分区做冒烟（上限 20 分钟） | 在正式分区排几小时队，结果一跑就报错 | 实测 |
| 安装包用 `pip install --no-cache-dir` | pip 缓存撑爆 home | 指南 §3.8 |
| 日志文件名带 `%j` | 重复提交互相覆盖 | 指南 §6 |
| 在脚本里激活环境 | batch 作业从干净环境启动，找不到 torch | 指南 §6 |

> ⚠️ **与官方指南不一致**：指南 §4.2 的示例写了 `--mem=16G`，§6 说"一定要显式设置 `--mem`"；但实际加了 `--mem` 作业就排不上。本文按**实测能跑通的写法**来。如果训练因主机内存不足被杀，再到 `a100-test` 上试一个较小的 `--mem` 值。
>
> 指南 §3.2 让你 `export TMPDIR=$HOME/tmp`——那只适用于安装 Miniconda 那一步。**训练和数据作业里不要这样设置**，否则临时文件会写进只有 16 GB 的 home，而系统本来就给了 197 GB 的本地盘。

---

## 三、存储

### 现状（2026-09-19）

| 目录 | 大小 | 处理 |
|---|---|---|
| `miniconda3`（含 torch 2.13.0+cu130 环境） | 4.5 GB | **复用**，与本地开发环境版本完全一致 |
| └ `pkgs` 包缓存 | 0.9 GB | `conda clean --all -y` 可安全清掉 |
| 课程作业目录（demo2） | 3.6 GB | demo2 已结束：按 §3.5 **备份到本地后删除** |
| `data` | 0.3 GB | 自行判断 |
| **可用** | **7.5 GB** → 清 conda 缓存后 8.4 GB → **删 demo2 后约 12 GB** | |

### 三层存储

| 位置 | 容量 | 寿命 | 放什么 |
|---|---|---|---|
| `$HOME` | 可用约 12 GB（清理 demo2 与 conda 缓存后） | 永久 | 代码、分词器、**P5 数据分片**、最新 checkpoint、编译缓存、指标日志 |
| `$TMPDIR`（计算节点本地） | ~197 GB | **作业结束即失效**，且下个作业可能在别的节点 | HF 下载缓存、原始 parquet、中间文件、wandb 本地文件 |
| 对象存储（云） | 按需 | 永久 | **P6 的 19 GB 数据分片**、历史 checkpoint |

### P5 阶段预算

| 项目 | 大小 |
|---|---|
| 数据（S3 需要 1.99B tokens，所有 P5 实验共用） | 4.0 GB |
| 最新 checkpoint + 写入时的临时副本（S3：1.2 GB × 2） | 2.4 GB |
| torch 编译缓存 | ~0.5 GB |
| **合计** | **~6.9 GB**；清理后可用约 12 GB，余量约 5 GB |

**写 checkpoint 要"先写临时文件、再改名替换"**，不能先删旧的再写新的：作业可能在写到一半时被杀，那样两份都没了。所以峰值是两份。历史 checkpoint 每次作业结束后同步回本地电脑，不留在 Rangpur。

**如果不清理 conda 缓存**：S3 改用 1B 不重复 token 训 2 轮（Muennighoff 等 [arXiv:2305.16264](https://arxiv.org/abs/2305.16264)：同一批数据重复 ≤4 轮，loss 与全新数据几乎无差别）。数据降到 2 GB，峰值约 4.9 GB。

**P7 阶段**：删掉 P5 的数据，腾出空间放 250M 的完整 checkpoint（2.84 GB × 2）。

### 防止撑爆配额

所有作业开头都设置：

```bash
export HF_HOME="$TMPDIR/hf"                               # HF 下载缓存 → 本地盘
export HF_DATASETS_CACHE="$TMPDIR/hf/datasets"
export PIP_NO_CACHE_DIR=1                                 # 不留 pip 缓存
export TORCHINDUCTOR_CACHE_DIR="$HOME/myTransformer/.cache/inductor"  # 编译缓存留 home，跨作业复用
export WANDB_DIR="$TMPDIR/wandb"                          # wandb 在线同步，本地文件放临时盘
```

用 `datasets` 库读数据时**必须 `streaming=True`**，或者确保 `HF_HOME` 指向 `$TMPDIR`。默认会把整个原始 parquet 缓存到 `~/.cache/huggingface`，一下就超配额。

查 home 用量要在 CPU 作业里用 `du -xh --max-depth=1 ~ | sort -h`。`du -sh ~` 在 NFS 上会扫描 conda 环境里的大量小文件，非常慢。

### 3.4 第一次上手：克隆仓库、配置环境

**1. 克隆仓库**（登录节点上可以做，这是"moving data"）：

```bash
git clone https://github.com/liquidtoy001/myTransformer.git ~/myTransformer
```

仓库如果是私有的，需要先在 Rangpur 上配 SSH key 或 GitHub token。以后更新代码：`cd ~/myTransformer && git pull`。

**2. 配置环境**（要装包，放在 CPU 交互作业里做）：

```bash
srun --partition=cpu --time=00:30:00 --pty bash
```

```bash
bash ~/myTransformer/scripts/rangpur/setup_env.sh && exit
```

脚本做的事：在课程的 `torch` 环境上叠一个 `~/mt-venv`，用 `--system-site-packages` 直接继承课程环境里的 torch，只额外安装本项目的小依赖。

**为什么不直接装进课程环境**：`miniconda3/envs/torch` 后面的课程作业还要用，往里装本项目的依赖可能引起版本冲突，把作业环境搞坏。`mt-venv` 只占几百 MB，收尾时 `rm -rf ~/mt-venv` 即可。

**本项目在 Rangpur 上的所有文件只放两处**：`~/myTransformer`（代码、数据、checkpoint、编译缓存、日志）和 `~/mt-venv`。

### 3.5 开工前：备份并删除 demo2（2026-09-19 已备份到本地，SHA256 校验一致）

tar 打成一个文件再传，比 `scp -r` 逐个传大量小文件快，而且能用校验和确认完整。指南 §1 说登录节点可以用来"moving data"，这一步不用开作业。

**1. 在 Rangpur 登录节点打包**（需要约 3.6 GB 临时空间，当前可用 7.5 GB，够）：

```bash
cd ~ && tar cf demo2_backup.tar COMP3710Rangpurfordemo2 && sha256sum demo2_backup.tar
```

**2. 在本地 PowerShell 下载并校验**：

```powershell
scp sXXXXXXX@rangpur.compute.eait.uq.edu.au:~/demo2_backup.tar E:\Documents\COMP3710\
```

```powershell
(Get-FileHash E:\Documents\COMP3710\demo2_backup.tar -Algorithm SHA256).Hash.ToLower()
```

两边的 SHA256 必须完全一致（Linux 输出小写，所以 PowerShell 这边转成小写再比）。

> ⚠️ 不要用 `ssh ... "tar cf - ..." > 文件` 这种流式写法：**Windows PowerShell 5.1 的 `>` 会把二进制流按文本重新编码**，文件会损坏且不报错。

**3. 校验一致后，在 Rangpur 上删除**：

```bash
rm -rf ~/COMP3710Rangpurfordemo2 ~/demo2_backup.tar && df -h ~
```

本地解包：`tar -xf E:\Documents\COMP3710\demo2_backup.tar -C E:\Documents\COMP3710\`（Windows 10 以后自带 `tar`）。

---

## 四、作业脚本

都在 `scripts/rangpur/`，**在 `~/myTransformer` 目录下提交**（日志路径是相对于提交目录的）。

| 脚本 | 用途 | 分区 |
|---|---|---|
| `setup_env.sh` | 一次性环境配置（§3.4） | `cpu` 交互作业里运行 |
| `smoke.sbatch` | 冒烟：测试 + `torch.compile` + 真实信号 + 续跑 | `a100-test`，15 分钟 |
| `train.sbatch` | 训练一段（到 225 分钟存档退出） | `comp3710`，4 小时 |
| `submit_chain.sh` | 把 `train.sbatch` 串成 N 个接力作业 | — |

数据准备作业（`data.sbatch`）随 P2 一起写。

### 4.1 冒烟（第一次上 A100 必做）

```bash
cd ~/myTransformer && mkdir -p logs && sbatch scripts/rangpur/smoke.sbatch
```

**`mkdir -p logs` 不能省**：Slurm 不会自动创建 `--output` 所在的目录，目录不存在时作业会直接失败，而且**没有任何日志**可看。

看进度和结果：

```bash
squeue --me
```

```bash
tail -f logs/smoke_*.out
```

冒烟分四步，每一步验证一件在本地验证不了的事：

| 步骤 | 验证什么 | 为什么本地验证不了 |
|---|---|---|
| [1/4] 全部测试 | A100 上测试照样全绿，包括分块交叉熵的显存测试 | 硬件不同 |
| [2/4] 合成数据 | 与本地冒烟相同的数据，loss 可以直接对比 | — |
| [3/4] `compile=true` 训练，中途发 `SIGUSR1` | `torch.compile` 能用；收到真实信号会存档退出 | Windows 没有 triton，也没有 `SIGUSR1` |
| [4/4] 同一条命令再启动 | 续跑、通过自检、训完 | 本地验证过，这里确认 Linux 上也一样 |

脚本最后会**明确检查**第 3 步是因 `SIGUSR1` 停下的、第 4 步是从 checkpoint 续跑的，任何一项不满足都判为失败（退出码 1，日志末尾是"冒烟失败"）。只看"最后训完了"是不够的：信号来得太早时进程会被直接杀掉，第 4 步从头训练照样能训完。

第一次跑时顺便确认 §八 的待确认事项：A100 节点上 `$TMPDIR` 有多大。

### 4.2 训练接力

```bash
cd ~/myTransformer && bash scripts/rangpur/submit_chain.sh configs/train/<配置>.yaml 3
```

同一配置跑不同种子（P5 的 seed 方差实验）：

```bash
SET="seed=1 run_dir=runs/<名字>_seed1" bash scripts/rangpur/submit_chain.sh configs/train/<配置>.yaml 1
```

**两道保险**（`train.sbatch` 里）：

1. `--max-minutes 225`：训练 225 分钟后主动存档退出，给启动、编译、存档留 15 分钟
2. `#SBATCH --signal=B:USR1@600`：万一没按时停，被杀前 10 分钟 Slurm 会发 `USR1`

**`exec` 不能省**：`B:` 表示信号发给"批处理 shell"。`exec` 让 python 直接取代这个 shell 成为作业主进程，信号才会送到 python；不用 `exec`，信号只到 bash，python 收不到，照样会在时限到时被强杀。

**用 `afterany` 串联**：前一个作业不管正常结束、到时被杀还是出错，下一个都照样启动。续跑逻辑本来就能处理这些情况；训练提前完成时，后面排着的作业一启动就会发现"已训完"并立即退出。如果某个作业报错（比如续跑自检失败），后面的作业也会很快失败：用 `scancel` 取消剩下的，查清原因再重新提交。

### 4.3 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| 一直 `PD (PartitionConfig)` | 没加 `--account=comp3710` | 脚本里已经加了；自己写新脚本时别忘 |
| 一直 `PD`，原因是资源 | 加了 `--mem`，或申请的时间太长 | 去掉 `--mem`；调试用 `a100-test` 和短时间 |
| 作业瞬间结束，找不到日志 | `logs/` 目录不存在 | `mkdir -p logs` |
| `ModuleNotFoundError: mytransformer` | 没激活 `mt-venv`，或没运行 `setup_env.sh` | 按 §3.4 配置 |
| `$'
': command not found` | 脚本是 CRLF 换行 | 仓库里的 `.gitattributes` 已强制 LF；自己在 Windows 上新写的脚本要注意 |
| `torch.compile` 报错 | 编译器、驱动或 triton 的问题 | 先用 `--set compile=false` 跑通，再单独排查 |
| `ResumeError` | 续跑自检失败：代码、权重或数据与存档不一致 | **不要**删掉 checkpoint 硬跑；看报错里是模型输出还是数据流，查清原因 |

---

## 五、作业时长

P5 在 Rangpur 上的纯训练时间约 14.7 小时（A100 MFU 按 35% 估），加开销约 17 GPU 小时。

**每个作业申请约 4 小时**，不申请满 12 小时：

- 指南 §5：*"A 5-minute job schedules far faster than a 10-hour one."* 申请时间越短，调度器越容易用空档先排进来。
- 作业短，GPU 释放得勤，同学的作业更容易插进来。
- 总 GPU 小时数不变，只是多重启几次——而续跑本来就要做对。

每个作业扣除约 15 分钟开销（启动、加载、编译、存档、安全余量），有效训练约 225 分钟。

---

## 六、共享资源礼仪

`comp3710` 是全班共用的 10 块 A100。本项目在 Rangpur 上总计约 **17 GPU 小时**，大致相当于跑几次课程作业的 GAN。仍然注意：

- **避开 demo 和作业截止前那几天**
- 调试用 `a100-test` 或本地 GPU，不在 `comp3710` 上试错
- 交互会话（`srun --pty`）用完立刻 `exit`
- 数据处理放 `cpu` 分区，不占 A100

P6 正式训练（约 47 GPU 小时的量）**刻意不放在 Rangpur**：没有助教的明确许可，这个量在共享的 10 块卡上太显眼。

---

## 七、收尾：给后续作业腾空间

项目结束后，按这个顺序处理：

| 步骤 | 内容 |
|---|---|
| 1. 带走成果 | 最终权重传 HuggingFace Hub；`reports/`、`logs/`、指标日志 `scp` 回本地（或 `git push`，它们本来就在仓库里） |
| 2. 确认带走了 | 本地能打开报告、能加载权重跑一次生成 |
| 3. 删除 | `rm -rf ~/myTransformer ~/mt-venv` |
| 4. 核对 | `df -h ~`，应回到开工前的水平 |

课程的 `miniconda3/envs/torch` 从头到尾没被改动，后续作业直接用。

**中途也要清理**：P5 做完、进入 P7 之前，删掉 P5 的数据（`~/myTransformer/data/tokens/`，4 GB）和缩放律模型的 checkpoint，只留汇总结果。P7 需要为 250M 的完整 checkpoint 腾出约 5.7 GB。

---

## 八、待确认

| 事项 | 怎么确认 |
|---|---|
| A100 节点是否也有 197 GB 的 `$TMPDIR` | 冒烟作业里的 `df -h "$TMPDIR"`（模板 4.1 已包含） |
| `$TMPDIR` 在作业结束后是否被清空 | 按"会清空"来设计，不依赖它 |
| `a100-test` 是否也需要 `--account=comp3710` | 模板里先加上；排不上再去掉试 |
| `cpu` 分区的时限与 CPU 核数上限 | `sinfo -p cpu -o "%P %l %c"` |
