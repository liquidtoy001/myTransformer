# P4-7 多卡数据并行（DDP）

代码：`src/mytransformer/train/distributed.py`（新）、`src/mytransformer/train/trainer.py`（多处，下文逐条列出）、`src/mytransformer/data/shards.py` 的 `peek(rank=)`
测试：`tests/test_ddp.py`（本机 CPU，2 个进程）
租机验收：`scripts/cloud/ddp_check.sh` + `configs/train/p6_check/ddp_250m.yaml`

---

## 1. 它解决什么问题

P6 在 8×H100 上训练。8 张卡要像一张"8 倍快的卡"一样工作，训练结果要和单卡在数学上等价。

DDP（DistributedDataParallel）的做法：

```
每张卡一个进程，各有一份完整的模型
每一步：
  每个进程读自己那份数据 → 前向 → 反向，得到"自己的梯度"
  反向过程中，各进程把梯度 all-reduce（求和再除以卡数）→ 每个进程拿到同一份平均梯度
  每个进程用同一份梯度做同一次更新 → 权重始终一致
```

**global batch 不随卡数变。** 配置里写的是每步 524,288 tokens。1 卡时一张卡累积 16 个 micro batch；8 卡时每张卡累积 2 个。学习率、步数、缩放律的口径都不用改：

```
global_batch_tokens = micro_batch × seq_len × 卡数 × 累积次数
     524,288        =     16      ×  2048   ×  8   ×    2
```

## 2. 怎么读代码

### 2.1 `distributed.py`：我是谁

`torchrun` 为每张卡起一个进程，并设好环境变量 `WORLD_SIZE`（几个进程）、`RANK`（我是第几个）、`LOCAL_RANK`（用本机第几张卡）。`init()` 读这几个变量，建立进程组：GPU 用 NCCL，CPU 用 gloo。

没有 `WORLD_SIZE` 时返回 `world=1` 的空壳。**单卡路径与以前逐位相同**，原来的 40 多个训练器测试一个没改、全部通过。

### 2.2 训练器里改了什么（按执行顺序）

| 位置 | 改动 | 为什么 |
|---|---|---|
| `__init__` | `device = cuda:LOCAL_RANK`；非主进程的 `log` 换成空函数 | 每个进程管一张卡；只让 rank 0 打印，否则每行出现 8 遍 |
| `__init__` | `grad_accum_steps(world)` | 卡多了，每卡累积次数按比例减少 |
| `__init__` | `ShardLoader(..., rank, world_size)` | 加载器早在 P4-3 就支持多卡：rank r 读 `offset + r×B×T`，所有 rank 共享同一个 offset |
| `__init__` | `DistributedDataParallel(model, broadcast_buffers=False)`，再 `torch.compile` | 先包 DDP 再编译，编译器会按 DDP 的梯度桶切图，让通信和反向重叠 |
| `_train_step` | 前 k−1 个 micro batch 在 `ddp.no_sync()` 里反向 | 梯度在本地累积，只在最后一次反向时同步。否则通信量翻 k 倍 |
| `_fit` | 记录前 `_mean(loss)` | 每个 rank 只看到自己数据上的 loss，记下的应是全体平均 |
| `_fit` | 每步 `_any(stop_reason is not None)` | **全体一起决定停不停**，见 §3.1 |
| `_save` | 只有 rank 0 写文件，写完 `_barrier()` | 各 rank 状态完全相同，写一份就够 |
| `_save` / `_verify_resume` | `loader.peek(rank=0)` | 续跑自检要一个与卡数无关的视角，见 §3.2 |
| `_resume` | CUDA 随机数状态只存、只恢复本卡的 | `get_rng_state_all()` 会在 rank 0 上给 8 张卡都建 CUDA 上下文，白占显存 |
| `_log_eval` | 验证 batch 轮流分给各 rank，各自求和再 all-reduce | 结果与单卡相同，耗时降到 1/8 |
| `_finish` | "是否已有最终评测"由 rank 0 判断后广播；压缩存档前后各一个 barrier | metrics.jsonl 只有 rank 0 写；删 checkpoint 前要等所有 rank 读完 |
| `_log_train` | MFU 的分母乘以卡数 | 吞吐是全体的，峰值也要按全体算 |

### 2.3 为什么不用同步梯度范数、Muon 也不用改

DDP 同步之后，**每个 rank 的梯度逐位相同**。所以梯度范数、"这一步要不要因为 NaN 跳过"、Muon 的正交化结果，在各 rank 上自然一致，不需要再通信。

代价：Muon 的 Newton-Schulz 迭代每张卡都算一遍全部矩阵，是重复劳动（P5 实测 Muon 比 AdamW 慢约 4%）。可以把矩阵分给各卡算完再广播，这是以后的优化项，不影响正确性。

## 3. 两个容易出错的地方

### 3.1 停不停必须全体一致

多卡训练里，每一次 all-reduce 都要求**所有进程都到场**。如果 rank 1 收到了停止信号、去存档退出，而 rank 0 没收到、进入下一步的反向，rank 0 会在梯度同步上一直等 rank 1，直到 30 分钟超时。

信号不一定同时到达每个进程；计时预算各进程也差几毫秒。所以每步做完后加一次很小的 all-reduce（取最大值）：任何一个进程想停，所有进程一起停。自己没收到请求的进程把退出原因记为 `peer`。

这次 all-reduce 只有一个数，耗时几十微秒，而一步要几百毫秒到几秒。

### 3.2 续跑自检用 rank 0 的视角

P4-4 的续跑自检会记下"下一个 batch 的前 16 个 token"，恢复后比对。多卡时每个 rank 的下一个 batch 不同。如果存档时记 rank 0 的、恢复后 rank 1 拿自己的去比，就会误报失败。

`peek(rank=0)` 看的是共享 offset 处的数据，这个位置**与卡数无关**：8 卡存的 checkpoint 换成 4 卡或 1 卡续跑，自检照样通过，后续数据也完全接得上（`test_checkpoint_from_two_ranks_resumes_on_one` 验证了这一点）。租的竞价机器被回收后，换一台卡数不同的机器也能接着训。

## 4. 怎么验证

```powershell
.venv/Scripts/python.exe -m pytest -q tests/test_ddp.py
```

4 个测试，每个都真的起 2 个进程（gloo 后端），约 40 秒：

| 测试 | 检查什么 |
|---|---|
| `test_two_ranks_match_one_process` | 2 卡各累积 2 次 vs 1 卡累积 4 次：每步 loss、梯度范数、验证 loss、最终权重相对误差 < 1e-4 |
| `test_stop_on_one_rank_stops_all_then_resume_is_identical` | 只让 rank 1 请求停止 → 两个进程都停在第 5 步（rank 0 记为 peer）；续跑后与不中断**逐位相同** |
| `test_checkpoint_from_two_ranks_resumes_on_one` | 2 卡存档、1 卡续跑，结果与一直 1 卡一致 |
| `test_final_weights_only_under_ddp` | 训完压缩 final.pt、补最终评测、重跑是空操作，都不会卡住 |

第一个测试能做到 1e-4 的原因：数据切分保证同一步里 1 卡读的 4 个 micro batch，正好就是 2 卡读的 2×2 个，只是分给谁算不同。剩下的差别只有浮点求和顺序。

### 故意改错（都实际跑过）

| 改错 | 结果 |
|---|---|
| 永远在 `no_sync()` 里反向（梯度从不同步） | `test_two_ranks_match_one_process` 失败 |
| `_mean` 里不除以卡数 | 同上失败：记录的 loss 翻倍 |
| 验证集各 rank 算完不 all-reduce | 同上失败：val loss 只是 rank 自己那部分 |
| 续跑自检用 `peek()` 而不是 `peek(rank=0)` | 续跑测试失败：rank 1 报"数据流自检失败" |
| 删掉每步的 `_any(...)` 停止同步 | 续跑测试在 300 秒超时处失败：两个进程互相等死 |

最后一条正是 §3.1 说的死锁。测试用 `communicate(timeout=300)` 等进程，所以死锁会表现为失败，而不是让测试永远挂着。

### Windows 上的两个坑

- **torchrun 在 Windows 上起不来**：Windows 版 PyTorch 没编译 libuv，而 torchrun 的启动代理不理会 `USE_LIBUV=0`。测试里自己设 `RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT`、各起一个进程，这和 torchrun 做的事一样
- **gloo 初始化卡住、不报错**：gloo 按主机名解析自己的地址，本机主机名先解析出 IPv6 链路本地地址（`fe80::…`），连不上。`MASTER_ADDR` 是回环地址时，`_gloo_options()` 把 gloo 绑到 127.0.0.1

这两条只影响本机测试；租的 Linux 机器上用 torchrun + NCCL。

## 5. 本机验证不了、要在租机上验证的

| 项目 | 原因 | 怎么验 |
|---|---|---|
| NCCL 通信 | Windows 没有 NCCL | `ddp_check.sh` 第 1 步：1 卡 vs 2 卡，相对差 < 2% |
| `torch.compile` + DDP + `no_sync` | 本机 CPU 编译需要 C++ 编译器，测试里关了 compile | 同上，配置里 compile=true |
| 8 卡吞吐和 MFU | 只有真机才有 | 第 2 步：打印 tokens/s、MFU，并换算 P6 需要几小时 |
| 真实信号下的存档退出 | | 第 3 步：训到第 20 步给各 worker 发 SIGUSR1，存档退出后续跑 |

**一个要记住的限制**：竞价实例被回收时，平台通常给整个容器发 SIGTERM。torchrun 收到后会转发给各 worker，但只等约 30 秒就强杀。250M 的完整 checkpoint（权重 + 优化器状态约 3 GB）在本地 NVMe 上写几秒钟，来得及；如果存到网络盘就要先测写入速度。

## 6. 下次自己从零搭

1. 先保证单卡训练器的**数据加载器能按 rank 切分**、状态可存可恢复（这是最难补的，P4-3 一开始就做了）
2. 写一个 `init()`：读 `WORLD_SIZE/RANK/LOCAL_RANK`，建进程组，单进程时什么都不做
3. 模型包 `DistributedDataParallel`；梯度累积时前 k−1 次用 `no_sync()`
4. 找出所有"只该做一次"的事：打印、写 metrics、写 checkpoint → 只让 rank 0 做，写完 barrier
5. 找出所有"各 rank 必须一致"的决定：停不停、要不要补评测 → 用 all-reduce 统一
6. 写一个"N 卡 ≈ 1 卡"的等价测试。它一个就能抓住梯度没同步、数据切错、loss 没平均这三类最常见的错误

## 7. 自己动手

1. 把 `write_cfg` 里的 `global_batch_tokens` 改成 `4 * SEQ * 2`（单卡累积 2 次、2 卡各 1 次），跑 `-k match_one`，测试仍然通过（实际跑过）。想想为什么这时 `no_sync()` 一次都不会被用到
2. 把 `broadcast_buffers=False` 改成 `True`，测试会不会变？想一想 RoPE 的 cos/sin 表在各 rank 上是不是本来就一样，以及真机上每步会多出什么
3. 在 `_save` 里把 `self._barrier()` 删掉，跑 `test_final_weights_only_under_ddp`，看它过不过。不管结果如何，想一想：主进程还在写文件、其他进程已经退出，在 Slurm 或租的机器上会发生什么？"测试通过"能不能说明没有竞态？
