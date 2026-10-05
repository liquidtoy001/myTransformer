"""多卡数据并行（DDP）的最小封装。

用 torchrun 启动时，每张卡一个进程，环境变量告诉进程"我是谁"：

    WORLD_SIZE  一共几个进程          RANK  我是第几个（全局）
    LOCAL_RANK  我是本机第几个（= 用哪张卡）

DDP 只做一件事：每个进程算自己那份数据的梯度，反向传播时把各进程的梯度**取平均**，
于是每个进程拿到完全相同的梯度、做完全相同的更新，权重始终一致。

单进程（直接 python 启动，没有 WORLD_SIZE）时这里的函数全部退化为空操作，
训练器的单卡路径与以前逐位相同。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistInfo:
    rank: int = 0
    world: int = 1
    local_rank: int = 0

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def enabled(self) -> bool:
        return self.world > 1


def init(device_type: str) -> DistInfo:
    """读 torchrun 的环境变量并建立进程组。重复调用安全。

    GPU 用 NCCL（走 NVLink/PCIe，快）；CPU 用 gloo（本机测试多卡逻辑用，Windows 也能跑）。
    超时设 30 分钟：存档、最终评测时其他进程会在 barrier 上等主进程，几百 MB 的写盘不能被当成卡死。
    """
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world == 1:
        return DistInfo()
    info = DistInfo(int(os.environ["RANK"]), world, int(os.environ.get("LOCAL_RANK", "0")))
    if not dist.is_initialized():
        timeout = timedelta(minutes=30)
        if device_type == "cuda":
            torch.cuda.set_device(info.local_rank)
            dist.init_process_group("nccl", timeout=timeout)
        else:
            dist.init_process_group("gloo", timeout=timeout, pg_options=_gloo_options())
    return info


def _gloo_options():
    """单机测试时把 gloo 绑在 127.0.0.1 上。

    gloo 默认按主机名解析自己的地址。Windows 上主机名常常先解析出 IPv6 链路本地地址（fe80::…），
    连不上，进程组初始化就一直卡住、也不报错。MASTER_ADDR 是回环地址时说明所有进程都在本机，直接用回环。
    """
    if os.environ.get("MASTER_ADDR") not in ("127.0.0.1", "localhost"):
        return None
    opts = dist.ProcessGroupGloo._Options()
    opts._devices = [dist.ProcessGroupGloo.create_device(hostname="127.0.0.1")]
    return opts


def shutdown() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
