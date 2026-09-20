"""训练循环。

职责：
- 组装：模型、优化器、学习率调度、数据加载器
- 一步：梯度累积 → 梯度裁剪 → 优化器更新（梯度出现 NaN/Inf 时跳过这一步）
- 续跑：启动先找最新 checkpoint；恢复后自检，确认模型和数据流都对得上
- 退出：训完、收到 SLURM 信号、时间预算快用完——三种情况都先存档再退出
- 记录：metrics.jsonl，每行一条 JSON，是训练过程的唯一真源
"""

from __future__ import annotations

import contextlib
import glob
import json
import math
import signal
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import torch

from ..data.pack import check_meta
from ..data.shards import ShardLoader, eval_batches
from ..model import ModelConfig, Transformer
from ..tokenizer import Tokenizer
from . import checkpoint as ck
from .config import TrainConfig
from .muon import Muon, split_params
from .schedule import LRSchedule

# bf16 稠密峰值（TFLOPS），用来算 MFU。A100/H100/4090 为厂商标称；4070 Ti SUPER 为本机实测。
PEAK_TFLOPS = {"H100": 989.0, "A100": 312.0, "4090": 165.0, "4070 Ti SUPER": 76.1}

MAX_NONFINITE_STREAK = 10  # 连续这么多步梯度非有限就停下，不再硬撑
PROBE_TOKENS = 512         # 续跑自检用的序列长度；限制存档时 logits 的临时显存（2×512×V×4 字节）
PROBE_RTOL = 1e-5          # 同型号硬件上重算应逐位相同，1e-5 是给极少数非确定性 kernel 的余量


class ResumeError(RuntimeError):
    """续跑自检失败：checkpoint 与当前的代码、权重或数据对不上。"""


def _glob(pattern: str) -> list[str]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"没有匹配 {pattern!r} 的数据分片")
    return paths


def _git_commit() -> str:
    try:
        root = Path(__file__).resolve().parents[3]
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 —— 没有 git 不影响训练
        return "unknown"


class Trainer:
    def __init__(
        self,
        cfg: TrainConfig,
        *,
        device: str | None = None,
        max_minutes: float | None = None,
        on_step: Callable[[Trainer], None] | None = None,
        log: Callable[[str], None] = print,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.run_dir = Path(cfg.run_dir)
        self.max_seconds = max_minutes * 60 if max_minutes else None
        self.on_step = on_step
        self.log = log
        self.stop_reason: str | None = None

        torch.manual_seed(cfg.seed)
        self.model_cfg = ModelConfig.from_yaml(cfg.model)
        if cfg.seq_len > self.model_cfg.max_seq_len:
            raise ValueError(f"seq_len={cfg.seq_len} 超过模型的 max_seq_len={self.model_cfg.max_seq_len}")
        self.model = Transformer(self.model_cfg).to(self.device)
        self.opts = self._build_optimizers()
        self.sched = LRSchedule(
            cfg.lr, cfg.warmup_steps, cfg.max_steps,
            kind=cfg.schedule, decay_start=cfg.decay_start, min_lr_ratio=cfg.min_lr_ratio,
        )
        self.accum = cfg.grad_accum_steps()
        train_paths = _glob(cfg.train_data)
        val_paths = _glob(cfg.val_data) if cfg.val_data else []
        self._check_data(train_paths + val_paths)
        self.loader = ShardLoader(train_paths, cfg.micro_batch_size, cfg.seq_len, seed=cfg.seed)
        # 每个验证文件单独算 loss（P2 为每个子集各写一个 val_<子集>.bin）：中文和英文的 loss
        # 走势常常不一样，混成一个数就看不出来。总的 eval 批数在各文件之间平分，开销不变
        per_file = math.ceil(cfg.eval_batches / max(1, len(val_paths)))
        self.val = {
            Path(p).stem.removeprefix("val_"): eval_batches([p], cfg.micro_batch_size, cfg.seq_len, per_file)
            for p in val_paths
        }
        self.fwd = torch.compile(self.model) if cfg.compile else self.model
        self.amp_dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else None
        self.flops_per_token = self.model_cfg.flops_per_token(cfg.seq_len)
        self.peak_flops = self._peak_flops()

        self.step = 0
        self.tokens_seen = 0
        self.nonfinite_streak = 0
        self.probe_ids: torch.Tensor | None = None

    # ------------------------------------------------------------------ 主循环

    def fit(self) -> str:
        """训练到 max_steps 或被要求停下。返回退出原因：done / time / signal:* / nonfinite / 自定义。"""
        old_handlers = self._install_signal_handlers()
        try:
            return self._fit()
        finally:
            for sig, h in old_handlers.items():
                signal.signal(sig, h)

    def _fit(self) -> str:
        cfg = self.cfg
        self._resume()
        if self.step >= cfg.max_steps:
            self.log(f"已训练到 {self.step}/{cfg.max_steps} 步，无需继续")
            return "done"
        self.log(self._banner())

        session_start = self.step
        t_start = time.perf_counter()
        window_t, window_steps = t_start, 0
        last_saved = None

        while self.step < cfg.max_steps:
            loss, grad_norm, lr = self._train_step()
            if self.step == session_start:  # 本次会话的第一步刚做完（torch.compile 在这一步里编译）
                self._reinstall_signal_handlers()
            self.step += 1
            self.tokens_seen += cfg.global_batch_tokens
            window_steps += 1

            if self.step % cfg.log_every == 0 or self.step == cfg.max_steps:
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                now = time.perf_counter()
                self._log_train(loss, grad_norm, lr, (now - window_t) / window_steps)
                window_t, window_steps = now, 0
            if self.val and self.step % cfg.eval_every == 0:
                self._log_eval()
            if self.step % cfg.ckpt_every == 0:
                self._save("periodic")
                last_saved = self.step
            if self.on_step:
                self.on_step(self)

            # 时间预算：留出两步的余量，别在最后一步中途被杀
            if self.stop_reason is None and self.max_seconds:
                elapsed = time.perf_counter() - t_start
                if elapsed + 2 * elapsed / (self.step - session_start) > self.max_seconds:
                    self.stop_reason = "time"
            if self.stop_reason:
                if last_saved != self.step:
                    self._save(self.stop_reason)
                self.log(f"在 step {self.step} 停下（{self.stop_reason}）")
                return self.stop_reason

        if last_saved != self.step:
            self._save("done")
        self.log(f"训练完成：{self.step} 步，{self.tokens_seen / 1e9:.3f}B tokens")
        return "done"

    def _build_optimizers(self) -> list[torch.optim.Optimizer]:
        """adamw：全部参数一个 AdamW。muon：二维权重给 Muon，嵌入和一维参数仍给 AdamW（见 muon.py）。"""
        cfg = self.cfg
        fused = self.device.type == "cuda"
        if cfg.optimizer == "adamw":
            return [torch.optim.AdamW(self.model.param_groups(cfg.weight_decay),
                                      lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, fused=fused)]
        if cfg.optimizer == "muon":
            muon_params, adamw_params = split_params(self.model)
            return [
                Muon(muon_params, lr=cfg.muon_lr, momentum=cfg.muon_momentum,
                     weight_decay=cfg.weight_decay),
                torch.optim.AdamW(adamw_params, lr=cfg.lr, betas=cfg.betas, eps=cfg.eps,
                                  weight_decay=0.0, fused=fused),
            ]
        raise ValueError(f"optimizer 只能是 adamw 或 muon，得到 {cfg.optimizer!r}")

    def _set_lr(self, lr: float) -> None:
        """调度给出的是 AdamW 的学习率。Muon 的量级不同（cfg.muon_lr），按同一比例缩放，
        这样 warmup 和衰减的形状对两者一致。"""
        ratio = lr / self.cfg.lr if self.cfg.lr else 0.0
        for opt in self.opts:
            base = self.cfg.muon_lr if isinstance(opt, Muon) else self.cfg.lr
            for group in opt.param_groups:
                group["lr"] = base * ratio

    def _train_step(self) -> tuple[torch.Tensor, torch.Tensor, float]:
        lr = self.sched(self.step)
        self._set_lr(lr)

        total = torch.zeros((), device=self.device)
        for _ in range(self.accum):
            x, y = self.loader.next_batch()
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            with self._autocast():
                loss = self.fwd(x, y, shift=False, ce_chunk=self.cfg.ce_chunk, z_loss=self.cfg.z_loss)["loss"]
            # 除以累积步数：累积 k 个 micro batch 的梯度，等于一个 k 倍大的 batch 的平均梯度
            (loss / self.accum).backward()
            total += loss.detach()

        # 返回的是裁剪**前**的总范数——它是训练要炸的最灵敏的先行指标
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        if torch.isfinite(grad_norm):
            for opt in self.opts:
                opt.step()
            self.nonfinite_streak = 0
        else:
            # 更新一次就可能把权重全毁成 NaN，所以跳过；数据照常消耗，调度照常前进
            self.nonfinite_streak += 1
            self.log(f"step {self.step + 1}：梯度范数为 {grad_norm.item()}，跳过这一步的更新（连续第 {self.nonfinite_streak} 次）")
            if self.nonfinite_streak >= MAX_NONFINITE_STREAK:
                self.stop_reason = "nonfinite"
        for opt in self.opts:
            opt.zero_grad(set_to_none=True)
        return total / self.accum, grad_norm, lr

    # ------------------------------------------------------------------ 存档与续跑

    def _save(self, reason: str) -> None:
        if self.probe_ids is None:
            # 自检用的固定小 batch，第一次存档时从数据里取，之后一直带在 checkpoint 里
            x, y = self.loader.peek()
            n = min(PROBE_TOKENS, x.size(1))
            self.probe_ids = torch.cat([x[:2, :n], y[:2, n - 1 : n]], dim=1).to(self.device)
        payload = {
            "model": self.model.state_dict(),
            "optimizer": [o.state_dict() for o in self.opts],
            "step": self.step,
            "tokens_seen": self.tokens_seen,
            "loader": self.loader.state_dict(),
            "rng": {
                "torch": torch.get_rng_state(),
                **({"cuda": torch.cuda.get_rng_state_all()} if self.device.type == "cuda" else {}),
            },
            "probe": {
                "ids": self.probe_ids.cpu(),
                **self._probe(),
                "next_tokens": self.loader.peek()[0][0, :16].tolist(),
            },
            "meta": {
                "reason": reason,
                "git": _git_commit(),
                "config": self.cfg.to_dict(),
                "model_config": asdict(self.model_cfg),
                "time": time.time(),
            },
        }
        path = ck.save(self.run_dir, self.step, payload)
        ck.prune(self.run_dir, self.cfg.ckpt_keep, self.cfg.ckpt_keep_every)
        self.log(f"已存档 {path.name}（{reason}）")

    def _resume(self) -> None:
        path, source = ck.latest(self.run_dir), "续跑"
        if path is None and self.cfg.init_from:
            path, source = Path(self.cfg.init_from), "从分叉起点开始"
        if path is None:
            return

        state = ck.load(path, map_location=self.device)
        self.model.load_state_dict(state["model"])
        saved = state["optimizer"]
        saved = saved if isinstance(saved, list) else [saved]  # 兼容只有一个优化器时的旧存档
        if len(saved) != len(self.opts):
            raise ResumeError(f"存档里有 {len(saved)} 个优化器状态，当前配置有 {len(self.opts)} 个（optimizer 改了？）")
        for opt, st in zip(self.opts, saved):
            opt.load_state_dict(st)
        self.step = state["step"]
        self.tokens_seen = state["tokens_seen"]
        self.loader.load_state_dict(state["loader"])
        torch.set_rng_state(state["rng"]["torch"].cpu())
        cuda_rng = state["rng"].get("cuda")
        if self.device.type == "cuda" and cuda_rng and len(cuda_rng) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([t.cpu() for t in cuda_rng])
        self.probe_ids = state["probe"]["ids"].to(self.device)

        self._verify_resume(state, path)
        self.log(f"{source}：{path.name}（step {self.step}，已见 {self.tokens_seen / 1e9:.3f}B tokens）")

    def _verify_resume(self, state: dict, path: Path) -> None:
        """续跑自检。两件事：模型算出来的对不对，数据流接得上接不上。

        模型输出同时比 loss 和 logits 的均方根。只比 loss 不够：训练初期模型接近均匀分布，
        权重整体偏 1% 时 loss 只变 2e-5，而 logits 的均方根变 1e-2——灵敏 500 倍。
        """
        got = self._probe()
        for key in ("loss", "logit_rms"):
            expected = state["probe"][key]
            if not math.isclose(got[key], expected, rel_tol=PROBE_RTOL, abs_tol=1e-7):
                raise ResumeError(
                    f"续跑自检失败（模型输出 {key}）：{path.name} 存档时算出 {expected:.8f}，"
                    f"现在算出 {got[key]:.8f}。模型代码或权重与存档时不一致。"
                )
        nxt = self.loader.peek()[0][0, :16].tolist()
        if nxt != state["probe"]["next_tokens"]:
            raise ResumeError(
                f"续跑自检失败（数据流）：{path.name} 存档时记录的下一个 batch 以 {state['probe']['next_tokens'][:4]}… 开头，"
                f"现在是 {nxt[:4]}…。数据分片被改动过，或加载器状态不对。"
            )

    @torch.no_grad()
    def _probe(self) -> dict[str, float]:
        """用 fp32、关掉 TF32，在固定小 batch 上算 loss 和 logits 均方根。同型号硬件上应逐位可复现。"""
        was_training = self.model.training
        prev_precision = torch.get_float32_matmul_precision()
        self.model.eval()
        torch.set_float32_matmul_precision("highest")
        try:
            ids = self.probe_ids
            logits = self.model(ids[:, :-1])["logits"].float()
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), ids[:, 1:].flatten())
            return {"loss": loss.item(), "logit_rms": logits.pow(2).mean().sqrt().item()}
        finally:
            torch.set_float32_matmul_precision(prev_precision)
            self.model.train(was_training)

    # ------------------------------------------------------------------ 记录

    def _log_train(self, loss: torch.Tensor, grad_norm: torch.Tensor, lr: float, step_s: float) -> None:
        tok_s = self.cfg.global_batch_tokens / step_s
        rec = {
            "type": "train", "step": self.step, "tokens": self.tokens_seen, "epoch": self.loader.epoch,
            "loss": loss.item(), "lr": lr, "grad_norm": grad_norm.item(), "step_s": step_s, "tok_s": tok_s,
        }
        if self.peak_flops:
            rec["mfu"] = self.flops_per_token * tok_s / self.peak_flops
        if self.device.type == "cuda":
            rec["mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
        self._append(rec)
        mfu = f"  MFU {rec['mfu']:5.1%}" if "mfu" in rec else ""
        self.log(
            f"step {self.step:>6}  loss {rec['loss']:.4f}  lr {lr:.2e}  gnorm {rec['grad_norm']:.3f}"
            f"  {step_s * 1000:6.0f} ms/step  {tok_s:,.0f} tok/s{mfu}"
        )

    @torch.no_grad()
    def _log_eval(self) -> None:
        self.model.eval()
        per_set = {}
        for name, batches in self.val.items():
            losses = []
            for x, y in batches:
                with self._autocast():
                    out = self.model(x.to(self.device), y.to(self.device), shift=False, ce_chunk=self.cfg.ce_chunk)
                losses.append(out["loss"].item())
            per_set[name] = sum(losses) / len(losses)
        self.model.train()
        val = sum(per_set.values()) / len(per_set)  # 各子集等权平均：不让 token 多的子集主导
        rec = {"type": "eval", "step": self.step, "tokens": self.tokens_seen, "val_loss": val}
        if len(per_set) > 1:
            rec["val"] = per_set
        self._append(rec)
        detail = "  " + "  ".join(f"{k} {v:.3f}" for k, v in per_set.items()) if len(per_set) > 1 else ""
        self.log(f"step {self.step:>6}  val_loss {val:.4f}{detail}")

    def _check_data(self, paths: list[str]) -> None:
        """训练前核对数据：分片的 meta.json 与配置的分词器一致、分片没有拷贝不完整，词表放得进模型。"""
        fingerprint = None
        if self.cfg.tokenizer:
            tok = Tokenizer.from_file(self.cfg.tokenizer)
            if tok.vocab_size > self.model_cfg.vocab_size:
                raise ValueError(f"分词器词表 {tok.vocab_size} 大于模型的 vocab_size {self.model_cfg.vocab_size}")
            fingerprint = tok.fingerprint
        check_meta(paths, fingerprint)

    def _append(self, rec: dict) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with open(self.run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _banner(self) -> str:
        c, n = self.cfg, self.model_cfg.count_params()
        return (
            f"模型 {n['total'] / 1e6:.1f}M 参数（非嵌入 {n['non_embedding'] / 1e6:.1f}M）  设备 {self.device}  "
            f"dtype {c.dtype}  compile {c.compile}\n"
            f"每步 {c.global_batch_tokens:,} tokens = micro {c.micro_batch_size} × seq {c.seq_len} × 累积 {self.accum}  "
            f"从 step {self.step} 训到 {c.max_steps}（共 {c.max_steps * c.global_batch_tokens / 1e9:.3f}B tokens）"
        )

    # ------------------------------------------------------------------ 杂项

    def request_stop(self, reason: str) -> None:
        """要求在当前这一步结束后存档退出。信号处理和测试钩子都走这里。"""
        self.stop_reason = reason

    def _on_signal(self, signum: int, _frame) -> None:
        self.request_stop(f"signal:{signal.Signals(signum).name}")

    @staticmethod
    def _stop_signals() -> list[int]:
        """Windows 没有 SIGUSR1，自动跳过。"""
        return [s for s in (getattr(signal, "SIGUSR1", None), getattr(signal, "SIGTERM", None)) if s is not None]

    def _install_signal_handlers(self) -> dict:
        """SLURM 在作业被杀前发 USR1（我们在 sbatch 里用 --signal 要求的）和 TERM。

        处理函数只设一个标志，真正的存档在当前这一步做完之后：
        信号可能在任何时刻到来，在反向传播中途存档是不安全的。

        还要解除屏蔽：被屏蔽（blocked）的信号到了只会挂起，既不触发处理函数、
        也不按默认行为杀掉进程。Rangpur 冒烟 600172 就是"信号发了、进程没停也没死"。
        """
        old = {}
        for sig in self._stop_signals():
            try:
                old[sig] = signal.signal(sig, self._on_signal)
            except ValueError:  # 不在主线程
                pass
        if old and hasattr(signal, "pthread_sigmask"):
            was_blocked = signal.pthread_sigmask(signal.SIG_UNBLOCK, list(old)) & set(old)
            if was_blocked:
                names = ", ".join(sorted(signal.Signals(s).name for s in was_blocked))
                self.log(f"注意：{names} 原本处于屏蔽状态（blocked），已解除——否则收到也不会生效")
        return old

    def _reinstall_signal_handlers(self) -> None:
        """本次会话第一步做完后再装一次处理函数。

        torch.compile 在第一次前向时才真正编译，期间会加载 triton 等底层库。
        如果某个库在 C 层替换了处理函数，Python 的 signal.getsignal 是看不出来的；
        重新调用 signal.signal 会重新登记，不管之前被谁换过都能恢复。

        Rangpur 上实测确实如此：冒烟 600172 没有这一步，SIGUSR1 发出后训练既没停也没死；
        冒烟 600202 加上这一步后在下一步就存档退出了。/proc 显示信号从未被屏蔽，
        Python 层也没看到处理函数被换——替换发生在 C 层，具体是哪个库尚未查明。
        """
        for sig in self._stop_signals():
            try:
                prev = signal.signal(sig, self._on_signal)
            except ValueError:
                continue
            if prev != self._on_signal:
                self.log(f"注意：{signal.Signals(sig).name} 的处理函数在第一步期间被替换成了 {prev!r}，已恢复")

    def _autocast(self):
        if self.amp_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(self.device.type, dtype=self.amp_dtype)

    def _peak_flops(self) -> float | None:
        if self.cfg.peak_tflops:
            return self.cfg.peak_tflops * 1e12
        if self.device.type == "cuda":
            name = torch.cuda.get_device_name(self.device)
            for key, tflops in PEAK_TFLOPS.items():
                if key in name:
                    return tflops * 1e12
        return None
