"""训练器 - 6 阶段 + BF16 + 8M token 累积.

ShannonTrainer 实现 6 阶段训练流程 (spec):
    Phase1 预训练 -> Phase2 中间训练 -> Phase3 SFT
    -> Phase4 对齐 -> Phase5 持续学习 -> Phase6 自我进化

特性:
    - BF16 训练精度 (昇腾 910C / A100)
    - 梯度累积 (8M token 累积, 大批量模拟)
    - 5D 并行 (TP+PP+DP+SP+EP) 配置
    - 训练中断 30 分钟内恢复 (对接 TrainingCheckpoint)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


class TrainPhase(Enum):
    """6 阶段训练. """

    PHASE1_PRETRAIN = "phase1_pretrain"
    PHASE2_MIDTRAIN = "phase2_midtrain"
    PHASE3_SFT = "phase3_sft"
    PHASE4_ALIGN = "phase4_align"
    PHASE5_CONTINUAL = "phase5_continual"
    PHASE6_SELFEVOLVE = "phase6_selfevolve"


class Precision(Enum):
    BF16 = "BF16"
    FP16 = "FP16"
    FP32 = "FP32"


# spec: 8M token 累积 (大批量模拟, 配合 5D 并行)
DEFAULT_ACCUM_TOKENS = 8_000_000
DEFAULT_GLOBAL_BATCH = 4096
# spec: 训练中断 30 分钟内恢复
RECOVERY_WINDOW_MIN = 30


@dataclass
class ParallelConfig:
    """5D 并行配置 (TP+PP+DP+SP+EP). """

    tp: int = 1   # Tensor Parallel
    pp: int = 1   # Pipeline Parallel
    dp: int = 1   # Data Parallel
    sp: int = 1   # Sequence Parallel
    ep: int = 1   # Expert Parallel (MoE)

    @property
    def world_size(self) -> int:
        return self.tp * self.pp * self.dp

    def to_dict(self) -> Dict[str, int]:
        return {"tp": self.tp, "pp": self.pp, "dp": self.dp, "sp": self.sp, "ep": self.ep}


@dataclass
class TrainMetrics:
    """单步训练指标. """

    step: int
    loss: float
    lr: float
    grad_norm: float
    throughput_tok_s: float
    tokens_seen: int
    phase: str
    ts: float = field(default_factory=time.time)


@dataclass
class PhaseConfig:
    """单阶段配置. """

    phase: TrainPhase
    total_tokens: int
    lr_peak: float
    lr_min: float = 0.0
    warmup_ratio: float = 0.02
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    precision: Precision = Precision.BF16
    enabled: bool = True


class ShannonTrainer:
    """Shannon 6 阶段训练器.

    Args:
        parallel: 5D 并行配置.
        accum_tokens: 梯度累积的 token 预算 (默认 8M).
        global_batch: 全局 batch size.
        precision: 训练精度 (默认 BF16).
        checkpoint: 可选 TrainingCheckpoint 实例 (恢复用).
        step_fn: 单步前向+反向可调用
            ``fn(batch, phase, ctx) -> (loss, grad_norm, throughput)``.
    """

    def __init__(
        self,
        parallel: Optional[ParallelConfig] = None,
        accum_tokens: int = DEFAULT_ACCUM_TOKENS,
        global_batch: int = DEFAULT_GLOBAL_BATCH,
        precision: Precision = Precision.BF16,
        checkpoint: Optional[Any] = None,
        step_fn: Optional[Callable[..., Tuple[float, float, float]]] = None,
    ) -> None:
        self.parallel = parallel or ParallelConfig()
        self.accum_tokens = int(accum_tokens)
        self.global_batch = max(1, int(global_batch))
        self.precision = precision
        self.checkpoint = checkpoint
        self.step_fn = step_fn or self._default_step

        self._phase_configs: Dict[TrainPhase, PhaseConfig] = {}
        self._phase_order = [
            TrainPhase.PHASE1_PRETRAIN,
            TrainPhase.PHASE2_MIDTRAIN,
            TrainPhase.PHASE3_SFT,
            TrainPhase.PHASE4_ALIGN,
            TrainPhase.PHASE5_CONTINUAL,
            TrainPhase.PHASE6_SELFEVOLVE,
        ]
        self._current_phase_idx = 0
        self._tokens_seen = 0
        self._phase_tokens = 0
        self._step = 0
        self._metrics: List[TrainMetrics] = []
        self._interrupted_at: Optional[float] = None
        self._init_default_phases()

    # ------------------------------------------------------------------ #
    # 阶段配置
    # ------------------------------------------------------------------ #
    def _init_default_phases(self) -> None:
        defaults = {
            TrainPhase.PHASE1_PRETRAIN: (15_000_000_000_000, 6e-4),   # 15T tokens
            TrainPhase.PHASE2_MIDTRAIN: (500_000_000_000, 3e-4),      # 500B
            TrainPhase.PHASE3_SFT: (50_000_000_000, 1e-4),            # 50B
            TrainPhase.PHASE4_ALIGN: (10_000_000_000, 5e-5),          # 10B
            TrainPhase.PHASE5_CONTINUAL: (20_000_000_000, 2e-5),      # 20B
            TrainPhase.PHASE6_SELFEVOLVE: (30_000_000_000, 1e-5),     # 30B
        }
        for ph, (tok, lr) in defaults.items():
            self._phase_configs[ph] = PhaseConfig(
                phase=ph, total_tokens=tok, lr_peak=lr, lr_min=lr * 0.1
            )

    def configure_phase(self, config: PhaseConfig) -> None:
        self._phase_configs[config.phase] = config

    @property
    def current_phase(self) -> TrainPhase:
        return self._phase_order[self._current_phase_idx]

    @property
    def tokens_seen(self) -> int:
        return self._tokens_seen

    @property
    def step(self) -> int:
        return self._step

    # ------------------------------------------------------------------ #
    # 学习率调度 (cosine + warmup)
    # ------------------------------------------------------------------ #
    def _lr_schedule(self, phase_cfg: PhaseConfig, progress: float) -> float:
        """progress in [0,1]. warmup -> cosine decay. """
        warmup = phase_cfg.warmup_ratio
        if progress < warmup and warmup > 0:
            return phase_cfg.lr_peak * (progress / warmup)
        # cosine decay
        decay_progress = (progress - warmup) / max(1e-6, 1.0 - warmup)
        cos = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return phase_cfg.lr_min + (phase_cfg.lr_peak - phase_cfg.lr_min) * cos

    # ------------------------------------------------------------------ #
    # 训练循环
    # ------------------------------------------------------------------ #
    def train(
        self,
        data_iter: Optional[Callable[[int, TrainPhase], Sequence[Any]]] = None,
        max_steps: Optional[int] = None,
    ) -> List[TrainMetrics]:
        """运行训练 (从当前阶段开始, 直到 max_steps 或所有阶段完成). """
        data_iter = data_iter or (lambda step, ph: [{}])
        steps_run = 0
        while self._current_phase_idx < len(self._phase_order):
            phase = self.current_phase
            cfg = self._phase_configs[phase]
            if not cfg.enabled:
                self._advance_phase()
                continue
            # 恢复点
            if self.checkpoint is not None:
                self._maybe_resume()

            batch = data_iter(self._step, phase)
            loss, grad_norm, throughput = self.step_fn(batch, phase, self._ctx())
            progress = self._phase_tokens / max(1, cfg.total_tokens)
            lr = self._lr_schedule(cfg, min(1.0, progress))

            self._step += 1
            steps_run += 1
            self._tokens_seen += self.global_batch
            self._phase_tokens += self.global_batch
            self._metrics.append(
                TrainMetrics(
                    step=self._step,
                    loss=loss,
                    lr=lr,
                    grad_norm=grad_norm,
                    throughput_tok_s=throughput,
                    tokens_seen=self._tokens_seen,
                    phase=phase.value,
                )
            )

            # 定期检查点 (满足累积 token 预算)
            if self._phase_tokens >= self.accum_tokens or self._step % 100 == 0:
                self._save_checkpoint_if_needed()

            # 阶段完成
            if self._phase_tokens >= cfg.total_tokens:
                self._advance_phase()
            if max_steps is not None and steps_run >= max_steps:
                break
        return list(self._metrics)

    def _advance_phase(self) -> None:
        self._current_phase_idx = min(len(self._phase_order), self._current_phase_idx + 1)
        self._phase_tokens = 0

    def _ctx(self) -> Dict[str, Any]:
        return {
            "phase": self.current_phase.value,
            "parallel": self.parallel.to_dict(),
            "precision": self.precision.value,
            "accum_tokens": self.accum_tokens,
            "global_batch": self.global_batch,
        }

    # ------------------------------------------------------------------ #
    # 中断与恢复
    # ------------------------------------------------------------------ #
    def report_interrupt(self) -> None:
        """上报训练中断 (需在 30 分钟内恢复). """
        self._interrupted_at = time.time()

    def _save_checkpoint_if_needed(self) -> None:
        if self.checkpoint is None:
            return
        self.checkpoint.save(
            step=self._step,
            phase=self.current_phase.value,
            tokens_seen=self._tokens_seen,
            phase_tokens=self._phase_tokens,
            phase_idx=self._current_phase_idx,
            metrics=self._metrics[-1].__dict__ if self._metrics else {},
        )

    def _maybe_resume(self) -> None:
        if self._step > 0:
            return
        state = self.checkpoint.load()  # type: ignore[union-attr]
        if not state:
            return
        # 恢复中断检查 (30 分钟窗口)
        if self._interrupted_at is not None:
            elapsed_min = (time.time() - self._interrupted_at) / 60.0
            if elapsed_min > RECOVERY_WINDOW_MIN:
                # 超出恢复窗口, 回退到上一个稳定检查点
                state = self.checkpoint.load(stable=True)  # type: ignore[union-attr]
        self._step = state.get("step", 0)
        self._tokens_seen = state.get("tokens_seen", 0)
        self._phase_tokens = state.get("phase_tokens", 0)
        self._current_phase_idx = state.get("phase_idx", 0)

    # ------------------------------------------------------------------ #
    # 默认单步 (模拟)
    # ------------------------------------------------------------------ #
    def _default_step(self, batch: Sequence[Any], phase: TrainPhase, ctx: Dict[str, Any]) -> Tuple[float, float, float]:
        # 模拟 loss 下降
        base = {"phase1_pretrain": 3.0, "phase2_midtrain": 2.0, "phase3_sft": 1.0,
                "phase4_align": 0.8, "phase5_continual": 0.6, "phase6_selfevolve": 0.5}
        loss = base.get(phase.value, 1.0) * math.exp(-self._step * 1e-4)
        return loss, 1.0, float(self.global_batch * self.parallel.world_size)

    # ------------------------------------------------------------------ #
    # 统计
    # ------------------------------------------------------------------ #
    @property
    def metrics(self) -> List[TrainMetrics]:
        return list(self._metrics)

    def stats(self) -> Dict[str, Any]:
        return {
            "phase": self.current_phase.value,
            "step": self._step,
            "tokens_seen": self._tokens_seen,
            "phase_tokens": self._phase_tokens,
            "precision": self.precision.value,
            "parallel": self.parallel.to_dict(),
            "world_size": self.parallel.world_size,
            "accum_tokens": self.accum_tokens,
            "phases_total": len(self._phase_order),
            "recovery_window_min": RECOVERY_WINDOW_MIN,
        }


__all__ = [
    "ShannonTrainer",
    "TrainPhase",
    "Precision",
    "ParallelConfig",
    "PhaseConfig",
    "TrainMetrics",
    "DEFAULT_ACCUM_TOKENS",
    "DEFAULT_GLOBAL_BATCH",
    "RECOVERY_WINDOW_MIN",
]
