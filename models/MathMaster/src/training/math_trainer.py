"""MathTrainer - 5 阶段数学训练.

MathMaster 5 阶段训练流程:

    Phase1 预训练      (pretrain)         数学语料 LM 预训练
    Phase2 形式化      (formalize)        Lean4/SymPy 形式化对齐训练
    Phase3 合成强化    (synth_reinforce)  合成数据 + 执行反馈强化
    Phase4 RLHF对齐    (rlhf_align)       MathRLHF 分层混合奖励对齐
    Phase5 Self-Play   (self_play)        自我对弈 + 猜想生成自我进化

特性:
    - 课程学习集成 (MathCurriculum, 螺旋升级)
    - 阶段配置 (token 预算 / 学习率 / 精度)
    - 单步回调 (step_fn) 注入真实训练逻辑
    - 训练中断恢复 (对接外部 checkpoint)
    - 各阶段触发对应评估 (MathEvaluator)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .curriculum import CurriculumLevel, MathCurriculum
from .math_evaluator import MathBenchmark, MathEvaluator
from .math_rlhf import MathRLHF


class MathTrainPhase(Enum):
    """5 阶段训练. """

    PHASE1_PRETRAIN = "phase1_pretrain"           # 预训练
    PHASE2_FORMALIZE = "phase2_formalize"         # 形式化对齐
    PHASE3_SYNTH_REINFORCE = "phase3_synth_reinforce"  # 合成强化
    PHASE4_RLHF_ALIGN = "phase4_rlhf_align"       # RLHF 对齐
    PHASE5_SELF_PLAY = "phase5_self_play"         # Self-Play 自我进化


class Precision(Enum):
    BF16 = "BF16"
    FP16 = "FP16"
    FP32 = "FP32"


# 默认累积 token (梯度累积预算)
DEFAULT_ACCUM_TOKENS = 4_000_000
DEFAULT_GLOBAL_BATCH = 2048
# 训练中断恢复窗口
RECOVERY_WINDOW_MIN = 30


# 各阶段默认配置: (总 token, 峰值学习率, 课程起始级别)
PHASE_DEFAULTS: Dict[MathTrainPhase, Tuple[int, float, CurriculumLevel]] = {
    MathTrainPhase.PHASE1_PRETRAIN: (500_000_000_000, 6e-4, CurriculumLevel.BASIC),
    MathTrainPhase.PHASE2_FORMALIZE: (80_000_000_000, 2e-4, CurriculumLevel.INTERMEDIATE),
    MathTrainPhase.PHASE3_SYNTH_REINFORCE: (60_000_000_000, 1e-4, CurriculumLevel.INTERMEDIATE),
    MathTrainPhase.PHASE4_RLHF_ALIGN: (30_000_000_000, 5e-5, CurriculumLevel.ADVANCED),
    MathTrainPhase.PHASE5_SELF_PLAY: (40_000_000_000, 2e-5, CurriculumLevel.FRONTIER),
}

# 各阶段触发的评估基准
PHASE_EVAL_BENCHMARKS: Dict[MathTrainPhase, Tuple[MathBenchmark, ...]] = {
    MathTrainPhase.PHASE1_PRETRAIN: (
        MathBenchmark.GSM8K, MathBenchmark.MATH,
    ),
    MathTrainPhase.PHASE2_FORMALIZE: (
        MathBenchmark.MATH, MathBenchmark.LEAN_PROOF,
    ),
    MathTrainPhase.PHASE3_SYNTH_REINFORCE: (
        MathBenchmark.MATH, MathBenchmark.AIME, MathBenchmark.PUTNAM,
    ),
    MathTrainPhase.PHASE4_RLHF_ALIGN: (
        MathBenchmark.MATH, MathBenchmark.AIME, MathBenchmark.IMO,
    ),
    MathTrainPhase.PHASE5_SELF_PLAY: (
        MathBenchmark.IMO, MathBenchmark.CONJECTURE, MathBenchmark.FRONTIER,
        MathBenchmark.LEAN_PROOF,
    ),
}


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
class MathPhaseConfig:
    """单阶段配置. """

    phase: MathTrainPhase
    total_tokens: int
    lr_peak: float
    lr_min: float = 0.0
    warmup_ratio: float = 0.02
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    precision: Precision = Precision.BF16
    init_curriculum: CurriculumLevel = CurriculumLevel.BASIC
    enabled: bool = True


@dataclass
class MathTrainMetrics:
    """单步训练指标. """

    step: int
    phase: str
    loss: float
    lr: float
    grad_norm: float
    throughput_tok_s: float
    tokens_seen: int
    curriculum_level: str = "basic"
    mastery: float = 0.0
    reward: float = 0.0              # RLHF 阶段平均奖励
    ts: float = field(default_factory=time.time)


class MathTrainer:
    """MathMaster 5 阶段训练器.

    Args:
        parallel: 5D 并行配置.
        accum_tokens: 梯度累积 token 预算.
        global_batch: 全局 batch size.
        precision: 训练精度 (默认 BF16).
        checkpoint: 可选外部 checkpoint 对象 (需实现 save/load).
        step_fn: 单步前向+反向可调用
            ``fn(batch, phase, ctx) -> (loss, grad_norm, throughput)``.
        curriculum: 课程学习调度器 (默认新建).
        rlhf: RLHF 奖励器 (默认新建, Phase4 使用).
        evaluator: 评估器 (默认新建, 各阶段触发).
    """

    def __init__(
        self,
        parallel: Optional[ParallelConfig] = None,
        accum_tokens: int = DEFAULT_ACCUM_TOKENS,
        global_batch: int = DEFAULT_GLOBAL_BATCH,
        precision: Precision = Precision.BF16,
        checkpoint: Optional[Any] = None,
        step_fn: Optional[Callable[..., Tuple[float, float, float]]] = None,
        curriculum: Optional[MathCurriculum] = None,
        rlhf: Optional[MathRLHF] = None,
        evaluator: Optional[MathEvaluator] = None,
    ) -> None:
        self.parallel = parallel or ParallelConfig()
        self.accum_tokens = int(accum_tokens)
        self.global_batch = max(1, int(global_batch))
        self.precision = precision
        self.checkpoint = checkpoint
        self.step_fn = step_fn or self._default_step
        self.curriculum = curriculum or MathCurriculum()
        self.rlhf = rlhf or MathRLHF()
        self.evaluator = evaluator or MathEvaluator()

        self._phase_configs: Dict[MathTrainPhase, MathPhaseConfig] = {}
        self._phase_order = [
            MathTrainPhase.PHASE1_PRETRAIN,
            MathTrainPhase.PHASE2_FORMALIZE,
            MathTrainPhase.PHASE3_SYNTH_REINFORCE,
            MathTrainPhase.PHASE4_RLHF_ALIGN,
            MathTrainPhase.PHASE5_SELF_PLAY,
        ]
        self._current_phase_idx = 0
        self._tokens_seen = 0
        self._phase_tokens = 0
        self._step = 0
        self._metrics: List[MathTrainMetrics] = []
        self._interrupted_at: Optional[float] = None
        self._init_default_phases()

    # ------------------------------------------------------------------ #
    # 阶段配置
    # ------------------------------------------------------------------ #
    def _init_default_phases(self) -> None:
        for ph, (tok, lr, lv) in PHASE_DEFAULTS.items():
            self._phase_configs[ph] = MathPhaseConfig(
                phase=ph,
                total_tokens=tok,
                lr_peak=lr,
                lr_min=lr * 0.1,
                init_curriculum=lv,
            )

    def configure_phase(self, config: MathPhaseConfig) -> None:
        self._phase_configs[config.phase] = config

    @property
    def current_phase(self) -> MathTrainPhase:
        return self._phase_order[self._current_phase_idx]

    @property
    def tokens_seen(self) -> int:
        return self._tokens_seen

    @property
    def step(self) -> int:
        return self._step

    @property
    def metrics(self) -> List[MathTrainMetrics]:
        return list(self._metrics)

    # ------------------------------------------------------------------ #
    # 学习率调度 (cosine + warmup)
    # ------------------------------------------------------------------ #
    def _lr_schedule(self, cfg: MathPhaseConfig, progress: float) -> float:
        warmup = cfg.warmup_ratio
        if progress < warmup and warmup > 0:
            return cfg.lr_peak * (progress / warmup)
        decay_progress = (progress - warmup) / max(1e-6, 1.0 - warmup)
        cos = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return cfg.lr_min + (cfg.lr_peak - cfg.lr_min) * cos

    # ------------------------------------------------------------------ #
    # 训练循环
    # ------------------------------------------------------------------ #
    def train(
        self,
        data_iter: Optional[Callable[[int, MathTrainPhase], Sequence[Any]]] = None,
        max_steps: Optional[int] = None,
    ) -> List[MathTrainMetrics]:
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

            # 阶段切换时初始化课程级别
            if self._phase_tokens == 0:
                self.curriculum.reset(cfg.init_curriculum)

            batch = data_iter(self._step, phase)
            loss, grad_norm, throughput = self.step_fn(batch, phase, self._ctx())
            progress = self._phase_tokens / max(1, cfg.total_tokens)
            lr = self._lr_schedule(cfg, min(1.0, progress))

            # 阶段特定后处理
            reward = 0.0
            mastery = self.curriculum.state.mastery
            if phase == MathTrainPhase.PHASE4_RLHF_ALIGN:
                # RLHF 阶段: 用奖励器对 batch 打分 (模拟)
                reward = self._rlhf_step(batch)
            elif phase == MathTrainPhase.PHASE3_SYNTH_REINFORCE:
                # 合成强化: 更新课程掌握度 (模拟)
                self.curriculum.update(mastery=max(0.0, min(1.0, 1.0 - loss)), samples=len(batch))
            elif phase == MathTrainPhase.PHASE5_SELF_PLAY:
                # Self-Play: 推进课程
                self.curriculum.update(mastery=max(0.0, min(1.0, 1.0 - loss)), samples=len(batch))

            self._step += 1
            steps_run += 1
            self._tokens_seen += self.global_batch
            self._phase_tokens += self.global_batch
            self._metrics.append(
                MathTrainMetrics(
                    step=self._step,
                    phase=phase.value,
                    loss=loss,
                    lr=lr,
                    grad_norm=grad_norm,
                    throughput_tok_s=throughput,
                    tokens_seen=self._tokens_seen,
                    curriculum_level=self.curriculum.state.level.value,
                    mastery=round(mastery, 4),
                    reward=round(reward, 4),
                )
            )

            # 定期检查点 (满足累积 token 预算)
            if self._phase_tokens >= self.accum_tokens or self._step % 100 == 0:
                self._save_checkpoint_if_needed()
                # 阶段内评估触发 (轻量)
                self.evaluator.maybe_evaluate(
                    step=self._step, loss=loss, checkpoint=self._ckpt_snapshot()
                )

            # 阶段完成
            if self._phase_tokens >= cfg.total_tokens:
                # 阶段切换全量评估该阶段基准
                self._eval_phase_benchmarks(phase)
                self._advance_phase()
            if max_steps is not None and steps_run >= max_steps:
                break
        return list(self._metrics)

    def _rlhf_step(self, batch: Sequence[Any]) -> float:
        """RLHF 阶段单步: 对 batch 计算平均奖励 (模拟). """
        if not batch:
            return 0.0
        total = 0.0
        n = 0
        for sample in batch:
            if not isinstance(sample, dict):
                continue
            out = self.rlhf.reward(
                problem=sample.get("problem", ""),
                response=sample.get("response", ""),
                gold_answer=sample.get("gold_answer", sample.get("answer", "")),
                gold_formal=sample.get("gold_formal", sample.get("formal", "")),
            )
            total += out.reward
            n += 1
        return total / n if n > 0 else 0.0

    def _eval_phase_benchmarks(self, phase: MathTrainPhase) -> None:
        """阶段切换时评估该阶段对应的基准. """
        benchmarks = PHASE_EVAL_BENCHMARKS.get(phase, ())
        ckpt = self._ckpt_snapshot()
        for bench in benchmarks:
            self.evaluator.evaluate(bench, checkpoint=ckpt)

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
            "curriculum_level": self.curriculum.state.level.value,
        }

    def _ckpt_snapshot(self) -> Dict[str, Any]:
        return {
            "step": self._step,
            "phase": self.current_phase.value,
            "tokens_seen": self._tokens_seen,
            "phase_tokens": self._phase_tokens,
            "phase_idx": self._current_phase_idx,
            "curriculum_level": self.curriculum.state.level.value,
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
        snap = self._ckpt_snapshot()
        last = self._metrics[-1] if self._metrics else None
        snap["metrics"] = {
            "loss": last.loss if last else 0.0,
            "lr": last.lr if last else 0.0,
            "grad_norm": last.grad_norm if last else 0.0,
            "reward": last.reward if last else 0.0,
        } if last else {}
        self.checkpoint.save(**snap)

    def _maybe_resume(self) -> None:
        if self._step > 0:
            return
        state = self.checkpoint.load()  # type: ignore[union-attr]
        if not state:
            return
        if self._interrupted_at is not None:
            elapsed_min = (time.time() - self._interrupted_at) / 60.0
            if elapsed_min > RECOVERY_WINDOW_MIN:
                state = self.checkpoint.load(stable=True)  # type: ignore[union-attr]
        self._step = state.get("step", 0)
        self._tokens_seen = state.get("tokens_seen", 0)
        self._phase_tokens = state.get("phase_tokens", 0)
        self._current_phase_idx = state.get("phase_idx", 0)
        lv = state.get("curriculum_level")
        if lv:
            try:
                self.curriculum.reset(lv)
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ #
    # 默认单步 (模拟)
    # ------------------------------------------------------------------ #
    def _default_step(
        self,
        batch: Sequence[Any],
        phase: MathTrainPhase,
        ctx: Dict[str, Any],
    ) -> Tuple[float, float, float]:
        base = {
            MathTrainPhase.PHASE1_PRETRAIN: 3.0,
            MathTrainPhase.PHASE2_FORMALIZE: 2.2,
            MathTrainPhase.PHASE3_SYNTH_REINFORCE: 1.5,
            MathTrainPhase.PHASE4_RLHF_ALIGN: 1.0,
            MathTrainPhase.PHASE5_SELF_PLAY: 0.7,
        }
        loss = base.get(phase, 1.0) * math.exp(-self._step * 1e-4)
        return loss, 1.0, float(self.global_batch * self.parallel.world_size)

    # ------------------------------------------------------------------ #
    # 统计
    # ------------------------------------------------------------------ #
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
            "curriculum": self.curriculum.summary(),
            "evaluator": self.evaluator.summary(),
            "rlhf": self.rlhf.summary(),
        }


__all__ = [
    "MathTrainer",
    "MathTrainPhase",
    "MathPhaseConfig",
    "MathTrainMetrics",
    "ParallelConfig",
    "Precision",
    "DEFAULT_ACCUM_TOKENS",
    "DEFAULT_GLOBAL_BATCH",
    "RECOVERY_WINDOW_MIN",
    "PHASE_DEFAULTS",
    "PHASE_EVAL_BENCHMARKS",
]
