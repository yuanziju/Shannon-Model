"""models.MathMaster.src.training - MathMaster 数学训练模块.

5 阶段训练 + 分层混合奖励 + 合成数据 + 课程学习 + 8 基准评估:

    - MathRLHF:          5 层分层混合奖励 (Lean4+SymPy+数值+数学家审查+启发式)
    - MathDataGenerator: 9 领域 + 竞赛 + 前沿 合成数学数据生成
    - MathTrainer:       5 阶段训练 (预训练→形式化→合成强化→RLHF对齐→Self-Play)
    - MathEvaluator:     8 基准评估 (MATH/GSM8K/AIME/IMO/Putnam/Lean证明率/猜想/前沿)
    - MathCurriculum:    课程学习 (basic→intermediate→advanced→frontier)
"""

from __future__ import annotations

from .curriculum import (
    LEVEL_DOMAIN_DIST,
    LEVEL_MIN_SAMPLES,
    LEVEL_MIX_RATIO,
    LEVEL_ORDER,
    MASTERY_THRESHOLD,
    CurriculumLevel,
    CurriculumState,
    MathCurriculum,
)
from .math_evaluator import (
    BENCHMARK_COST,
    BENCHMARK_SAMPLES,
    BENCHMARK_TARGETS,
    HEAVY_BENCHMARKS,
    LIGHT_BENCHMARKS,
    MathBenchmark,
    MathEvalResult,
    MathEvaluator,
)
from .math_rlhf import (
    DEFAULT_LAYER_WEIGHTS,
    DEFAULT_LEAN_BONUS,
    DEFAULT_LEAN_PENALTY,
    DEFAULT_NUMERIC_SAMPLES,
    DEFAULT_NUMERIC_TOL,
    REWARD_CLIP_MAX,
    REWARD_CLIP_MIN,
    LayerResult,
    MathRLHF,
    RewardLayer,
    RewardOutput,
    WRONG_ANSWER_PENALTY,
)
from .math_trainer import (
    DEFAULT_ACCUM_TOKENS,
    DEFAULT_GLOBAL_BATCH,
    PHASE_DEFAULTS,
    PHASE_EVAL_BENCHMARKS,
    RECOVERY_WINDOW_MIN,
    MathPhaseConfig,
    MathTrainMetrics,
    MathTrainPhase,
    MathTrainer,
    ParallelConfig,
    Precision,
)
from .synth_data import (
    DOMAIN_DEFAULT_DIFFICULTY,
    MATH_DOMAINS,
    MathDataGenerator,
    ProblemSpec,
)

__all__ = [
    # 核心类
    "MathRLHF",
    "MathDataGenerator",
    "MathTrainer",
    "MathEvaluator",
    "MathCurriculum",
    # curriculum
    "CurriculumLevel",
    "CurriculumState",
    "LEVEL_MIX_RATIO",
    "LEVEL_DOMAIN_DIST",
    "MASTERY_THRESHOLD",
    "LEVEL_MIN_SAMPLES",
    "LEVEL_ORDER",
    # synth_data
    "ProblemSpec",
    "MATH_DOMAINS",
    "DOMAIN_DEFAULT_DIFFICULTY",
    # math_rlhf
    "RewardLayer",
    "LayerResult",
    "RewardOutput",
    "DEFAULT_LAYER_WEIGHTS",
    "DEFAULT_LEAN_BONUS",
    "DEFAULT_LEAN_PENALTY",
    "DEFAULT_NUMERIC_TOL",
    "DEFAULT_NUMERIC_SAMPLES",
    "REWARD_CLIP_MIN",
    "REWARD_CLIP_MAX",
    "WRONG_ANSWER_PENALTY",
    # math_trainer
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
    # math_evaluator
    "MathBenchmark",
    "MathEvalResult",
    "BENCHMARK_TARGETS",
    "BENCHMARK_SAMPLES",
    "BENCHMARK_COST",
    "LIGHT_BENCHMARKS",
    "HEAVY_BENCHMARKS",
]
