"""common.training - Shannon 训练引擎模块.

训练引擎层 (Layer 2) 核心组件:
    - ShannonTrainer:        6 阶段训练 + BF16 + 8M token 累积
    - TrainingCheckpoint:    分片 + RNG 状态 + 啊哈时刻
    - DynamicLossWeighter:   多任务损失动态加权
    - MoEBalanceLoss:        双层 MoE 负载均衡损失
    - MTPLoss:               多 token 预测损失 (k=2-4)
    - MultiOptimizer:        SAGE/Muon/AdEMAMix/SCALE 多优化器
    - PhasedTrainer:         1a->1b->1c->1d 架构渐进引入, 上下文 32K->5M
    - Evaluator:             10 基准评估 + 智能触发
"""

from .checkpoint import AhaMoment, CheckpointShard, TrainingCheckpoint
from .evaluator import (
    BENCHMARK_COST,
    BENCHMARK_TARGETS,
    Benchmark,
    EvalResult,
    Evaluator,
)
from .losses import DynamicLossWeighter, ExpertLoad, MoEBalanceLoss, MTPLoss
from .optimizers import DEFAULT_GROUP_MAPPING, MultiOptimizer, OptimizerKind, ParamGroup
from .phased_train import (
    STAGE_CONTEXT,
    STAGE_NEW_MODULES,
    ArchStage,
    PhasedTrainer,
    StageConfig,
)
from .trainer import (
    DEFAULT_ACCUM_TOKENS,
    DEFAULT_GLOBAL_BATCH,
    RECOVERY_WINDOW_MIN,
    ParallelConfig,
    PhaseConfig,
    Precision,
    ShannonTrainer,
    TrainMetrics,
    TrainPhase,
)

__all__ = [
    # trainer
    "ShannonTrainer",
    "TrainPhase",
    "Precision",
    "ParallelConfig",
    "PhaseConfig",
    "TrainMetrics",
    "DEFAULT_ACCUM_TOKENS",
    "DEFAULT_GLOBAL_BATCH",
    "RECOVERY_WINDOW_MIN",
    # checkpoint
    "TrainingCheckpoint",
    "CheckpointShard",
    "AhaMoment",
    # losses
    "DynamicLossWeighter",
    "MoEBalanceLoss",
    "MTPLoss",
    "ExpertLoad",
    # optimizers
    "MultiOptimizer",
    "ParamGroup",
    "OptimizerKind",
    "DEFAULT_GROUP_MAPPING",
    # phased train
    "PhasedTrainer",
    "StageConfig",
    "ArchStage",
    "STAGE_CONTEXT",
    "STAGE_NEW_MODULES",
    # evaluator
    "Evaluator",
    "EvalResult",
    "Benchmark",
    "BENCHMARK_TARGETS",
    "BENCHMARK_COST",
]
