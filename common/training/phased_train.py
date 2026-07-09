"""分阶段训练 - 1a->1b->1c->1d, 上下文 32K->5M.

PhasedTrainer 实现模型架构的渐进式引入 (spec):
    1a Dense   -> 1b MoE    -> 1c 门控  -> 1d RDT
    上下文窗口: 32K -> 128K -> 1M -> 5M

每阶段在前一阶段检查点基础上:
    - 扩展架构 (新增模块)
    - 扩展上下文 (LongRoPE2 / Ring Attention)
    - 微调适配
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


class ArchStage(Enum):
    """架构引入阶段. """

    S1A_DENSE = "1a_dense"     # Dense 模型
    S1B_MOE = "1b_moe"         # 引入双层 MoE
    S1C_GATE = "1c_gate"       # 引入门控注意力 + CTM
    S1D_RDT = "1d_rdt"         # 引入 RDT 循环块 (1-32 次)


# 上下文窗口演进 (spec: 32K -> 5M)
STAGE_CONTEXT = {
    ArchStage.S1A_DENSE: 32_768,      # 32K
    ArchStage.S1B_MOE: 131_072,       # 128K
    ArchStage.S1C_GATE: 1_048_576,    # 1M
    ArchStage.S1D_RDT: 5_242_880,     # 5M
}

# 各阶段新增模块
STAGE_NEW_MODULES = {
    ArchStage.S1A_DENSE: ["encoder", "dense_blocks", "decoder"],
    ArchStage.S1B_MOE: ["dual_layer_moe", "expert_router"],
    ArchStage.S1C_GATE: ["gated_attention", "ctm_nlm", "mla_sync"],
    ArchStage.S1D_RDT: ["rdt_loop_block", "act_stop", "latent_decoder"],
}


@dataclass
class StageConfig:
    """单阶段配置. """

    stage: ArchStage
    context_length: int
    new_modules: List[str]
    tokens: int                          # 该阶段训练 token 数
    lr: float
    # 冻结策略: 哪些已有模块在该阶段冻结
    freeze: List[str] = field(default_factory=list)
    # RoPE 扩展参数
    rope_scaling: str = "linear"
    yarn_factor: float = 1.0
    enabled: bool = True


class PhasedTrainer:
    """分阶段架构引入训练器.

    Args:
        base_checkpoint: 前一阶段检查点 (字典) 供继承.
        stages: 自定义阶段配置列表 (默认按 spec 1a->1d).
        train_fn: 单阶段训练回调
            ``fn(stage, config, ckpt) -> (new_ckpt, metrics)``.
    """

    def __init__(
        self,
        base_checkpoint: Optional[Dict[str, Any]] = None,
        stages: Optional[Sequence[StageConfig]] = None,
        train_fn: Optional[Callable[[ArchStage, StageConfig, Dict[str, Any]], Tuple[Dict[str, Any], Dict[str, Any]]]] = None,
    ) -> None:
        self.base_checkpoint = base_checkpoint or {}
        self.train_fn = train_fn or self._default_train
        self._stages: List[StageConfig] = list(stages) if stages else self._default_stages()
        self._stage_idx = 0
        self._checkpoint: Dict[str, Any] = dict(self.base_checkpoint)
        self._stage_metrics: List[Dict[str, Any]] = []
        self._completed: List[ArchStage] = []

    def _default_stages(self) -> List[StageConfig]:
        return [
            StageConfig(
                stage=ArchStage.S1A_DENSE,
                context_length=STAGE_CONTEXT[ArchStage.S1A_DENSE],
                new_modules=STAGE_NEW_MODULES[ArchStage.S1A_DENSE],
                tokens=500_000_000_000,  # 500B
                lr=6e-4,
            ),
            StageConfig(
                stage=ArchStage.S1B_MOE,
                context_length=STAGE_CONTEXT[ArchStage.S1B_MOE],
                new_modules=STAGE_NEW_MODULES[ArchStage.S1B_MOE],
                tokens=300_000_000_000,  # 300B
                lr=3e-4,
                freeze=["encoder"],
                rope_scaling="yarn",
                yarn_factor=2.0,
            ),
            StageConfig(
                stage=ArchStage.S1C_GATE,
                context_length=STAGE_CONTEXT[ArchStage.S1C_GATE],
                new_modules=STAGE_NEW_MODULES[ArchStage.S1C_GATE],
                tokens=150_000_000_000,  # 150B
                lr=1e-4,
                freeze=["encoder", "expert_router"],
                rope_scaling="longrope2",
                yarn_factor=4.0,
            ),
            StageConfig(
                stage=ArchStage.S1D_RDT,
                context_length=STAGE_CONTEXT[ArchStage.S1D_RDT],
                new_modules=STAGE_NEW_MODULES[ArchStage.S1D_RDT],
                tokens=100_000_000_000,  # 100B
                lr=5e-5,
                freeze=["encoder", "expert_router", "gated_attention"],
                rope_scaling="longrope2",
                yarn_factor=8.0,
            ),
        ]

    # ------------------------------------------------------------------ #
    # 阶段执行
    # ------------------------------------------------------------------ #
    @property
    def current_stage(self) -> ArchStage:
        if self._stage_idx >= len(self._stages):
            return self._stages[-1].stage
        return self._stages[self._stage_idx].stage

    @property
    def current_context_length(self) -> int:
        if self._stage_idx >= len(self._stages):
            return self._stages[-1].context_length
        return self._stages[self._stage_idx].context_length

    def run_stage(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """执行当前阶段训练, 返回 (新检查点, 指标). """
        if self._stage_idx >= len(self._stages):
            return self._checkpoint, {}
        cfg = self._stages[self._stage_idx]
        if not cfg.enabled:
            self._stage_idx += 1
            return self.run_stage()
        self._checkpoint, metrics = self.train_fn(cfg.stage, cfg, self._checkpoint)
        # 继承: 新模块加入检查点
        self._checkpoint.setdefault("modules", [])
        for m in cfg.new_modules:
            if m not in self._checkpoint["modules"]:
                self._checkpoint["modules"].append(m)
        self._checkpoint["stage"] = cfg.stage.value
        self._checkpoint["context_length"] = cfg.context_length
        metrics["stage"] = cfg.stage.value
        metrics["context_length"] = cfg.context_length
        metrics["new_modules"] = list(cfg.new_modules)
        self._stage_metrics.append(metrics)
        self._completed.append(cfg.stage)
        self._stage_idx += 1
        return self._checkpoint, metrics

    def run_all(self) -> List[Dict[str, Any]]:
        """依次执行所有阶段. """
        all_metrics: List[Dict[str, Any]] = []
        while self._stage_idx < len(self._stages):
            _, m = self.run_stage()
            all_metrics.append(m)
        return all_metrics

    # ------------------------------------------------------------------ #
    # 默认训练 (模拟)
    # ------------------------------------------------------------------ #
    def _default_train(
        self, stage: ArchStage, cfg: StageConfig, ckpt: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        modules = list(ckpt.get("modules", []))
        new_ckpt = {
            "modules": modules,
            "prev_stage": ckpt.get("stage"),
            "params_b": 15.0,  # 目标 15B MoE
            "context_length": cfg.context_length,
        }
        metrics = {
            "loss": 2.0 * math.exp(-len(self._completed) * 0.3),
            "tokens": cfg.tokens,
            "frozen": list(cfg.freeze),
            "rope_scaling": cfg.rope_scaling,
        }
        return new_ckpt, metrics

    # ------------------------------------------------------------------ #
    # 统计
    # ------------------------------------------------------------------ #
    @property
    def checkpoint(self) -> Dict[str, Any]:
        return dict(self._checkpoint)

    @property
    def completed_stages(self) -> List[ArchStage]:
        return list(self._completed)

    @property
    def stage_metrics(self) -> List[Dict[str, Any]]:
        return list(self._stage_metrics)

    def summary(self) -> Dict[str, Any]:
        return {
            "current_stage": self.current_stage.value,
            "current_context": self.current_context_length,
            "completed": [s.value for s in self._completed],
            "modules": self._checkpoint.get("modules", []),
            "total_stages": len(self._stages),
            "context_progression": {
                s.stage.value: s.context_length for s in self._stages
            },
        }


__all__ = [
    "PhasedTrainer",
    "StageConfig",
    "ArchStage",
    "STAGE_CONTEXT",
    "STAGE_NEW_MODULES",
]
