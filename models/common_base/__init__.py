"""common_base — Shannon 与 MathMaster 共用的底子模块.

提取两个模型共享的底层组件, 实现"同一底子"架构:

  * ``BaseConfig``            — 共享配置基类 (150B MoE 目标规格)
  * ``ResidentExpertConfig``  — 常驻专家子配置 (6 常驻: 4 固定 + 2 可学习)
  * ``ExpertFFN``             — 标准专家 FFN (SwiGLU + down-proj)
  * ``EmptyExpert``           — 零初始化空专家 (Shannon 风格)
  * ``ResidentExpertPool``    — 6 常驻专家池 (4 固定 + 2 可学习)
  * ``DualMoEPool``           — 16 大 x 16 小 双层 MoE (Top-k 路由)
  * ``ExpertPool``            — 完整专家池 (6 常驻 + 16x16 MoE)
  * ``ResidualPool``          — 残差池 (AttnRes + mHC + attention 检索)
  * ``ABStack``               — AB 堆叠 (Shannon 简化版 / MathMaster 完整版)
  * ``SimplifiedABStack``     — Shannon 简化 AB 堆叠
  * ``ABBlock``               — 完整 AB 块 (5路注意力 + 元路由器 + 子agent)
  * ``MetaRouter``            — 元路由器 (1对1置换)
  * ``SubAgent``              — 子agent (不同路由策略)
  * ``FivePathAttention``     — 五路注意力 (Hybrid-M3)

Shannon 和 MathMaster 都从本包导入这些共享组件, 实现共用底子架构.
参考: AGENTS.md 项目结构全景, Shannon 融合决策.
"""

from __future__ import annotations

from .base_config import BaseConfig, ResidentExpertConfig
from .expert_pool import (
    ExpertFFN,
    EmptyExpert,
    ResidentExpertPool,
    DualMoEPool,
    ExpertPool,
)
from .residual_pool import ResidualPool
from .ab_stack import (
    ABStack,
    SimplifiedABStack,
    SimplifiedABBlock,
    ABBlock,
    MetaRouter,
    SubAgent,
    FivePathAttention,
)

__all__ = [
    # config
    "BaseConfig",
    "ResidentExpertConfig",
    # expert pool
    "ExpertFFN",
    "EmptyExpert",
    "ResidentExpertPool",
    "DualMoEPool",
    "ExpertPool",
    # residual pool
    "ResidualPool",
    # ab stack
    "ABStack",
    "SimplifiedABStack",
    "SimplifiedABBlock",
    "ABBlock",
    "MetaRouter",
    "SubAgent",
    "FivePathAttention",
]
