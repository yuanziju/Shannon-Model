"""循环主体包 (RecurrentBody) — RDT 循环块 1-32 次动态迭代.

公开API:
    RecurrentBody            - 循环主体 (管理 1-32 次迭代 + Silent Thinking)
    RecurrentBlock           - 单次循环迭代块 (注意力 + MoE + 深度信号)
    HybridM3AttentionLayer   - Hybrid-M3 4层周期注意力层
    DepthEmbedding           - 循环深度位置嵌入
    LTIStability             - LTI 稳定性约束 (谱半径<1)
    ResidualStabilizer       - 残差稳定器
    DepthLoRALinear          - 深度相关 LoRA 线性层
    DepthLoRAAdapter         - 深度 LoRA 适配器集合
    ACTStop                  - ACT 自适应停止
"""

from __future__ import annotations

from .depth_embed import DepthEmbedding
from .lti import LTIStability, ResidualStabilizer
from .lora_adapter import DepthLoRALinear, DepthLoRAAdapter, apply_depth_lora
from .act import ACTStop
from .body import (
    RecurrentBody,
    RecurrentBlock,
    HybridM3AttentionLayer,
)

__all__ = [
    "RecurrentBody",
    "RecurrentBlock",
    "HybridM3AttentionLayer",
    "DepthEmbedding",
    "LTIStability",
    "ResidualStabilizer",
    "DepthLoRALinear",
    "DepthLoRAAdapter",
    "apply_depth_lora",
    "ACTStop",
]
