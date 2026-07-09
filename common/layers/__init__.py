"""common.layers - 模型基础层模块.

提供归一化、激活、旋转位置编码、掩码工具、残差连接与梯度检查点等基础层实现.
"""

from .norm import RMSNorm, GatedRMSNorm
from .activation import SwiGLU, GeGLU
from .rope import (
    RoPE,
    RoPE2D,
    RoPE3D,
    YaRN,
    LongRoPE2,
    TemporalDecayRoPE,
)
from .mask_utils import (
    make_causal_mask,
    make_bidirectional_mask,
    make_mma_mask,
    make_sliding_mask,
    HybridMaskGenerator,
)
from .residual import AttnRes, mHC
from .checkpoint_utils import GradientCheckpoint, checkpoint_sequential

__all__ = [
    # norm
    "RMSNorm",
    "GatedRMSNorm",
    # activation
    "SwiGLU",
    "GeGLU",
    # rope
    "RoPE",
    "RoPE2D",
    "RoPE3D",
    "YaRN",
    "LongRoPE2",
    "TemporalDecayRoPE",
    # mask
    "make_causal_mask",
    "make_bidirectional_mask",
    "make_mma_mask",
    "make_sliding_mask",
    "HybridMaskGenerator",
    # residual
    "AttnRes",
    "mHC",
    # checkpoint
    "GradientCheckpoint",
    "checkpoint_sequential",
]
