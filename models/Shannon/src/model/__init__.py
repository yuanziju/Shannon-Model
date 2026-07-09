"""Shannon 模型包 — 顶层模型与三大子模块.

公开API:
    ShannonModel           - 顶层模型 (编码器 + 循环主体 + 解码器)
    ShannonEncoder         - 编码器 (3% 参数)
    ShannonRecurrentBody   - 循环主体包装 (94% 参数)
    ShannonDecoderWrapper  - 解码器包装 (3% 参数, 含多任务头)
"""

from __future__ import annotations

from .model import (
    ShannonModel,
    ShannonEncoder,
    ShannonRecurrentBody,
    ShannonDecoderWrapper,
)

__all__ = [
    "ShannonModel",
    "ShannonEncoder",
    "ShannonRecurrentBody",
    "ShannonDecoderWrapper",
]
