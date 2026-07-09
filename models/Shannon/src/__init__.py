"""Shannon 15B MoE 模型 — 编码器(3%) + 循环主体(94%) + 解码器(3%).

顶层包导出:
    ShannonConfig          - 统一配置
    ShannonModel           - 顶层模型
    ShannonEncoder         - 编码器
    ShannonRecurrentBody   - 循环主体
    ShannonDecoderWrapper  - 解码器

子包:
    config    - 配置
    encoder   - 多模态编码器 (文本/图像/视频/文档/SVG)
    recurrent - 循环主体 (RDT 1-32 次动态迭代)
    moe       - 双层 MoE (16大×16小专家 + 空专家)
    decoder   - B+C 融合解码器 + 多任务输出头
    model     - 顶层模型
"""

from __future__ import annotations

from .config.config import ShannonConfig
from .model.model import (
    ShannonModel,
    ShannonEncoder,
    ShannonRecurrentBody,
    ShannonDecoderWrapper,
)

__all__ = [
    "ShannonConfig",
    "ShannonModel",
    "ShannonEncoder",
    "ShannonRecurrentBody",
    "ShannonDecoderWrapper",
]
