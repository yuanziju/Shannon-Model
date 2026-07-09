"""解码器包 (Decoder) — B+C 融合隐空间解码 + 多任务输出头.

公开API:
    ShannonDecoder     - 主解码器 (B+C 融合: HierarchicalNAR + MaskRefine + Flow + AR)
    SVGDecoder         - SVG 矢量图解码器 (自回归生成 SVG token)
    StructuredOutput   - 结构化输出 (JSON / 工具调用 / TTS)
    ImageEditRouter    - 图像编辑路由器 (编辑类型 + 区域 + VAE 增量)
"""

from __future__ import annotations

from .decoder import ShannonDecoder
from .svg_decoder import SVGDecoder
from .structured import (
    StructuredOutput,
    STRUCT_TEXT,
    STRUCT_JSON,
    STRUCT_TOOL,
    STRUCT_TTS,
    NUM_STRUCT_TYPES,
)
from .image_edit import (
    ImageEditRouter,
    EDIT_GLOBAL_STYLE,
    EDIT_LOCAL_EDIT,
    EDIT_INPAINTING,
    EDIT_SUPER_RES,
    NUM_EDIT_TYPES,
)

__all__ = [
    # 主解码器
    "ShannonDecoder",
    # SVG
    "SVGDecoder",
    # 结构化输出
    "StructuredOutput",
    "STRUCT_TEXT",
    "STRUCT_JSON",
    "STRUCT_TOOL",
    "STRUCT_TTS",
    "NUM_STRUCT_TYPES",
    # 图像编辑
    "ImageEditRouter",
    "EDIT_GLOBAL_STYLE",
    "EDIT_LOCAL_EDIT",
    "EDIT_INPAINTING",
    "EDIT_SUPER_RES",
    "NUM_EDIT_TYPES",
]
