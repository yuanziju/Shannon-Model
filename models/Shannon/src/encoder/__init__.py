"""编码器包 (Encoder) — 多模态统一编码.

公开API:
    TextEmbedding            - 文本嵌入 (9 特殊 token)
    ModalityEmbedding        - 模态统一投影
    SVGTokenizer             - SVG 矢量图分词器
    ViTQFormerEncoder        - ViT + Q-Former 视觉编码器
    VAEEncoder               - VAE 视觉编码器
    DualChannelVisionEncoder - 双通道视觉编码器 (ViT+Q-Former AND VAE)
    VideoEncoder             - 视频编码器 (空间 ViT + 时序 SSM)
    DocParser                - 文档解析器 (PDF/Word/PPT 双通道)
    DocPageEncoder           - 单页文档编码器
    PatchEmbed, ViTBlock     - 视觉基础块
"""

from __future__ import annotations

from .text_embed import (
    TextEmbedding,
    SPECIAL_TOKENS,
    NUM_SPECIAL_TOKENS,
    PAD_TOKEN_ID,
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    UNK_TOKEN_ID,
    MASK_TOKEN_ID,
    IMG_TOKEN_ID,
    VID_TOKEN_ID,
    DOC_TOKEN_ID,
    THINK_TOKEN_ID,
)
from .modality_embed import (
    ModalityEmbedding,
    MODALITY_TEXT,
    MODALITY_IMAGE,
    MODALITY_VIDEO,
    MODALITY_DOC,
    MODALITY_SVG,
    NUM_MODALITIES,
)
from .svg_tokenizer import SVGTokenizer
from .image_encoder import (
    PatchEmbed,
    ViTBlock,
    ViTQFormerEncoder,
    VAEEncoder,
    DualChannelVisionEncoder,
)
from .video_encoder import VideoEncoder, TemporalSSM
from .doc_parser import DocParser, DocPageEncoder

__all__ = [
    # text
    "TextEmbedding",
    "SPECIAL_TOKENS",
    "NUM_SPECIAL_TOKENS",
    "PAD_TOKEN_ID",
    "BOS_TOKEN_ID",
    "EOS_TOKEN_ID",
    "UNK_TOKEN_ID",
    "MASK_TOKEN_ID",
    "IMG_TOKEN_ID",
    "VID_TOKEN_ID",
    "DOC_TOKEN_ID",
    "THINK_TOKEN_ID",
    # modality
    "ModalityEmbedding",
    "MODALITY_TEXT",
    "MODALITY_IMAGE",
    "MODALITY_VIDEO",
    "MODALITY_DOC",
    "MODALITY_SVG",
    "NUM_MODALITIES",
    # svg
    "SVGTokenizer",
    # image
    "PatchEmbed",
    "ViTBlock",
    "ViTQFormerEncoder",
    "VAEEncoder",
    "DualChannelVisionEncoder",
    # video
    "VideoEncoder",
    "TemporalSSM",
    # doc
    "DocParser",
    "DocPageEncoder",
]
