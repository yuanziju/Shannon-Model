"""模态嵌入 (ModalityEmbedding) — 多模态统一投影.

将不同模态 (文本/图像/视频/文档/SVG) 的特征投影到统一的 hidden_dim,
并附加模态类型嵌入 (modality type embedding), 供循环主体区分来源.

模态类型:
  0: text   文本
  1: image  图像
  2: video  视频
  3: doc    文档
  4: svg    SVG 矢量图

参考: spec §3 编码器多模态位置编码, AGENTS.md 模态对齐.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.layers import RMSNorm


# 模态类型 id
MODALITY_TEXT = 0
MODALITY_IMAGE = 1
MODALITY_VIDEO = 2
MODALITY_DOC = 3
MODALITY_SVG = 4
NUM_MODALITIES = 5


class ModalityEmbedding(nn.Module):
    """模态统一嵌入层.

    将各模态特征投影到 hidden_dim 并叠加模态类型嵌入.

    Args:
        hidden_dim: 统一输出维度.
        modality_dims: dict {modality_id: input_dim} 各模态输入维度.
        num_modalities: 模态类型数.
        dropout: dropout.
    """

    def __init__(
        self,
        hidden_dim: int,
        modality_dims: Optional[dict] = None,
        num_modalities: int = NUM_MODALITIES,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities

        # 默认: 各模态输入维度 = hidden_dim (已投影)
        if modality_dims is None:
            modality_dims = {
                MODALITY_TEXT: hidden_dim,
                MODALITY_IMAGE: hidden_dim,
                MODALITY_VIDEO: hidden_dim,
                MODALITY_DOC: hidden_dim,
                MODALITY_SVG: hidden_dim,
            }
        self.modality_dims = dict(modality_dims)

        # 各模态投影层 (若输入维度 != hidden_dim)
        self.modality_projs = nn.ModuleDict({
            str(mid): nn.Linear(dim, hidden_dim, bias=False)
            for mid, dim in self.modality_dims.items()
            if dim != hidden_dim
        })

        # 模态类型嵌入
        self.modality_type_embed = nn.Embedding(num_modalities, hidden_dim)
        nn.init.normal_(self.modality_type_embed.weight, std=0.02)

        # 归一化
        self.norm = RMSNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    # ------------------------------------------------------------------
    def forward(
        self,
        features: torch.Tensor,
        modality_id: int,
    ) -> torch.Tensor:
        """投影并叠加模态类型嵌入.

        Args:
            features: [B, S, D] 某模态特征 (D = modality_dims[modality_id]).
            modality_id: 模态类型 id.

        Returns:
            [B, S, hidden_dim] 统一模态嵌入.
        """
        key = str(modality_id)
        if key in self.modality_projs:
            h = self.modality_projs[key](features)
        else:
            # 已是 hidden_dim, 直接透传 (若维度不匹配则投影)
            if features.shape[-1] != self.hidden_dim:
                h = F.linear(
                    features,
                    torch.zeros(
                        self.hidden_dim, features.shape[-1],
                        device=features.device, dtype=features.dtype,
                    ),
                )
            else:
                h = features

        # 叠加模态类型嵌入
        B, S, _ = h.shape
        type_emb = self.modality_type_embed(
            torch.full((B, S), modality_id, device=h.device, dtype=torch.long)
        )  # [B, S, H]
        h = h + type_emb

        h = self.norm(h)
        h = self.dropout(h)
        return h

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, "
            f"num_modalities={self.num_modalities}, "
            f"modality_dims={self.modality_dims}"
        )
