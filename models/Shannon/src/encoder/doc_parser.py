"""文档解析器 (DocParser) — PDF/Word/PPT 双通道解析.

支持 PDF / xlsx / docx / pptx 双通道文档解析:
  - 通道 1: 文本提取 (结构化文本 token)
  - 通道 2: 视觉渲染 (页面图像 -> ViT 编码)

将文档统一编码为 token 序列, 供循环主体处理.

参考: spec §3 编码器文档解析管道, AGENTS.md 文档解析.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.layers import RMSNorm
from .image_encoder import PatchEmbed, ViTBlock
from .text_embed import TextEmbedding


class DocPageEncoder(nn.Module):
    """单页文档编码器: 文本通道 + 视觉通道融合.

    每页提取文本 token 与页面图像 patch, 融合后输出固定数量 token.
    """

    def __init__(
        self,
        hidden_dim: int,
        text_vocab_size: int,
        patch_size: int = 16,
        num_vit_layers: int = 4,
        num_heads: int = 8,
        max_text_tokens: int = 512,
        page_memory_tokens: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_text_tokens = max_text_tokens
        self.page_memory_tokens = page_memory_tokens
        num_heads = max(1, min(num_heads, hidden_dim))

        # 文本通道
        self.text_embed = TextEmbedding(
            vocab_size=text_vocab_size,
            hidden_dim=hidden_dim,
            max_seq_len=max_text_tokens,
            dropout=dropout,
        )

        # 视觉通道 (页面图像)
        self.patch_embed = PatchEmbed(patch_size, 3, hidden_dim)
        num_vit_layers = max(1, num_vit_layers)
        self.vit_blocks = nn.ModuleList([
            ViTBlock(hidden_dim, num_heads, dropout=dropout)
            for _ in range(num_vit_layers)
        ])
        self.vit_norm = nn.LayerNorm(hidden_dim)

        # 页面 memory token (压缩页内信息)
        self.page_queries = nn.Parameter(torch.zeros(page_memory_tokens, hidden_dim))
        nn.init.normal_(self.page_queries, std=0.02)
        self.page_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.page_norm = nn.LayerNorm(hidden_dim)

        # 文本/视觉融合
        self.fuse_proj = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.norm = RMSNorm(hidden_dim)

    # ------------------------------------------------------------------
    def forward(
        self,
        text_ids: torch.Tensor,
        page_image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """单页编码.

        Args:
            text_ids: [B, S] 页面文本 token id.
            page_image: [B, C, H, W] 页面渲染图像 (可选).

        Returns:
            [B, page_memory_tokens, hidden_dim] 页面 token.
        """
        B = text_ids.shape[0]
        # 文本通道
        text_feat = self.text_embed(text_ids)  # [B, S, H]

        # 视觉通道 (若有)
        if page_image is not None:
            vis_feat = self.patch_embed(page_image)  # [B, P, H]
            for blk in self.vit_blocks:
                vis_feat = blk(vis_feat)
            vis_feat = self.vit_norm(vis_feat)
            # 拼接文本与视觉特征
            all_feat = torch.cat([text_feat, vis_feat], dim=1)  # [B, S+P, H]
        else:
            all_feat = text_feat

        # 页面 memory 压缩
        q = self.page_queries.unsqueeze(0).expand(B, -1, -1)
        mem_out, _ = self.page_attn(q, all_feat, all_feat, need_weights=False)
        mem_out = self.page_norm(mem_out)
        mem_out = self.norm(mem_out)
        return mem_out


class DocParser(nn.Module):
    """文档解析器: 多页文档双通道编码.

    Args:
        hidden_dim: 输出维度.
        text_vocab_size: 文本词表大小.
        max_pages: 最大页数.
        page_memory_tokens: 每页输出 token 数.
    """

    def __init__(
        self,
        hidden_dim: int,
        text_vocab_size: int,
        patch_size: int = 16,
        num_vit_layers: int = 4,
        num_heads: int = 8,
        max_pages: int = 128,
        max_text_tokens: int = 512,
        page_memory_tokens: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_pages = max_pages
        self.page_memory_tokens = page_memory_tokens

        # 共享的单页编码器
        self.page_encoder = DocPageEncoder(
            hidden_dim=hidden_dim,
            text_vocab_size=text_vocab_size,
            patch_size=patch_size,
            num_vit_layers=num_vit_layers,
            num_heads=num_heads,
            max_text_tokens=max_text_tokens,
            page_memory_tokens=page_memory_tokens,
            dropout=dropout,
        )

        # 页间位置编码
        self.page_pos = nn.Embedding(max_pages, hidden_dim)
        nn.init.normal_(self.page_pos.weight, std=0.02)

        # 文档级 memory token (压缩全文档)
        self.doc_queries = nn.Parameter(torch.zeros(page_memory_tokens, hidden_dim))
        nn.init.normal_(self.doc_queries, std=0.02)
        num_heads_d = max(1, min(num_heads, hidden_dim))
        self.doc_attn = nn.MultiheadAttention(
            hidden_dim, num_heads_d, dropout=dropout, batch_first=True
        )
        self.doc_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = RMSNorm(hidden_dim)

    # ------------------------------------------------------------------
    def forward(
        self,
        page_text_ids: List[torch.Tensor],
        page_images: Optional[List[torch.Tensor]] = None,
    ) -> dict:
        """多页文档编码.

        Args:
            page_text_ids: list of [B, S_i] 每页文本 token id.
            page_images: list of [B, C, H, W] 每页图像 (可选).

        Returns:
            dict 含:
              - hidden: [B, doc_memory_tokens, hidden_dim] 文档 token.
              - page_features: list of [B, page_memory_tokens, hidden_dim].
        """
        num_pages = min(len(page_text_ids), self.max_pages)
        page_features = []
        for i in range(num_pages):
            text_ids = page_text_ids[i]
            img = page_images[i] if page_images is not None else None
            feat = self.page_encoder(text_ids, img)  # [B, M, H]
            # 页间位置编码
            pos = self.page_pos(torch.tensor(i, device=feat.device))
            feat = feat + pos.unsqueeze(0).unsqueeze(0)
            page_features.append(feat)

        # 拼接所有页特征
        all_feat = torch.cat(page_features, dim=1)  # [B, num_pages*M, H]
        B = all_feat.shape[0]

        # 文档级 memory 压缩
        q = self.doc_queries.unsqueeze(0).expand(B, -1, -1)
        doc_out, _ = self.doc_attn(q, all_feat, all_feat, need_weights=False)
        doc_out = self.doc_norm(doc_out)
        doc_out = self.out_proj(doc_out)
        doc_out = self.norm(doc_out)

        return {
            "hidden": doc_out,
            "page_features": page_features,
            "num_pages": num_pages,
        }

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, "
            f"max_pages={self.max_pages}, "
            f"page_memory_tokens={self.page_memory_tokens}"
        )
