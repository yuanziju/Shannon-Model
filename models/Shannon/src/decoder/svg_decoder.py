"""SVG 解码器 (SVGDecoder) — 矢量图生成.

将循环主体输出的隐状态解码为 SVG token 序列, 进而还原为 SVG path 字符串.
采用自回归方式生成 SVG token, 复用 SVGTokenizer 的词表与 detokenize.

工作流:
    hidden [B, S, H] ──→ svg_logits [B, S, svg_vocab] ──→ svg_tokens
    svg_tokens ──→ SVGTokenizer.detokenize ──→ SVG path 字符串

参考: spec §9 多任务输出头 (SVG), encoder/svg_tokenizer.py.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.layers import RMSNorm

from ..encoder.svg_tokenizer import (
    SVGTokenizer,
    SVG_BOS,
    SVG_EOS,
    SVG_PAD,
    SVG_NUM_BASE,
)


class SVGDecoder(nn.Module):
    """SVG 矢量图解码器.

    将隐状态解码为 SVG token 序列. 独立于主文本 lm_head, 使用 SVG 专属词表.

    Args:
        hidden_dim: 模型隐维度.
        svg_hidden_dim: SVG 解码器内部隐维度.
        coord_bins: 坐标量化 bin 数.
        max_paths: 最大路径数.
        num_layers: SVG 解码 transformer 层数.
        num_heads: 注意力头数.
    """

    def __init__(
        self,
        hidden_dim: int,
        svg_hidden_dim: int = 512,
        coord_bins: int = 256,
        max_paths: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.svg_hidden_dim = svg_hidden_dim

        # SVG tokenizer (复用其词表与 embed)
        self.tokenizer = SVGTokenizer(
            hidden_dim=svg_hidden_dim,
            coord_bins=coord_bins,
            max_paths=max_paths,
        )
        self.svg_vocab_size = self.tokenizer.vocab_size

        # 从主模型隐维度投影到 SVG 隐维度
        self.ctx_proj = nn.Linear(hidden_dim, svg_hidden_dim, bias=False)

        # SVG 解码 transformer (自回归)
        self.pos_embed = nn.Embedding(2048, svg_hidden_dim)
        nn.init.normal_(self.pos_embed.weight, std=0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=svg_hidden_dim,
            nhead=max(1, min(num_heads, svg_hidden_dim // 32)),
            dim_feedforward=svg_hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        # 输出归一化 + logits 头
        self.norm = RMSNorm(svg_hidden_dim)
        self.lm_head = nn.Linear(svg_hidden_dim, self.svg_vocab_size, bias=False)
        nn.init.normal_(self.lm_head.weight, std=0.02)
        # 权重共享: lm_head 与 tokenizer embed
        self.lm_head.weight = self.tokenizer.embed.weight

    # ------------------------------------------------------------------
    def forward(
        self,
        hidden: torch.Tensor,
        svg_ids: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
    ) -> dict:
        """SVG 解码前向.

        训练模式 (svg_ids 提供): teacher forcing 计算 logits + loss.
        推理模式 (svg_ids=None): 自回归生成 SVG token.

        Args:
            hidden: [B, S, H] 循环主体输出 (context).
            svg_ids: [B, T] 训练目标 SVG token id (teacher forcing).
            max_new_tokens: 推理时最大生成 token 数.

        Returns:
            dict 含 logits (训练) 或 tokens (推理), 以及 loss (训练).
        """
        B = hidden.shape[0]
        device = hidden.device

        # context 投影
        ctx = self.ctx_proj(hidden)  # [B, S, svg_H]
        ctx = self.norm(ctx)

        if svg_ids is not None:
            # ---- 训练: teacher forcing ----
            T = svg_ids.shape[1]
            svg_emb = self.tokenizer.embed(svg_ids)  # [B, T, svg_H]
            pos = self.pos_embed(torch.arange(T, device=device).unsqueeze(0).expand(B, T))
            svg_emb = svg_emb + pos

            # 因果掩码
            causal_mask = torch.triu(
                torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
            )
            h = self.transformer(
                tgt=svg_emb,
                memory=ctx,
                tgt_mask=causal_mask,
            )
            h = self.norm(h)
            logits = self.lm_head(h)  # [B, T, svg_vocab]

            # loss: 预测下一个 token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = svg_ids[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.svg_vocab_size),
                shift_labels.view(-1),
                ignore_index=SVG_PAD,
            )
            return {"logits": logits, "loss": loss}

        # ---- 推理: 自回归生成 ----
        tokens = torch.full(
            (B, 1), SVG_BOS, dtype=torch.long, device=device
        )
        for _ in range(max_new_tokens):
            T = tokens.shape[1]
            svg_emb = self.tokenizer.embed(tokens)
            pos = self.pos_embed(
                torch.arange(T, device=device).unsqueeze(0).expand(B, T)
            )
            svg_emb = svg_emb + pos
            causal_mask = torch.triu(
                torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
            )
            h = self.transformer(
                tgt=svg_emb, memory=ctx, tgt_mask=causal_mask
            )
            h = self.norm(h)
            logits = self.lm_head(h[:, -1, :])  # [B, svg_vocab]
            next_tok = logits.argmax(dim=-1, keepdim=True)  # [B, 1]
            tokens = torch.cat([tokens, next_tok], dim=1)
            # 全部序列到达 EOS 则停止
            if (next_tok == SVG_EOS).all():
                break

        return {"tokens": tokens, "logits": None}

    # ------------------------------------------------------------------
    def decode_to_path(self, token_ids: torch.Tensor) -> List[str]:
        """将生成的 token id 还原为 SVG path 字符串.

        Args:
            token_ids: [B, T] 或 [T] token id.

        Returns:
            list of SVG path 字符串 (每个 batch 元素一个).
        """
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        paths = []
        for row in token_ids:
            ids = row.cpu().tolist()
            path = self.tokenizer.detokenize(ids)
            paths.append(path)
        return paths

    def extra_repr(self) -> str:
        return (
            f"hidden={self.hidden_dim}, svg_hidden={self.svg_hidden_dim}, "
            f"vocab={self.svg_vocab_size}"
        )
