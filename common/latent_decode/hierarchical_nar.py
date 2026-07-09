"""层次化非自回归解码 (HierarchicalNAR).

方案 B (Hierarchical NAR): 三层金字塔式并行解码, 从粗到细:

    段落 (paragraph) ──→ 句子 (sentence) ──→ token

每一层均以非自回归方式一次性生成全部位置 (基于掩码占位), 上层结果作为
下层条件. 通过层次化解耦, 既保留 NAR 的并行速度, 又借助层级结构缓解
"全局一致性差" 的固有缺陷.

设计参考: spec.md §14.3 决策 L1-L15, B+C 融合架构.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HierarchicalNARConfig:
    """层次化 NAR 配置."""

    vocab_size: int = 128000
    hidden_dim: int = 1024
    num_heads: int = 16
    num_layers_per_level: int = 2
    max_paragraphs: int = 64
    max_sentences_per_paragraph: int = 16
    max_tokens_per_sentence: int = 64
    mask_token_id: int = 4           # 词表中 <MASK> 占位符
    dropout: float = 0.1
    # 跨层级条件: 上层隐状态如何注入下层
    cross_level_conditioning: str = "add"  # "add" | "concat"


class MaskedBlock(nn.Module):
    """单层 NAR 解码块: 自注意力 + FFN, 支持 mask token 占位."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(
            h, h, h, key_padding_mask=key_padding_mask, need_weights=False
        )
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class HierarchicalNAR(nn.Module):
    """段落 → 句子 → token 三层并行 NAR 解码器."""

    LEVELS = ("paragraph", "sentence", "token")

    def __init__(self, config: HierarchicalNARConfig | None = None, **kwargs):
        super().__init__()
        cfg = config or HierarchicalNARConfig(**kwargs)
        self.cfg = cfg

        # ---- 嵌入层 ----
        # 段落级: 段落槽位嵌入 + 类型嵌入
        self.paragraph_pos = nn.Embedding(cfg.max_paragraphs, cfg.hidden_dim)
        # 句子级: (段内) 句子位置嵌入
        self.sentence_pos = nn.Embedding(
            cfg.max_sentences_per_paragraph, cfg.hidden_dim
        )
        # token 级: 词表嵌入 + (句内) token 位置嵌入
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.token_pos = nn.Embedding(cfg.max_tokens_per_sentence, cfg.hidden_dim)
        self.mask_embed = nn.Parameter(torch.randn(cfg.hidden_dim) * 0.02)

        # ---- 三层解码主干 ----
        self.paragraph_blocks = nn.ModuleList([
            MaskedBlock(cfg.hidden_dim, cfg.num_heads, cfg.dropout)
            for _ in range(cfg.num_layers_per_level)
        ])
        self.sentence_blocks = nn.ModuleList([
            MaskedBlock(cfg.hidden_dim, cfg.num_heads, cfg.dropout)
            for _ in range(cfg.num_layers_per_level)
        ])
        self.token_blocks = nn.ModuleList([
            MaskedBlock(cfg.hidden_dim, cfg.num_heads, cfg.dropout)
            for _ in range(cfg.num_layers_per_level)
        ])

        # ---- 跨层级条件投影 ----
        if cfg.cross_level_conditioning == "concat":
            self.paragraph_to_sentence = nn.Linear(
                cfg.hidden_dim * 2, cfg.hidden_dim
            )
            self.sentence_to_token = nn.Linear(
                cfg.hidden_dim * 2, cfg.hidden_dim
            )
        else:
            self.paragraph_to_sentence = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
            self.sentence_to_token = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)

        # ---- 输出头 ----
        # 段落级: 预测该段的"主题" 分布 (主题词表, 此处复用 vocab 简化)
        self.paragraph_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size)
        # 句子级: 预测句子的长度 / 边界类型
        self.sentence_length_head = nn.Linear(
            cfg.hidden_dim, cfg.max_tokens_per_sentence + 1
        )
        # token 级: 预测每个 token 的词表分布
        self.token_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    # ------------------------------------------------------------------
    # 段落级解码
    # ------------------------------------------------------------------
    def decode_paragraphs(
        self,
        num_paragraphs: int,
        batch_size: int = 1,
        device: torch.device | None = None,
        condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """并行解码所有段落的主题向量.

        Returns:
            logits: [B, P, vocab] 段落主题分布.
            hidden: [B, P, H] 段落级隐状态 (供下层条件).
        """
        device = device or torch.device("cpu")
        pos = torch.arange(num_paragraphs, device=device)
        x = self.paragraph_pos(pos).unsqueeze(0).expand(batch_size, -1, -1)
        if condition is not None:
            # condition: [B, H] -> 广播到 [B, P, H]
            x = x + condition.unsqueeze(1)
        for blk in self.paragraph_blocks:
            x = blk(x)
        logits = self.paragraph_head(x)
        return logits, x

    # ------------------------------------------------------------------
    # 句子级解码
    # ------------------------------------------------------------------
    def decode_sentences(
        self,
        paragraph_hidden: torch.Tensor,
        num_sentences: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """并行解码每段内的句子长度与边界.

        Args:
            paragraph_hidden: [B, P, H] 段落级隐状态.
            num_sentences: 每段最大句子数.

        Returns:
            length_logits: [B, P, S, max_tokens+1] 每句长度分布.
            sentence_hidden: [B, P, S, H] 句子级隐状态.
        """
        B, P, H = paragraph_hidden.shape
        device = paragraph_hidden.device
        pos = torch.arange(num_sentences, device=device)
        sent_pos = self.sentence_pos(pos)  # [S, H]
        # 扩展: [B, P, S, H]
        x = sent_pos.unsqueeze(0).unsqueeze(0).expand(B, P, -1, -1).clone()
        # 注入段落条件
        cond = self.paragraph_to_sentence(paragraph_hidden)  # [B, P, H]
        if self.cfg.cross_level_conditioning == "concat":
            cond_rep = cond.unsqueeze(2).expand(-1, -1, num_sentences, -1)
            x = torch.cat([x, cond_rep], dim=-1)
            x = self.paragraph_to_sentence(x)
        else:
            x = x + cond.unsqueeze(2)
        x = x.reshape(B * P, num_sentences, H)
        for blk in self.sentence_blocks:
            x = blk(x)
        x = x.reshape(B, P, num_sentences, H)
        length_logits = self.sentence_length_head(x)
        return length_logits, x

    # ------------------------------------------------------------------
    # token 级解码
    # ------------------------------------------------------------------
    def decode_tokens(
        self,
        sentence_hidden: torch.Tensor,
        max_tokens: int,
        mask_ratio: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """并行解码每句内的 token 序列.

        Args:
            sentence_hidden: [B, P, S, H] 句子级隐状态.
            max_tokens: 每句最大 token 数.
            mask_ratio: 初始 mask 比例 (1.0 = 全部 mask).

        Returns:
            token_logits: [B, P, S, T, vocab].
            token_hidden: [B, P, S, T, H].
        """
        B, P, S, H = sentence_hidden.shape
        device = sentence_hidden.device
        # 初始: 全部为 mask token
        T = max_tokens
        x = self.mask_embed.reshape(1, 1, 1, 1, H).expand(B, P, S, T, H).clone()
        # token 位置嵌入
        tok_pos = self.token_pos(torch.arange(T, device=device))  # [T, H]
        x = x + tok_pos.reshape(1, 1, 1, T, H)
        # 注入句子条件
        cond = self.sentence_to_token(sentence_hidden)  # [B, P, S, H]
        if self.cfg.cross_level_conditioning == "concat":
            cond_rep = cond.unsqueeze(3).expand(-1, -1, -1, T, -1)
            x = torch.cat([x, cond_rep], dim=-1)
            x = self.sentence_to_token(x)
        else:
            x = x + cond.unsqueeze(3)
        x = x.reshape(B * P * S, T, H)
        for blk in self.token_blocks:
            x = blk(x)
        x = x.reshape(B, P, S, T, H)
        logits = self.token_head(x)
        return logits, x

    # ------------------------------------------------------------------
    # 完整三层解码 (推理入口)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        num_paragraphs: int = 4,
        num_sentences: int = 8,
        max_tokens: int = 32,
        batch_size: int = 1,
        device: torch.device | None = None,
        condition: torch.Tensor | None = None,
    ) -> dict:
        """三层并行解码完整流程: 段落 → 句子 → token."""
        device = device or torch.device("cpu")
        # 1. 段落级
        para_logits, para_hidden = self.decode_paragraphs(
            num_paragraphs, batch_size, device, condition
        )
        # 2. 句子级
        sent_len_logits, sent_hidden = self.decode_sentences(
            para_hidden, num_sentences
        )
        # 3. token 级
        tok_logits, tok_hidden = self.decode_tokens(
            sent_hidden, max_tokens, mask_ratio=1.0
        )
        # 取 argmax 作为最终 token
        tokens = tok_logits.argmax(dim=-1)  # [B, P, S, T]
        sentence_lengths = sent_len_logits.argmax(dim=-1)  # [B, P, S]
        paragraph_topics = para_logits.argmax(dim=-1)  # [B, P]
        return {
            "tokens": tokens,
            "sentence_lengths": sentence_lengths,
            "paragraph_topics": paragraph_topics,
            "para_hidden": para_hidden,
            "sent_hidden": sent_hidden,
            "tok_hidden": tok_hidden,
            "para_logits": para_logits,
            "sent_len_logits": sent_len_logits,
            "tok_logits": tok_logits,
        }

    def extra_repr(self) -> str:
        return (
            f"vocab_size={self.cfg.vocab_size}, "
            f"hidden_dim={self.cfg.hidden_dim}, "
            f"levels={self.LEVELS}"
        )
