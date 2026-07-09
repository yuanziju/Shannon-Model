"""文本嵌入 (TextEmbedding) — 含 9 个特殊 token.

Shannon 词表 (128K) 包含 9 个特殊 token, 控制多模态与推理流程:
  0: <PAD>      填充
  1: <BOS>      序列起始
  2: <EOS>      序列结束
  3: <UNK>      未知 token
  4: <MASK>     掩码占位 (NAR/掩码精化使用)
  5: <IMG>      图像模态边界
  6: <VID>      视频模态边界
  7: <DOC>      文档模态边界
  8: <THINK>    Silent Thinking 边界

文本嵌入 = token embedding + 位置投影, 输出对齐到 hidden_dim.

参考: spec §3 编码器, AGENTS.md MUTANT Tokenizer.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.layers import RMSNorm


# 9 个特殊 token 定义
SPECIAL_TOKENS = (
    "<PAD>",    # 0
    "<BOS>",    # 1
    "<EOS>",    # 2
    "<UNK>",    # 3
    "<MASK>",   # 4
    "<IMG>",    # 5
    "<VID>",    # 6
    "<DOC>",    # 7
    "<THINK>",  # 8
)
NUM_SPECIAL_TOKENS = len(SPECIAL_TOKENS)

PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
UNK_TOKEN_ID = 3
MASK_TOKEN_ID = 4
IMG_TOKEN_ID = 5
VID_TOKEN_ID = 6
DOC_TOKEN_ID = 7
THINK_TOKEN_ID = 8


class TextEmbedding(nn.Module):
    """文本嵌入层: token embedding + 位置编码投影.

    Args:
        vocab_size: 词表大小 (含特殊 token).
        hidden_dim: 输出维度.
        max_seq_len: 最大序列长度.
        padding_idx: 填充 token id (该位置 embedding 恒为 0).
        dropout: embedding dropout.
        use_pos_proj: 是否使用可学习位置投影 (True) 或仅 sinusoidal.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        max_seq_len: int = 32768,
        padding_idx: int = PAD_TOKEN_ID,
        dropout: float = 0.0,
        use_pos_proj: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.padding_idx = padding_idx
        self.num_special_tokens = NUM_SPECIAL_TOKENS

        # token embedding
        self.token_embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=padding_idx)
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        # padding 位置置零
        with torch.no_grad():
            self.token_embed.weight[padding_idx].fill_(0.0)

        # 位置编码: sinusoidal (不可学习) + 可学习投影
        self.use_pos_proj = use_pos_proj
        inv_freq = 1.0 / (
            10000.0 ** (torch.arange(0, hidden_dim, 2).float() / hidden_dim)
        )
        self.register_buffer("pos_inv_freq", inv_freq, persistent=False)
        if use_pos_proj:
            self.pos_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
            nn.init.normal_(self.pos_proj.weight, std=0.02)

        # 归一化与 dropout
        self.norm = RMSNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    # ------------------------------------------------------------------
    def _sinusoidal_pos(self, seq_len: int, device, dtype) -> torch.Tensor:
        """生成 sinusoidal 位置编码 [seq_len, hidden_dim]."""
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(pos, self.pos_inv_freq.to(device))  # [S, H/2]
        emb = torch.cat([freqs.sin(), freqs.cos()], dim=-1)  # [S, H]
        return emb.to(dtype)

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """文本嵌入前向.

        Args:
            input_ids: [B, S] token id.
            position_ids: [B, S] 显式位置 id (None 用 0..S-1).

        Returns:
            [B, S, hidden_dim] 文本嵌入.
        """
        B, S = input_ids.shape
        device = input_ids.device

        tok = self.token_embed(input_ids)  # [B, S, H]

        # 位置编码
        if position_ids is None:
            pos = self._sinusoidal_pos(S, device, tok.dtype)  # [S, H]
        else:
            # 显式位置: 用 sinusoidal 编码 position_ids
            pos = self._sinusoidal_pos(int(position_ids.max().item()) + 1, device, tok.dtype)
            pos = pos[position_ids]  # [B, S, H]
        if self.use_pos_proj:
            pos = self.pos_proj(pos)
        h = tok + pos.unsqueeze(0) if position_ids is not None else tok + pos

        h = self.norm(h)
        h = self.dropout(h)
        return h

    def extra_repr(self) -> str:
        return (
            f"vocab_size={self.vocab_size}, hidden_dim={self.hidden_dim}, "
            f"max_seq_len={self.max_seq_len}, "
            f"num_special={self.num_special_tokens}"
        )
