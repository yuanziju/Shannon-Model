"""循环深度位置嵌入 (DepthEmbedding).

为RDT循环主体的每次迭代提供深度位置编码, 区分不同循环步骤.
使用正弦位置编码 + 可学习深度嵌入.

参考: spec 循环索引嵌入 (正弦深度位置编码).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class DepthEmbedding(nn.Module):
    """循环深度位置嵌入.

    结合正弦位置编码与可学习嵌入, 为每次循环迭代提供深度信号:
      depth_embed(iter) = sin_pos(iter) + learnable_embed(iter)
    """

    def __init__(
        self,
        hidden_dim: int,
        max_iterations: int = 32,
        embed_dim: int = 64,
        learnable: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_iterations = max_iterations
        self.embed_dim = min(embed_dim, hidden_dim)

        # 可学习深度嵌入
        if learnable:
            self.learnable_embed = nn.Parameter(
                torch.zeros(max_iterations, self.embed_dim) * 0.02
            )
            nn.init.normal_(self.learnable_embed, std=0.02)
        else:
            self.register_buffer(
                "learnable_embed",
                torch.zeros(max_iterations, self.embed_dim),
            )
        self.learnable = learnable

        # 投影到 hidden_dim
        self.proj = nn.Linear(self.embed_dim, hidden_dim, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)

        # 正弦位置编码 (预计算)
        inv_freq = 1.0 / (
            10000.0
            ** (torch.arange(0, self.embed_dim, 2).float() / self.embed_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _sinusoidal(self, depth: int, device, dtype) -> torch.Tensor:
        """计算正弦深度位置编码 [embed_dim]."""
        t = torch.tensor([float(depth)], device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))  # [1, embed_dim/2]
        emb = torch.cat([freqs.sin(), freqs.cos()], dim=-1)  # [1, embed_dim]
        return emb.squeeze(0).to(dtype)

    def forward(self, depth: int, batch_size: int = 1, seq_len: int = 1) -> torch.Tensor:
        """计算指定深度的位置嵌入.

        Args:
            depth: 循环迭代索引 (0-indexed).
            batch_size: B.
            seq_len: S.

        Returns:
            [B, S, hidden_dim] 深度位置嵌入.
        """
        depth = min(depth, self.max_iterations - 1)
        device = self.proj.weight.device
        dtype = self.proj.weight.dtype

        # 正弦 + 可学习
        sin_emb = self._sinusoidal(depth, device, dtype)
        learn_emb = self.learnable_embed[depth].to(device=device, dtype=dtype)
        combined = sin_emb + learn_emb  # [embed_dim]

        # 投影到 hidden_dim
        emb = self.proj(combined)  # [hidden_dim]

        # 广播到 [B, S, H]
        return emb.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, "
            f"max_iter={self.max_iterations}, "
            f"embed_dim={self.embed_dim}, "
            f"learnable={self.learnable}"
        )
