"""视频编码器 (VideoEncoder) — 时序多帧编码.

将视频 (帧序列) 编码为 token 序列, 结合空间 (ViT) 与时序 (SSM) 特征:
  1. 每帧经 ViT 提取空间 patch 特征
  2. 时序维度用 SSM (状态空间模型) 聚合, 动态循环状态
  3. 输出 memory tokens + 时序状态

参考: spec §3 编码器视频模态, AGENTS.md 视频 SSM 状态.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.layers import RMSNorm
from .image_encoder import PatchEmbed, ViTBlock


class TemporalSSM(nn.Module):
    """时序状态空间模型 (简化 SSM).

    对每帧的特征做时序递归: s_t = A * s_{t-1} + B * x_t, o_t = C * s_t.
    使用可学习衰减保证稳定.
    """

    def __init__(self, hidden_dim: int, state_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        # 输入到状态
        self.B = nn.Linear(hidden_dim, state_dim, bias=False)
        # 状态到输出
        self.C = nn.Linear(state_dim, hidden_dim, bias=False)
        # 可学习衰减 (sigmoid -> (0,1), 谱半径<1)
        self.decay = nn.Parameter(torch.zeros(state_dim))
        # 状态初始化投影
        self.state_init = nn.Linear(hidden_dim, state_dim, bias=False)
        self.norm = RMSNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """时序 SSM 前向.

        Args:
            x: [B, T, S, H] (B=batch, T=帧数, S=每帧token数, H=hidden).

        Returns:
            [B, T, S, H] 时序聚合特征.
        """
        B, T, S, H = x.shape
        device = x.device
        dtype = x.dtype
        gate = torch.sigmoid(self.decay)  # [state_dim]

        # 初始状态: 用第一帧均值初始化
        s = self.state_init(x[:, 0].mean(dim=1))  # [B, state_dim]
        outputs = []
        for t in range(T):
            xt = x[:, t]  # [B, S, H]
            # 状态更新: s = gate * s + (1-gate) * B(xt_mean)
            s_update = self.B(xt.mean(dim=1))  # [B, state_dim]
            s = gate * s + (1.0 - gate) * s_update
            # 输出: C(s) 广播到每帧
            o = self.C(s).unsqueeze(1).expand(-1, S, -1)  # [B, S, H]
            outputs.append(xt + o)
        out = torch.stack(outputs, dim=1)  # [B, T, S, H]
        return self.norm(out)


class VideoEncoder(nn.Module):
    """视频编码器: 空间 ViT + 时序 SSM.

    Args:
        hidden_dim: 输出维度.
        patch_size: ViT patch 大小.
        num_layers: ViT 层数.
        num_heads: 注意力头数.
        max_frames: 最大帧数.
        ssm_state_dim: SSM 状态维度.
        memory_tokens: 输出 memory token 数.
    """

    def __init__(
        self,
        hidden_dim: int,
        patch_size: int = 16,
        num_layers: int = 6,
        num_heads: int = 8,
        max_frames: int = 32,
        ssm_state_dim: int = 64,
        memory_tokens: int = 16,
        in_channels: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_frames = max_frames
        self.memory_tokens = memory_tokens
        num_heads = max(1, min(num_heads, hidden_dim))

        # 空间编码 (共享 ViT)
        self.patch_embed = PatchEmbed(patch_size, in_channels, hidden_dim)
        num_layers = max(1, num_layers)
        self.vit_blocks = nn.ModuleList([
            ViTBlock(hidden_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.vit_norm = nn.LayerNorm(hidden_dim)

        # 时序 SSM
        self.temporal_ssm = TemporalSSM(hidden_dim, state_dim=ssm_state_dim)

        # 时序位置编码
        self.temporal_pos = nn.Embedding(max_frames, hidden_dim)
        nn.init.normal_(self.temporal_pos.weight, std=0.02)

        # memory tokens (learnable, 压缩时序信息)
        self.memory_queries = nn.Parameter(torch.zeros(memory_tokens, hidden_dim))
        nn.init.normal_(self.memory_queries, std=0.02)
        self.memory_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.memory_norm = nn.LayerNorm(hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = RMSNorm(hidden_dim)

    # ------------------------------------------------------------------
    def forward(self, video: torch.Tensor) -> dict:
        """视频编码前向.

        Args:
            video: [B, T, C, H, W] 视频帧序列 (T <= max_frames).

        Returns:
            dict 含:
              - hidden: [B, memory_tokens, hidden_dim] 视频 token.
              - frame_features: [B, T, S, hidden_dim] 每帧特征.
              - temporal_state: SSM 最终状态.
        """
        B, T, C, H, W = video.shape
        T = min(T, self.max_frames)
        video = video[:, :T]

        # 每帧空间编码
        frames = video.reshape(B * T, C, H, W)
        x = self.patch_embed(frames)  # [B*T, S, H]
        for blk in self.vit_blocks:
            x = blk(x)
        x = self.vit_norm(x)  # [B*T, S, H]
        S = x.shape[1]
        x = x.reshape(B, T, S, -1)  # [B, T, S, H]

        # 时序位置编码
        t_pos = self.temporal_pos(torch.arange(T, device=x.device))  # [T, H]
        x = x + t_pos.unsqueeze(0).unsqueeze(2)  # 广播

        # 时序 SSM 聚合
        x = self.temporal_ssm(x)  # [B, T, S, H]

        # memory token 压缩: 用 learnable query 对所有帧特征做注意力
        flat = x.reshape(B, T * S, -1)  # [B, T*S, H]
        q = self.memory_queries.unsqueeze(0).expand(B, -1, -1)  # [B, M, H]
        mem_out, _ = self.memory_attn(q, flat, flat, need_weights=False)  # [B, M, H]
        mem_out = self.memory_norm(mem_out)
        mem_out = self.out_proj(mem_out)
        mem_out = self.norm(mem_out)

        return {
            "hidden": mem_out,
            "frame_features": x,
            "num_frames": T,
        }

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, "
            f"max_frames={self.max_frames}, "
            f"memory_tokens={self.memory_tokens}"
        )
