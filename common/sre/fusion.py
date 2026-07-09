"""跨注意力融合 (CrossAttentionFusion).

spec §7.1: 在统一 Transformer 骨干的 Layer 8 / 16 / 24 / 32 插入
Cross-Attention 层, 将 SymPy / Lean / Python 三通道的工具输出结果向量
注入主干隐状态, 实现工具知识与模型推理的融合.

架构:
    主干 hidden ──→ Cross-Attention(Q=hidden, K=V=tool_vectors) ──→ 残差
                 ──→ Self-Attention ──→ FFN ──→ 输出

每层融合:
    - 多头交叉注意力 (Q 来自主干, K/V 来自工具通道)
    - 工具类型感知的位置编码
    - 与 ToolGating 联动 (门控决定是否启用融合)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# 默认插入层 (与 spec §7.1 一致)
DEFAULT_FUSION_LAYERS = (8, 16, 24, 32)

# 工具通道类型
TOOL_CHANNELS = ("sympy", "lean", "python")
NUM_TOOL_CHANNELS = len(TOOL_CHANNELS)


@dataclass
class CrossAttentionFusionConfig:
    """Cross-Attention Fusion 配置."""

    hidden_dim: int = 1024
    num_heads: int = 16
    # 插入层索引 (1-indexed 层号)
    fusion_layers: tuple = DEFAULT_FUSION_LAYERS
    # 工具向量维度 (各通道 output_dim)
    tool_dim: int = 1024
    # 每层 cross-attention 的层数
    num_cross_layers: int = 1
    dropout: float = 0.1
    # 是否每层共享参数 (False=每层独立)
    shared_params: bool = False
    # 残差缩放
    residual_scale: float = 1.0


class CrossAttentionBlock(nn.Module):
    """单层 Cross-Attention 块: Q=主干, K=V=工具向量."""

    def __init__(self, hidden_dim: int, num_heads: int, tool_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim 必须能被 num_heads 整除"

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(tool_dim, hidden_dim)
        self.v_proj = nn.Linear(tool_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(tool_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

        # FFN
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        # 工具类型嵌入 (区分 sympy/lean/python)
        self.channel_embed = nn.Embedding(NUM_TOOL_CHANNELS, hidden_dim)

    def forward(
        self,
        hidden: torch.Tensor,           # [B, T, H] 主干 Q
        tool_vectors: torch.Tensor,     # [B, M, tool_dim] 工具 K/V
        channel_ids: torch.Tensor | None = None,  # [B, M] 每个工具向量所属通道
        key_padding_mask: torch.Tensor | None = None,  # [B, M] True=valid
        gate: torch.Tensor | None = None,  # [B, 1] 或 [B, T, 1] 门控
    ) -> torch.Tensor:
        B, T, H = hidden.shape
        M = tool_vectors.shape[1]

        q = self.q_proj(self.norm_q(hidden))  # [B, T, H]
        kv_input = self.norm_kv(tool_vectors)
        if channel_ids is not None:
            kv_input = kv_input + self.channel_embed(channel_ids)
        k = self.k_proj(kv_input)  # [B, M, H]
        v = self.v_proj(kv_input)  # [B, M, H]

        # 多头 reshape
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, T, d]
        k = k.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, M, d]
        v = v.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, M, d]

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [B, h, T, M]
        if key_padding_mask is not None:
            # key_padding_mask: [B, M] True=valid. 转为 attention mask
            mask = (~key_padding_mask).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, M]
            attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B, h, T, d]
        out = out.transpose(1, 2).contiguous().view(B, T, H)
        out = self.out_proj(out)

        # 残差 (含可选门控)
        if gate is not None:
            out = out * gate
        hidden = hidden + out
        # FFN
        hidden = hidden + self.ffn(self.ffn_norm(hidden))
        return hidden


class CrossAttentionFusion(nn.Module):
    """Layer 8/16/24/32 插入的 Cross-Attention 融合层."""

    def __init__(self, config: CrossAttentionFusionConfig | None = None, **kwargs):
        super().__init__()
        cfg = config or CrossAttentionFusionConfig(**kwargs)
        self.cfg = cfg
        self.fusion_layers = list(cfg.fusion_layers)

        if cfg.shared_params:
            # 共享一个 CrossAttentionBlock
            self.shared_block = CrossAttentionBlock(
                cfg.hidden_dim, cfg.num_heads, cfg.tool_dim, cfg.dropout
            )
            self.layer_blocks = None
        else:
            # 每层独立
            self.layer_blocks = nn.ModuleDict({
                str(layer): nn.ModuleList([
                    CrossAttentionBlock(
                        cfg.hidden_dim, cfg.num_heads, cfg.tool_dim, cfg.dropout
                    )
                    for _ in range(cfg.num_cross_layers)
                ])
                for layer in self.fusion_layers
            })
            self.shared_block = None

        # 层级嵌入 (区分不同 fusion 层)
        self.layer_level_embed = nn.Embedding(
            max(self.fusion_layers) + 1, cfg.hidden_dim
        )

    # ------------------------------------------------------------------
    def _get_block(self, layer_idx: int) -> nn.ModuleList:
        if self.shared_block is not None:
            return nn.ModuleList([self.shared_block])
        return self.layer_blocks[str(layer_idx)]

    # ------------------------------------------------------------------
    def is_fusion_layer(self, layer_idx: int) -> bool:
        """判断指定层是否为融合插入层."""
        return layer_idx in self.fusion_layers

    # ------------------------------------------------------------------
    # 前向: 在指定层应用融合
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden: torch.Tensor,           # [B, T, H] 主干隐状态
        layer_idx: int,                  # 当前层索引
        tool_vectors: torch.Tensor | None = None,  # [B, M, tool_dim]
        channel_ids: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """在指定层应用 Cross-Attention 融合.

        若 layer_idx 不在 fusion_layers 中, 直接返回 hidden (无操作).
        若 tool_vectors 为 None, 也直接返回 (无工具输出时).
        """
        if not self.is_fusion_layer(layer_idx):
            return hidden
        if tool_vectors is None or tool_vectors.shape[1] == 0:
            return hidden

        h = hidden
        # 注入层级嵌入
        h = h + self.layer_level_embed(torch.tensor(layer_idx, device=h.device)).reshape(1, 1, -1)

        for block in self._get_block(layer_idx):
            h = block(
                h, tool_vectors, channel_ids, key_padding_mask, gate
            )
        return h

    # ------------------------------------------------------------------
    # 批量: 对所有融合层依次应用 (用于已展开的 hidden 列表)
    # ------------------------------------------------------------------
    def apply_all(
        self,
        hidden_states: list[torch.Tensor],
        tool_vectors: torch.Tensor | None = None,
        channel_ids: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        gates: dict | None = None,
    ) -> list[torch.Tensor]:
        """对主干各层隐状态列表应用融合.

        Args:
            hidden_states: 每层隐状态 [B, T, H] 的列表.
            tool_vectors: [B, M, tool_dim].
            gates: {layer_idx: gate_tensor} 每层门控.
        """
        out = []
        gates = gates or {}
        for i, h in enumerate(hidden_states):
            layer_idx = i + 1  # 1-indexed
            if self.is_fusion_layer(layer_idx):
                g = gates.get(layer_idx)
                h = self.forward(
                    h, layer_idx, tool_vectors, channel_ids,
                    key_padding_mask, g,
                )
            out.append(h)
        return out

    def extra_repr(self) -> str:
        return (
            f"fusion_layers={self.fusion_layers}, "
            f"hidden_dim={self.cfg.hidden_dim}, "
            f"num_heads={self.cfg.num_heads}, "
            f"shared_params={self.cfg.shared_params}"
        )
