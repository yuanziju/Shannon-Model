"""双层MoE专家实现.

专家类型:
  - StandardExpert: 标准FFN专家 (SwiGLU)
  - BigExpert: 粗粒度大专家 (大FFN, 可选NLM增强)
  - SmallExpert: 细粒度小专家 (小FFN)
  - SharedExpert: 常驻共享专家 (DeepSeek模式, 始终激活)

参考: AGENTS.md Agent 9 (MoEAgent), spec 双层MoE设计.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.layers import RMSNorm, SwiGLU
from common.ctm import NLMLayer


class StandardExpert(nn.Module):
    """标准FFN专家: SwiGLU 门控前馈网络.

    计算: out = down_proj(silu(gate_proj(x)) * up_proj(x))
    """

    def __init__(self, hidden_dim: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.gate_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._init_weights()

    def _init_weights(self):
        for m in [self.gate_proj, self.up_proj]:
            nn.init.kaiming_uniform_(m.weight, a=5.0 / 3)
        nn.init.zeros_(self.down_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.gate_proj(x)) * self.up_proj(x)
        h = self.dropout(h)
        return self.down_proj(h)

    def extra_repr(self) -> str:
        return f"hidden={self.hidden_dim}, ffn={self.ffn_dim}"


class BigExpert(nn.Module):
    """粗粒度大专家 (16个大专家之一).

    使用更大的FFN维度, 可选CTM NLM增强激活函数 (决策C7/C10:
    NLM仅增强MoE专家内激活函数, 不主导状态转移).

    分层归属: 浅层8 / 中层16 / 深层24 (通过 layer_idx 配置).
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        use_nlm: bool = False,
        nlm_num_neurons: int = 8,
        nlm_d_state: int = 16,
        nlm_warmup_freeze: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.use_nlm = use_nlm and nlm_num_neurons > 0
        self.ffn = StandardExpert(hidden_dim, ffn_dim, dropout=dropout)
        self.norm = RMSNorm(hidden_dim)
        if self.use_nlm:
            self.nlm = NLMLayer(
                d_model=hidden_dim,
                num_neurons=nlm_num_neurons,
                d_state=nlm_d_state,
                warmup_freeze=nlm_warmup_freeze,
            )
        else:
            self.nlm = None

    def forward(
        self, x: torch.Tensor, nlm_states: Optional[list] = None
    ) -> tuple:
        """前向计算.

        Args:
            x: [N, hidden_dim] 被 dispatch 到此专家的 token.
            nlm_states: 上一 tick 的 NLM 神经元状态列表 (可选).

        Returns:
            (output [N, hidden_dim], new_nlm_states or None).
        """
        h = self.ffn(x)
        if self.nlm is not None:
            h_nlm, new_states = self.nlm(h, nlm_states)
            h = self.norm(h + h_nlm)
            return h, new_states
        return h, None

    def extra_repr(self) -> str:
        return (
            f"hidden={self.hidden_dim}, ffn={self.ffn_dim}, "
            f"nlm={self.use_nlm}"
        )


class SmallExpert(nn.Module):
    """细粒度小专家 (16个小专家之一).

    使用更小的FFN维度, 不使用NLM增强 (决策C10: 仅实体大专家使用NLM).
    """

    def __init__(self, hidden_dim: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.ffn = StandardExpert(hidden_dim, ffn_dim, dropout=dropout)
        self.norm = RMSNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.ffn(x))

    def extra_repr(self) -> str:
        return f"hidden={self.hidden_dim}, ffn={self.ffn_dim}"


class SharedExpert(nn.Module):
    """常驻共享专家 (DeepSeek模式).

    始终激活, 捕获通用知识, 不参与路由.
    """

    def __init__(self, hidden_dim: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.ffn = StandardExpert(hidden_dim, ffn_dim, dropout=dropout)
        self.norm = RMSNorm(hidden_dim)
        self.gate = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.gate * self.ffn(x))

    def extra_repr(self) -> str:
        return f"hidden={self.hidden_dim}, ffn={self.ffn_dim}, gate={float(self.gate):.3f}"
