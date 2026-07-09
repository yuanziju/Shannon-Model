"""激活函数模块.

提供 SwiGLU 与 GeGLU 门控线性单元实现, 常用于 MoE/FFN 专家的前馈网络.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU 门控线性单元.

    out = Swish(x @ W_gate) * (x @ W_up)
    其中 Swish(x) = x * sigmoid(x) = SiLU(x).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gate_proj = nn.Linear(in_features, out_features, bias=bias)
        self.up_proj = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.gate_proj(x)) * self.up_proj(x)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


class GeGLU(nn.Module):
    """GeGLU 门控线性单元.

    out = GELU(x @ W_gate) * (x @ W_up)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gate_proj = nn.Linear(in_features, out_features, bias=bias)
        self.up_proj = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.gate_proj(x)) * self.up_proj(x)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"
