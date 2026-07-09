"""归一化层模块.

提供标准 RMSNorm 与 DeepNorm 风格门控 RMSNorm 实现.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """标准 RMS 归一化.

    计算: y = (x / sqrt(mean(x^2) + eps)) * weight
    使用 float32 计算方差以保证数值稳定性, 输出恢复输入 dtype.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        out = self._norm(x)
        return (self.weight * out).to(input_dtype)

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, eps={self.eps}"


class GatedRMSNorm(nn.Module):
    """DeepNorm 风格门控 RMSNorm.

    在 RMSNorm 之上引入可学习门控, 自适应控制归一化强度:
        y = gate * RMSNorm(x) + (1 - gate) * x
    其中 gate = sigmoid(W_g · x), 由小线性层从输入计算得到.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        # 门控初始化偏向于使用归一化路径 (gate -> 1)
        nn.init.zeros_(self.gate_proj.weight)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        normed = self._norm(x_fp32)
        normed = (self.weight * normed).to(input_dtype)
        gate = torch.sigmoid(self.gate_proj(x))
        return gate * normed + (1.0 - gate) * x

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, eps={self.eps}"
