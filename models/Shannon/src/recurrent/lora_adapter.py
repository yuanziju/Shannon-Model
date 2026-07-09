"""深度 LoRA 适配器 (Depth-Dependent LoRA).

为循环主体的每次迭代提供低秩适配, 使同一组循环权重在不同深度
表现出不同行为 (spec: 深度LoRA, rank=32).

参考: latent_decode.mode_switch.LoRALinear 的设计, 但此处按深度索引
      而非按模式索引, 为循环迭代提供深度特异化的低秩增量.

决策: 深度LoRA 复用循环主体权重, 不引入独立网络.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthLoRALinear(nn.Module):
    """深度相关的 LoRA 线性层.

    包装一个基础线性层, 并为每个循环深度提供独立的低秩增量:
      out = base(x) + sum_d indicator(depth==d) * scaling_d * B_d @ A_d @ x

    为节省参数, A 跨深度共享, 仅 B 按深度独立; 同时维护一个深度门控
    gate[depth] 控制该深度 LoRA 的强度.

    Args:
        in_features: 输入维度.
        out_features: 输出维度.
        max_depths: 最大循环深度数 (dynamic_iterations[1]).
        rank: LoRA 秩.
        alpha: LoRA 缩放系数 (scaling = alpha / rank).
        dropout: LoRA dropout.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        max_depths: int = 32,
        rank: int = 32,
        alpha: float = 32.0,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_depths = max(1, int(max_depths))
        self.rank = max(1, int(rank))
        self.scaling = float(alpha) / float(self.rank)

        # 基础线性层
        self.base = nn.Linear(in_features, out_features, bias=bias)

        # 共享 A: [rank, in_features]
        self.lora_A = nn.Parameter(torch.zeros(self.rank, in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # 深度独立 B: [max_depths, out_features, rank]
        self.lora_B = nn.Parameter(torch.zeros(self.max_depths, out_features, self.rank))

        # 深度门控: 控制每个深度 LoRA 强度 (sigmoid -> (0,1))
        self.depth_gate = nn.Parameter(torch.zeros(self.max_depths))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        depth: int = 0,
    ) -> torch.Tensor:
        """前向计算.

        Args:
            x: [..., in_features].
            depth: 当前循环深度 (0-indexed), 用于选择对应 LoRA.

        Returns:
            [..., out_features].
        """
        out = self.base(x)
        depth = min(max(depth, 0), self.max_depths - 1)

        # 低秩增量: B_d @ A @ x
        d = self.dropout(x)
        # A: [rank, in] -> d @ A^T: [..., rank]
        h = F.linear(d, self.lora_A)
        # B_d: [out, rank] -> h @ B_d^T: [..., out]
        b_d = self.lora_B[depth]
        delta = F.linear(h, b_d)
        # 深度门控
        gate = torch.sigmoid(self.depth_gate[depth])
        return out + gate * self.scaling * delta

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.rank}, max_depths={self.max_depths}, "
            f"scaling={self.scaling:.4f}"
        )


class DepthLoRAAdapter(nn.Module):
    """深度 LoRA 适配器集合.

    为循环主体的多个子模块 (Q/K/V/O 投影, FFN 等) 提供深度相关 LoRA,
    统一管理 depth 参数的传递.

    用法:
        adapter = DepthLoRAAdapter(hidden_dim, max_depths, rank)
        # 在循环中
        q = adapter.apply("q_proj", x, depth)
    """

    def __init__(
        self,
        hidden_dim: int,
        max_depths: int = 32,
        rank: int = 32,
        alpha: float = 32.0,
        dropout: float = 0.0,
        num_adapted: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_depths = max_depths
        self.num_adapted = num_adapted

        self.adapters = nn.ModuleDict({
            f"adapter_{i}": DepthLoRALinear(
                hidden_dim, hidden_dim,
                max_depths=max_depths,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
            for i in range(num_adapted)
        })

    def apply(
        self,
        name: str,
        x: torch.Tensor,
        depth: int = 0,
    ) -> torch.Tensor:
        """对输入应用指定名称的深度 LoRA 适配."""
        if name not in self.adapters:
            return x
        return self.adapters[name](x, depth=depth)

    def forward(
        self,
        x: torch.Tensor,
        depth: int = 0,
        name: str = "adapter_0",
    ) -> torch.Tensor:
        """便捷前向: 应用单个适配器."""
        return self.apply(name, x, depth)

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, max_depths={self.max_depths}, "
            f"num_adapted={self.num_adapted}"
        )


def apply_depth_lora(
    base_module: nn.Module,
    x: torch.Tensor,
    lora_layer: Optional[DepthLoRALinear],
    depth: int = 0,
) -> torch.Tensor:
    """工具函数: 对任意基础模块的输出叠加深度 LoRA 增量.

    Args:
        base_module: 基础模块 (如 nn.Linear), 已对 x 计算完毕或将被调用.
        x: 输入张量.
        lora_layer: 深度 LoRA 层 (None 则仅返回 base 输出).
        depth: 循环深度.

    Returns:
        base(x) + lora(x, depth).
    """
    if lora_layer is None:
        return base_module(x)
    # 复用 base_module 计算, 再叠加 lora 增量
    out = base_module(x)
    depth = min(max(depth, 0), lora_layer.max_depths - 1)
    d = lora_layer.dropout(x)
    h = F.linear(d, lora_layer.lora_A)
    b_d = lora_layer.lora_B[depth]
    delta = F.linear(h, b_d)
    gate = torch.sigmoid(lora_layer.depth_gate[depth])
    return out + gate * lora_layer.scaling * delta
