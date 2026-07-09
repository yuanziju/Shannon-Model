"""双层MoE路由器 (DualLayerRouter).

实现 16大专家 Top-4 + 16小专家 Top-4 的双层路由:
  1. 第一层: 从 num_big_experts 个大专家中选 Top-k_big 个
  2. 第二层: 从 num_small_experts 个小专家中选 Top-k_small 个
  3. 常驻共享专家: 始终激活

路由权重经 softmax 归一化, 训练时加噪声 (load balancing).
参考: AGENTS.md Agent 9, spec 双层MoE Top-4×Top-4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RouterOutput:
    """路由器输出容器."""

    # 大专家路由 [N, k_big]: 选中的大专家索引
    big_indices: torch.Tensor
    # 大专家权重 [N, k_big]
    big_weights: torch.Tensor
    # 小专家路由 [N, k_small]
    small_indices: torch.Tensor
    # 小专家权重 [N, k_small]
    small_weights: torch.Tensor
    # 全量路由概率 (用于 load balance loss)
    big_scores: torch.Tensor       # [N, num_big]
    small_scores: torch.Tensor     # [N, num_small]
    # 负载均衡辅助损失
    aux_loss: torch.Tensor


class DualLayerRouter(nn.Module):
    """双层路由器: 16 Top-4 (大) + 16 Top-4 (小).

    大专家(粗粒度)与小专家(细粒度)独立路由, 输出加权合并.
    常驻共享专家不参与路由, 由调用方单独处理.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_big_experts: int = 16,
        num_small_experts: int = 16,
        top_k_big: int = 4,
        top_k_small: int = 4,
        noise_std: float = 1.0,
        load_balance_alpha: float = 0.01,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_big_experts = num_big_experts
        self.num_small_experts = num_small_experts
        self.top_k_big = min(top_k_big, num_big_experts)
        self.top_k_small = min(top_k_small, num_small_experts)
        self.noise_std = noise_std
        self.load_balance_alpha = load_balance_alpha

        # 路由投影 (token -> expert logits)
        self.big_router = nn.Linear(hidden_dim, num_big_experts, bias=False)
        self.small_router = nn.Linear(hidden_dim, num_small_experts, bias=False)
        nn.init.normal_(self.big_router.weight, std=0.02)
        nn.init.normal_(self.small_router.weight, std=0.02)

    def _route_single(
        self,
        x_flat: torch.Tensor,
        router: nn.Linear,
        num_experts: int,
        top_k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """单层路由: 返回 (indices [N,k], weights [N,k], scores [N,E])."""
        logits = router(x_flat)  # [N, E]
        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std
        scores = F.softmax(logits, dim=-1)  # [N, E]
        k = min(top_k, num_experts)
        topk_scores, topk_idx = scores.topk(k, dim=-1)  # [N, k]
        # 重归一化
        topk_weights = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-9)
        return topk_idx, topk_weights, scores

    def _load_balance_loss(
        self,
        big_scores: torch.Tensor,
        small_scores: torch.Tensor,
    ) -> torch.Tensor:
        """计算负载均衡辅助损失.

        L = E_b * (mean(f_b) * mean(p_b)) + E_s * (mean(f_s) * mean(p_s))
        其中 f = 每个专家被选中的 token 比例, p = 每个专家的平均路由概率.
        """
        loss = torch.tensor(0.0, device=big_scores.device)
        for scores, k in [(big_scores, self.top_k_big), (small_scores, self.top_k_small)]:
            N, E = scores.shape
            # 每个专家被选中的 token 数比例
            topk_idx = scores.topk(min(k, E), dim=-1).indices  # [N, k]
            mask = F.one_hot(topk_idx, E).float().sum(dim=1)   # [N, E]
            frac = mask.mean(dim=0)                              # [E]
            # 每个专家的平均路由概率
            prob = scores.mean(dim=0)                            # [E]
            loss = loss + E * (frac * prob).sum()
        return self.load_balance_alpha * loss

    def forward(self, x: torch.Tensor) -> RouterOutput:
        """路由 token 到大/小专家.

        Args:
            x: [B, S, H] 或 [N, H].

        Returns:
            RouterOutput.
        """
        orig_shape = x.shape
        if x.dim() == 3:
            B, S, H = x.shape
            x_flat = x.reshape(B * S, H)
        else:
            x_flat = x

        big_idx, big_w, big_scores = self._route_single(
            x_flat, self.big_router, self.num_big_experts, self.top_k_big
        )
        small_idx, small_w, small_scores = self._route_single(
            x_flat, self.small_router, self.num_small_experts, self.top_k_small
        )
        aux = self._load_balance_loss(big_scores, small_scores)

        return RouterOutput(
            big_indices=big_idx,
            big_weights=big_w,
            small_indices=small_idx,
            small_weights=small_w,
            big_scores=big_scores,
            small_scores=small_scores,
            aux_loss=aux,
        )

    def extra_repr(self) -> str:
        return (
            f"big={self.num_big_experts}x{self.top_k_big}, "
            f"small={self.num_small_experts}x{self.top_k_small}"
        )
