"""MoE 负载均衡器 (LoadBalancer).

实现专家级与专家类别级的负载均衡:
  1. 专家级: 确保每个专家被路由的 token 数均衡
  2. 容量管理: 限制每个专家的最大 token 数 (防过载/饥饿)
  3. 辅助损失: 负载均衡 aux loss

参考: AGENTS.md Agent 9, spec 负载均衡与专家容量管理.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoadBalancer(nn.Module):
    """MoE 负载均衡器.

    计算:
      - 负载均衡损失: L_aux = alpha * E * sum(f_i * p_i)
        其中 f_i = 被路由到专家i的token比例, p_i = 专家i的平均路由概率
      - 容量因子: 每个专家的最大 token 数 = capacity_factor * (N / E)
      - 溢出处理: 超出容量的 token 被丢弃或路由到共享专家
    """

    def __init__(
        self,
        num_experts: int,
        capacity_factor: float = 1.25,
        alpha: float = 0.01,
        balance_type: str = "expert",  # "expert" | "category"
    ):
        super().__init__()
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.alpha = alpha
        self.balance_type = balance_type

    def compute_aux_loss(
        self,
        routing_scores: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        """计算负载均衡辅助损失.

        Args:
            routing_scores: [N, E] 每个token到每个专家的路由概率.
            topk_indices: [N, k] 每个token选中的专家索引.

        Returns:
            标量辅助损失.
        """
        N, E = routing_scores.shape
        if N == 0:
            return torch.tensor(0.0, device=routing_scores.device)
        # 每个专家被选中的 token 数比例
        mask = F.one_hot(topk_indices, E).float().sum(dim=1)  # [N, E]
        frac = mask.mean(dim=0)  # [E]
        # 每个专家的平均路由概率
        prob = routing_scores.mean(dim=0)  # [E]
        # L_aux = E * sum(f_i * p_i)
        aux_loss = E * (frac * prob).sum()
        return self.alpha * aux_loss

    def compute_capacity(
        self, num_tokens: int, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """计算每个专家的容量 (最大 token 数).

        Args:
            num_tokens: 总 token 数.

        Returns:
            [E] 每个专家的容量.
        """
        capacity = int(self.capacity_factor * num_tokens / max(self.num_experts, 1))
        capacity = max(capacity, 1)
        return torch.full(
            (self.num_experts,), capacity,
            dtype=torch.long, device=device,
        )

    def apply_capacity(
        self,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        capacities: torch.Tensor,
    ) -> tuple:
        """应用容量限制: 超出容量的 token 被丢弃.

        Args:
            topk_indices: [N, k] 选中的专家索引.
            topk_weights: [N, k] 路由权重.
            capacities: [E] 每个专家容量.

        Returns:
            (mask [N, k] bool, 溢出 token 数).
        """
        N, k = topk_indices.shape
        mask = torch.ones(N, k, dtype=torch.bool, device=topk_indices.device)
        # 统计每个专家的当前负载
        counts = torch.zeros(self.num_experts, dtype=torch.long, device=topk_indices.device)
        overflow = 0
        for i in range(N):
            for j in range(k):
                ei = int(topk_indices[i, j].item())
                if counts[ei] < capacities[ei]:
                    counts[ei] += 1
                else:
                    mask[i, j] = False
                    overflow += 1
        return mask, overflow

    def forward(
        self,
        routing_scores: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: Optional[torch.Tensor] = None,
        apply_cap: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """前向: 计算辅助损失, 可选应用容量限制.

        Returns:
            dict 含 aux_loss, capacities (可选), overflow (可选), mask (可选).
        """
        aux_loss = self.compute_aux_loss(routing_scores, topk_indices)
        result = {"aux_loss": aux_loss}
        if apply_cap and topk_weights is not None:
            N = topk_indices.shape[0]
            caps = self.compute_capacity(N, topk_indices.device)
            mask, overflow = self.apply_capacity(topk_indices, topk_weights, caps)
            result["capacities"] = caps
            result["overflow"] = overflow
            result["capacity_mask"] = mask
        return result

    def extra_repr(self) -> str:
        return (
            f"num_experts={self.num_experts}, "
            f"capacity_factor={self.capacity_factor}, "
            f"alpha={self.alpha}, type={self.balance_type}"
        )
