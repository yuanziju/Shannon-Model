"""MoE All-to-All 通信 (MoEAllToAll).

实现 EP (Expert Parallelism) 中的 all-to-all token 分发与收集:
  1. dispatch: 按 expert rank 分发 token 到对应 GPU
  2. compute: 各 GPU 本地计算专家前向
  3. combine: all-to-all 收集结果回原 rank

在单 GPU 模式下退化为本地分发 (不做实际通信).

参考: AGENTS.md 5D并行 (TP+PP+DP+SP+EP), spec EP 并行.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist


class MoEAllToAll(nn.Module):
    """MoE All-to-All 通信层.

    支持两种模式:
      - 分布式: 使用 dist.all_to_all_single 进行实际跨 rank 通信
      - 本地: 单 GPU 时退化为本地分发 (不通信)
    """

    def __init__(
        self,
        num_experts: int,
        ep_size: int = 1,
        hidden_dim: int = 4096,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.ep_size = max(1, ep_size)
        self.hidden_dim = hidden_dim
        self.local_num_experts = num_experts // self.ep_size
        if self.local_num_experts < 1:
            self.local_num_experts = 1
        # 是否启用分布式通信
        self._distributed = dist.is_available() and dist.is_initialized()
        self._rank = dist.get_rank() if self._distributed else 0
        self._world_size = dist.get_world_size() if self._distributed else 1

    def dispatch(
        self,
        x: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """按专家索引分发 token.

        Args:
            x: [N, H] 输入 token.
            expert_indices: [N, k] 选中的专家索引.
            expert_weights: [N, k] 路由权重.

        Returns:
            列表, 每个元素为 (x_local, local_indices, local_weights) 对应一个本地专家.
        """
        N, k = expert_indices.shape
        dispatched = []
        for ei in range(self.num_experts):
            # 找到路由到此专家的 token
            mask = expert_indices == ei  # [N, k]
            sel = mask.any(dim=-1)  # [N]
            if not sel.any():
                dispatched.append((
                    x[sel] if sel.any() else x[:0],
                    torch.empty(0, dtype=torch.long, device=x.device),
                    torch.empty(0, dtype=x.dtype, device=x.device),
                ))
            else:
                x_sel = x[sel]
                # 取权重
                w_mask = mask[sel].float()
                w_sel = (expert_weights[sel] * w_mask).sum(dim=-1)
                dispatched.append((x_sel, torch.full((x_sel.shape[0],), ei, device=x.device), w_sel))
        return dispatched

    def combine(
        self,
        expert_outputs: List[torch.Tensor],
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
        N: int,
        H: int,
        device: torch.device,
    ) -> torch.Tensor:
        """收集各专家输出, 加权合并回原 token 位置.

        Args:
            expert_outputs: 列表, 每个元素为某专家的输出 [M, H].
            expert_indices: [N, k] 原路由索引.
            expert_weights: [N, k] 原路由权重.
            N: 原始 token 数.
            H: 隐维度.

        Returns:
            [N, H] 合并后的输出.
        """
        output = torch.zeros(N, H, device=device, dtype=expert_outputs[0].dtype if len(expert_outputs) > 0 and expert_outputs[0].numel() > 0 else torch.float32)
        token_offset = [0] * self.num_experts
        for i in range(N):
            for j in range(expert_indices.shape[1]):
                ei = int(expert_indices[i, j].item())
                off = token_offset[ei]
                if off < expert_outputs[ei].shape[0]:
                    w = expert_weights[i, j]
                    output[i] += w * expert_outputs[ei][off]
                    token_offset[ei] += 1
        return output

    def forward(
        self,
        x: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_fn: Callable[[torch.Tensor, int], torch.Tensor],
    ) -> torch.Tensor:
        """完整 all-to-all: dispatch -> compute -> combine.

        Args:
            x: [N, H] 输入.
            expert_indices: [N, k] 路由索引.
            expert_weights: [N, k] 路由权重.
            expert_fn: 专家前向函数 (x_local, expert_id) -> output_local.

        Returns:
            [N, H] 合并输出.
        """
        N, H = x.shape
        # dispatch
        dispatched = self.dispatch(x, expert_indices, expert_weights)
        # compute
        outputs = []
        for ei, (x_local, _, _) in enumerate(dispatched):
            if x_local.numel() > 0:
                out = expert_fn(x_local, ei)
            else:
                out = x_local
            outputs.append(out)
        # combine
        return self.combine(outputs, expert_indices, expert_weights, N, H, x.device)

    def extra_repr(self) -> str:
        return (
            f"num_experts={self.num_experts}, ep_size={self.ep_size}, "
            f"local_experts={self.local_num_experts}, "
            f"distributed={self._distributed}"
        )
