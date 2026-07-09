"""残差连接模块.

提供 Kimi AttnRes (深度方向块级注意力残差) 与 DeepSeek mHC
(Sinkhorn-Knopp 双随机矩阵流形约束超链接) 实现.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn

from .norm import RMSNorm


class AttnRes(nn.Module):
    """Kimi AttnRes (深度方向注意力残差).

    标准残差:  h_L = h_1 + sum_{i=1}^{L-1} f_i(h_i)
    AttnRes:   h_l = sum_{i=0}^{l-1} alpha_{i->l} * f_i(h_i)
      其中 alpha_{i->l} = softmax_l( w^T · RMSNorm(f_i(h_i)) )

    Block AttnRes: 将 L 层分为 N 个 block, block 内标准求和,
    block 间使用注意力聚合 (训练开销 < 4%, 推理延迟 < 2%).
    """

    def __init__(self, hidden_size: int, num_blocks: int = 8, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.eps = eps
        self.norm = RMSNorm(hidden_size, eps=eps)
        self.score_proj = nn.Linear(hidden_size, 1, bias=False)
        nn.init.zeros_(self.score_proj.weight)

    def _stack(self, hidden_states: Union[Sequence[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        if isinstance(hidden_states, (list, tuple)):
            return torch.stack(list(hidden_states), dim=-2)
        return hidden_states

    def forward(
        self,
        hidden_states: Union[Sequence[torch.Tensor], torch.Tensor],
        block_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """聚合来自前序层的隐藏状态.

        hidden_states: 前序层输出列表 [(..., hidden)] 或张量 (..., L, hidden).
        block_ids: (L,) 每层所属 block id, 提供时启用 block 级聚合
                   (block 内均匀求和, block 间注意力).
        返回聚合后的 (..., hidden).
        """
        h = self._stack(hidden_states)  # (..., L, hidden)
        L = h.shape[-2]

        normed = self.norm(h)  # (..., L, hidden)
        scores = self.score_proj(normed).squeeze(-1)  # (..., L)

        if block_ids is None or self.num_blocks <= 1:
            # 全局注意力聚合
            weights = torch.softmax(scores.float(), dim=-1).to(h.dtype)
        else:
            # Block AttnRes: block 内均匀, block 间注意力
            weights = self._block_weights(scores, block_ids, L, h.dtype)

        out = (weights.unsqueeze(-1) * h).sum(dim=-2)  # (..., hidden)
        return out

    def _block_weights(
        self,
        scores: torch.Tensor,  # (..., L)
        block_ids: torch.Tensor,  # (L,)
        L: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        device = scores.device
        block_ids = block_ids.to(device)
        num_blocks = int(block_ids.max().item()) + 1
        # 收集每个 block 的聚合分数 (block 内分数均值), 用于 block 间 softmax
        block_agg = []
        for b in range(num_blocks):
            mask = block_ids == b
            if not mask.any():
                block_agg.append(torch.zeros(scores.shape[:-1], device=device, dtype=torch.float32))
            else:
                block_agg.append(scores[..., mask].float().mean(dim=-1))
        block_logits = torch.stack(block_agg, dim=-1)  # (..., num_blocks)
        block_attn = torch.softmax(block_logits, dim=-1)  # (..., num_blocks)
        # 将 block 注意力均分到 block 内各层
        weights = torch.zeros_like(scores)
        for b in range(num_blocks):
            mask = block_ids == b
            cnt = int(mask.sum().item())
            if cnt == 0:
                continue
            weights[..., mask] = block_attn[..., b].unsqueeze(-1) / cnt
        return weights.to(dtype)

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, num_blocks={self.num_blocks}"


class mHC(nn.Module):
    """DeepSeek mHC (流形约束超链接).

    将残差连接映射投影到 Birkhoff polytope (双随机矩阵集合),
    通过 Sinkhorn-Knopp 迭代 (默认 20 次) 得到行/列和均为 1 的非负矩阵 M.
    双随机矩阵的谱范数 <= 1, 从而保证深层网络信号不爆炸.

    计算 (spec §4.11):
        H_raw = alpha * tanh(x_proj @ x^T) + b
        M = exp(H_raw)
        for _ in range(num_iters):
            M = M / row_sum(M)   # 行归一化
            M = M / col_sum(M)   # 列归一化
        out = M @ x              # 用双随机矩阵聚合各层

    映射计算保持 float32 以确保数值稳定性.
    """

    def __init__(self, hidden_size: int, num_iters: int = 20, eps: float = 1e-12):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_iters = num_iters
        self.eps = eps
        self.theta = nn.Linear(hidden_size, hidden_size, bias=False)
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def _sinkhorn(self, M: torch.Tensor) -> torch.Tensor:
        """对 (..., L, L) 矩阵做 Sinkhorn-Knopp 双随机投影."""
        M = M / M.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        for _ in range(self.num_iters):
            M = M / M.sum(dim=-1, keepdim=True).clamp_min(self.eps)  # 行归一化
            M = M / M.sum(dim=-2, keepdim=True).clamp_min(self.eps)  # 列归一化
        return M

    def forward(self, hidden_states: Union[Sequence[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """对各层隐藏状态做流形约束聚合.

        hidden_states: 层输出列表 [(..., hidden)] 或张量 (..., L, hidden).
        返回聚合后的 (..., L, hidden) (每层被重新混合).
        """
        if isinstance(hidden_states, (list, tuple)):
            x = torch.stack(list(hidden_states), dim=-2)
        else:
            x = hidden_states  # (..., L, hidden)

        x_fp32 = x.to(torch.float32)
        proj = self.theta(x_fp32)  # (..., L, hidden)
        # 层间交互: (..., L, hidden) x (..., hidden, L) -> (..., L, L)
        H_raw = self.alpha * torch.tanh(torch.matmul(x_fp32, proj.transpose(-1, -2))) + self.bias
        # 数值稳定的 exp
        H_raw = H_raw - H_raw.max(dim=-1, keepdim=True).values.detach()
        M = torch.exp(H_raw)
        M = self._sinkhorn(M)  # (..., L, L) 双随机

        out = torch.matmul(M, x_fp32)  # (..., L, hidden)
        return out.to(x.dtype)

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, num_iters={self.num_iters}"
