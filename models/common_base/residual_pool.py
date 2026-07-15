"""ResidualPool — 共享残差池 (AttnRes + mHC + attention 检索).

提取 Shannon 与 MathMaster 共享的残差池组件:

  每轮迭代:
    1. 将当前 hidden 与池中残差 (仅 AB/循环残差) 堆叠
    2. AttnRes 注意力聚合 + mHC 流形约束聚合
    3. attention 检索: 用 hidden 查询池中残差, 取最相关的"有用笔记"
    4. 删除非 AB 残差 + 压缩 (线性投影降维)
    5. 每 pool_compress_every 轮: 注意力索引 + top-k 筛选

参考: MathMaster ResidualPool 设计, common.layers.AttnRes / mHC.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.layers import RMSNorm, AttnRes, mHC

from .base_config import BaseConfig


class ResidualPool(nn.Module):
    """残差池: AttnRes+mHC 约束 + attention 检索 + 删除压缩 + 每 N 轮 top-k.

    组件:
      * AttnRes (Kimi 深度方向块级注意力残差)
      * mHC (DeepSeek Sinkhorn-Knopp 双随机矩阵流形约束)
      * attention 检索 (hidden 查询池中残差, 取最相关"有用笔记")
      * 压缩投影 (删除非 AB 残差后压缩)
      * top-k 筛选 (每 pool_compress_every 轮)
    """

    def __init__(self, cfg: BaseConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim

        # AttnRes + mHC (从 common.layers 导入复用)
        self.attn_res = (
            AttnRes(d, num_blocks=cfg.attn_res_num_blocks, eps=cfg.rms_eps)
            if cfg.use_attn_res else None
        )
        self.mhc = (
            mHC(d, num_iters=cfg.mhc_num_iters)
            if cfg.use_mhc else None
        )

        # attention 检索 "有用笔记" (hidden x pool -> 相关性分数)
        self.note_query = nn.Linear(d, d, bias=False)
        self.note_key = nn.Linear(d, d, bias=False)
        self.note_scale = d ** -0.5

        # 压缩投影 (删除非 AB 残差后压缩)
        self.compress_proj = nn.Linear(d, d, bias=False)
        self.compress_norm = RMSNorm(d, eps=cfg.rms_eps)

        # top-k 筛选的可学习门控
        self.topk_gate = nn.Linear(d, 1, bias=False)

        self.topk = cfg.pool_topk
        self.compress_every = cfg.pool_compress_every

    def forward(
        self,
        hidden: torch.Tensor,
        ab_residuals: List[torch.Tensor],
        iteration: int,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """处理残差池.

        Args:
            hidden: [b, s, d] 当前隐藏状态.
            ab_residuals: 池中残差列表 (每项 [b, s, d]).
            iteration: 当前迭代索引.

        Returns:
            (pooled_hidden, new_ab_residuals, aux_loss)
        """
        b, s, d = hidden.shape
        aux_loss = hidden.new_zeros(())

        # --- 1. AttnRes + mHC 聚合残差 ---
        if len(ab_residuals) > 0 and self.attn_res is not None:
            stacked = torch.stack(ab_residuals, dim=-2)  # [b, s, L, d]
            attn_agg = self.attn_res(stacked)             # [b, s, d]
            hidden = hidden + attn_agg
        if len(ab_residuals) > 0 and self.mhc is not None:
            stacked = torch.stack(ab_residuals, dim=-2)  # [b, s, L, d]
            mhc_agg = self.mhc(stacked)                   # [b, s, L, d]
            # 取最后一层 (最近残差) 融合
            hidden = hidden + mhc_agg[..., -1, :]

        # --- 2. attention 检索 "有用笔记" ---
        if len(ab_residuals) > 0:
            q = self.note_query(hidden)                    # [b, s, d]
            keys = torch.stack(
                [self.note_key(r) for r in ab_residuals], dim=-2
            )                                               # [b, s, L, d]
            scores = torch.einsum("bsd,bsld->bsl", q, keys) * self.note_scale
            attn_weights = F.softmax(scores, dim=-1)       # [b, s, L]
            retrieved = torch.einsum(
                "bsl,bsld->bsd", attn_weights,
                torch.stack(ab_residuals, dim=-2),
            )
            hidden = hidden + retrieved

        # --- 3. 压缩 (投影 + 归一化) ---
        compressed = self.compress_norm(self.compress_proj(hidden))

        # --- 4. 每 compress_every 轮: top-k 筛选 ---
        if (iteration + 1) % self.compress_every == 0 and len(ab_residuals) > self.topk:
            gate_scores = torch.stack(
                [self.topk_gate(r).squeeze(-1).mean(dim=-1) for r in ab_residuals],
                dim=-1,
            )                                               # [b, L]
            k = min(self.topk, len(ab_residuals))
            _, top_idx = gate_scores.topk(k, dim=-1)       # [b, k]
            new_residuals: List[torch.Tensor] = []
            stacked_res = torch.stack(ab_residuals, dim=1)  # [b, L, s, d]
            for ki in range(k):
                idx_ki = top_idx[:, ki]                     # [b]
                gathered = stacked_res[torch.arange(b), idx_ki]  # [b, s, d]
                new_residuals.append(gathered)
            ab_residuals = new_residuals

        return hidden, ab_residuals, aux_loss

    def extra_repr(self) -> str:
        return (
            f"topk={self.topk}, compress_every={self.compress_every}, "
            f"attn_res={self.attn_res is not None}, mhc={self.mhc is not None}"
        )
