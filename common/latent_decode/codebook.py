"""神经码本 (NeuroCodebook).

将神经语空间的连续表示离散化为码本向量, 同时区分连续主表示与 5 种边界类型
(句子边界 / 段落边界 / 章节边界 / 文档边界 / 续接). 采用 Straight-Through
Estimator (STE) 保证梯度回传, 码本向量使用 EMA 在线更新 (VQ-VAE 风格).

设计参考: spec.md §14.3 决策 L (隐空间解码), B+C 融合架构.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# 5 种边界类型 + 1 个连续(主)类型, 共 6 个语义槽位.
BOUNDARY_TYPES: Tuple[str, ...] = (
    "continuous",     # 连续主表示 (无边界)
    "sentence_end",   # 句子边界
    "paragraph_end",  # 段落边界
    "section_end",    # 章节边界
    "document_end",   # 文档边界
    "continuation",   # 跨片段续接
)
NUM_BOUNDARY_TYPES = len(BOUNDARY_TYPES)


@dataclass
class CodebookConfig:
    """神经码本配置."""

    codebook_size: int = 8192          # 码本条目数
    latent_dim: int = 1024             # 单条码本向量维度
    num_boundary_types: int = NUM_BOUNDARY_TYPES
    ema_decay: float = 0.99            # EMA 衰减系数
    ema_eps: float = 1e-5              # EMA 数值稳定项
    commitment_beta: float = 0.25      # commitment loss 权重
    ste: bool = True                   # 是否使用 Straight-Through Estimator
    threshold_ema_dead_code: float = 1e-9  # 死码复活阈值
    reset_usage: int = 32              # 每多少步重置 usage 计数


class NeuroCodebook(nn.Module):
    """神经码本: 连续 + 5 种边界类型, STE 反传, EMA 更新.

    每个边界类型共享同一张码表, 但通过额外的边界类型嵌入 (type embedding)
    偏置进行区分, 避免边界符号占据主码本过多容量.
    """

    def __init__(self, config: CodebookConfig | None = None, **kwargs):
        super().__init__()
        cfg = config or CodebookConfig(**kwargs)
        self.cfg = cfg

        # 主码本: [codebook_size, latent_dim]
        self.codebook = nn.Parameter(
            torch.empty(cfg.codebook_size, cfg.latent_dim)
        )
        nn.init.kaiming_uniform_(self.codebook, a=5.0 / 3)

        # 边界类型嵌入: 区分 continuous / 5 种边界
        self.boundary_embed = nn.Parameter(
            torch.empty(cfg.num_boundary_types, cfg.latent_dim)
        )
        nn.init.normal_(self.boundary_embed, std=0.02)

        # EMA 更新所需的统计量 (非梯度缓冲区)
        self.register_buffer(
            "ema_cluster_size", torch.zeros(cfg.codebook_size)
        )
        self.register_buffer(
            "ema_weight", self.codebook.data.clone()
        )
        self.register_buffer(
            "usage_count", torch.zeros(cfg.codebook_size, dtype=torch.long)
        )
        self.register_buffer("step", torch.zeros(1, dtype=torch.long))
        # 死码累计计数, 用于复活
        self.register_buffer(
            "dead_count", torch.zeros(cfg.codebook_size, dtype=torch.long)
        )

    # ------------------------------------------------------------------
    # 前向量化
    # ------------------------------------------------------------------
    def forward(
        self,
        z: torch.Tensor,
        boundary_type: torch.Tensor | None = None,
        return_loss: bool = True,
    ) -> dict:
        """将连续 latent z 量化为码本向量.

        Args:
            z: [..., latent_dim] 连续表示.
            boundary_type: [...,] long 张量, 每个位置的边界类型索引.
                若为 None, 默认为 continuous (0).
            return_loss: 是否计算 commitment loss.

        Returns:
            dict 含 quantized / indices / commit_loss / boundary_bias.
        """
        orig_shape = z.shape
        flat = z.reshape(-1, self.cfg.latent_dim)

        # 计算到每个码字的距离 (欧氏距离平方)
        dist = (
            flat.pow(2).sum(dim=-1, keepdim=True)
            - 2 * flat @ self.codebook.t()
            + self.codebook.pow(2).sum(dim=-1)
        )
        indices = dist.argmin(dim=-1)
        quantized = self.codebook[indices]  # [N, D]

        # STE: 前向用 quantized, 反向用 flat
        if self.cfg.ste:
            quantized_ste = flat + (quantized - flat).detach()
        else:
            quantized_ste = quantized

        # 边界类型偏置
        if boundary_type is None:
            btype = torch.zeros(
                flat.shape[0], dtype=torch.long, device=z.device
            )
        else:
            btype = boundary_type.reshape(-1).to(torch.long)
        boundary_bias = self.boundary_embed[btype]  # [N, D]
        quantized_final = quantized_ste + boundary_bias

        # EMA 更新 (仅在训练阶段)
        if self.training:
            with torch.no_grad():
                self._ema_update(flat, indices)

        # commitment loss (仅引导编码器输出靠近码本)
        commit_loss = torch.tensor(0.0, device=z.device)
        if return_loss:
            commit_loss = self.cfg.commitment_beta * F.mse_loss(
                quantized_final, flat.detach()
            )

        self.step += 1
        return {
            "quantized": quantized_final.reshape(orig_shape),
            "indices": indices.reshape(orig_shape[:-1]),
            "commit_loss": commit_loss,
            "boundary_bias": boundary_bias.reshape(*orig_shape),
        }

    # ------------------------------------------------------------------
    # EMA 更新
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _ema_update(self, flat: torch.Tensor, indices: torch.Tensor) -> None:
        """VQ-VAE 风格 EMA 码本更新 + 死码复活."""
        one_hot = F.one_hot(
            indices, num_classes=self.cfg.codebook_size
        ).float()  # [N, K]
        cluster_size = one_hot.sum(dim=0)  # [K]

        # EMA 更新 cluster size
        self.ema_cluster_size.mul_(self.cfg.ema_decay).add_(
            cluster_size, alpha=1 - self.cfg.ema_decay
        )
        # Laplace 平滑
        smoothed = (
            self.ema_cluster_size + self.cfg.ema_eps
        ) / (
            self.ema_cluster_size.sum() + self.cfg.codebook_size * self.cfg.ema_eps
        )
        # 更新 ema_weight
        embed_sum = one_hot.t() @ flat  # [K, D]
        self.ema_weight.mul_(self.cfg.ema_decay).add_(
            embed_sum, alpha=1 - self.cfg.ema_decay
        )
        normalized = self.ema_weight / smoothed.unsqueeze(-1)
        self.codebook.data.copy_(normalized)

        # usage 统计
        self.usage_count.scatter_add_(
            0, indices, torch.ones_like(indices)
        )

        # 死码复活: 长期未使用的码字重新初始化为随机输入
        dead_mask = self.ema_cluster_size < self.cfg.threshold_ema_dead_code
        if dead_mask.any():
            self.dead_count[dead_mask] += 1
            num_dead = int(dead_mask.sum().item())
            if num_dead > 0 and flat.shape[0] > 0:
                # 从当前 batch 中随机采样替换
                rand_idx = torch.randint(
                    0, flat.shape[0], (num_dead,), device=flat.device
                )
                self.codebook.data[dead_mask] = flat[rand_idx].detach()

        # 周期性重置 usage, 避免长期累积失真
        if int(self.step.item()) % self.cfg.reset_usage == 0:
            self.usage_count.zero_()

    # ------------------------------------------------------------------
    # 解码时按索引取码字
    # ------------------------------------------------------------------
    def lookup(
        self,
        indices: torch.Tensor,
        boundary_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """按索引取码本向量 (解码阶段使用)."""
        flat_idx = indices.reshape(-1).to(torch.long)
        quantized = self.codebook[flat_idx]
        if boundary_type is not None:
            btype = boundary_type.reshape(-1).to(torch.long)
            quantized = quantized + self.boundary_embed[btype]
        out_shape = (*indices.shape, self.cfg.latent_dim)
        return quantized.reshape(out_shape)

    def extra_repr(self) -> str:
        return (
            f"codebook_size={self.cfg.codebook_size}, "
            f"latent_dim={self.cfg.latent_dim}, "
            f"num_boundary_types={self.cfg.num_boundary_types}, "
            f"ema_decay={self.cfg.ema_decay}"
        )
