"""视觉编码器 — ViT+Q-Former AND VAE 双通道.

Shannon 视觉编码采用双通道架构 (spec: ViT+Q-Former AND VAE双通道视觉编码):
  - ViTQFormerEncoder: ViT 提取 patch 特征 + Q-Former 压缩到固定数量 query
    (适合语义理解, 原生任意分辨率)
  - VAEEncoder: VAE 编码到连续隐空间 (适合图像生成/编辑)
  - DualChannelVisionEncoder: 融合两通道, 输出统一 hidden_dim

参考: spec §3 编码器视觉双通道, AGENTS.md ViT+Q-Former AND VAE.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.layers import RMSNorm


class PatchEmbed(nn.Module):
    """图像分块嵌入: [B, C, H, W] -> [B, num_patches, hidden_dim]."""

    def __init__(self, patch_size: int = 16, in_channels: int = 3, hidden_dim: int = 4096):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, H', W', D]
        B, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, D]
        return x


class ViTBlock(nn.Module):
    """单层 ViT 编码块 (self-attention + FFN)."""

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class ViTQFormerEncoder(nn.Module):
    """ViT + Q-Former 视觉编码器.

    流程: 图像 -> ViT patch embedding -> ViT 编码块 -> Q-Former
    (用固定数量 learnable query 交叉注意力压缩 patch 特征).

    适合语义理解, 输出固定长度 num_queries 个 token.

    Args:
        hidden_dim: 输出维度.
        patch_size: ViT patch 大小.
        num_layers: ViT 层数.
        num_heads: 注意力头数.
        num_queries: Q-Former query 数量.
        qformer_layers: Q-Former 层数.
    """

    def __init__(
        self,
        hidden_dim: int,
        patch_size: int = 16,
        num_layers: int = 12,
        num_heads: int = 12,
        num_queries: int = 32,
        qformer_layers: int = 4,
        in_channels: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        # 确保 num_heads <= hidden_dim
        num_heads = max(1, min(num_heads, hidden_dim))

        self.patch_embed = PatchEmbed(patch_size, in_channels, hidden_dim)
        # 确保至少1层
        num_layers = max(1, num_layers)
        self.vit_blocks = nn.ModuleList([
            ViTBlock(hidden_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.vit_norm = nn.LayerNorm(hidden_dim)

        # Q-Former: learnable queries
        self.queries = nn.Parameter(torch.zeros(num_queries, hidden_dim))
        nn.init.normal_(self.queries, std=0.02)

        qformer_layers = max(1, qformer_layers)
        self.qformer_blocks = nn.ModuleList([
            nn.ModuleDict({
                "self_attn_norm": nn.LayerNorm(hidden_dim),
                "self_attn": nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True),
                "cross_attn_norm": nn.LayerNorm(hidden_dim),
                "cross_attn": nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True),
                "ffn_norm": nn.LayerNorm(hidden_dim),
                "ffn": nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                ),
            })
            for _ in range(qformer_layers)
        ])
        self.qformer_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """ViT+Q-Former 编码.

        Args:
            images: [B, C, H, W] 图像.

        Returns:
            [B, num_queries, hidden_dim] 压缩后的视觉 token.
        """
        B = images.shape[0]
        # ViT 编码
        x = self.patch_embed(images)  # [B, P, D]
        for blk in self.vit_blocks:
            x = blk(x)
        x = self.vit_norm(x)  # [B, P, D] patch 特征

        # Q-Former: learnable queries 交叉注意力
        q = self.queries.unsqueeze(0).expand(B, -1, -1)  # [B, Q, D]
        for blk in self.qformer_blocks:
            # self-attention on queries
            h = blk["self_attn_norm"](q)
            sa_out, _ = blk["self_attn"](h, h, h, need_weights=False)
            q = q + sa_out
            # cross-attention: queries attend to patch features
            h = blk["cross_attn_norm"](q)
            ca_out, _ = blk["cross_attn"](h, x, x, need_weights=False)
            q = q + ca_out
            # FFN
            q = q + blk["ffn"](blk["ffn_norm"](q))
        q = self.qformer_norm(q)
        return self.out_proj(q)

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, "
            f"num_queries={self.num_queries}, "
            f"vit_layers={len(self.vit_blocks)}, "
            f"qformer_layers={len(self.qformer_blocks)}"
        )


class VAEEncoder(nn.Module):
    """VAE 视觉编码器 (连续隐空间).

    将图像编码到连续隐空间 latent, 适合图像生成/编辑.
    输出均值与方差, 采样得到 latent.

    Args:
        latent_dim: 隐空间维度.
        downsample: 下采样倍数.
        in_channels: 输入通道.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        downsample: int = 8,
        in_channels: int = 3,
        hidden_dim: int = 4096,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.downsample = downsample
        self.hidden_dim = hidden_dim

        # 卷积下采样 backbone
        layers = []
        ch = in_channels
        cur = 32
        ds = 1
        while ds < downsample:
            layers.append(nn.Conv2d(ch, cur, kernel_size=3, stride=2, padding=1))
            layers.append(nn.GroupNorm(min(8, cur), cur))
            layers.append(nn.SiLU())
            ch = cur
            cur = min(cur * 2, 512)
            ds *= 2
        self.backbone = nn.Sequential(*layers)

        # 输出 mean / logvar
        self.norm_out = nn.GroupNorm(min(8, ch), ch)
        self.conv_mean = nn.Conv2d(ch, latent_dim, kernel_size=1)
        self.conv_logvar = nn.Conv2d(ch, latent_dim, kernel_size=1)

        # 投影到 hidden_dim
        self.to_hidden = nn.Linear(latent_dim, hidden_dim, bias=False)
        self.norm = RMSNorm(hidden_dim)

    def forward(
        self,
        images: torch.Tensor,
        sample: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """VAE 编码.

        Args:
            images: [B, C, H, W].
            sample: 是否采样 (训练 True, 推理可用 mean).

        Returns:
            (latent_tokens [B, S, hidden_dim], mean [B, latent_dim, h, w], logvar [...]).
        """
        h = self.backbone(images)  # [B, C', H', W']
        h = self.norm_out(h)
        mean = self.conv_mean(h)    # [B, latent_dim, H', W']
        logvar = self.conv_logvar(h)  # [B, latent_dim, H', W']

        if sample:
            std = torch.exp(0.5 * logvar)
            z = mean + std * torch.randn_like(mean)
        else:
            z = mean

        # 展平为 token 序列: [B, latent_dim, H', W'] -> [B, H'*W', latent_dim]
        B, D, H, W = z.shape
        z_tokens = z.flatten(2).transpose(1, 2)  # [B, H*W, latent_dim]
        # 投影到 hidden_dim
        z_tokens = self.to_hidden(z_tokens)
        z_tokens = self.norm(z_tokens)
        return z_tokens, mean, logvar

    def kl_loss(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """计算 KL 散度损失."""
        return -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())

    def extra_repr(self) -> str:
        return (
            f"latent_dim={self.latent_dim}, "
            f"downsample={self.downsample}"
        )


class DualChannelVisionEncoder(nn.Module):
    """双通道视觉编码器: 融合 ViT+Q-Former 与 VAE.

    输出 = concat(vit_qformer_tokens, vae_tokens) 或加权融合.

    Args:
        hidden_dim: 输出维度.
        config: ShannonConfig (取视觉参数).
    """

    def __init__(
        self,
        hidden_dim: int,
        patch_size: int = 16,
        vit_num_layers: int = 12,
        vit_num_heads: int = 12,
        qformer_num_queries: int = 32,
        qformer_num_layers: int = 4,
        vae_latent_dim: int = 256,
        vae_downsample: int = 8,
        fusion: str = "concat",  # "concat" | "mean"
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fusion = fusion

        # 通道 1: ViT + Q-Former
        self.vit_qformer = ViTQFormerEncoder(
            hidden_dim=hidden_dim,
            patch_size=patch_size,
            num_layers=vit_num_layers,
            num_heads=vit_num_heads,
            num_queries=qformer_num_queries,
            qformer_layers=qformer_num_layers,
            dropout=dropout,
        )

        # 通道 2: VAE
        self.vae = VAEEncoder(
            latent_dim=vae_latent_dim,
            downsample=vae_downsample,
            hidden_dim=hidden_dim,
        )

        # 融合
        if fusion == "concat":
            self.fuse_proj = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        else:
            self.fuse_proj = None
        # 通道权重 (可学习)
        self.channel_weight = nn.Parameter(torch.tensor([0.5, 0.5]))

        self.norm = RMSNorm(hidden_dim)

    # ------------------------------------------------------------------
    def forward(
        self,
        images: torch.Tensor,
        sample_vae: bool = True,
    ) -> dict:
        """双通道视觉编码.

        Args:
            images: [B, C, H, W].
            sample_vae: VAE 是否采样.

        Returns:
            dict 含:
              - hidden: [B, S, hidden_dim] 融合后视觉 token.
              - vae_mean, vae_logvar: VAE 隐空间参数.
              - kl_loss: VAE KL 散度.
              - vit_tokens, vae_tokens: 各通道原始输出.
        """
        # 通道 1: ViT + Q-Former
        vit_tokens = self.vit_qformer(images)  # [B, Q, H]

        # 通道 2: VAE
        vae_tokens, vae_mean, vae_logvar = self.vae(images, sample=sample_vae)  # [B, S2, H]
        kl_loss = self.vae.kl_loss(vae_mean, vae_logvar)

        # 融合 (对齐序列长度: 取较短的, 或 pad)
        B1, S1, H = vit_tokens.shape
        B2, S2, H2 = vae_tokens.shape
        S = min(S1, S2)
        # 截断到相同长度后融合
        v1 = vit_tokens[:, :S]
        v2 = vae_tokens[:, :S]

        w = torch.softmax(self.channel_weight, dim=0)
        if self.fusion == "concat":
            combined = self.fuse_proj(torch.cat([v1, v2], dim=-1))
        else:
            combined = w[0] * v1 + w[1] * v2

        combined = self.norm(combined)
        return {
            "hidden": combined,
            "vit_tokens": vit_tokens,
            "vae_tokens": vae_tokens,
            "vae_mean": vae_mean,
            "vae_logvar": vae_logvar,
            "kl_loss": kl_loss,
        }

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, "
            f"fusion={self.fusion}, "
            f"queries={self.vit_qformer.num_queries}"
        )
