"""图像编辑路由器 (ImageEditRouter) — 图像编辑指令解析与操作路由.

将用户图像编辑指令 (文本) 路由为具体的图像编辑操作, 在 VAE 隐空间中
预测编辑增量. 支持:
  - 编辑类型分类 (全局风格 / 局部编辑 / inpainting / super-res)
  - 编辑区域预测 (bounding box / mask)
  - VAE 隐空间增量预测 (delta latent)

工作流:
    text_hidden + image_latent ──→ edit_type + region + delta_latent
    delta_latent + image_latent ──→ VAE decode ──→ edited image

参考: spec §9 图像编辑输出头, decoder_output.image_edit_enabled.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.layers import RMSNorm


# 编辑类型
EDIT_GLOBAL_STYLE = 0      # 全局风格迁移
EDIT_LOCAL_EDIT = 1         # 局部编辑 (指定区域修改)
EDIT_INPAINTING = 2         # 区域重绘
EDIT_SUPER_RES = 3          # 超分辨率
NUM_EDIT_TYPES = 4


class ImageEditRouter(nn.Module):
    """图像编辑路由器.

    接收文本隐状态 (编辑指令) 与图像 VAE 隐变量, 预测:
      1. 编辑类型 (分类)
      2. 编辑区域 (bbox 回归: [cx, cy, w, h] 归一化)
      3. VAE 隐空间增量 (delta_latent)

    Args:
        hidden_dim: 文本隐状态维度.
        latent_dim: VAE 隐空间维度.
        num_edit_types: 编辑类型数.
        num_region_heads: 区域预测头数 (多区域编辑).
    """

    def __init__(
        self,
        hidden_dim: int,
        latent_dim: int = 256,
        num_edit_types: int = NUM_EDIT_TYPES,
        num_region_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_edit_types = num_edit_types
        self.num_region_heads = num_region_heads

        # 文本-图像跨注意力 (文本 query 图像 latent)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=max(1, min(8, hidden_dim // 32)),
            dropout=dropout,
            batch_first=True,
            kdim=latent_dim,
            vdim=latent_dim,
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)

        # 编辑类型分类头
        self.type_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_edit_types),
        )
        nn.init.normal_(self.type_head[-1].weight, std=0.02)

        # 编辑区域回归头 (每个 head 预测 [cx, cy, w, h] + confidence)
        self.region_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_region_heads * 5),  # 4 coords + 1 conf
        )
        nn.init.normal_(self.region_head[-1].weight, std=0.02)

        # VAE 隐空间增量预测 (从文本隐状态投影到 latent_dim)
        self.delta_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        nn.init.zeros_(self.delta_proj[-1].weight)
        nn.init.zeros_(self.delta_proj[-1].bias)

        # 归一化
        self.text_norm = RMSNorm(hidden_dim)

    # ------------------------------------------------------------------
    def forward(
        self,
        text_hidden: torch.Tensor,
        image_latent: Optional[torch.Tensor] = None,
        edit_type_label: Optional[torch.Tensor] = None,
        region_label: Optional[torch.Tensor] = None,
        delta_label: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """图像编辑路由前向.

        Args:
            text_hidden: [B, S, H] 编辑指令的文本隐状态.
            image_latent: [B, P, latent_dim] 图像 VAE 隐变量 (可选, 用于
                          跨注意力条件; None 则仅用文本).
            edit_type_label: [B] 编辑类型标签 (训练).
            region_label: [B, num_region_heads, 5] 区域标签 (训练).
            delta_label: [B, latent_dim] 或 [B, P, latent_dim] 增量标签 (训练).

        Returns:
            dict 含:
              - type_logits: [B, num_edit_types]
              - regions: [B, num_region_heads, 5] (cx, cy, w, h, conf)
              - delta_latent: [B, latent_dim] 隐空间增量
              - loss: (训练) 总损失
        """
        h = self.text_norm(text_hidden)  # [B, S, H]

        # 跨注意力: 文本 attend to 图像 latent
        if image_latent is not None:
            h_attn, _ = self.cross_attn(
                query=h, key=image_latent, value=image_latent,
                need_weights=False,
            )
            h = self.cross_norm(h + h_attn)

        # 池化得到序列级表示
        pooled = h.mean(dim=1)  # [B, H]

        # 编辑类型
        type_logits = self.type_head(pooled)  # [B, num_types]

        # 编辑区域 (bbox + confidence)
        regions = self.region_head(pooled)  # [B, num_heads*5]
        regions = regions.view(-1, self.num_region_heads, 5)
        # 坐标 sigmoid 到 [0, 1], confidence sigmoid
        regions_coords = torch.sigmoid(regions[..., :4])
        regions_conf = torch.sigmoid(regions[..., 4:5])
        regions_out = torch.cat([regions_coords, regions_conf], dim=-1)

        # VAE 隐空间增量
        delta_latent = self.delta_proj(pooled)  # [B, latent_dim]

        result: Dict[str, torch.Tensor] = {
            "type_logits": type_logits,
            "regions": regions_out,
            "delta_latent": delta_latent,
        }

        # ---- 计算损失 ----
        if edit_type_label is not None:
            type_loss = F.cross_entropy(type_logits, edit_type_label)
            result["type_loss"] = type_loss
            total_loss = type_loss

            if region_label is not None:
                # 区域回归: MSE on coords + BCE on confidence
                coord_loss = F.mse_loss(
                    regions_out[..., :4], region_label[..., :4]
                )
                conf_loss = F.binary_cross_entropy(
                    regions_out[..., 4], region_label[..., 4]
                )
                result["region_loss"] = coord_loss + conf_loss
                total_loss = total_loss + 0.5 * (coord_loss + conf_loss)

            if delta_label is not None:
                if delta_label.dim() == 2:
                    delta_loss = F.mse_loss(delta_latent, delta_label)
                else:
                    # [B, P, latent_dim]: 广播 delta_latent
                    delta_loss = F.mse_loss(
                        delta_latent.unsqueeze(1), delta_label
                    )
                result["delta_loss"] = delta_loss
                total_loss = total_loss + delta_loss

            result["loss"] = total_loss

        return result

    # ------------------------------------------------------------------
    @torch.no_grad()
    def route(
        self,
        text_hidden: torch.Tensor,
        image_latent: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """推理: 路由编辑操作.

        Args:
            text_hidden: [B, S, H].
            image_latent: [B, P, latent_dim] (可选).

        Returns:
            dict 含 edit_type, regions, delta_latent.
        """
        out = self.forward(text_hidden, image_latent=image_latent)
        return {
            "edit_type": out["type_logits"].argmax(dim=-1),
            "regions": out["regions"],
            "delta_latent": out["delta_latent"],
        }

    def extra_repr(self) -> str:
        return (
            f"hidden={self.hidden_dim}, latent={self.latent_dim}, "
            f"types={self.num_edit_types}, region_heads={self.num_region_heads}"
        )
