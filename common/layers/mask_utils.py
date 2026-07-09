"""掩码工具模块.

提供因果/双向/MMA/滑动窗口掩码构造函数, 以及动态混合掩码生成器 HybridMaskGenerator.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

# 模态 id 约定
TEXT_MODALITY = 0
IMAGE_MODALITY = 1


def make_causal_mask(
    seq_len: int, device=None, dtype: torch.dtype = torch.bool
) -> torch.Tensor:
    """标准因果掩码, 下三角 (含对角) 为 True (允许注意).

    返回形状 (seq_len, seq_len).
    """
    return torch.tril(
        torch.ones(seq_len, seq_len, device=device, dtype=dtype), diagonal=0
    )


def make_bidirectional_mask(
    seq_len: int, device=None, dtype: torch.dtype = torch.bool
) -> torch.Tensor:
    """双向掩码: 全部位置互相允许注意."""
    return torch.ones(seq_len, seq_len, device=device, dtype=dtype)


def make_mma_mask(
    modality_ids: torch.Tensor,
    device=None,
    dtype: torch.dtype = torch.bool,
) -> torch.Tensor:
    """模态互注意力掩码 (MMA).

    规则 (spec §4.5):
      - 同模态内: 保持因果
      - image -> text: 允许双向 (解锁 image 对其之前 text 的注意)
      - text -> image: 保持因果

    modality_ids: (seq_len,) 每个位置的模态 id (0=text, 1=image).
    返回 (seq_len, seq_len) bool 掩码, True 表示允许注意.
    """
    if device is None:
        device = modality_ids.device
    seq_len = modality_ids.shape[0]
    causal = make_causal_mask(seq_len, device=device, dtype=dtype)
    is_image = modality_ids == IMAGE_MODALITY
    is_text = modality_ids == TEXT_MODALITY
    # image (query, 行) 允许 attend 到任意 text (key, 列)
    img_to_text = is_image.unsqueeze(1) & is_text.unsqueeze(0)
    return causal | img_to_text.to(dtype)


def make_sliding_mask(
    seq_len: int,
    window: int,
    bidirectional: bool = True,
    device=None,
    dtype: torch.dtype = torch.bool,
) -> torch.Tensor:
    """滑动窗口掩码: |i - j| <= window 允许注意.

    bidirectional=True 时为双向滑动窗口 (图像 patch 局部建模);
    bidirectional=False 时仅允许注意过去窗口内的位置 (因果滑动).
    """
    i = torch.arange(seq_len, device=device).unsqueeze(1)
    j = torch.arange(seq_len, device=device).unsqueeze(0)
    if bidirectional:
        mask = (i - j).abs() <= window
    else:
        mask = ((i - j) >= 0) & ((i - j) <= window)
    return mask.to(dtype)


class HybridMaskGenerator(nn.Module):
    """动态混合掩码生成器.

    根据层索引 (与可选的模态信息) 动态选择并组合不同的掩码模式,
    对应 spec §4.8 的分层异构注意力路由 (每 4 层一个周期):
      - Layer 4k+1 / 4k+3: KDA 线性注意力 -> 因果掩码
      - Layer 4k+2       : KDA + MoH      -> 因果掩码 (稀疏头)
      - Layer 4k+4       : MLA + MMA      -> MMA 掩码 (多模态对齐)

    可通过 layer_types 显式指定每层的掩码类型, 支持:
      'causal' | 'bidirectional' | 'sliding' | 'mma'.
    """

    SUPPORTED = ("causal", "bidirectional", "sliding", "mma")

    def __init__(
        self,
        num_layers: int,
        sliding_window: int = 512,
        layer_types: Optional[list] = None,
        mma_period: int = 4,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.sliding_window = sliding_window
        if layer_types is None:
            # 默认: 每 mma_period 层的最后一层使用 MMA, 其余因果
            layer_types = []
            for idx in range(num_layers):
                if mma_period > 0 and (idx % mma_period) == (mma_period - 1):
                    layer_types.append("mma")
                else:
                    layer_types.append("causal")
        assert len(layer_types) == num_layers, "layer_types 长度需与 num_layers 一致"
        for lt in layer_types:
            assert lt in self.SUPPORTED, f"不支持的掩码类型: {lt}"
        self.layer_types = layer_types

    def forward(
        self,
        seq_len: int,
        layer_idx: int,
        modality_ids: Optional[torch.Tensor] = None,
        device=None,
        dtype: torch.dtype = torch.bool,
    ) -> torch.Tensor:
        if device is None:
            device = modality_ids.device if modality_ids is not None else None
        mtype = self.layer_types[layer_idx % len(self.layer_types)]
        if mtype == "causal":
            return make_causal_mask(seq_len, device=device, dtype=dtype)
        if mtype == "bidirectional":
            return make_bidirectional_mask(seq_len, device=device, dtype=dtype)
        if mtype == "sliding":
            return make_sliding_mask(
                seq_len, self.sliding_window, bidirectional=True, device=device, dtype=dtype
            )
        if mtype == "mma":
            if modality_ids is None:
                # 无模态信息时退化为因果
                return make_causal_mask(seq_len, device=device, dtype=dtype)
            return make_mma_mask(modality_ids, device=device, dtype=dtype)
        raise ValueError(f"未知掩码类型: {mtype}")

    def extra_repr(self) -> str:
        return f"num_layers={self.num_layers}, sliding_window={self.sliding_window}"
