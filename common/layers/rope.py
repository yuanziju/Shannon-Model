"""旋转位置编码模块.

提供标准 RoPE、二维/三维 RoPE、YaRN 长度外推、LongRoPE2 超长上下文与时序衰减 RoPE.
所有实现按需在 forward 中计算 cos/sin, 避免预分配超大缓存.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


def _build_inv_freq(
    dim: int, base: float, device=None, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """构造逆频率表, 形状 (dim // 2,)."""
    return 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """将最后一维拆成两半并旋转: [-x2, x1]."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _broadcast_to_x(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """将形状 (seq, dim) 的张量广播到 x (..., seq, dim) 的形状."""
    extra = x.dim() - 2
    for _ in range(extra):
        t = t.unsqueeze(0)
    return t


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """对 x (..., seq, dim) 应用旋转, cos/sin 形状 (seq, dim)."""
    cos = _broadcast_to_x(cos, x)
    sin = _broadcast_to_x(sin, x)
    return x * cos + rotate_half(x) * sin


class RoPE(nn.Module):
    """标准旋转位置编码.

    对最后一个维度的相邻频率对施加旋转, 为序列位置编码相对位置信息.
    """

    def __init__(self, dim: int, base: float = 10000.0, max_seq_len: int = 2048):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        inv_freq = _build_inv_freq(dim, base)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _cos_sin(
        self,
        seq_len: int,
        positions: Optional[torch.Tensor],
        device,
        dtype: torch.dtype,
    ):
        if positions is None:
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
        else:
            t = positions.to(device=device, dtype=torch.float32)
        inv_freq = self.inv_freq.to(device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # (seq, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq, dim)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_len = x.shape[-2]
        cos, sin = self._cos_sin(seq_len, positions, x.device, x.dtype)
        return _apply_rotary(x, cos, sin)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, base={self.base}"


def _cos_sin_from_idx(
    idx: torch.Tensor, inv_freq: torch.Tensor, device, dtype: torch.dtype
):
    """对任意形状的坐标 idx (..., seq) 计算对应 cos/sin, 返回 (..., seq, dim)."""
    t = idx.to(device=device, dtype=torch.float32)
    inv_freq = inv_freq.to(device=device, dtype=torch.float32)
    # einsum: (..., seq) x (dim/2,) -> (..., seq, dim/2)
    freqs = torch.einsum("...s,d->...sd", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)  # (..., seq, dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


class RoPE2D(nn.Module):
    """二维旋转位置编码 (T/H 解耦, 用于图像空间位置).

    将 dim 拆成两半: 前半用行坐标编码, 后半用列坐标编码.
    forward(x, positions): positions 形状 (..., seq, 2), 最后一维为 (h, w).
    """

    def __init__(self, dim: int, base: float = 10000.0, max_seq_len: int = 4096):
        super().__init__()
        assert dim % 2 == 0, "RoPE2D dim 必须为偶数"
        self.dim = dim
        self.half = dim // 2
        self.base = base
        self.max_seq_len = max_seq_len
        inv_freq = _build_inv_freq(self.half, base)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # positions: (..., seq, 2) -> (h, w)
        h_idx = positions[..., 0]
        w_idx = positions[..., 1]
        cos_h, sin_h = _cos_sin_from_idx(h_idx, self.inv_freq, x.device, x.dtype)
        cos_w, sin_w = _cos_sin_from_idx(w_idx, self.inv_freq, x.device, x.dtype)

        x_h = x[..., : self.half]
        x_w = x[..., self.half :]
        out_h = x_h * cos_h + rotate_half(x_h) * sin_h
        out_w = x_w * cos_w + rotate_half(x_w) * sin_w
        return torch.cat((out_h, out_w), dim=-1)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, base={self.base}"


class RoPE3D(nn.Module):
    """三维旋转位置编码 (T/H/W 解耦, 用于视频时空位置).

    将 dim 拆成三个大致相等的偶数块, 分别用 t/h/w 坐标编码.
    forward(x, positions): positions 形状 (..., seq, 3), 最后一维为 (t, h, w).
    """

    def __init__(self, dim: int, base: float = 10000.0, max_seq_len: int = 8192):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        # 划分三个偶数块: 前两块取最大的偶数等分, 余数归入第三块
        part = (dim // 6) * 2
        self.part_sizes = [part, part, dim - 2 * part]
        for ps in self.part_sizes:
            assert ps > 0 and ps % 2 == 0, f"RoPE3D 分块必须为正偶数, 得到 {self.part_sizes}"
        inv_freq_t = _build_inv_freq(self.part_sizes[0], base)
        inv_freq_h = _build_inv_freq(self.part_sizes[1], base)
        inv_freq_w = _build_inv_freq(self.part_sizes[2], base)
        self.register_buffer("inv_freq_t", inv_freq_t, persistent=False)
        self.register_buffer("inv_freq_h", inv_freq_h, persistent=False)
        self.register_buffer("inv_freq_w", inv_freq_w, persistent=False)

    def _rotate_part(
        self,
        x_part: torch.Tensor,
        idx: torch.Tensor,
        inv_freq: torch.Tensor,
    ) -> torch.Tensor:
        cos, sin = _cos_sin_from_idx(idx, inv_freq, x_part.device, x_part.dtype)
        return x_part * cos + rotate_half(x_part) * sin

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # positions: (..., seq, 3) -> (t, h, w)
        t_idx = positions[..., 0]
        h_idx = positions[..., 1]
        w_idx = positions[..., 2]

        s0, s1, s2 = self.part_sizes
        x_t = x[..., :s0]
        x_h = x[..., s0 : s0 + s1]
        x_w = x[..., s0 + s1 :]

        out_t = self._rotate_part(x_t, t_idx, self.inv_freq_t)
        out_h = self._rotate_part(x_h, h_idx, self.inv_freq_h)
        out_w = self._rotate_part(x_w, w_idx, self.inv_freq_w)
        return torch.cat((out_t, out_h, out_w), dim=-1)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, base={self.base}, parts={self.part_sizes}"


# ---------------------------------------------------------------------------
# YaRN 长度外推
# ---------------------------------------------------------------------------

def _yarn_find_correction_dim(
    num_rotations: int, dim: int, base: float, max_position_embeddings: int
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def _yarn_find_correction_range(
    low_rot: int, high_rot: int, dim: int, base: float, max_position_embeddings: int
):
    low = math.floor(_yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp_mask(low: float, high: float, dim: int, device, dtype) -> torch.Tensor:
    if low == high:
        high += 0.001
    linear = (torch.arange(dim, device=device, dtype=dtype) - low) / (high - low)
    return torch.clamp(linear, 0, 1)


class YaRN(nn.Module):
    """YaRN 长度外推.

    基于波长将频率分为三段: 外推区 (短波长, 不缩放)、插值区 (过渡, 线性混合)、
    不变区 (长波长, 位置无关, 强制插值), 并引入注意力温度因子 attn_factor.
    """

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        original_max_position_embeddings: int = 8192,
        max_seq_len: int = 32768,
        beta_fast: int = 32,
        beta_slow: int = 1,
        attn_factor: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.base = base
        self.original_max = original_max_position_embeddings
        self.max_seq_len = max_seq_len
        self.scaling_factor = float(max_seq_len) / float(original_max_position_embeddings)
        self.attn_factor = attn_factor

        half = dim // 2
        inv_freq = _build_inv_freq(dim, base)
        low, high = _yarn_find_correction_range(
            beta_fast, beta_slow, dim, base, original_max_position_embeddings
        )
        inv_freq_extrapolation = inv_freq
        inv_freq_interpolation = inv_freq / self.scaling_factor
        mask = _yarn_linear_ramp_mask(low, high, half, None, torch.float32)
        inv_freq_mask = 1.0 - mask
        # 插值区使用插值频率, 外推区使用原始频率
        scaled_inv_freq = inv_freq_interpolation * mask + inv_freq_extrapolation * inv_freq_mask
        self.register_buffer("inv_freq", scaled_inv_freq, persistent=False)

    def _cos_sin(
        self,
        seq_len: int,
        positions: Optional[torch.Tensor],
        device,
        dtype: torch.dtype,
    ):
        if positions is None:
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
        else:
            t = positions.to(device=device, dtype=torch.float32)
        inv_freq = self.inv_freq.to(device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return (emb.cos() * self.attn_factor).to(dtype), (emb.sin() * self.attn_factor).to(dtype)

    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_len = x.shape[-2]
        cos, sin = self._cos_sin(seq_len, positions, x.device, x.dtype)
        return _apply_rotary(x, cos, sin)

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, base={self.base}, scaling={self.scaling_factor:.3f}, "
            f"attn_factor={self.attn_factor}"
        )


class LongRoPE2(nn.Module):
    """LongRoPE2: 非均匀频率缩放, 支持超长上下文 (目标 5M).

    对每个频率维度引入独立缩放因子 lambda (非均匀), 并叠加全局缩放
    scale = sqrt(1 + ln(L/L0)/ln(L0)), 兼顾短距离精度与长距离外推.
    lambda 可学习, 初始化为渐进式缩放 (高频更大).
    """

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        max_seq_len: int = 5_000_000,
        original_max_position_embeddings: int = 8192,
        long_factor=None,
        learnable: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.original_max = original_max_position_embeddings
        half = dim // 2
        inv_freq = _build_inv_freq(dim, base)

        if long_factor is None:
            # 高频维度赋予更大缩放, 低频接近 1
            long_factor = torch.linspace(1.0, 4.0, half, dtype=torch.float32)
        long_factor = torch.as_tensor(long_factor, dtype=torch.float32)
        if learnable:
            self.lambda_long = nn.Parameter(long_factor.clone())
        else:
            self.register_buffer("lambda_long", long_factor, persistent=False)

        # 全局缩放, 保证注意力分布稳定
        scale = math.sqrt(
            1.0 + math.log(max_seq_len / original_max_position_embeddings)
            / math.log(original_max_position_embeddings)
        )
        self.scale = float(scale)
        scaled_inv_freq = inv_freq / (self.lambda_long.detach() * self.scale)
        self.register_buffer("inv_freq_base", inv_freq, persistent=False)
        # 实际使用的 inv_freq 在 forward 中按当前 lambda 计算 (支持学习)

    def _current_inv_freq(self, device, dtype) -> torch.Tensor:
        lam = self.lambda_long.to(device=device, dtype=torch.float32)
        inv_freq = self.inv_freq_base.to(device=device, dtype=torch.float32)
        return inv_freq / (lam * self.scale)

    def _cos_sin(
        self,
        seq_len: int,
        positions: Optional[torch.Tensor],
        device,
        dtype: torch.dtype,
    ):
        if positions is None:
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
        else:
            t = positions.to(device=device, dtype=torch.float32)
        inv_freq = self._current_inv_freq(device, dtype)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_len = x.shape[-2]
        cos, sin = self._cos_sin(seq_len, positions, x.device, x.dtype)
        return _apply_rotary(x, cos, sin)

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, base={self.base}, max_seq_len={self.max_seq_len}, "
            f"scale={self.scale:.3f}"
        )


class TemporalDecayRoPE(nn.Module):
    """时序衰减 RoPE (1D RoPE + 时序衰减偏置).

    在标准 RoPE 之上, 提供按时间距离指数衰减的注意力偏置:
        bias[i, j] = -|decay| * (i - j),   i >= j (因果方向)
        bias[i, j] = -inf,                  i < j
    forward 返回旋转后的 x; 通过 get_temporal_bias 获取可叠加到 attention scores 的偏置.
    """

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        max_seq_len: int = 2048,
        decay_init: float = 0.01,
    ):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        inv_freq = _build_inv_freq(dim, base)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.decay = nn.Parameter(torch.tensor(float(decay_init)))

    def _cos_sin(
        self,
        seq_len: int,
        positions: Optional[torch.Tensor],
        device,
        dtype: torch.dtype,
    ):
        if positions is None:
            t = torch.arange(seq_len, device=device, dtype=torch.float32)
        else:
            t = positions.to(device=device, dtype=torch.float32)
        inv_freq = self.inv_freq.to(device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_len = x.shape[-2]
        cos, sin = self._cos_sin(seq_len, positions, x.device, x.dtype)
        return _apply_rotary(x, cos, sin)

    def get_temporal_bias(
        self,
        seq_len: int,
        device=None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """返回 (seq_len, seq_len) 的加性注意力偏置, 因果上三角为 -inf."""
        if device is None:
            device = self.inv_freq.device
        i = torch.arange(seq_len, device=device).unsqueeze(1)
        j = torch.arange(seq_len, device=device).unsqueeze(0)
        dist = (i - j).clamp_min(0).to(dtype)
        bias = -self.decay.abs() * dist
        future = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1
        )
        bias = bias.masked_fill(future, float("-inf"))
        return bias

    def extra_repr(self) -> str:
        return f"dim={self.dim}, base={self.base}, decay={float(self.decay):.4f}"
