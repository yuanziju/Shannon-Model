"""注意力工具函数 (Shannon / MathMaster 共享基础设施).

包含:
    - 各类注意力掩码 (causal / bidirectional / MMA / sliding-window / prefix)
    - 旋转位置编码 RoPE (1D / 2D / 3D)
    - QK-RMSNorm 归一化
    - 缩放点积注意力 (兼容 bool 与 float 掩码)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


# =========================================================================
# 掩码函数: 约定 bool 掩码中 True = 允许注意, False = 屏蔽
# =========================================================================
def causal_mask(seq_len: int, device=None, dtype: torch.dtype = torch.bool) -> torch.Tensor:
    """下三角因果掩码 (含对角线), shape ``(seq_len, seq_len)``."""
    return torch.ones(seq_len, seq_len, dtype=dtype, device=device).tril().bool()


def bidirectional_mask(seq_len: int, device=None, dtype: torch.dtype = torch.bool) -> torch.Tensor:
    """全双向掩码, 所有位置互相可见."""
    return torch.ones(seq_len, seq_len, dtype=dtype, device=device).bool()


def mma_mask(seq_len: int, block_size: int, device=None) -> torch.Tensor:
    """Mixed Multi-granular Attention 掩码: 块内因果, 块间屏蔽.

    将序列按 ``block_size`` 分块, 只有同一块内的位置可互相 (因果) 注意,
    跨块位置被屏蔽. 适合长文本的局部 + 全局混合注意力中的局部分支.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    idx = torch.arange(seq_len, device=device)
    block = idx // block_size
    same_block = block[:, None] == block[None, :]
    causal = idx[:, None] >= idx[None, :]
    return (same_block & causal).bool()


def sliding_window_mask(
    seq_len: int, window: int, causal: bool = True, device=None
) -> torch.Tensor:
    """滑动窗口掩码.

    ``causal=True`` (默认): 仅注意过去 ``window`` 个 token (含自身).
    ``causal=False``: 注意以自身为中心、半径 ``window`` 的双向窗口.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    idx = torch.arange(seq_len, device=device)
    diff = idx[:, None] - idx[None, :]
    if causal:
        return (diff >= 0) & (diff < window)
    return diff.abs() < window


def prefix_mask(seq_len: int, prefix_len: int, device=None) -> torch.Tensor:
    """Prefix-LM 掩码: 前 ``prefix_len`` 个 token 双向可见, 之后为因果.

    常用于编码器-解码器式前缀微调与文档理解任务.
    """
    if prefix_len < 0 or prefix_len > seq_len:
        raise ValueError("prefix_len out of range")
    idx = torch.arange(seq_len, device=device)
    is_prefix_col = idx[None, :] < prefix_len
    causal = idx[:, None] >= idx[None, :]
    return (is_prefix_col | causal).bool()


# =========================================================================
# 旋转位置编码 RoPE
# =========================================================================
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """标准 RoPE 的 rotate_half: 将后半取负后与前半拼接."""
    d = x.shape[-1]
    half = d // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotation(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """对 ``x`` 应用旋转. cos/sin shape ``(seq, head_dim)``."""
    seq = x.shape[-2]
    cos = cos[:seq].to(x.dtype).unsqueeze(0).unsqueeze(0)
    sin = sin[:seq].to(x.dtype).unsqueeze(0).unsqueeze(0)
    return x * cos + _rotate_half(x) * sin


def rope_freqs(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算 1D RoPE 的 cos/sin, shape ``(seq_len, head_dim)``."""
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim)
    )
    pos = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.outer(pos, inv_freq)              # (seq, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)         # (seq, head_dim)
    return emb.cos(), emb.sin()


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """应用 1D RoPE.

    ``x``: ``(..., heads, seq, head_dim)``; ``cos``/``sin``: ``(seq, head_dim)``.
    """
    return _apply_rotation(x, cos, sin)


def rope_freqs_2d(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算 2D RoPE 的 cos/sin (head_dim 等分给 x/y 两轴).

    适用于视觉/空间序列: 每个位置编码 (x, y) 两个坐标. ``seq_len`` 个位置
    假定按行优先排列于一个正方形网格 (边长 ceil(sqrt(seq_len))). 每个轴分得
    ``head_dim // 2`` 个旋转维度, 共同拼成完整的 ``head_dim``.
    """
    if head_dim % 4 != 0:
        raise ValueError("head_dim must be divisible by 4 for 2D RoPE")
    side = math.isqrt(seq_len)
    if side * side < seq_len:
        side += 1
    axis_dim = head_dim // 2  # 每个轴分得的旋转维度 (必须为偶数)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, axis_dim, 2, device=device, dtype=dtype) / axis_dim)
    )
    ys = torch.arange(seq_len, device=device, dtype=dtype) // side
    xs = torch.arange(seq_len, device=device, dtype=dtype) % side
    freqs_x = torch.outer(xs, inv_freq)            # (seq, axis_dim/2)
    freqs_y = torch.outer(ys, inv_freq)
    emb_x = torch.cat((freqs_x, freqs_x), dim=-1)  # (seq, axis_dim)
    emb_y = torch.cat((freqs_y, freqs_y), dim=-1)
    emb = torch.cat((emb_x, emb_y), dim=-1)        # (seq, head_dim)
    return emb.cos(), emb.sin()


def apply_rope_2d(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """应用 2D RoPE: 对 head_dim 的 x / y 两段分别做 rotate_half 旋转.

    ``cos``/``sin`` 形状 ``(seq, head_dim)``, 前半为 x 轴频率、后半为 y 轴频率
    (由 :func:`rope_freqs_2d` 生成). 分段旋转保证每段内部 cos/sin 配对一致,
    从而保范数.
    """
    d = x.shape[-1] // 2
    o1 = _apply_rotation(x[..., :d], cos[..., :d], sin[..., :d])
    o2 = _apply_rotation(x[..., d:], cos[..., d:], sin[..., d:])
    return torch.cat((o1, o2), dim=-1)


def rope_freqs_3d(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算 3D RoPE 的 cos/sin (head_dim 三等分给 t/h/w).

    适用于视频/时空序列. 位置按 (t, h, w) 立方体排列: t = idx // side^2,
    h = (idx % side^2) // side, w = idx % side. 每个轴分得 ``head_dim // 3``
    个旋转维度 (必须为偶数), 共同拼成完整的 ``head_dim``.
    """
    if head_dim % 6 != 0:
        raise ValueError("head_dim must be divisible by 6 for 3D RoPE")
    side = math.isqrt(seq_len)
    if side * side < seq_len:
        side += 1
    axis_dim = head_dim // 3  # 每个轴分得的旋转维度 (head_dim%6==0 保证为偶数)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, axis_dim, 2, device=device, dtype=dtype) / axis_dim)
    )
    idx = torch.arange(seq_len, device=device, dtype=dtype)
    ts = idx // (side * side)
    rem = idx % (side * side)
    hs = rem // side
    ws = rem % side
    ft = torch.outer(ts, inv_freq)              # (seq, axis_dim/2)
    fh = torch.outer(hs, inv_freq)
    fw = torch.outer(ws, inv_freq)
    emb_t = torch.cat((ft, ft), dim=-1)         # (seq, axis_dim)
    emb_h = torch.cat((fh, fh), dim=-1)
    emb_w = torch.cat((fw, fw), dim=-1)
    emb = torch.cat((emb_t, emb_h, emb_w), dim=-1)  # (seq, head_dim)
    return emb.cos(), emb.sin()


def apply_rope_3d(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """应用 3D RoPE: 对 head_dim 的 t / h / w 三段分别做 rotate_half 旋转.

    ``cos``/``sin`` 形状 ``(seq, head_dim)``, 三等分分别对应 t/h/w 轴频率
    (由 :func:`rope_freqs_3d` 生成). 分段旋转保证每段内部 cos/sin 配对一致,
    从而保范数.
    """
    d = x.shape[-1] // 3
    o1 = _apply_rotation(x[..., :d], cos[..., :d], sin[..., :d])
    o2 = _apply_rotation(x[..., d:2 * d], cos[..., d:2 * d], sin[..., d:2 * d])
    o3 = _apply_rotation(x[..., 2 * d:], cos[..., 2 * d:], sin[..., 2 * d:])
    return torch.cat((o1, o2, o3), dim=-1)


# =========================================================================
# QK-Norm (RMSNorm on head_dim)
# =========================================================================
def qk_norm(q: torch.Tensor, k: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """对 Q/K 沿 ``head_dim`` 做 RMSNorm 归一化.

    稳定注意力 logits 的数值范围, 计算在 float32 下进行, 结果转回原 dtype.
    """
    def _rms(t: torch.Tensor) -> torch.Tensor:
        o = t.to(torch.float32)
        o = o * torch.rsqrt(o.pow(2).mean(dim=-1, keepdim=True) + eps)
        return o.to(t.dtype)
    return _rms(q), _rms(k)


# =========================================================================
# 缩放点积注意力
# =========================================================================
def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """缩放点积注意力.

    ``q``: ``(..., heads, seq_q, head_dim)``
    ``k``: ``(..., heads, seq_k, head_dim)``
    ``v``: ``(..., heads, seq_k, head_dim)``
    ``attn_mask``: bool (True=注意) 或 float (加到 logits). ``is_causal`` 与
    ``attn_mask`` 可同时使用.
    """
    head_dim = q.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale

    if is_causal:
        seq_q, seq_k = q.shape[-2], k.shape[-2]
        causal = torch.ones(seq_q, seq_k, dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~causal, float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask.to(scores.dtype)

    attn = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
    if dropout > 0.0 and torch.is_grad_enabled():
        attn = F.dropout(attn, p=dropout)
    return torch.matmul(attn, v)
