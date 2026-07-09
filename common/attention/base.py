"""Base attention classes for Hybrid-M3.

Defines the shared ``AttentionConfig`` / ``AttentionOutput`` containers,
the abstract ``BaseAttention`` base class, and common utilities
(GQA-aware Q/K/V projections, KV-cache concatenation, RoPE, RMSNorm,
causal-mask construction) reused by every concrete attention module.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AttentionConfig:
    """Shared configuration for all Hybrid-M3 attention modules.

    Field defaults target the reference 15B-MoE config
    (d_model=4096, n_heads=32, GQA with n_kv_heads=8).
    """
    d_model: int = 4096
    n_heads: int = 32
    n_kv_heads: Optional[int] = None      # GQA; None -> n_heads
    d_kv: Optional[int] = None            # None -> d_model // n_heads
    d_c: Optional[int] = None             # MLA latent dim; None -> d_model // 4
    max_seq_len: int = 32768
    rope_theta: float = 10000.0
    rope_base_scale: float = 1.0
    dropout: float = 0.0
    layer_idx: int = 0
    bias: bool = False
    rms_eps: float = 1e-6
    # MMA / multimodal
    n_modalities: int = 2
    # MoH
    moh_top_k: int = 4
    moh_n_shared: int = 2
    # Gated attention
    gated_rank: int = 64
    # Sliding window
    window_size: int = 512
    # Lightning attention
    lightning_block_size: int = 64
    # KDA
    kda_chunk_size: int = 64

    def __post_init__(self):
        if self.d_kv is None:
            self.d_kv = self.d_model // self.n_heads
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.d_c is None:
            self.d_c = self.d_model // 4
        assert self.d_model % self.n_heads == 0, \
            f"d_model={self.d_model} not divisible by n_heads={self.n_heads}"
        assert self.n_heads % self.n_kv_heads == 0, \
            f"n_heads={self.n_heads} not divisible by n_kv_heads={self.n_kv_heads}"
        assert self.n_heads > self.moh_n_shared, \
            "MoH needs more total heads than shared heads"


@dataclass
class AttentionOutput:
    """Container for attention forward outputs."""
    output: torch.Tensor
    present_kv: Any = None
    aux: dict = field(default_factory=dict)


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (QK-Norm compatible)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x.to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    theta: float = 10000.0,
    scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply Rotary Position Embedding to ``q`` and ``k``.

    q, k: [b, h, s, d].  position_ids: [b, s] or [s] or None.
    """
    if position_ids is None:
        seq = q.shape[-2]
        position_ids = torch.arange(seq, device=q.device)
    if position_ids.dim() == 1:
        position_ids = position_ids.unsqueeze(0)
    d = q.shape[-1]
    inv_freq = 1.0 / (
        theta
        ** (torch.arange(0, d, 2, device=q.device, dtype=torch.float32) / d)
    ) * scale
    pos = position_ids.float()
    freqs = torch.einsum("bi,j->bij", pos, inv_freq)        # [b, s, d/2]
    emb = torch.cat([freqs, freqs], dim=-1)                 # [b, s, d]
    cos = emb.cos().unsqueeze(1)                            # [b, 1, s, d]
    sin = emb.sin().unsqueeze(1)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot.to(q.dtype), k_rot.to(k.dtype)


def sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Wrapper around ``F.scaled_dot_product_attention``.

    Picks ``is_causal`` automatically when no explicit mask is supplied.
    Handles three regimes correctly:

    * **No cache, multi-token** (``s_q == s_k > 1``): standard causal.
    * **Single-token decode** (``s_q == 1``): attend to all cached keys
      (no mask needed — the lone query is the most recent position).
    * **Chunked decode with cache** (``s_q < s_k``): the first
      ``s_k - s_q`` keys are always visible; the trailing ``s_q`` keys
      use a lower-triangular causal mask.
    """
    s_q = q.shape[-2]
    s_k = k.shape[-2]
    if attn_mask is not None:
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=False, dropout_p=dropout_p
        )
    if s_q == s_k and s_q > 1:
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=True, dropout_p=dropout_p
        )
    if s_q == 1:
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=False, dropout_p=dropout_p
        )
    # Chunked decode with cache: build a [s_q, s_k] mask.
    device, dtype = q.device, q.dtype
    mask = torch.zeros(s_q, s_k, device=device, dtype=dtype)
    causal = torch.triu(
        torch.full((s_q, s_q), float("-inf"), device=device, dtype=dtype),
        diagonal=1,
    )
    mask[:, s_k - s_q:] = causal
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, is_causal=False, dropout_p=dropout_p
    )


class BaseAttention(nn.Module, ABC):
    """Abstract base class for all Hybrid-M3 attention modules.

    Provides:
      * GQA-aware Q/K/V projection helpers
      * KV-cache concatenation
      * causal-mask construction
      * rotary embedding application
    """

    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.d_kv = config.d_kv
        self.n_rep = self.n_heads // self.n_kv_heads
        self.layer_idx = config.layer_idx

    # ----- GQA helpers -------------------------------------------------
    def _qkv_proj(
        self,
        hidden: torch.Tensor,
        wq: nn.Linear,
        wk: nn.Linear,
        wv: nn.Linear,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project hidden states to Q, K, V with GQA shapes.

        Returns q: [b, n_heads, s, d_kv],
                k: [b, n_kv_heads, s, d_kv],
                v: [b, n_kv_heads, s, d_kv].
        """
        b, s, _ = hidden.shape
        q = wq(hidden).view(b, s, self.n_heads, self.d_kv).transpose(1, 2)
        k = wk(hidden).view(b, s, self.n_kv_heads, self.d_kv).transpose(1, 2)
        v = wv(hidden).view(b, s, self.n_kv_heads, self.d_kv).transpose(1, 2)
        return q, k, v

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Expand KV heads to match query heads for GQA."""
        if self.n_rep == 1:
            return x
        b, h, s, d = x.shape
        return (
            x[:, :, None, :, :]
            .expand(b, h, self.n_rep, s, d)
            .reshape(b, h * self.n_rep, s, d)
        )

    # ----- cache helpers -----------------------------------------------
    @staticmethod
    def cat_kv(
        past: Optional[Tuple[torch.Tensor, torch.Tensor]],
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if past is None:
            return k_new, v_new
        k_prev, v_prev = past
        return (
            torch.cat([k_prev, k_new], dim=-2),
            torch.cat([v_prev, v_new], dim=-2),
        )

    # ----- mask helpers ------------------------------------------------
    @staticmethod
    def make_causal_mask(
        seq: int, device: torch.device, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        return torch.triu(
            torch.full((seq, seq), float("-inf"), device=device, dtype=dtype),
            diagonal=1,
        )

    # ----- abstract ----------------------------------------------------
    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        past_kv: Any = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> AttentionOutput:
        ...

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"n_kv={self.n_kv_heads}, layer={self.layer_idx}"
        )
