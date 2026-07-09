"""DeepSeek Multi-head Latent Attention (MLA).

Caches the low-rank latent ``c_kv`` of shape ``[b, s, d_c]`` with
``d_c = d_model / 4`` instead of the full K/V tensors, reducing KV-cache
memory by 75-90%.  This implementation also includes:

* **QK-Norm** (Qwen-style): an RMSNorm applied to the content portion of
  Q and K *before* the dot product, preventing attention-logit blow-up in
  deep networks.
* **Decoupled RoPE**: a separate half-head ``q_pe`` / ``k_pe`` path
  carries positional information while leaving the cached latent free of
  positional encoding (so the latent stays rotation-invariant).

Forward cache format: ``(c_kv, k_pe)`` — both are the minimal tensors
needed to reconstruct the full K, V at the next step.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from .base import (
    AttentionConfig,
    AttentionOutput,
    BaseAttention,
    RMSNorm,
    apply_rope,
    sdpa,
)


class MLAAttention(BaseAttention):
    """DeepSeek Multi-head Latent Attention with QK-Norm + decoupled RoPE."""

    def __init__(self, config: AttentionConfig):
        super().__init__(config)
        self.d_c = config.d_c
        # Decoupled RoPE uses half-head width per group (rotary dim = d_kv // 2)
        self.d_rope = self.d_kv // 2

        # Down-projections into the latent space (the part actually cached)
        self.q_a = nn.Linear(self.d_model, self.d_c, bias=config.bias)
        self.kv_a = nn.Linear(self.d_model, self.d_c, bias=config.bias)
        # Decoupled RoPE latents (NOT cached as part of c_kv; k_pe is cached)
        self.q_rope_a = nn.Linear(
            self.d_model, self.n_heads * self.d_rope, bias=config.bias
        )
        self.k_rope_a = nn.Linear(
            self.d_model, self.n_kv_heads * self.d_rope, bias=config.bias
        )
        # Up-projections back to head space
        self.q_b = nn.Linear(
            self.d_c, self.n_heads * self.d_kv, bias=config.bias
        )
        self.k_b = nn.Linear(
            self.d_c, self.n_kv_heads * self.d_kv, bias=config.bias
        )
        self.v_b = nn.Linear(
            self.d_c, self.n_kv_heads * self.d_kv, bias=config.bias
        )
        # QK-Norm (per-head RMSNorm on the content portion only)
        self.q_norm = RMSNorm(self.d_kv, eps=config.rms_eps)
        self.k_norm = RMSNorm(self.d_kv, eps=config.rms_eps)
        # Output projection
        self.o_proj = nn.Linear(
            self.n_heads * self.d_kv, self.d_model, bias=config.bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        past_kv: Any = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> AttentionOutput:
        b, s, _ = hidden_states.shape

        # --- Latent projections --------------------------------------
        c_q = self.q_a(hidden_states)                          # [b, s, d_c]
        c_kv = self.kv_a(hidden_states)                        # [b, s, d_c]  (cached)
        q = (
            self.q_b(c_q)
            .view(b, s, self.n_heads, self.d_kv)
            .transpose(1, 2)
        )                                                       # [b, h, s, d_kv]
        # Decoupled RoPE path (separate from content)
        q_pe = (
            self.q_rope_a(hidden_states)
            .view(b, s, self.n_heads, self.d_rope)
            .transpose(1, 2)
        )                                                       # [b, h, s, d_rope]
        k_pe = (
            self.k_rope_a(hidden_states)
            .view(b, s, self.n_kv_heads, self.d_rope)
            .transpose(1, 2)
        )                                                       # [b, hk, s, d_rope]
        q_pe, k_pe = apply_rope(
            q_pe, k_pe, position_ids,
            self.config.rope_theta, self.config.rope_base_scale,
        )

        # --- Concatenate latent cache -------------------------------
        if past_kv is not None:
            c_kv_prev, k_pe_prev = past_kv
            c_kv_full = torch.cat([c_kv_prev, c_kv], dim=1)
            k_pe_full = torch.cat([k_pe_prev, k_pe], dim=2)
        else:
            c_kv_full = c_kv
            k_pe_full = k_pe

        # --- Re-expand K, V from the (possibly cached) latent ------
        k_full = (
            self.k_b(c_kv_full)
            .view(b, -1, self.n_kv_heads, self.d_kv)
            .transpose(1, 2)
        )                                                       # [b, hk, S, d_kv]
        v_full = (
            self.v_b(c_kv_full)
            .view(b, -1, self.n_kv_heads, self.d_kv)
            .transpose(1, 2)
        )

        # --- QK-Norm on the content portion -------------------------
        q_cont = self.q_norm(q)
        k_cont = self.k_norm(k_full)

        # --- Concat content + decoupled RoPE components -------------
        q_full = torch.cat([q_cont, q_pe], dim=-1)             # [b, h, s, d_kv+d_rope]
        k_full = torch.cat([k_cont, k_pe_full], dim=-1)        # [b, hk, S, d_kv+d_rope]

        # GQA: expand KV heads to match Q heads
        k_full = self._repeat_kv(k_full)
        v_full = self._repeat_kv(v_full)

        # --- Scaled dot-product attention ---------------------------
        out = sdpa(q_full, k_full, v_full, attn_mask=attention_mask)
        out = out.transpose(1, 2).contiguous().view(b, s, -1)
        out = self.o_proj(out)

        present = (c_kv_full, k_pe_full) if use_cache else None
        return AttentionOutput(output=out, present_kv=present)
