"""Rank-64 Gated Attention (Hybrid-M3 8th head).

Augments standard GQA attention with a low-rank (rank-64) gating path
that scales each head's output by a per-token, per-head gate computed
from the input:

    gate(x) = sigmoid( scale * (W_up * SiLU(W_down * x)) )
    out = (attention_output) * gate(x)

* ``W_down``: ``[d_model, 64]`` — compresses the input to a rank-64
  bottleneck.
* ``W_up``: ``[64, n_heads]`` — produces a per-head gate.
* ``scale``: per-head learnable scalar (init 1).

At the reference config (d_model=4096, n_heads=32, GQA n_kv=8, d_kv=128)
the gating path adds ~0.27M parameters on top of the Q/K/V/O projection
cost, giving a total module size on the order of ~27M parameters with a
shared-QKV variant.  The rank-64 gate is the architectural feature that
distinguishes this head from a plain GQA layer.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import (
    AttentionConfig,
    AttentionOutput,
    BaseAttention,
    apply_rope,
    sdpa,
)


class GatedAttention(BaseAttention):
    """Rank-64 low-rank gated attention."""

    def __init__(self, config: AttentionConfig):
        super().__init__(config)
        self.rank = config.gated_rank
        self.q_proj = nn.Linear(
            self.d_model, self.n_heads * self.d_kv, bias=config.bias
        )
        self.k_proj = nn.Linear(
            self.d_model, self.n_kv_heads * self.d_kv, bias=config.bias
        )
        self.v_proj = nn.Linear(
            self.d_model, self.n_kv_heads * self.d_kv, bias=config.bias
        )
        self.o_proj = nn.Linear(
            self.n_heads * self.d_kv, self.d_model, bias=config.bias
        )
        # Low-rank gating path
        self.gate_down = nn.Linear(self.d_model, self.rank, bias=False)
        self.gate_up = nn.Linear(self.rank, self.n_heads, bias=False)
        self.gate_scale = nn.Parameter(torch.ones(self.n_heads))

    # ------------------------------------------------------------------
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
        q, k, v = self._qkv_proj(
            hidden_states, self.q_proj, self.k_proj, self.v_proj
        )
        if position_ids is not None:
            q, k = apply_rope(
                q, k, position_ids,
                self.config.rope_theta, self.config.rope_base_scale,
            )
        # Cache the *unexpanded* (n_kv_heads) K/V so the next chunk's
        # freshly-projected K/V match the head dimension on concat.
        k_cat, v_cat = self.cat_kv(past_kv, k, v)
        k = self._repeat_kv(k_cat)
        v = self._repeat_kv(v_cat)

        # Standard scaled-dot-product attention per head
        out = sdpa(q, k, v, attn_mask=attention_mask)           # [b, h, s, d_kv]

        # ---- Low-rank gate ------------------------------------------
        g = self.gate_down(hidden_states)                       # [b, s, rank]
        g = F.silu(g)
        g = self.gate_up(g) * self.gate_scale                   # [b, s, n_heads]
        # Apply gate (sigmoid) to each head's output
        gate = torch.sigmoid(g).transpose(1, 2).unsqueeze(-1)   # [b, h, s, 1]
        out = out * gate

        out = out.transpose(1, 2).contiguous().view(b, s, -1)
        out = self.o_proj(out)
        present = (k_cat, v_cat) if use_cache else None
        return AttentionOutput(
            output=out, present_kv=present,
            aux={"gate_mean": gate.mean().item()},
        )
