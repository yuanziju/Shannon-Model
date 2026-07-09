"""Mixture-of-Head Attention (MoH).

Treats attention heads as experts in a Mixture-of-Experts sense and
routes each token to its Top-K heads:

    O = sum_{i in shared}            Head_i(x)
      + sum_{i in TopK_dynamic(x)}   g_i(x) * Head_i(x)

    router(x) = softmax( W_r * RMSNorm(x) )          # head-level routing
    g_i(x)    = renormalised Top-K router weight

* **Shared heads** (``n_shared``) are always active with weight 1 and
  capture generic knowledge.
* **Dynamic heads** (``n_heads - n_shared``) are sparsely activated via
  Top-K selection; only the chosen heads are computed at inference time.
* LLaMA3-8B uses only 75% of heads and gains +2.4% on 14 benchmarks.

Implementation note: heads are always projected through a single
``o_proj`` of shape ``[n_heads * d_kv, d_model]``; non-selected dynamic
heads are zeroed out so the projection stays consistent across tokens.
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
    RMSNorm,
    apply_rope,
    sdpa,
)


class MoHAttention(BaseAttention):
    """Mixture-of-Head Attention with shared + Top-K dynamic heads."""

    def __init__(self, config: AttentionConfig):
        super().__init__(config)
        self.top_k = config.moh_top_k
        self.n_shared = config.moh_n_shared
        self.n_dynamic = self.n_heads - self.n_shared
        assert self.n_dynamic >= self.top_k, (
            f"n_dynamic={self.n_dynamic} must be >= top_k={self.top_k}"
        )

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
        # Router
        self.router_norm = RMSNorm(self.d_model, eps=config.rms_eps)
        self.router = nn.Linear(
            self.d_model, self.n_dynamic, bias=False
        )
        # Per-head learnable bias added to router logits (encourages
        # specialisation); initialised so that all heads start equally
        # likely.
        self.router_bias = nn.Parameter(torch.zeros(self.n_dynamic))

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

        # ---- Router --------------------------------------------------
        normed = self.router_norm(hidden_states)               # [b, s, d]
        logits = self.router(normed) + self.router_bias        # [b, s, n_dyn]
        weights = F.softmax(logits, dim=-1)                    # [b, s, n_dyn]
        topk_w, topk_idx = weights.topk(self.top_k, dim=-1)    # [b, s, K]
        # Renormalise the Top-K weights so they sum to 1
        topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-6)
        # Build the full per-token weight vector for the dynamic heads
        full_w = torch.zeros_like(weights)
        full_w.scatter_(2, topk_idx, topk_w)                   # [b, s, n_dyn]

        # ---- Per-head attention --------------------------------------
        # Shared heads: weight 1; Dynamic heads: Top-K weight (else 0)
        q_shared = q[:, : self.n_shared]
        q_dynamic = q[:, self.n_shared :]
        k_shared = k[:, : self.n_shared]
        k_dynamic = k[:, self.n_shared :]
        v_shared = v[:, : self.n_shared]
        v_dynamic = v[:, self.n_shared :]

        out_shared = sdpa(q_shared, k_shared, v_shared, attn_mask=attention_mask)
        out_dynamic = sdpa(
            q_dynamic, k_dynamic, v_dynamic, attn_mask=attention_mask
        )
        # out_shared:  [b, n_shared, s, d_kv]
        # out_dynamic: [b, n_dynamic, s, d_kv]

        # Apply per-token router weights to dynamic head outputs
        # full_w: [b, s, n_dyn] -> [b, n_dyn, s, 1]
        w = full_w.permute(0, 2, 1).unsqueeze(-1)
        out_dynamic = out_dynamic * w

        # Reassemble full [b, n_heads, s, d_kv] tensor (shared weights = 1)
        out = torch.cat([out_shared, out_dynamic], dim=1)
        out = out.transpose(1, 2).contiguous().view(b, s, -1)
        out = self.o_proj(out)

        present = (k_cat, v_cat) if use_cache else None
        # Router entropy (auxiliary signal for load-balancing loss)
        with torch.no_grad():
            entropy = -(weights * (weights + 1e-9).log()).sum(-1).mean()
        return AttentionOutput(
            output=out, present_kv=present,
            aux={"router_entropy": entropy.item(), "topk_idx": topk_idx},
        )
