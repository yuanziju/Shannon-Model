"""Sliding-Window Attention.

Standard scaled-dot-product attention restricted to a local window of
size ``W`` (default 512 for text, 64x64 for image patches).  Used for
image patches and video frames where local spatial modelling matters.

Two modes:
* ``bidirectional=False`` (default, text / causal): position ``i`` may
  attend to ``[i - W, i]``.
* ``bidirectional=True`` (image patches / non-causal): position ``i``
  may attend to ``[i - W, i + W]``.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from .base import (
    AttentionConfig,
    AttentionOutput,
    BaseAttention,
    apply_rope,
    sdpa,
)


class SlidingWindowAttention(BaseAttention):
    """Sliding-window attention with configurable window size."""

    def __init__(self, config: AttentionConfig):
        super().__init__(config)
        self.window = config.window_size
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

    def _build_window_mask(
        self,
        s_q: int,
        s_k: int,
        bidirectional: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build an additive [s_q, s_k] mask with -inf outside the window.

        ``s_q`` is the number of new queries and ``s_k`` is the total
        number of keys (cached + new).  The query at local index ``i``
        has global position ``(s_k - s_q) + i``; the window is applied
        in *global* coordinates so cached keys are handled correctly.
        """
        offset = s_k - s_q                                    # cached key count
        i_idx = (
            torch.arange(s_q, device=device) + offset
        ).unsqueeze(1)                                       # [s_q, 1] global
        j_idx = torch.arange(s_k, device=device).unsqueeze(0)  # [1, s_k]
        if bidirectional:
            # image patches: window both directions
            allowed = (j_idx >= i_idx - self.window) & (
                j_idx <= i_idx + self.window
            )
        else:
            # text / causal: j in [i - W, i]
            allowed = (j_idx <= i_idx) & (j_idx >= i_idx - self.window)
        mask = torch.zeros(s_q, s_k, device=device, dtype=dtype)
        mask = mask.masked_fill(~allowed, float("-inf"))
        return mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        past_kv: Any = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        bidirectional: bool = False,
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

        # Append new K/V to the cache (used for autoregressive decoding).
        # Cache the *unexpanded* (n_kv_heads) tensors so the next chunk's
        # freshly-projected K/V match the head dimension on concat.
        k_cat, v_cat = self.cat_kv(past_kv, k, v)
        s_k = k_cat.shape[-2]
        k_exp = self._repeat_kv(k_cat)
        v_exp = self._repeat_kv(v_cat)

        # Build the sliding-window mask and merge with any external mask
        win_mask = self._build_window_mask(
            s, s_k, bidirectional, q.device, q.dtype
        )                                                         # [s, s_k]
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            # broadcast-add
            win_mask = win_mask.unsqueeze(0).unsqueeze(0) + attention_mask
        else:
            win_mask = win_mask.unsqueeze(0).unsqueeze(0)        # [1, 1, s, s_k]

        out = sdpa(q, k_exp, v_exp, attn_mask=win_mask)
        out = out.transpose(1, 2).contiguous().view(b, s, -1)
        out = self.o_proj(out)
        present = (k_cat, v_cat) if use_cache else None
        return AttentionOutput(output=out, present_kv=present)
