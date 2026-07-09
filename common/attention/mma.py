"""Modality-Mutual Attention (MMA).

Unlocks the image-token -> text-token attention path that the standard
causal mask forbids.  The mask rules are:

* Same modality (text<->text, image<->image): keep causal.
* image -> text: **bidirectional allowed** (image queries can attend to
  all text keys, including future text).
* text -> image: keep causal (text queries still see only past image).

This wraps any inner ``BaseAttention`` module and only modifies the
attention mask passed to it.  Adds **zero parameters** — the +5.5%
average gain on 12 multimodal benchmarks comes entirely from the mask
reconfiguration.

Per the spec, MMA is enabled during SFT and disabled during
pretraining: pass ``modality_ids=None`` to fall back to the inner
module's default (causal) behaviour.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from .base import (
    AttentionConfig,
    AttentionOutput,
    BaseAttention,
)


class MMAAttention(BaseAttention):
    """Modality-Mutual Attention wrapper around an inner attention module."""

    def __init__(
        self,
        config: AttentionConfig,
        inner: Optional[BaseAttention] = None,
    ):
        super().__init__(config)
        # Lazy import to avoid a hard dependency cycle at module load time.
        if inner is None:
            from .mla import MLAAttention
            inner = MLAAttention(config)
        self.inner = inner

    # ------------------------------------------------------------------
    def _build_mma_mask(
        self,
        modality_ids: torch.Tensor,    # [b, s]  (0=text, 1=image)
        seq: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build an additive [b, 1, s, s] MMA mask (-inf blocks attention)."""
        b = modality_ids.shape[0]
        is_image = modality_ids == 1                      # [b, s]
        is_text = ~is_image

        i_idx = torch.arange(seq, device=device).view(-1, 1)   # [s, 1]
        j_idx = torch.arange(seq, device=device).view(1, -1)   # [1, s]
        j_gt_i = (j_idx > i_idx).unsqueeze(0)                  # [1, s, s]

        i_is_text = is_text.unsqueeze(2)                       # [b, s, 1]
        i_is_image = is_image.unsqueeze(2)                     # [b, s, 1]
        j_is_image = is_image.unsqueeze(1)                     # [b, 1, s]

        # A position is blocked when:
        #   (i is text  AND j > i)                 -- text query, future key
        #   OR
        #   (i is image AND j is image AND j > i) -- image query, future image key
        blocked = (i_is_text & j_gt_i) | (i_is_image & j_is_image & j_gt_i)
        blocked = blocked.unsqueeze(1)                         # [b, 1, s, s]
        mask = torch.zeros(b, 1, seq, seq, device=device, dtype=dtype)
        mask = mask.masked_fill(blocked, float("-inf"))
        return mask

    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        past_kv: Any = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        modality_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> AttentionOutput:
        # If no modality info is supplied (e.g. pretraining), behave as the
        # inner module with its default causal mask.
        if modality_ids is not None:
            seq = hidden_states.shape[1]
            mma_mask = self._build_mma_mask(
                modality_ids, seq, hidden_states.device, hidden_states.dtype
            )
            if attention_mask is None:
                attention_mask = mma_mask
            else:
                if attention_mask.dim() == 2:
                    attention_mask = (
                        attention_mask.unsqueeze(1).unsqueeze(1)
                    )
                attention_mask = attention_mask + mma_mask

        return self.inner(
            hidden_states,
            position_ids=position_ids,
            past_kv=past_kv,
            attention_mask=attention_mask,
            use_cache=use_cache,
            **kwargs,
        )
