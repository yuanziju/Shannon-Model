"""Unified Attention Scheduler.

Implements the 4-layer hybrid-attention cycle of Hybrid-M3:

    Layer 4k+1 : KDA                       (long-sequence memory)
    Layer 4k+2 : KDA + MoH Top-4           (adaptive computation)
    Layer 4k+3 : KDA                       (long-sequence memory)
    Layer 4k+4 : MLA + QK-Norm + MMA       (global alignment)

The scheduler instantiates one KDA module (shared across the two KDA
phases to save parameters), one MoH module (used in phase 4k+2 only),
and one MLA+MMA module (used in phase 4k+4).  A cross-layer projection
``W_proj * concat([h_kda, h_moh])`` aligns the heterogeneous KV
representations (KDA state matrix vs. MLA low-rank latent) when both
paths contribute to the same layer's output.

The scheduler is deterministic given the layer index — there is no
runtime routing overhead at inference time.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Optional

import torch
import torch.nn as nn

from .base import (
    AttentionConfig,
    AttentionOutput,
    BaseAttention,
)
from .kda import KDAAttention
from .mla import MLAAttention
from .mma import MMAAttention
from .moh import MoHAttention


class UnifiedAttentionScheduler(BaseAttention):
    """Routes attention computation based on ``layer_idx % 4``."""

    def __init__(self, config: AttentionConfig):
        super().__init__(config)
        # KDA for layers 4k+1, 4k+2, 4k+3 (shared across the three phases).
        self.kda = KDAAttention(config)
        # MoH for layer 4k+2 (Top-4 dynamic heads).
        moh_cfg = dataclasses.replace(
            config, moh_top_k=4, moh_n_shared=2
        )
        self.moh = MoHAttention(moh_cfg)
        # MLA + QK-Norm + MMA for layer 4k+4.  MLA already includes QK-Norm
        # internally; MMA wraps it and applies the multimodal mask.
        mla_cfg = dataclasses.replace(config)
        self.mla = MLAAttention(mla_cfg)
        self.mma = MMAAttention(mla_cfg, inner=self.mla)
        # Cross-layer projection: aligns KDA (state) and MLA/MoH (KV)
        # representations when they co-occur in a single layer.
        self.cross_proj = nn.Linear(
            config.d_model * 2, config.d_model, bias=False
        )

    # ------------------------------------------------------------------
    @property
    def phase(self) -> int:
        """Phase in {0,1,2,3} = (layer_idx mod 4).

        Mapping (1-indexed layer IDs in the spec, 0-indexed here):
            phase 0 -> Layer 4k+1 : KDA
            phase 1 -> Layer 4k+2 : KDA + MoH
            phase 2 -> Layer 4k+3 : KDA
            phase 3 -> Layer 4k+4 : MLA + QK-Norm + MMA
        """
        return self.layer_idx % 4

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
        phase = self.phase

        if phase in (0, 2):
            # 4k+1 / 4k+3 : pure KDA
            return self.kda(
                hidden_states,
                position_ids=position_ids,
                past_kv=past_kv,
                attention_mask=attention_mask,
                use_cache=use_cache,
                **kwargs,
            )

        if phase == 1:
            # 4k+2 : KDA + MoH Top-4 (cross-projected)
            out_kda = self.kda(
                hidden_states,
                position_ids=position_ids,
                past_kv=past_kv,
                attention_mask=attention_mask,
                use_cache=False,
                **kwargs,
            )
            out_moh = self.moh(
                hidden_states,
                position_ids=position_ids,
                past_kv=past_kv,
                attention_mask=attention_mask,
                use_cache=False,
                **kwargs,
            )
            combined = self.cross_proj(
                torch.cat([out_kda.output, out_moh.output], dim=-1)
            )
            return AttentionOutput(
                output=combined,
                present_kv=out_kda.present_kv,
                aux={"phase": 1, "modules": ["kda", "moh"]},
            )

        # phase == 3 : 4k+4 : MLA + QK-Norm + MMA
        return self.mma(
            hidden_states,
            position_ids=position_ids,
            past_kv=past_kv,
            attention_mask=attention_mask,
            use_cache=use_cache,
            modality_ids=modality_ids,
            **kwargs,
        )
