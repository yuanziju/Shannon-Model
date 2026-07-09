"""Dynamic Attention Controller.

A lightweight router (<1% of total attention parameters) that softly
mixes the outputs of a pool of candidate attention modules based on:

* pooled hidden-state statistics (what the input "looks like")
* the (log) sequence length (short -> MLA, long -> KDA / Lightning)

The router is deliberately tiny: a single down-projection to
``d_model / 16``, an SiLU non-linearity, a length embedding, and a
linear head over the candidate modules.  At inference time a hard
Top-1 routing can be derived from the softmax weights; at training time
the differentiable soft mixture is used so all modules receive gradient.

This is the "Dynamic controller" head of Hybrid-M3 — it does not
replace the per-layer routing of ``UnifiedAttentionScheduler`` (which
is deterministic and layer-id driven); instead it provides *runtime*
adaptation within a single layer when multiple attention paths are
available.
"""
from __future__ import annotations

from typing import Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import (
    AttentionConfig,
    AttentionOutput,
    BaseAttention,
    RMSNorm,
)


class DynamicAttentionController(nn.Module):
    """Soft-routing controller over a pool of attention modules.

    Parameters
    ----------
    config : AttentionConfig
    attention_modules : list[BaseAttention]
        Pool of candidate attention modules to route between.
    """

    def __init__(
        self,
        config: AttentionConfig,
        attention_modules: List[BaseAttention],
    ):
        super().__init__()
        self.config = config
        self.modules_list = nn.ModuleList(attention_modules)
        self.n_choices = len(attention_modules)
        d_model = config.d_model
        d_hidden = max(d_model // 16, 16)
        # Router: pooled hidden + log(seq_len) -> logits over modules
        self.norm = RMSNorm(d_model, eps=config.rms_eps)
        self.proj = nn.Linear(d_model, d_hidden, bias=False)
        self.len_proj = nn.Linear(1, d_hidden, bias=False)
        self.head = nn.Linear(d_hidden * 2, self.n_choices, bias=False)
        # Temperature for sharper / softer routing
        self.log_temperature = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------
    def _route(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        b, s, d = hidden_states.shape
        pooled = self.norm(hidden_states).mean(dim=1)           # [b, d]
        h = F.silu(self.proj(pooled))                           # [b, d_hidden]
        len_feat = torch.log1p(
            torch.tensor(
                float(s), device=hidden_states.device, dtype=hidden_states.dtype
            )
        ).view(1, 1).expand(b, 1)
        len_h = self.len_proj(len_feat)                         # [b, d_hidden]
        logits = self.head(torch.cat([h, len_h], dim=-1))       # [b, n_choices]
        temp = self.log_temperature.exp().clamp(min=0.1, max=10.0)
        return F.softmax(logits / temp, dim=-1)

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
        weights = self._route(hidden_states)                    # [b, n_choices]

        outputs = []
        for module in self.modules_list:
            try:
                out = module(
                    hidden_states,
                    position_ids=position_ids,
                    past_kv=past_kv,
                    attention_mask=attention_mask,
                    use_cache=False,
                    modality_ids=modality_ids,
                    **kwargs,
                )
            except TypeError:
                # Module does not accept ``modality_ids``.
                out = module(
                    hidden_states,
                    position_ids=position_ids,
                    past_kv=past_kv,
                    attention_mask=attention_mask,
                    use_cache=False,
                    **kwargs,
                )
            outputs.append(out.output)

        out_tensor = torch.stack(outputs, dim=1)                # [b, n_choices, s, d]
        w = weights.unsqueeze(-1).unsqueeze(-1)                 # [b, n_choices, 1, 1]
        final = (out_tensor * w).sum(dim=1)                     # [b, s, d]

        with torch.no_grad():
            top1 = weights.argmax(dim=-1)
        return AttentionOutput(
            output=final,
            present_kv=None,
            aux={
                "router_weights": weights,
                "top1": top1,
                "entropy": -(weights * (weights + 1e-9).log()).sum(-1).mean().item(),
            },
        )
