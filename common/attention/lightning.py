"""Lightning Attention.

Eliminates the ``cumsum`` bottleneck of causal linear attention via a
**block decomposition**:

    O_t = [(Q_t K_t^T) ⊙ Mask] V_t            # intra-block: standard (block size B)
        + Q_t (KV_accumulated)                  # inter-block: linear kernel trick

    KV_accumulated <- KV_accumulated + K_t^T V_t

* Intra-block uses a left-product (standard causal attention within the
  block of size ``B``).
* Inter-block uses a right-product (linear-attention kernel trick,
  ``phi(Q) (phi(K)^T V)``) where the accumulator ``KV`` is a
  ``[d_k, d_v]`` state matrix per head.

With this decomposition, training throughput (TGS) stays essentially
constant from 1K to 128K tokens.  An exponential decay per head is
applied to keys for long-range forgetting.
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
)


class LightningAttention(BaseAttention):
    """Lightning Attention with intra/inter-block decomposition."""

    def __init__(self, config: AttentionConfig):
        super().__init__(config)
        self.block_size = config.lightning_block_size
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
        # Per-head log-decay (init 0 => decay = 1, no forgetting)
        self.decay_log = nn.Parameter(torch.zeros(self.n_heads))

    @staticmethod
    def _feature(x: torch.Tensor) -> torch.Tensor:
        """Non-negative feature map for the linear kernel trick (elu+1)."""
        return F.elu(x) + 1.0

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
        # Lightning uses its own feature map in lieu of RoPE.
        q = self._feature(q)
        k = self._feature(k)

        # GQA: expand KV heads BEFORE applying per-head decay so the
        # decay (shape [1, n_heads, 1, 1]) broadcasts correctly.
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)
        # Per-head exponential decay (so |decay| <= 1).
        decay = torch.exp(-self.decay_log.abs()).view(1, self.n_heads, 1, 1)
        k = k * decay

        # Inter-block accumulator (state).  past_kv is [b, h, d, d] or None.
        if past_kv is not None:
            kv_state = past_kv
        else:
            kv_state = torch.zeros(
                b, self.n_heads, self.d_kv, self.d_kv,
                device=q.device, dtype=q.dtype,
            )

        block = self.block_size
        outputs = []
        for start in range(0, s, block):
            end = min(start + block, s)
            qc = q[:, :, start:end]                              # [b, h, T, d]
            kc = k[:, :, start:end]
            vc = v[:, :, start:end]
            T = end - start

            # Inter-block: o_inter = qc @ kv_state
            o_inter = torch.einsum("bhid,bhde->bhie", qc, kv_state)

            # Intra-block: causal linear attention within the block
            qk = torch.einsum("bhid,bhjd->bhij", qc, kc)        # [b, h, T, T]
            mask = torch.tril(
                torch.ones(T, T, device=q.device, dtype=qk.dtype)
            )
            qk = qk * mask
            o_intra = torch.einsum("bhij,bhjd->bhid", qk, vc)

            outputs.append(o_inter + o_intra)

            # Update state: kv_state += kc^T @ vc
            kv_state = kv_state + torch.einsum(
                "bhjd,bhje->bhde", kc, vc
            )

        out = torch.cat(outputs, dim=2)                         # [b, h, s, d]

        # Honour a 2-D padding mask if supplied
        if attention_mask is not None and attention_mask.dim() == 2:
            m = attention_mask.bool().view(b, 1, s, 1)
            out = out.masked_fill(~m, 0.0)

        out = out.transpose(1, 2).contiguous().view(b, s, -1)
        out = self.o_proj(out)
        present = kv_state if use_cache else None
        return AttentionOutput(output=out, present_kv=present)
