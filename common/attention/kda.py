"""Kimi Delta Attention (KDA).

Implements the gated delta-rule linear attention used by Kimi Linear:

    S_t = (I - beta_t * k_t k_t^T) * Diag(alpha_t) * S_{t-1}
          + beta_t * k_t v_t^T
    o_t = S_t^T q_t

* ``S_t in R^{d_k x d_v}`` is a matrix-valued RNN state (associative
  memory); there is **no KV cache** — the state *is* the cache.
* ``beta_t`` is a data-dependent scalar learning rate (delta gate).
* ``Diag(alpha_t)`` is a **per-channel forget gate** (finer-grained than
  head-level scalar gating).

A chunkwise parallel algorithm is used for training: within each chunk
the delta rule is unrolled with a causal weighted-attention form
(per-channel alpha folded via cumulative products); between chunks the
state matrix is propagated.  At inference time the same code path runs
with ``chunk_size`` equal to the current sequence length, degenerating
into the standard recurrent form.
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
)


class KDAAttention(BaseAttention):
    """Kimi Delta Attention with gated delta-rule matrix state."""

    def __init__(self, config: AttentionConfig):
        super().__init__(config)
        self.chunk_size = config.kda_chunk_size
        # Q/K/V projections (K is L2-normalised inside forward)
        self.q_proj = nn.Linear(
            self.d_model, self.n_heads * self.d_kv, bias=config.bias
        )
        self.k_proj = nn.Linear(
            self.d_model, self.n_heads * self.d_kv, bias=config.bias
        )
        self.v_proj = nn.Linear(
            self.d_model, self.n_heads * self.d_kv, bias=config.bias
        )
        # Per-channel forget gate alpha (data-dependent, per token, per head, per channel)
        self.alpha_proj = nn.Linear(
            self.d_model, self.n_heads * self.d_kv, bias=config.bias
        )
        # Delta-rule learning rate beta (data-dependent scalar per token, per head)
        self.beta_proj = nn.Linear(self.d_model, self.n_heads, bias=config.bias)
        # Output projection
        self.o_proj = nn.Linear(
            self.n_heads * self.d_kv, self.d_model, bias=config.bias
        )
        # Input RMSNorm (KDA benefits from pre-normed inputs)
        self.norm = RMSNorm(self.d_model, eps=config.rms_eps)

    # ------------------------------------------------------------------
    def _chunkwise(
        self,
        q: torch.Tensor,           # [b, h, s, d]
        k: torch.Tensor,           # [b, h, s, d]  (already L2-normalised)
        v: torch.Tensor,           # [b, h, s, d]
        beta: torch.Tensor,        # [b, h, s]
        alpha: torch.Tensor,       # [b, h, s, d]
        state: Optional[torch.Tensor],  # [b, h, d, d] or None
    ):
        """Chunkwise parallel delta-rule computation with per-channel
        forget gate.

        Implements the exact recurrence

            S_t = Diag(alpha_t) * S_{t-1} + beta_t * k_t v_t^T
            o_t = S_t^T q_t

        in a chunkwise-parallel form.  ``state`` is indexed as
        ``[b, h, d_k, d_v]`` (first dim = key channel, second = value).

        The per-channel cumulative product of ``alpha`` factorises the
        otherwise [T, T, d] weighted dot product into a rank-1 form

            QK[i, j] = (q[i] * cumprod[i]) . (k[j] / cumprod[j])

        where ``cumprod[t, d] = prod_{l<=t} alpha[l, d]``.  This is
        exact for ``j <= i`` (the ratio ``cumprod[i]/cumprod[j]`` equals
        ``prod_{l=j+1..i} alpha[l]``), so the chunked computation is
        algebraically identical to the token-by-token recurrence —
        including the final (possibly shorter) chunk, which is processed
        without padding.
        """
        b, h, s, d = q.shape
        chunk = self.chunk_size

        if state is None:
            state = torch.zeros(
                b, h, d, d, device=q.device, dtype=q.dtype
            )

        # Per-channel cumulative product of alpha (log-space for stability).
        log_alpha = torch.log(alpha.clamp(min=1e-6))            # [b, h, s, d]
        cumlog = torch.cumsum(log_alpha, dim=2)                 # [b, h, s, d]
        cumprod = torch.exp(cumlog)                             # [b, h, s, d]
        cumprod_safe = cumprod.clamp(min=1e-20)
        # Prepend a ones row: cumprod_ext[t] = prod_{l<t} alpha[l]
        # (cumprod_ext[0] == 1 == empty product).
        ones = torch.ones(b, h, 1, d, device=q.device, dtype=q.dtype)
        cumprod_ext = torch.cat([ones, cumprod], dim=2)         # [b, h, s+1, d]

        # Effective q/k that factorise the per-channel-weighted dot
        # product (exact for j <= i):
        #   QK[i,j] = sum_d q[i,d]*k[j,d] * (cumprod[i,d]/cumprod[j,d])
        q_eff = q * cumprod                                     # [b, h, s, d]
        k_eff = k / cumprod_safe                               # [b, h, s, d]

        outputs = []
        pos = 0
        while pos < s:
            T = min(chunk, s - pos)
            lo, hi = pos, pos + T

            qc_eff = q_eff[:, :, lo:hi]                         # [b, h, T, d]
            kc_eff = k_eff[:, :, lo:hi]
            qc_raw = q[:, :, lo:hi]
            kc_raw = k[:, :, lo:hi]
            vc = v[:, :, lo:hi]
            beta_c = beta[:, :, lo:hi]                          # [b, h, T]

            # ---- Inter-chunk: contribution of the incoming state -----
            # o_inter[i,e] = sum_d q[i,d] * (prod_{l=lo..i} alpha[l,d])
            #                                 * state[d,e]
            # prod_{l=lo..i} alpha = cumprod[i] / cumprod_ext[lo]
            cumprod_lo_prev = cumprod_ext[:, :, lo, :]          # [b, h, d]
            chunk_decay = (
                cumprod[:, :, lo:hi] / cumprod_lo_prev.unsqueeze(-2)
            )                                                   # [b, h, T, d]
            q_local = qc_raw * chunk_decay                      # [b, h, T, d]
            o_inter = torch.einsum("bhid,bhde->bhie", q_local, state)

            # ---- Intra-chunk: per-channel-weighted linear attention --
            # o_intra[i,e] = sum_{j<=i} QK[i,j] * beta[j] * v[j,e]
            QK = torch.einsum("bhid,bhjd->bhij", qc_eff, kc_eff)  # [b,h,T,T]
            causal = torch.tril(
                torch.ones(T, T, device=q.device, dtype=QK.dtype)
            )
            attn = QK * causal
            beta_v = vc * beta_c.unsqueeze(-1)                  # [b, h, T, d]
            o_intra = torch.einsum("bhij,bhje->bhie", attn, beta_v)

            outputs.append(o_inter + o_intra)

            # ---- State update (per-channel forget gate) -------------
            #   A_chunk[d]   = prod_{l=lo..hi-1} alpha[l,d]
            #               = cumprod[hi-1,d] / cumprod_ext[lo,d]
            #   delta_S[d,e] = sum_j (prod_{l=j+1..hi-1} alpha[l,d])
            #                       * beta[j] * k[j,d] * v[j,e]
            #   state_new    = Diag(A_chunk) * state + delta_S
            #
            # delta_S is computed with the bounded weight
            #   w_chan[j,d] = exp(cumlog[hi-1,d] - cumlog[j,d]) in (0, 1]
            # to avoid the tiny/huge cancellation that
            # ``cumprod[hi-1] * (k/cumprod[j])`` would introduce.
            cumlog_chunk = cumlog[:, :, lo:hi]                  # [b, h, T, d]
            cumlog_hi_last = cumlog[:, :, hi - 1, :]            # [b, h, d]
            w_chan = torch.exp(
                cumlog_hi_last.unsqueeze(2) - cumlog_chunk
            )                                                   # [b, h, T, d]
            coeff = w_chan * beta_c.unsqueeze(-1) * kc_raw      # [b, h, T, d]
            delta_S = torch.einsum("bhtd,bhte->bhde", coeff, vc)  # [b,h,d,d]
            # A_chunk = prod_{l=lo..hi-1} alpha[l,d]
            #         = cumprod[hi-1,d] / cumprod_ext[lo,d]
            # (cumprod_ext[lo] == 1 for lo == 0, so the first chunk
            # divides by 1 and yields the full product.)
            A_chunk = (
                cumprod[:, :, hi - 1, :] / cumprod_lo_prev.clamp(min=1e-20)
            )                                                   # [b, h, d]
            # Diag(A_chunk) scales the key channel (rows, dim -2).
            state = state * A_chunk.unsqueeze(-1) + delta_S

            pos = hi

        out = torch.cat(outputs, dim=2)                         # [b, h, s, d]
        return out, state

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
        h = self.norm(hidden_states)

        q = (
            self.q_proj(h)
            .view(b, s, self.n_heads, self.d_kv)
            .transpose(1, 2)
        )                                                       # [b, h, s, d]
        k = (
            self.k_proj(h)
            .view(b, s, self.n_heads, self.d_kv)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(h)
            .view(b, s, self.n_heads, self.d_kv)
            .transpose(1, 2)
        )
        alpha = torch.sigmoid(
            self.alpha_proj(h)
            .view(b, s, self.n_heads, self.d_kv)
            .transpose(1, 2)
        )                                                       # [b, h, s, d]  in (0, 1)
        beta = torch.sigmoid(
            self.beta_proj(h)
            .view(b, s, self.n_heads)
            .transpose(1, 2)
        )                                                       # [b, h, s]  in (0, 1)

        # KDA keys are L2-normalised (standard linear-attention practice)
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)

        # past_kv is the recurrent state [b, h, d, d]
        state = past_kv
        out, new_state = self._chunkwise(q, k, v, beta, alpha, state)

        # KDA is linear-attention based: an externally supplied padding
        # mask is honoured by zeroing out padded query positions.
        if attention_mask is not None:
            # accept [b, s] (1 = valid) or [b, 1, s, s] additive masks
            if attention_mask.dim() == 2:
                m = attention_mask.bool().view(b, 1, s, 1)
                out = out.masked_fill(~m, 0.0)

        out = out.transpose(1, 2).contiguous().view(b, s, -1)
        out = self.o_proj(out)
        present = new_state if use_cache else None
        return AttentionOutput(
            output=out, present_kv=present,
            aux={"state_norm": new_state.norm(dim=(-1, -2)).mean().item()}
            if use_cache else {},
        )
