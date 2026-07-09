"""MLA latent-variable synchronization for CTM (decisions C5/C14).

Per decision C5: the synchronization matrix reuses MLA latent variables
(c_kv) and does not introduce an independent module. Per decision C14:
synchronization is trained with the c_kv . c_kv^T product and validated on
code tasks.
"""
from __future__ import annotations

from typing import List, Tuple

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:  # torch optional at import / py_compile time
    torch = None
    nn = None
    F = None
    _HAS_TORCH = False

if _HAS_TORCH:
    _Module = nn.Module
else:
    class _Module:  # pragma: no cover - torch-free import path
        def __init__(self, *args, **kwargs):
            pass


def _require_torch():
    if not _HAS_TORCH:
        raise RuntimeError("torch is required to run MLA sync operations")


class MLASync(_Module):
    """c_kv . c_kv^T synchronization matrix.

    Given MLA low-rank latent variables c_kv in R^{seq x d_c}, the
    synchronization matrix S = c_kv @ c_kv^T in R^{seq x seq} measures
    pairwise latent alignment across positions. The matrix is reused from
    MLA (no new parameters for the matrix itself) and used to gate the
    aggregation of neuron states across the sequence.
    """

    def __init__(self, d_c: int, num_neurons: int = 8, dropout: float = 0.0):
        super().__init__()
        _require_torch()
        self.d_c = d_c
        self.num_neurons = num_neurons
        self.scale = d_c ** -0.5
        # Only small projections on top of the *reused* c_kv (no independent
        # sync module - decision C5).
        self.q_proj = nn.Linear(d_c, d_c, bias=False)
        self.k_proj = nn.Linear(d_c, d_c, bias=False)
        # Per-neuron sync gate derived from the reused latent.
        self.neuron_gate = nn.Linear(d_c, num_neurons)
        self.dropout = nn.Dropout(dropout)

    def sync_matrix(self, c_kv):
        """Compute the c_kv . c_kv^T synchronization matrix.

        Args:
            c_kv: [batch, seq, d_c] MLA latent variables (reused, not
                  learned here).
        Returns:
            S: [batch, seq, seq] synchronization matrix (softmax-normalised).
        """
        _require_torch()
        q = self.q_proj(c_kv)  # [b, s, d_c]
        k = self.k_proj(c_kv)  # [b, s, d_c]
        # c_kv . c_kv^T style product, scaled.
        S = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        S = F.softmax(S, dim=-1)
        return S

    def forward(self, c_kv, neuron_states):
        """Synchronize per-neuron states across the sequence using c_kv . c_kv^T.

        Args:
            c_kv: [batch, seq, d_c] reused MLA latent variables.
            neuron_states: either a list of [batch, seq, d_state] per-neuron
                           state tensors, or a single
                           [batch, seq, d_state, num_neurons] tensor.
        Returns:
            (synced, S) where S is the [batch, seq, seq] sync matrix and
            synced matches the input structure (list or tensor).
        """
        _require_torch()
        S = self.sync_matrix(c_kv)  # [b, s, s]
        S = self.dropout(S)
        gates = torch.sigmoid(self.neuron_gate(c_kv))  # [b, s, num_neurons]

        if isinstance(neuron_states, (list, tuple)):
            synced: List = []
            for i, st in enumerate(neuron_states):
                synced_i = torch.matmul(S, st)  # [b, s, d_state]
                synced.append(synced_i * gates[..., i:i + 1])
            return synced, S

        # Tensor path: [b, s, d_state, n]
        b, s, d_state, n = neuron_states.shape
        perm = neuron_states.permute(0, 3, 2, 1)            # [b, n, d_state, s]
        flat = perm.reshape(b * n, d_state, s)              # [b*n, d_state, s]
        Sexp = S.unsqueeze(1).expand(b, n, s, s).reshape(b * n, s, s)
        out = torch.matmul(flat, Sexp)                     # [b*n, d_state, s]
        out = out.reshape(b, n, d_state, s).permute(0, 3, 2, 1)  # [b, s, d_state, n]
        out = out * gates.unsqueeze(2)                      # [b, s, 1, n] broadcast
        return out, S
