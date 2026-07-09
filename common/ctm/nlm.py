"""NLM (Neuron-Level Model) for CTM integration (decisions C7/C9/C10/C13).

Per decision C7: NLM only enhances MoE expert activation functions and does
not dominate state transition. Per decision C10: only entity experts use
NLM; empty experts keep the standard design. Per T3.10.2: NLM parameters
are frozen during the warmup window (~first 10% of training) for stable
convergence.
"""
from __future__ import annotations

from typing import List, Optional

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
        raise RuntimeError("torch is required to run NLM operations")


class NLMNeuron(_Module):
    """A single neuron-level model.

    Each neuron maintains its own latent state that evolves across CTM
    "ticks". The neuron output is a function of (i) the shared MLP
    projection of the input, (ii) a neuron-specific adapter, and (iii) the
    neuron's previous state. This realises the neuron-level model of CTM.
    """

    def __init__(self, d_model: int, d_state: int = 16, neuron_id: int = 0):
        super().__init__()
        _require_torch()
        self.d_model = d_model
        self.d_state = d_state
        self.neuron_id = neuron_id

        # Neuron-specific state projection (per-neuron weights).
        self.state_in = nn.Linear(d_model, d_state, bias=False)
        self.state_out = nn.Linear(d_state, d_model, bias=False)
        # Neuron-specific adapter gate modulating the shared representation.
        self.adapter = nn.Linear(d_model, d_model, bias=False)
        # Learnable per-neuron decay for state stability (LTI-like,
        # spectral radius < 1 enforced via tanh).
        self.decay = nn.Parameter(torch.zeros(d_state))
        self._frozen = False

    def init_state(self, batch_size: int, device=None):
        _require_torch()
        return torch.zeros(batch_size, self.d_state, device=device)

    def forward(self, x, state):
        """Compute one tick for this neuron.

        Args:
            x: input features [batch, d_model] (shared MLP output slice).
            state: previous neuron state [batch, d_state].
        Returns:
            (output [batch, d_model], new_state [batch, d_state]).
        """
        _require_torch()
        # Bounded decay keeps the state-update spectral radius < 1 (LTI stable).
        gate = torch.tanh(self.decay)
        new_state = gate * state + (1.0 - gate) * torch.tanh(self.state_in(x))
        # Neuron-specific adapter modulates the shared representation.
        output = self.adapter(x) + self.state_out(new_state)
        return output, new_state

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self._frozen = True

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True
        self._frozen = False


class NLMLayer(_Module):
    """NLM layer = shared_mlp + neuron_adapter (per-neuron models).

    Replaces/enhances the activation inside an MoE expert with a bank of
    neuron-level models. The shared MLP is shared across neurons for
    parameter efficiency; each neuron contributes its own adapter and state.

    Per T3.10.2 the NLM-specific parameters are frozen during warmup so the
    base expert converges first.
    """

    def __init__(self, d_model: int, num_neurons: int = 8, d_state: int = 16,
                 warmup_freeze: bool = True):
        super().__init__()
        _require_torch()
        self.d_model = d_model
        self.num_neurons = num_neurons
        self.d_state = d_state
        self.warmup_freeze = warmup_freeze

        # Shared MLP (parameter-efficient; shared across neurons).
        self.shared_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        # Per-neuron neuron-level models.
        self.neurons = nn.ModuleList([
            NLMNeuron(d_model, d_state=d_state, neuron_id=i)
            for i in range(num_neurons)
        ])
        # Aggregation weights mixing neuron outputs back to d_model.
        self.neuron_weights = nn.Parameter(torch.ones(num_neurons) / num_neurons)
        self.norm = nn.LayerNorm(d_model)
        self._frozen = False
        if warmup_freeze:
            self.freeze_nlm()

    def freeze_nlm(self):
        """Freeze NLM-specific params during warmup (shared MLP may train)."""
        for n in self.neurons:
            n.freeze()
        self._frozen = True

    def unfreeze_nlm(self):
        for n in self.neurons:
            n.unfreeze()
        self._frozen = False

    def is_frozen(self) -> bool:
        return getattr(self, "_frozen", False)

    def init_states(self, batch_size: int, device=None) -> List:
        return [n.init_state(batch_size, device) for n in self.neurons]

    def forward(self, x, states: Optional[List] = None):
        """Run one CTM tick over the NLM-enhanced activation.

        Args:
            x: [batch, d_model] input to the expert activation.
            states: list of per-neuron states from previous tick
                    (None -> freshly zero-initialised).
        Returns:
            (output [batch, d_model], new_states list).
        """
        _require_torch()
        bsz = x.shape[0]
        if states is None:
            states = self.init_states(bsz, device=x.device)
        shared = self.shared_mlp(x)
        outputs: List = []
        new_states: List = []
        for neuron, st in zip(self.neurons, states):
            out, ns = neuron(shared, st)
            outputs.append(out)
            new_states.append(ns)
        # Weighted aggregation of neuron outputs.
        w = F.softmax(self.neuron_weights, dim=0)
        stacked = torch.stack(outputs, dim=0)  # [num_neurons, batch, d_model]
        agg = (w.view(-1, 1, 1) * stacked).sum(dim=0)
        # Residual + norm: NLM *enhances* (not replaces) the activation (C7).
        out = self.norm(shared + agg)
        return out, new_states
