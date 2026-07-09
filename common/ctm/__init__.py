"""CTM (Continuous Thought Machine) integration module.

Implements the CTM building blocks for the Shannon project (spec v4.0,
decisions C1-C17). All components are designed to reuse, not duplicate,
existing model state (decisions C5/C7).

Public API:
    NLMNeuron        - single neuron-level model (ctm.nlm)
    NLMLayer         - shared_mlp + neuron_adapter, warmup-frozen (ctm.nlm)
    MLASync          - c_kv . c_kv^T synchronization matrix (ctm.mla_sync)
    CTMDynamicLoss   - min-loss + max-certainty tick (ctm.ctm_loss)
    CTMRouter        - complexity-driven 3-way router (ctm.router)
"""
from .nlm import NLMNeuron, NLMLayer
from .mla_sync import MLASync
from .ctm_loss import CTMDynamicLoss
from .router import CTMRouter

__all__ = [
    "NLMNeuron",
    "NLMLayer",
    "MLASync",
    "CTMDynamicLoss",
    "CTMRouter",
]
