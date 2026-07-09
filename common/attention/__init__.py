"""Hybrid-M3 attention modules.

Eight attention types plus a dynamic controller and a unified scheduler:

* ``MLAAttention``                — DeepSeek Multi-head Latent Attention
                                    (QK-Norm + decoupled RoPE, d_c = d_model/4)
* ``KDAAttention``                — Kimi Delta Attention
                                    (gated delta-rule matrix state)
* ``LightningAttention``          — Lightning Attention
                                    (intra/inter-block decomposition)
* ``SlidingWindowAttention``      — Sliding-window attention
                                    (window = 512, optional bidirectional)
* ``MMAAttention``                — Modality-Mutual Attention
                                    (image<->text bidirectional mask wrapper)
* ``MoHAttention``                — Mixture-of-Head Attention
                                    (shared + Top-K dynamic heads)
* ``GatedAttention``              — Rank-64 low-rank gated attention
* ``DynamicAttentionController``  — <1% parameter soft-routing controller
* ``UnifiedAttentionScheduler``   — 4-layer cycle scheduler
                                    (4k+1=KDA, 4k+2=KDA+MoH,
                                     4k+3=KDA, 4k+4=MLA+QKNorm+MMA)

Shared utilities (``AttentionConfig``, ``AttentionOutput``, ``RMSNorm``,
``apply_rope``, ``BaseAttention``) are re-exported from :mod:`.base`.
"""

from .base import (
    AttentionConfig,
    AttentionOutput,
    BaseAttention,
    RMSNorm,
    apply_rope,
    rotate_half,
    sdpa,
)
from .mla import MLAAttention
from .kda import KDAAttention
from .lightning import LightningAttention
from .sliding import SlidingWindowAttention
from .mma import MMAAttention
from .moh import MoHAttention
from .gated import GatedAttention
from .dynamic import DynamicAttentionController
from .unified import UnifiedAttentionScheduler

__all__ = [
    # base
    "AttentionConfig",
    "AttentionOutput",
    "BaseAttention",
    "RMSNorm",
    "apply_rope",
    "rotate_half",
    "sdpa",
    # concrete attention modules
    "MLAAttention",
    "KDAAttention",
    "LightningAttention",
    "SlidingWindowAttention",
    "MMAAttention",
    "MoHAttention",
    "GatedAttention",
    # controller + scheduler
    "DynamicAttentionController",
    "UnifiedAttentionScheduler",
]
