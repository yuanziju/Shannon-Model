"""latent_decode: 隐空间解码模块 (B+C 融合架构).

实现层次化 NAR + 掩码精化 + 流匹配可选 + AR 保底的四层解码架构,
以及拟人流式输出前端与 Lean 验证器.

参考: spec.md §14.3 决策 L1-L15, AGENTS.md Agent 11 (LatentDecodeAgent).
"""

from __future__ import annotations

from .codebook import NeuroCodebook, CodebookConfig, BOUNDARY_TYPES, NUM_BOUNDARY_TYPES
from .mode_switch import (
    ModeSwitch,
    ModeSwitchConfig,
    LoRALinear,
    MODES,
    REASONING,
    DECODING,
)
from .hierarchical_nar import HierarchicalNAR, HierarchicalNARConfig
from .mask_refine import MaskRefinement, MaskRefinementConfig
from .flow_planner import FlowPlanner, FlowPlannerConfig, VelocityNet, SinusoidalTimeEmbedding
from .ar_fallback import (
    ARFallback,
    ARFallbackConfig,
    TOKEN_THRESHOLD,
    BLOCK_THRESHOLD,
    GLOBAL_THRESHOLD,
)
from .spec_decode import SpeculativeDecoder, SpeculativeDecoderConfig
from .human_stream import (
    HumanStream,
    HumanStreamConfig,
    StreamAction,
    StreamEvent,
    REVISION_RATE_CAP,
    MIN_DELAY_MS,
    MAX_DELAY_MS,
)
from .lean_verifier import LeanVerifier, LeanVerifierConfig, VerificationResult

__all__ = [
    # codebook
    "NeuroCodebook",
    "CodebookConfig",
    "BOUNDARY_TYPES",
    "NUM_BOUNDARY_TYPES",
    # mode_switch
    "ModeSwitch",
    "ModeSwitchConfig",
    "LoRALinear",
    "MODES",
    "REASONING",
    "DECODING",
    # hierarchical_nar
    "HierarchicalNAR",
    "HierarchicalNARConfig",
    # mask_refine
    "MaskRefinement",
    "MaskRefinementConfig",
    # flow_planner
    "FlowPlanner",
    "FlowPlannerConfig",
    "VelocityNet",
    "SinusoidalTimeEmbedding",
    # ar_fallback
    "ARFallback",
    "ARFallbackConfig",
    "TOKEN_THRESHOLD",
    "BLOCK_THRESHOLD",
    "GLOBAL_THRESHOLD",
    # spec_decode
    "SpeculativeDecoder",
    "SpeculativeDecoderConfig",
    # human_stream
    "HumanStream",
    "HumanStreamConfig",
    "StreamAction",
    "StreamEvent",
    "REVISION_RATE_CAP",
    "MIN_DELAY_MS",
    "MAX_DELAY_MS",
    # lean_verifier
    "LeanVerifier",
    "LeanVerifierConfig",
    "VerificationResult",
]
