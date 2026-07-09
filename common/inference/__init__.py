"""common.inference - Shannon 推理引擎模块.

推理引擎层 (Layer 4) 核心组件:
    - RequestScheduler:       P0-P4 优先级 + 抢占 + 连续批处理
    - CacheManager:           PagedAttention + 4 层 KV 压缩 (~300x)
    - Quantizer:              逐组件量化 + 动态切换
    - MemoryManager:          Paged KV + SSM swap, 24-48GB 目标
    - StreamDecoder:          SSE 流式 + 拟人修订
    - MoESpeculativeDecoder:  MoE 层投机解码 (draft+verify, 1.5-2x)
"""

from .cache import (
    BLOCK_SIZE,
    LAYER_RATIO,
    CacheManager,
    CompressionLayer,
    DType,
    KVBlock,
    KVSequence,
)
from .memory import (
    DEFAULT_CPU_BUDGET,
    DEFAULT_GPU_BUDGET,
    MAX_LOOP_DEPTH,
    MIN_LOOP_DEPTH,
    Device,
    MemoryManager,
    MemoryPool,
    SSMState,
)
from .moe_spec_decode import (
    DEFAULT_DRAFT_K,
    MoESpeculativeDecoder,
    SpecStep,
    TARGET_SPEEDUP_MAX,
    TARGET_SPEEDUP_MIN,
)
from .quantizer import (
    DEFAULT_POLICY,
    PRECISION_PROFILE,
    Component,
    ComponentType,
    Precision,
    Quantizer,
)
from .scheduler import InferRequest, Priority, RequestScheduler
from .stream import (
    MAX_DELAY_MS,
    MAX_REVISION_RATE,
    MIN_DELAY_MS,
    StreamDecoder,
    StreamEvent,
    TokenChunk,
)

__all__ = [
    # scheduler
    "RequestScheduler",
    "InferRequest",
    "Priority",
    # cache
    "CacheManager",
    "KVBlock",
    "KVSequence",
    "CompressionLayer",
    "DType",
    "BLOCK_SIZE",
    "LAYER_RATIO",
    # quantizer
    "Quantizer",
    "Component",
    "Precision",
    "ComponentType",
    "DEFAULT_POLICY",
    "PRECISION_PROFILE",
    # memory
    "MemoryManager",
    "MemoryPool",
    "SSMState",
    "Device",
    "DEFAULT_GPU_BUDGET",
    "DEFAULT_CPU_BUDGET",
    "MIN_LOOP_DEPTH",
    "MAX_LOOP_DEPTH",
    # stream
    "StreamDecoder",
    "TokenChunk",
    "StreamEvent",
    "MAX_REVISION_RATE",
    "MIN_DELAY_MS",
    "MAX_DELAY_MS",
    # moe spec decode
    "MoESpeculativeDecoder",
    "SpecStep",
    "TARGET_SPEEDUP_MIN",
    "TARGET_SPEEDUP_MAX",
    "DEFAULT_DRAFT_K",
]
