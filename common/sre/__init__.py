"""sre: 特化推理引擎 (Specialized Reasoning Engine).

实现 SymPy / Lean / Python 三通道工具编码, Cross-Attention Fusion (Layer
8/16/24/32), 动态工具门控, 工具协调器 (Kahn 拓扑排序 + [IF:tool_failed])
与跨工具记忆.

参考: spec.md §5.6 / §7.1 / §7.3, AGENTS.md (T2.3 SRE 引擎).
"""

from __future__ import annotations

from .sympy_channel import (
    SymPyChannel,
    SymPyChannelConfig,
    ASTNodeEncoder,
    TreeTransformer,
    SYMPY_NODE_TYPES,
    NODE_TYPE_TO_ID,
    NUM_NODE_TYPES,
)
from .lean_channel import (
    LeanChannel,
    LeanChannelConfig,
    GoalContextTransformer,
    LEAN_TACTICS,
    TACTIC_TO_ID,
    NUM_TACTICS,
)
from .python_channel import (
    PythonChannel,
    PythonChannelConfig,
    TextEncoder,
    DataFrameEncoder,
    PlotEncoder,
    OUTPUT_TYPES,
    OUTPUT_TYPE_TO_ID,
    NUM_OUTPUT_TYPES,
)
from .fusion import (
    CrossAttentionFusion,
    CrossAttentionFusionConfig,
    CrossAttentionBlock,
    DEFAULT_FUSION_LAYERS,
    TOOL_CHANNELS,
    NUM_TOOL_CHANNELS,
)
from .tool_gating import (
    ToolGating,
    ToolGatingConfig,
    TASK_TYPES,
    TASK_TYPE_TO_ID,
    NUM_TASK_TYPES,
)
from .coordinator import (
    ToolCoordinator,
    CoordinatorConfig,
    ToolNode,
    ToolStatus,
)
from .tool_memory import (
    ToolMemory,
    ToolMemoryConfig,
    MemoryEntry,
    VariableType,
)

__all__ = [
    # sympy_channel
    "SymPyChannel",
    "SymPyChannelConfig",
    "ASTNodeEncoder",
    "TreeTransformer",
    "SYMPY_NODE_TYPES",
    "NODE_TYPE_TO_ID",
    "NUM_NODE_TYPES",
    # lean_channel
    "LeanChannel",
    "LeanChannelConfig",
    "GoalContextTransformer",
    "LEAN_TACTICS",
    "TACTIC_TO_ID",
    "NUM_TACTICS",
    # python_channel
    "PythonChannel",
    "PythonChannelConfig",
    "TextEncoder",
    "DataFrameEncoder",
    "PlotEncoder",
    "OUTPUT_TYPES",
    "OUTPUT_TYPE_TO_ID",
    "NUM_OUTPUT_TYPES",
    # fusion
    "CrossAttentionFusion",
    "CrossAttentionFusionConfig",
    "CrossAttentionBlock",
    "DEFAULT_FUSION_LAYERS",
    "TOOL_CHANNELS",
    "NUM_TOOL_CHANNELS",
    # tool_gating
    "ToolGating",
    "ToolGatingConfig",
    "TASK_TYPES",
    "TASK_TYPE_TO_ID",
    "NUM_TASK_TYPES",
    # coordinator
    "ToolCoordinator",
    "CoordinatorConfig",
    "ToolNode",
    "ToolStatus",
    # tool_memory
    "ToolMemory",
    "ToolMemoryConfig",
    "MemoryEntry",
    "VariableType",
]
