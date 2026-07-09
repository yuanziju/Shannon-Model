"""common.agent - Shannon Agent 能力架构模块.

提供单 Agent ReAct+CRA 框架与多 Agent 协作能力:
    - AgentRuntime:        ReAct 推理循环 (复用 15B 循环主体权重)
    - ToolOrchestrator:    CHAIN/PARALLEL/MIXED 工具编排
    - LongTermMemory:      cosine top-k 长程记忆检索
    - SelfReflect:         错误分类与重试
    - SocialDeploy:        多 Agent 角色扮演 / 图灵测试
    - SelfPlay:            proposer/solver/judge 自我对弈
    - ReActCRAAgent:       意图识别+状态跟踪+任务规划 统一对话 Agent
"""

from .memory import LongTermMemory, MemoryEntry, cosine_similarity
from .orchestrator import OrchestratorMode, ToolCall, ToolOrchestrator
from .react_cra import CRAEntry, Intent, ReActCRAAgent, SlotState
from .runtime import AgentRuntime, ReActState
from .self_play import GameRole, Outcome, PlayEpisode, SelfPlay
from .self_reflect import (
    ErrorCategory,
    ReflectResult,
    RetryStrategy,
    SelfReflect,
)
from .social import Persona, SocialDeploy, SocialMessage, TuringVerdict

__all__ = [
    # runtime
    "AgentRuntime",
    "ReActState",
    # orchestrator
    "ToolOrchestrator",
    "ToolCall",
    "OrchestratorMode",
    # memory
    "LongTermMemory",
    "MemoryEntry",
    "cosine_similarity",
    # self reflect
    "SelfReflect",
    "ReflectResult",
    "ErrorCategory",
    "RetryStrategy",
    # social
    "SocialDeploy",
    "Persona",
    "SocialMessage",
    "TuringVerdict",
    # self play
    "SelfPlay",
    "PlayEpisode",
    "GameRole",
    "Outcome",
    # react+cra
    "ReActCRAAgent",
    "Intent",
    "SlotState",
    "CRAEntry",
]
