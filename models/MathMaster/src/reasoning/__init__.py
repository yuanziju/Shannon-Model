"""MathMaster reasoning 推理模块.

提供数学推理所需的全套能力, 支撑 Shannon 项目的理科推理与代码生成终极目标:

    - LongReasoningEngine:   超长推理引擎 (1M-10M 上下文 + 断点续推 + 压缩)
    - SelfPlayDebate:        Self-Play 多 Agent 辩论 (正方/反方/裁判)
    - CoTDistillation:       CoT 自我蒸馏 (长链 -> 压缩表示 -> 学生模型)
    - SelfPlaySolver:        Self-Play 迭代逼近求解器 (开放问题多轮逼近)
    - MathToT:               Tree-of-Thought 数学搜索 (BFS + 价值 + 剪枝 + 回溯)
    - ConjectureGenerator:   数学猜想生成器 (生成 + 数值验证 + 新颖性评估)
    - ReasoningCheckpoint:   推理检查点管理 (断点续推支持)

各模块均复用 15B (MoE) 循环主体权重, 通过角色 prompt 切换扮演不同推理角色,
与 ``common.agent`` 的 ReAct+CRA 单 Agent 框架及多 Agent 协作能力互补.
"""

from .conjecture_generator import (
    Conjecture,
    ConjectureGenerator,
    VerificationResult,
)
from .cot_distillation import CoTDistillation, DistillResult
from .long_reasoning import LongReasoningEngine, LongReasoningResult
from .reasoning_checkpoint import ReasoningCheckpoint
from .self_play_debate import (
    DebateResult,
    DebateRound,
    DebateSide,
    Judgment,
    SelfPlayDebate,
)
from .self_play_solver import (
    SelfPlaySolver,
    SolverIteration,
    SolverJudgment,
    SolverResult,
)
from .tree_of_thought import MathToT, ThoughtNode, ToTResult

__all__ = [
    # long reasoning
    "LongReasoningEngine",
    "LongReasoningResult",
    # self-play debate
    "SelfPlayDebate",
    "DebateResult",
    "DebateRound",
    "DebateSide",
    "Judgment",
    # cot distillation
    "CoTDistillation",
    "DistillResult",
    # self-play solver
    "SelfPlaySolver",
    "SolverResult",
    "SolverJudgment",
    "SolverIteration",
    # tree of thought
    "MathToT",
    "ThoughtNode",
    "ToTResult",
    # conjecture generator
    "ConjectureGenerator",
    "Conjecture",
    "VerificationResult",
    # checkpoint
    "ReasoningCheckpoint",
]
