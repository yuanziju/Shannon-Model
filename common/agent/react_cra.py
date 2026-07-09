"""ReAct+CRA Agent - 意图识别 + 状态跟踪 + 任务规划.

ReActCRAAgent 是统一对话 Agent 框架 (决策11): 内嵌 ReAct 推理引擎,
采用 CRA (Conversational Reasoning with Actions) 格式管理交错多轮对话
与工具调用. 单 Agent 框架与多 Agent 协作层解耦, 可独立运行.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

from .orchestrator import OrchestratorMode, ToolCall, ToolOrchestrator
from .runtime import AgentRuntime, ReActState


class Intent(Enum):
    """用户意图分类. """

    QA = "QA"                # 问答
    REASONING = "REASONING"  # 理科推理/数学
    GENERATION = "GENERATION"  # 代码/文本生成
    EDITING = "EDITING"      # 编辑/重构
    TOOL_USE = "TOOL_USE"    # 工具调用
    CHITCHAT = "CHITCHAT"    # 闲聊
    UNKNOWN = "UNKNOWN"


@dataclass
class SlotState:
    """对话槽位状态 (状态跟踪). """

    slots: Dict[str, Any] = field(default_factory=dict)
    task_progress: float = 0.0  # 0-1
    pending_tools: List[ToolCall] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def fill(self, key: str, value: Any) -> None:
        self.slots[key] = value

    def reset(self) -> None:
        self.slots.clear()
        self.task_progress = 0.0
        self.pending_tools.clear()
        self.errors.clear()


@dataclass
class CRAEntry:
    """CRA 格式单条记录. """

    role: str  # user / assistant / observation
    content: str = ""
    think: str = ""
    action: str = ""
    respond: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        d = {"role": self.role, "content": self.content}
        if self.think:
            d["think"] = self.think
        if self.action:
            d["action"] = self.action
        if self.respond:
            d["respond"] = self.respond
        return d


class ReActCRAAgent:
    """ReAct+CRA 统一对话 Agent.

    组合三个子模块:
        1. 意图识别 (轻量分类头 + 循环主体特征)
        2. 状态跟踪 (槽位/任务进度, 复用长程记忆模块)
        3. 任务规划 (多步分解, 对接 AgentRuntime + ToolOrchestrator)

    Args:
        model_call: 15B 循环主体调用 (见 AgentRuntime).
        intent_classifier: 自定义意图分类器
            ``fn(user_input: str, ctx: dict) -> Intent``.
        tools: 工具编排器实例或注册表.
        max_iter: ReAct 最大迭代.
        memory: 可选长程记忆 (LongTermMemory), 用于状态跟踪持久化.
    """

    def __init__(
        self,
        model_call: Optional[Callable[..., str]] = None,
        intent_classifier: Optional[Callable[[str, Dict[str, Any]], Intent]] = None,
        tools: Optional[ToolOrchestrator] = None,
        max_iter: int = 8,
        memory: Optional[Any] = None,
    ) -> None:
        self.runtime = AgentRuntime(model_call=model_call, max_iter=max_iter)
        self.orchestrator = tools if isinstance(tools, ToolOrchestrator) else ToolOrchestrator(tools or {})
        self._classifier = intent_classifier or self._default_classifier
        self.memory = memory
        self._session: List[CRAEntry] = []
        self._state = SlotState()
        self._context: Deque[Dict[str, Any]] = deque(maxlen=32)

    # ------------------------------------------------------------------ #
    # 意图识别
    # ------------------------------------------------------------------ #
    def recognize_intent(self, user_input: str) -> Intent:
        ctx = {"history": list(self._context)}
        intent = self._classifier(user_input, ctx)
        return intent

    def _default_classifier(self, user_input: str, ctx: Dict[str, Any]) -> Intent:
        text = (user_input or "").lower()
        if any(k in text for k in ("证明", "求解", "计算", "prove", "solve", "compute", "求")):
            return Intent.REASONING
        if any(k in text for k in ("写代码", "实现", "生成", "write", "implement", "generate", "code")):
            return Intent.GENERATION
        if any(k in text for k in ("修改", "重构", "编辑", "edit", "refactor", "fix", "修复")):
            return Intent.EDITING
        if any(k in text for k in ("查询", "调用", "搜索", "search", "call", "api", "工具")):
            return Intent.TOOL_USE
        if any(k in text for k in ("你好", "嗨", "hello", "hi", "谢谢")):
            return Intent.CHITCHAT
        if "?" in text or "？" in text or "什么" in text or "怎么" in text:
            return Intent.QA
        return Intent.UNKNOWN

    # ------------------------------------------------------------------ #
    # 状态跟踪
    # ------------------------------------------------------------------ #
    @property
    def state(self) -> SlotState:
        return self._state

    def update_state(self, user_input: str, intent: Intent) -> None:
        # 槽位填充: 简化实现, 记录意图与最近输入
        self._state.fill("intent", intent.value)
        self._state.fill("last_input", user_input)
        self._state.task_progress = min(1.0, self._state.task_progress + 0.2)

    def reset_state(self) -> None:
        self._state.reset()
        self._session.clear()
        self._context.clear()

    # ------------------------------------------------------------------ #
    # 任务规划
    # ------------------------------------------------------------------ #
    def plan(self, user_input: str, intent: Intent) -> List[ToolCall]:
        """根据意图生成工具调用计划 (多步任务分解). """
        plan: List[ToolCall] = []
        if intent == Intent.TOOL_USE:
            plan.append(ToolCall("search", args={"query": user_input}, call_id="step_1"))
        elif intent == Intent.REASONING:
            plan.append(ToolCall("calculator", args={"expr": user_input}, call_id="step_1"))
        elif intent == Intent.GENERATION:
            plan.append(ToolCall("code_runner", args={"spec": user_input}, call_id="step_1"))
        elif intent == Intent.EDITING:
            plan.append(ToolCall("diff", args={"input": user_input}, call_id="step_1"))
        # 简单链式: QA/CHITCHAT 无需工具
        self._state.pending_tools = list(plan)
        return plan

    # ------------------------------------------------------------------ #
    # 主对话入口 (CRA 格式)
    # ------------------------------------------------------------------ #
    def chat(self, user_input: str) -> Dict[str, Any]:
        """处理一轮用户输入, 返回 CRA 格式响应. """
        intent = self.recognize_intent(user_input)
        self.update_state(user_input, intent)
        plan = self.plan(user_input, intent)

        # 执行工具计划 (若有)
        observations: List[Dict[str, Any]] = []
        if plan:
            mode = OrchestratorMode.CHAIN if len(plan) == 1 else OrchestratorMode.MIXED
            observations = list(self.orchestrator.execute(plan, mode=mode).values())

        ctx = {
            "intent": intent.value,
            "observations": observations,
            "slots": self._state.slots,
            "history": list(self._context),
        }
        # 接入 ReAct 推理引擎
        respond = self.runtime.run(user_input, context=ctx)

        # 写入 CRA 会话
        user_entry = CRAEntry(role="user", content=user_input)
        assistant_entry = CRAEntry(
            role="assistant",
            content=respond,
            think=self._extract(ReActState.THINK),
            action=";".join(c.name for c in plan),
            respond=respond,
        )
        self._session.append(user_entry)
        if observations:
            self._session.append(
                CRAEntry(role="observation", content=str(observations))
            )
        self._session.append(assistant_entry)
        self._context.append({"role": "user", "content": user_input})
        self._context.append({"role": "assistant", "content": respond})
        self._state.task_progress = 1.0

        return assistant_entry.to_dict()

    # ------------------------------------------------------------------ #
    # 会话导出 (CRA 数据集格式, 用于 SFT/对齐训练)
    # ------------------------------------------------------------------ #
    def export_cra(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self._session]

    @property
    def session(self) -> List[CRAEntry]:
        return list(self._session)

    # ------------------------------------------------------------------ #
    # 内部
    # ------------------------------------------------------------------ #
    def _extract(self, state: ReActState) -> str:
        for t in reversed(self.runtime.trace):
            if t.get("state") == state.value:
                return t.get("content", "")
        return ""


__all__ = ["ReActCRAAgent", "Intent", "SlotState", "CRAEntry"]
