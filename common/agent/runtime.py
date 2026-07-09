"""Agent运行时 - ReAct循环.

实现 ReAct (Reason+Act) 推理循环，复用 15B (MoE) 循环主体权重。
循环四阶段: <THINK> -> <ACTION> -> <OBSERVATION> -> <RESPOND>，
与本项目动态循环深度 (1-32 次) 天然契合。
"""

from __future__ import annotations

import time
from collections import deque
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple


class ReActState(Enum):
    """ReAct 四阶段状态机. """

    THINK = "THINK"
    ACTION = "ACTION"
    OBSERVATION = "OBSERVATION"
    RESPOND = "RESPOND"
    DONE = "DONE"


# 循环深度约束 (spec: RDT 循环块 1-32 次动态迭代)
MIN_LOOP_DEPTH = 1
MAX_LOOP_DEPTH = 32
DEFAULT_MAX_ITER = 8


class AgentRuntime:
    """单 Agent ReAct 推理运行时.

    Args:
        model_call: 调用 15B 循环主体的可调用对象, 签名
            ``model_call(prompt: str, state: ReActState, context: dict) -> str``.
        tool_executor: 工具执行器, 签名
            ``tool_executor(action: str) -> str``. 默认为空实现.
        max_iter: 单次任务最大 ReAct 迭代次数.
        silent_thinking: 是否启用 Silent Thinking (仅最终步计算 loss).
    """

    STATE_ORDER = (
        ReActState.THINK,
        ReActState.ACTION,
        ReActState.OBSERVATION,
        ReActState.RESPOND,
    )

    def __init__(
        self,
        model_call: Optional[Callable[..., str]] = None,
        tool_executor: Optional[Callable[[str], str]] = None,
        max_iter: int = DEFAULT_MAX_ITER,
        silent_thinking: bool = True,
    ) -> None:
        self.model_call = model_call or (lambda *a, **k: "")
        self.tool_executor = tool_executor or (lambda action: f"<obs:{action}>")
        self.max_iter = max(1, int(max_iter))
        self.silent_thinking = silent_thinking

        # 动态循环深度 (1-32), 由 ACT 自适应停止模块反馈调节
        self._loop_depth: int = 1
        self._trace: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # 公共接口
    # ------------------------------------------------------------------ #
    @property
    def loop_depth(self) -> int:
        """当前动态循环深度, 取值 [1, 32]. """
        return self._loop_depth

    def set_loop_depth(self, depth: int) -> None:
        self._loop_depth = max(MIN_LOOP_DEPTH, min(MAX_LOOP_DEPTH, int(depth)))

    @property
    def trace(self) -> List[Dict[str, Any]]:
        """返回本轮推理的完整轨迹 (供自我反思/记忆写入). """
        return list(self._trace)

    def reset(self) -> None:
        self._trace.clear()
        self._loop_depth = 1

    def run(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> str:
        """执行一次完整 ReAct 循环, 返回最终 <RESPOND> 内容.

        循环在以下情况终止:
            1. 产生 <RESPOND> 输出 (任务完成)
            2. 达到 ``max_iter`` 上限 (强制 RESPOND 保底)
            3. ACT 自适应停止信号触发
        """
        self.reset()
        ctx = dict(context or {})
        ctx["user_input"] = user_input

        for iteration in range(1, self.max_iter + 1):
            # 动态循环深度: 简单任务浅迭代, 复杂任务深迭代
            self._adapt_depth(user_input, iteration, ctx)

            thought = self._step(ReActState.THINK, user_input, ctx)
            action = self._step(ReActState.ACTION, thought, ctx)

            if self._is_terminal(thought, action):
                # 无需工具调用, 直接进入 RESPOND
                respond = self._step(ReActState.RESPOND, thought, ctx)
                self._record(ReActState.RESPOND, respond, iteration, final=True)
                return respond

            observation = self.tool_executor(action)
            self._record(ReActState.OBSERVATION, observation, iteration)
            ctx.setdefault("observations", []).append(observation)

            # 尝试 RESPOND
            respond = self._step(ReActState.RESPOND, observation, ctx)
            if self._is_complete(respond):
                self._record(ReActState.RESPOND, respond, iteration, final=True)
                return respond
            self._record(ReActState.RESPOND, respond, iteration, final=False)
            # 未完成 -> 回到 THINK 继续

        # 达到上限, 强制保底响应
        respond = self._step(ReActState.RESPOND, ctx.get("observations", [""])[-1], ctx)
        self._record(ReActState.RESPOND, respond, self.max_iter, final=True, forced=True)
        return respond

    # ------------------------------------------------------------------ #
    # 内部方法
    # ------------------------------------------------------------------ #
    def _step(self, state: ReActState, prompt: str, ctx: Dict[str, Any]) -> str:
        out = self.model_call(prompt, state, ctx)
        self._record(state, out, len(self._trace) // 4 + 1)
        return out

    def _record(
        self,
        state: ReActState,
        content: str,
        iteration: int,
        final: bool = False,
        forced: bool = False,
    ) -> None:
        self._trace.append(
            {
                "state": state.value,
                "content": content,
                "iteration": iteration,
                "loop_depth": self._loop_depth,
                "final": final,
                "forced": forced,
                "ts": time.time(),
            }
        )

    def _adapt_depth(self, user_input: str, iteration: int, ctx: Dict[str, Any]) -> None:
        """根据输入复杂度与迭代轮次自适应调节循环深度 (1-32). """
        # 启发式: token 长度 + 是否含推理关键词
        complexity = len(user_input.split())
        keywords = ("证明", "求解", "推导", "prove", "solve", "derive", "debug", "重构")
        if any(kw in user_input.lower() for kw in keywords):
            complexity += 8
        depth = MIN_LOOP_DEPTH + min(MAX_LOOP_DEPTH - 1, complexity // 4 + iteration - 1)
        self.set_loop_depth(depth)

    @staticmethod
    def _is_terminal(thought: str, action: str) -> bool:
        a = (action or "").strip().lower()
        return a in ("", "none", "noop", "no-op", "n/a") or "<respond>" in (thought or "").lower()

    @staticmethod
    def _is_complete(respond: str) -> bool:
        return bool(respond) and not respond.strip().endswith("...")


__all__ = ["AgentRuntime", "ReActState"]
