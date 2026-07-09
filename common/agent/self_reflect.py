"""自我反思 - 错误分类与重试策略.

SelfReflect 对 ReAct 循环轨迹进行回顾, 分类错误类型, 选择重试策略并
更新长程记忆. 与 ACT 自适应停止、Self-Play 经验回放强化联动.
"""

from __future__ import annotations

import time
from collections import Counter
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


class ErrorCategory(Enum):
    """错误分类, 决定重试策略. """

    NONE = "NONE"                      # 无错误
    TOOL_ERROR = "TOOL_ERROR"          # 工具调用失败/参数错误
    HALLUCINATION = "HALLUCINATION"    # 虚构事实/观测不符
    PLAN_ERROR = "PLAN_ERROR"          # 任务规划错误/死循环
    CONTEXT_LOSS = "CONTEXT_LOSS"      # 上下文丢失/遗忘关键信息
    TIMEOUT = "TIMEOUT"                # 循环超时/强制保底
    UNKNOWN = "UNKNOWN"


class RetryStrategy(Enum):
    NONE = "NONE"
    RETRY_SAME = "RETRY_SAME"            # 原样重试 (瞬时错误)
    RETRY_REPLAN = "RETRY_REPLAN"        # 重新规划任务分解
    RETRY_REFINE = "RETRY_REFINE"        # 精化工具参数
    RETRY_MEMORY = "RETRY_MEMORY"        # 注入长程记忆后再试
    ESCALATE = "ESCALATE"                # 升级到多 Agent 协作
    ABORT = "ABORT"                      # 放弃, 返回最佳尝试


# 错误类别 -> 推荐重试策略 映射
_DEFAULT_STRATEGY: Dict[ErrorCategory, RetryStrategy] = {
    ErrorCategory.NONE: RetryStrategy.NONE,
    ErrorCategory.TOOL_ERROR: RetryStrategy.RETRY_REFINE,
    ErrorCategory.HALLUCINATION: RetryStrategy.RETRY_MEMORY,
    ErrorCategory.PLAN_ERROR: RetryStrategy.RETRY_REPLAN,
    ErrorCategory.CONTEXT_LOSS: RetryStrategy.RETRY_MEMORY,
    ErrorCategory.TIMEOUT: RetryStrategy.RETRY_REPLAN,
    ErrorCategory.UNKNOWN: RetryStrategy.RETRY_SAME,
}

MAX_RETRIES = 3


class ReflectResult:
    """单次反思结论. """

    __slots__ = ("category", "strategy", "confidence", "reason", "feedback", "timestamp")

    def __init__(
        self,
        category: ErrorCategory,
        strategy: RetryStrategy,
        confidence: float,
        reason: str,
        feedback: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.category = category
        self.strategy = strategy
        self.confidence = confidence
        self.reason = reason
        self.feedback = feedback or {}
        self.timestamp = time.time()

    def should_retry(self) -> bool:
        return self.strategy not in (RetryStrategy.NONE, RetryStrategy.ABORT)


class SelfReflect:
    """自我反思模块.

    Args:
        classifier: 自定义错误分类器, 签名
            ``fn(trace: list[dict]) -> (ErrorCategory, float, str)``.
            默认使用基于规则的启发式分类.
        max_retries: 最大重试次数 (spec 建议 <=3).
    """

    def __init__(
        self,
        classifier: Optional[Callable[[List[Dict[str, Any]]], Tuple[ErrorCategory, float, str]]] = None,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        self._classifier = classifier or self._default_classify
        self.max_retries = max(0, int(max_retries))
        self._history: List[ReflectResult] = []
        self._error_stats: Counter = Counter()

    # ------------------------------------------------------------------ #
    # 反思入口
    # ------------------------------------------------------------------ #
    def reflect(self, trace: Sequence[Dict[str, Any]]) -> ReflectResult:
        """对一次 ReAct 轨迹进行反思. """
        category, confidence, reason = self._classifier(list(trace))
        strategy = self._select_strategy(category, confidence, len(self._history))
        feedback = self._build_feedback(category, trace)
        result = ReflectResult(category, strategy, confidence, reason, feedback)
        self._history.append(result)
        self._error_stats[category] += 1
        return result

    def reflect_and_retry(
        self,
        run_fn: Callable[[Optional[Dict[str, Any]]], Tuple[str, List[Dict[str, Any]]]],
        initial_trace: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Tuple[str, List[ReflectResult], List[Dict[str, Any]]]:
        """反思 + 自动重试循环.

        Args:
            run_fn: 执行一次 ReAct 任务, 接收反思 feedback, 返回
                ``(respond, trace)``.
        Returns:
            ``(final_respond, reflections, final_trace)``.
        """
        reflections: List[ReflectResult] = []
        feedback: Optional[Dict[str, Any]] = None
        trace: List[Dict[str, Any]] = list(initial_trace or [])
        respond = ""
        for attempt in range(self.max_retries + 1):
            respond, trace = run_fn(feedback)
            result = self.reflect(trace)
            reflections.append(result)
            if not result.should_retry():
                break
            feedback = result.feedback
        return respond, reflections, trace

    # ------------------------------------------------------------------ #
    # 默认启发式分类器
    # ------------------------------------------------------------------ #
    def _default_classify(
        self, trace: List[Dict[str, Any]]
    ) -> Tuple[ErrorCategory, float, str]:
        if not trace:
            return ErrorCategory.UNKNOWN, 0.3, "empty_trace"
        last = trace[-1]
        # 强制保底 -> 超时
        if last.get("forced"):
            return ErrorCategory.TIMEOUT, 0.8, "forced_respond_max_iter"
        # 统计错误观测
        obs = [t for t in trace if t.get("state") == "OBSERVATION"]
        err_obs = [o for o in obs if "error" in str(o.get("content", "")).lower()]
        if err_obs:
            return ErrorCategory.TOOL_ERROR, 0.75, f"{len(err_obs)}_tool_errors"
        # 死循环: 多次相同 ACTION
        actions = [t.get("content") for t in trace if t.get("state") == "ACTION"]
        if len(actions) - len(set(actions)) >= 2:
            return ErrorCategory.PLAN_ERROR, 0.7, "repeated_actions_loop"
        # 上下文丢失: RESPOND 内容为空或截断
        responds = [t for t in trace if t.get("state") == "RESPOND"]
        if responds and not responds[-1].get("content", "").strip():
            return ErrorCategory.CONTEXT_LOSS, 0.6, "empty_respond"
        return ErrorCategory.NONE, 0.9, "success"

    def _select_strategy(
        self, category: ErrorCategory, confidence: float, history_len: int
    ) -> RetryStrategy:
        base = _DEFAULT_STRATEGY.get(category, RetryStrategy.RETRY_SAME)
        # 多次失败后升级
        if history_len >= self.max_retries:
            return RetryStrategy.ABORT
        if category in (ErrorCategory.PLAN_ERROR, ErrorCategory.TIMEOUT) and history_len >= 1:
            return RetryStrategy.ESCALATE
        return base

    @staticmethod
    def _build_feedback(
        category: ErrorCategory, trace: Sequence[Dict[str, Any]]
    ) -> Dict[str, Any]:
        return {
            "error_category": category.value,
            "hint": _DEFAULT_STRATEGY[category].value,
            "failing_steps": [
                {"state": t.get("state"), "iteration": t.get("iteration")}
                for t in trace
                if t.get("final") is False
            ],
        }

    # ------------------------------------------------------------------ #
    # 统计
    # ------------------------------------------------------------------ #
    @property
    def history(self) -> List[ReflectResult]:
        return list(self._history)

    @property
    def error_stats(self) -> Dict[str, int]:
        return dict(self._error_stats)

    def clear(self) -> None:
        self._history.clear()
        self._error_stats.clear()


__all__ = ["SelfReflect", "ReflectResult", "ErrorCategory", "RetryStrategy", "MAX_RETRIES"]
