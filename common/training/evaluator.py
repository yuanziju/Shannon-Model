"""评估器 - 10 基准 + 智能触发.

Evaluator 在训练过程中按需触发评估 (避免每步全量评估的开销):
    - 10 个基准 (MMLU/GSM8K/HumanEval/SWE-bench/LiveCodeBench/...)
    - 智能触发: 基于训练步/loss 变化/啊哈时刻触发增量评估
    - 全量评估仅在阶段切换时进行
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


class Benchmark(Enum):
    """10 项评估基准. """

    MMLU = "MMLU"                       # 综合知识
    GSM8K = "GSM8K"                     # 数学推理
    HUMANEVAL = "HumanEval"             # 代码生成
    SWE_BENCH = "SWE-bench"             # 全库代码修复
    LIVECODEBENCH = "LiveCodeBench"     # 实时代码
    BBH = "BBH"                         # BIG-Bench Hard
    MATH = "MATH"                       # 高等数学
    NEEDLE_IN_CODEBASE = "Needle-in-Codebase"  # 全库理解 (5M 上下文)
    TURING_TEST = "TuringTest"          # 社交拟人
    SAFETY = "Safety"                   # 安全对齐


# spec 代码生成目标
BENCHMARK_TARGETS: Dict[Benchmark, float] = {
    Benchmark.HUMANEVAL: 0.85,
    Benchmark.SWE_BENCH: 0.30,
    Benchmark.LIVECODEBENCH: 0.60,
    Benchmark.NEEDLE_IN_CODEBASE: 0.90,  # 召回 >90%
    Benchmark.GSM8K: 0.90,
    Benchmark.MATH: 0.60,
    Benchmark.MMLU: 0.85,
    Benchmark.BBH: 0.80,
    Benchmark.TURING_TEST: 0.70,
    Benchmark.SAFETY: 0.99,
}

# 各基准评估开销 (相对单位, 决定触发频率)
BENCHMARK_COST: Dict[Benchmark, int] = {
    Benchmark.MMLU: 10,
    Benchmark.GSM8K: 8,
    Benchmark.HUMANEVAL: 15,
    Benchmark.SWE_BENCH: 50,
    Benchmark.LIVECODEBENCH: 20,
    Benchmark.BBH: 12,
    Benchmark.MATH: 10,
    Benchmark.NEEDLE_IN_CODEBASE: 40,
    Benchmark.TURING_TEST: 30,
    Benchmark.SAFETY: 25,
}


@dataclass
class EvalResult:
    """单次评估结果. """

    benchmark: Benchmark
    score: float                      # 0-1
    target: float
    samples: int
    triggered_by: str                 # 触发原因
    passed: bool = False
    timestamp: float = field(default_factory=time.time)
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.passed = self.score >= self.target


class Evaluator:
    """训练过程评估器 (智能触发).

    Args:
        eval_fn: 评估单个基准的可调用
            ``fn(benchmark, model_ckpt) -> (score, samples)``.
        trigger_interval: 步数触发间隔 (每 N 步触发轻量评估).
        loss_delta_threshold: loss 变化触发阈值.
        aha_trigger: 是否在啊哈时刻触发评估.
    """

    # 轻量基准 (低开销, 频繁触发)
    LIGHT_BENCHMARKS = (
        Benchmark.MMLU, Benchmark.GSM8K, Benchmark.HUMANEVAL, Benchmark.BBH
    )
    # 重量基准 (高开销, 仅阶段切换/啊哈时触发)
    HEAVY_BENCHMARKS = (
        Benchmark.SWE_BENCH, Benchmark.LIVECODEBENCH, Benchmark.NEEDLE_IN_CODEBASE,
        Benchmark.TURING_TEST, Benchmark.SAFETY, Benchmark.MATH,
    )

    def __init__(
        self,
        eval_fn: Optional[Callable[[Benchmark, Dict[str, Any]], Tuple[float, int]]] = None,
        trigger_interval: int = 500,
        loss_delta_threshold: float = 0.05,
        aha_trigger: bool = True,
    ) -> None:
        self.eval_fn = eval_fn or self._default_eval
        self.trigger_interval = max(1, int(trigger_interval))
        self.loss_delta_threshold = float(loss_delta_threshold)
        self.aha_trigger = bool(aha_trigger)

        self._results: List[EvalResult] = []
        self._latest: Dict[Benchmark, EvalResult] = {}
        self._last_eval_step: Dict[Benchmark, int] = {}
        self._loss_history: List[Tuple[int, float]] = []
        self._total_eval_cost = 0

    # ------------------------------------------------------------------ #
    # 智能触发
    # ------------------------------------------------------------------ #
    def maybe_evaluate(
        self,
        step: int,
        loss: float,
        checkpoint: Optional[Dict[str, Any]] = None,
        aha_moment: bool = False,
    ) -> List[EvalResult]:
        """根据触发策略决定是否评估, 返回本次触发的评估结果. """
        self._loss_history.append((step, loss))
        triggers: List[Tuple[Benchmark, str]] = []

        # 1. 步数触发 (轻量基准)
        if step % self.trigger_interval == 0:
            for b in self.LIGHT_BENCHMARKS:
                triggers.append((b, "step_interval"))

        # 2. loss 突变触发
        if self._loss_changed(step):
            triggers.append((Benchmark.GSM8K, "loss_delta"))

        # 3. 啊哈时刻触发 (重量基准)
        if aha_moment and self.aha_trigger:
            for b in self.HEAVY_BENCHMARKS:
                triggers.append((b, "aha_moment"))

        if not triggers:
            return []

        results: List[EvalResult] = []
        ckpt = checkpoint or {}
        for bench, reason in triggers:
            # 去重: 同一基准本步不重复评估
            if any(b == bench for b, _ in triggers[:triggers.index((bench, reason))]):
                continue
            res = self._evaluate_one(bench, ckpt, reason)
            results.append(res)
        return results

    def evaluate_full(self, checkpoint: Optional[Dict[str, Any]] = None) -> List[EvalResult]:
        """全量评估 (所有 10 基准), 阶段切换时调用. """
        ckpt = checkpoint or {}
        results: List[EvalResult] = []
        for bench in Benchmark:
            res = self._evaluate_one(bench, ckpt, "phase_switch")
            results.append(res)
        return results

    def _evaluate_one(
        self, bench: Benchmark, checkpoint: Dict[str, Any], reason: str
    ) -> EvalResult:
        score, samples = self.eval_fn(bench, checkpoint)
        res = EvalResult(
            benchmark=bench,
            score=max(0.0, min(1.0, float(score))),
            target=BENCHMARK_TARGETS[bench],
            samples=int(samples),
            triggered_by=reason,
            extra={"cost": BENCHMARK_COST[bench]},
        )
        self._results.append(res)
        self._latest[bench] = res
        self._total_eval_cost += BENCHMARK_COST[bench]
        return res

    def _loss_changed(self, step: int) -> bool:
        if len(self._loss_history) < 2:
            return False
        prev = self._loss_history[-2][1]
        cur = self._loss_history[-1][1]
        if prev == 0:
            return False
        return abs(cur - prev) / abs(prev) > self.loss_delta_threshold

    # ------------------------------------------------------------------ #
    # 默认评估 (模拟, 分数随 step 提升)
    # ------------------------------------------------------------------ #
    def _default_eval(self, bench: Benchmark, checkpoint: Dict[str, Any]) -> Tuple[float, int]:
        step = checkpoint.get("step", 0)
        target = BENCHMARK_TARGETS[bench]
        # 模拟: 分数渐进逼近目标
        progress = 1.0 - math.exp(-step / 5000.0)
        score = target * (0.3 + 0.7 * progress)
        samples = {Benchmark.HUMANEVAL: 164, Benchmark.SWE_BENCH: 300,
                   Benchmark.GSM8K: 1319, Benchmark.MMLU: 14042}.get(bench, 500)
        return score, samples

    # ------------------------------------------------------------------ #
    # 统计
    # ------------------------------------------------------------------ #
    @property
    def results(self) -> List[EvalResult]:
        return list(self._results)

    @property
    def latest_scores(self) -> Dict[str, float]:
        return {b.value: r.score for b, r in self._latest.items()}

    @property
    def pass_rate(self) -> float:
        if not self._latest:
            return 0.0
        passed = sum(1 for r in self._latest.values() if r.passed)
        return passed / len(self._latest)

    def summary(self) -> Dict[str, Any]:
        return {
            "evaluations_run": len(self._results),
            "benchmarks_evaluated": len(self._latest),
            "total_benchmarks": len(list(Benchmark)),
            "pass_rate": round(self.pass_rate, 3),
            "total_eval_cost": self._total_eval_cost,
            "latest_scores": self.latest_scores,
            "targets": {b.value: t for b, t in BENCHMARK_TARGETS.items()},
        }


__all__ = ["Evaluator", "EvalResult", "Benchmark", "BENCHMARK_TARGETS", "BENCHMARK_COST"]
