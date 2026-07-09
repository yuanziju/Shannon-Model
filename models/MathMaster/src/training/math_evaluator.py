"""MathEvaluator - 8 基准数学评估.

8 项数学评估基准, 覆盖从基础到前沿:

    MATH           高等数学竞赛题 (Hendrix/MATH)
    GSM8K          小学应用题 (多步算术推理)
    AIME           美国数学邀请赛 (整数答案 0-999)
    IMO            国际数学奥林匹克 (高难证明)
    Putnam         普特南数学竞赛 (本科级)
    LEAN_PROOF     Lean4 证明率 (形式化证明闭合率)
    CONJECTURE     猜想生成 (新颖性 + 可验证性)
    FRONTIER       前沿研究问题 (开放问题探索)

特性:
    - 各基准目标分数 (target) 与样本量
    - 智能触发: 训练步触发轻量基准, 阶段切换 / 答题突变触发重量基准
    - 默认评估器 (模拟, 分数随训练步渐进逼近目标)
    - 支持外部 eval_fn 注入真实评估逻辑
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


class MathBenchmark(Enum):
    """8 项数学评估基准. """

    MATH = "MATH"                 # 高等数学
    GSM8K = "GSM8K"               # 小学应用题
    AIME = "AIME"                 # 美国数学邀请赛
    IMO = "IMO"                   # 国际数学奥林匹克
    PUTNAM = "Putnam"             # 普特南
    LEAN_PROOF = "LeanProof"      # Lean4 证明率
    CONJECTURE = "Conjecture"     # 猜想生成
    FRONTIER = "Frontier"         # 前沿研究


# 各基准目标分数
BENCHMARK_TARGETS: Dict[MathBenchmark, float] = {
    MathBenchmark.MATH: 0.70,
    MathBenchmark.GSM8K: 0.95,
    MathBenchmark.AIME: 0.40,
    MathBenchmark.IMO: 0.15,
    MathBenchmark.PUTNAM: 0.25,
    MathBenchmark.LEAN_PROOF: 0.50,
    MathBenchmark.CONJECTURE: 0.35,
    MathBenchmark.FRONTIER: 0.10,
}

# 各基准默认样本量
BENCHMARK_SAMPLES: Dict[MathBenchmark, int] = {
    MathBenchmark.MATH: 5000,
    MathBenchmark.GSM8K: 1319,
    MathBenchmark.AIME: 30,
    MathBenchmark.IMO: 30,
    MathBenchmark.PUTNAM: 60,
    MathBenchmark.LEAN_PROOF: 200,
    MathBenchmark.CONJECTURE: 100,
    MathBenchmark.FRONTIER: 50,
}

# 各基准评估开销 (相对单位, 决定触发频率)
BENCHMARK_COST: Dict[MathBenchmark, int] = {
    MathBenchmark.MATH: 12,
    MathBenchmark.GSM8K: 8,
    MathBenchmark.AIME: 15,
    MathBenchmark.IMO: 25,
    MathBenchmark.PUTNAM: 20,
    MathBenchmark.LEAN_PROOF: 30,
    MathBenchmark.CONJECTURE: 28,
    MathBenchmark.FRONTIER: 40,
}

# 轻量基准 (低开销, 频繁触发)
LIGHT_BENCHMARKS: Tuple[MathBenchmark, ...] = (
    MathBenchmark.GSM8K, MathBenchmark.MATH, MathBenchmark.AIME,
)
# 重量基准 (高开销, 仅阶段切换 / 突变触发)
HEAVY_BENCHMARKS: Tuple[MathBenchmark, ...] = (
    MathBenchmark.IMO, MathBenchmark.PUTNAM, MathBenchmark.LEAN_PROOF,
    MathBenchmark.CONJECTURE, MathBenchmark.FRONTIER,
)


@dataclass
class MathEvalResult:
    """单次评估结果. """

    benchmark: MathBenchmark
    score: float                       # 0-1
    target: float
    samples: int
    triggered_by: str                  # 触发原因
    passed: bool = False
    timestamp: float = field(default_factory=time.time)
    extra: Dict[str, Any] = field(default_factory=dict)
    # 细分指标 (可选)
    sub_metrics: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.passed = self.score >= self.target


class MathEvaluator:
    """数学训练过程评估器 (智能触发).

    Args:
        eval_fn: 评估单个基准的可调用
            ``fn(benchmark, model_ckpt) -> (score, samples)``.
        trigger_interval: 步数触发间隔 (每 N 步触发轻量评估).
        loss_delta_threshold: loss 变化触发阈值.
        aha_trigger: 是否在啊哈时刻触发重量评估.
        targets: 各基准目标分数覆盖 (可选).
    """

    def __init__(
        self,
        eval_fn: Optional[Callable[[MathBenchmark, Dict[str, Any]], Tuple[float, int]]] = None,
        trigger_interval: int = 500,
        loss_delta_threshold: float = 0.05,
        aha_trigger: bool = True,
        targets: Optional[Dict[MathBenchmark, float]] = None,
    ) -> None:
        self.eval_fn = eval_fn or self._default_eval
        self.trigger_interval = max(1, int(trigger_interval))
        self.loss_delta_threshold = float(loss_delta_threshold)
        self.aha_trigger = bool(aha_trigger)
        self._targets: Dict[MathBenchmark, float] = dict(BENCHMARK_TARGETS)
        if targets:
            for k, v in targets.items():
                self._targets[k] = float(v)

        self._results: List[MathEvalResult] = []
        self._latest: Dict[MathBenchmark, MathEvalResult] = {}
        self._last_eval_step: Dict[MathBenchmark, int] = {}
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
    ) -> List[MathEvalResult]:
        """根据触发策略决定是否评估, 返回本次触发的评估结果. """
        self._loss_history.append((step, loss))
        triggers: List[Tuple[MathBenchmark, str]] = []

        # 1. 步数触发 (轻量基准)
        if step > 0 and step % self.trigger_interval == 0:
            for b in LIGHT_BENCHMARKS:
                triggers.append((b, "step_interval"))

        # 2. loss 突变触发 (轻量基准)
        if self._loss_changed(step):
            triggers.append((MathBenchmark.GSM8K, "loss_delta"))

        # 3. 啊哈时刻触发 (重量基准)
        if aha_moment and self.aha_trigger:
            for b in HEAVY_BENCHMARKS:
                triggers.append((b, "aha_moment"))

        if not triggers:
            return []

        # 去重: 同一基准本步只评估一次
        seen: set = set()
        results: List[MathEvalResult] = []
        ckpt = checkpoint or {}
        for bench, reason in triggers:
            if bench in seen:
                continue
            seen.add(bench)
            res = self._evaluate_one(bench, ckpt, reason)
            results.append(res)
        return results

    def evaluate_full(
        self,
        checkpoint: Optional[Dict[str, Any]] = None,
    ) -> List[MathEvalResult]:
        """全量评估 (所有 8 基准), 阶段切换时调用. """
        ckpt = checkpoint or {}
        results: List[MathEvalResult] = []
        for bench in MathBenchmark:
            results.append(self._evaluate_one(bench, ckpt, "phase_switch"))
        return results

    def evaluate(
        self,
        benchmark: MathBenchmark,
        checkpoint: Optional[Dict[str, Any]] = None,
    ) -> MathEvalResult:
        """评估单个基准. """
        return self._evaluate_one(benchmark, checkpoint or {}, "manual")

    def _evaluate_one(
        self,
        bench: MathBenchmark,
        checkpoint: Dict[str, Any],
        reason: str,
    ) -> MathEvalResult:
        score, samples = self.eval_fn(bench, checkpoint)
        score = max(0.0, min(1.0, float(score)))
        res = MathEvalResult(
            benchmark=bench,
            score=score,
            target=self._targets[bench],
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
    # 默认评估 (模拟, 分数随 step 渐进逼近目标)
    # ------------------------------------------------------------------ #
    def _default_eval(
        self,
        bench: MathBenchmark,
        checkpoint: Dict[str, Any],
    ) -> Tuple[float, int]:
        step = checkpoint.get("step", 0)
        target = self._targets[bench]
        # 各基准收敛速度不同
        speed = {
            MathBenchmark.GSM8K: 3000.0,
            MathBenchmark.MATH: 5000.0,
            MathBenchmark.AIME: 8000.0,
            MathBenchmark.IMO: 15000.0,
            MathBenchmark.PUTNAM: 12000.0,
            MathBenchmark.LEAN_PROOF: 10000.0,
            MathBenchmark.CONJECTURE: 18000.0,
            MathBenchmark.FRONTIER: 25000.0,
        }.get(bench, 8000.0)
        progress = 1.0 - math.exp(-step / speed)
        score = target * (0.25 + 0.75 * progress)
        samples = BENCHMARK_SAMPLES.get(bench, 100)
        return score, samples

    # ------------------------------------------------------------------ #
    # 统计
    # ------------------------------------------------------------------ #
    @property
    def results(self) -> List[MathEvalResult]:
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

    def latest(self, benchmark: MathBenchmark) -> Optional[MathEvalResult]:
        return self._latest.get(benchmark)

    def summary(self) -> Dict[str, Any]:
        return {
            "evaluations_run": len(self._results),
            "benchmarks_evaluated": len(self._latest),
            "total_benchmarks": len(list(MathBenchmark)),
            "pass_rate": round(self.pass_rate, 3),
            "total_eval_cost": self._total_eval_cost,
            "latest_scores": self.latest_scores,
            "targets": {b.value: t for b, t in self._targets.items()},
        }

    def gap_to_target(self, benchmark: MathBenchmark) -> Optional[float]:
        """返回指定基准当前分数距目标的差距 (负值表示已达标). """
        r = self._latest.get(benchmark)
        if r is None:
            return None
        return self._targets[benchmark] - r.score


__all__ = [
    "MathEvaluator",
    "MathEvalResult",
    "MathBenchmark",
    "BENCHMARK_TARGETS",
    "BENCHMARK_SAMPLES",
    "BENCHMARK_COST",
    "LIGHT_BENCHMARKS",
    "HEAVY_BENCHMARKS",
]
