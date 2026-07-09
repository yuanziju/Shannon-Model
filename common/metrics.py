"""训练 / 评估指标追踪 (Shannon / MathMaster 共享基础设施).

维护基准分数 (MMLU / GSM8K / HumanEval / MATH / AIME 等) 与训练动态指标
(loss / grad_norm / lr ...), 并提供 "啊哈时刻" 检测:
    1. 梯度范数突增 > 3× 移动平均;
    2. loss 曲率反转 (二阶差分符号变化).
"""

from __future__ import annotations

import collections
import logging
import statistics
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

DEFAULT_BENCHMARKS: Tuple[str, ...] = (
    "MMLU", "GSM8K", "HumanEval", "MATH", "AIME",
    "BBH", "GPQA", "LiveCodeBench", "SWE-bench",
    "CMMLU", "C-Eval", "AGIEval",
)


class MetricsTracker:
    """指标追踪器.

    基准指标 (benchmark) 与训练动态指标分别存储. 每次 :meth:`update` 会把
    ``(step, value)`` 追加到对应历史中, 便于后续绘曲线 / 检测拐点.
    """

    def __init__(self, benchmarks: Optional[Iterable[str]] = None) -> None:
        self.benchmarks: List[str] = list(benchmarks) if benchmarks is not None else list(DEFAULT_BENCHMARKS)
        # benchmark 名 -> [(step, value)]
        self.metrics: Dict[str, List[Tuple[int, float]]] = {
            name: [] for name in self.benchmarks
        }
        # 训练动态指标名 -> [(step, value)]
        self.training_metrics: Dict[str, List[Tuple[int, float]]] = collections.defaultdict(list)
        self._grad_norm_window: Deque[float] = collections.deque(maxlen=100)

    # ------------------------------------------------------------------
    # 更新 / 查询
    # ------------------------------------------------------------------
    def update(self, name: str, value: Any, step: int = 0) -> None:
        """更新某项指标. ``value`` 支持标量或 0-d / 1-elem torch.Tensor."""
        v = self._to_float(value)
        if name in self.metrics:
            self.metrics[name].append((step, v))
        else:
            self.training_metrics[name].append((step, v))
        if name == "grad_norm":
            self._grad_norm_window.append(v)
        logger.debug("metric %s = %s @step %d", name, v, step)

    def get_metric(self, name: str, default: Optional[float] = None) -> Optional[float]:
        """返回某指标的最新值, 不存在则返回 ``default``."""
        hist = self.metrics.get(name) or self.training_metrics.get(name)
        if hist:
            return hist[-1][1]
        return default

    def get_history(self, name: str) -> List[Tuple[int, float]]:
        """返回某指标的完整历史 ``[(step, value), ...]``."""
        if name in self.metrics:
            return list(self.metrics[name])
        return list(self.training_metrics.get(name, []))

    def all_latest(self) -> Dict[str, float]:
        """返回所有指标的最新值."""
        out: Dict[str, float] = {}
        for name, hist in self.metrics.items():
            if hist:
                out[name] = hist[-1][1]
        for name, hist in self.training_metrics.items():
            if hist:
                out[name] = hist[-1][1]
        return out

    # ------------------------------------------------------------------
    # 日志
    # ------------------------------------------------------------------
    def log_metrics(self, step: Optional[int] = None, log=None) -> str:
        """把当前所有指标格式化为一行并写入日志, 返回该行字符串."""
        log = log or logger
        parts: List[str] = []
        if step is not None:
            parts.append(f"step={step}")
        for name in self.benchmarks:
            v = self.get_metric(name)
            if v is not None:
                parts.append(f"{name}={v:.4f}")
        for name, hist in self.training_metrics.items():
            if hist:
                parts.append(f"{name}={hist[-1][1]:.6g}")
        line = " | ".join(parts) if parts else "(no metrics)"
        log.info(line)
        return line

    # ------------------------------------------------------------------
    # 啊哈时刻检测
    # ------------------------------------------------------------------
    def detect_aha_moment(
        self,
        grad_norm: Optional[float] = None,
        loss: Optional[float] = None,
        step: Optional[int] = None,
        spike_factor: float = 3.0,
    ) -> bool:
        """检测啊哈时刻.

        触发条件 (满足其一):
            1. ``grad_norm`` 超过其移动平均的 ``spike_factor`` (默认 3×);
            2. loss 曲率反转: 最近若干步二阶差分符号变化.

        若传入 ``grad_norm`` / ``loss`` 会同时更新对应历史, 便于连续调用.
        """
        triggered = False
        reasons: List[str] = []

        # 1. 梯度范数突增
        if grad_norm is not None:
            self.update("grad_norm", grad_norm, step if step is not None else 0)
            if len(self._grad_norm_window) >= 10:
                history = list(self._grad_norm_window)
                ma = statistics.mean(history[:-1])
                if ma > 0 and grad_norm > spike_factor * ma:
                    triggered = True
                    reasons.append(
                        f"grad_norm {grad_norm:.4f} > {spike_factor}x MA {ma:.4f}"
                    )

        # 2. loss 曲率反转
        loss_hist = self.training_metrics.get("loss", [])
        if loss is not None:
            self.update("loss", loss, step if step is not None else 0)
            loss_hist = self.training_metrics.get("loss", [])
        if len(loss_hist) >= 5:
            vals = [v for _, v in loss_hist[-5:]]
            d1 = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
            d2 = [d1[i + 1] - d1[i] for i in range(len(d1) - 1)]
            for i in range(1, len(d2)):
                # 排除接近 0 的微小波动
                if abs(d2[i]) > 1e-12 and d2[i] * d2[i - 1] < 0:
                    triggered = True
                    reasons.append("loss curvature reversal")
                    break

        if triggered:
            logger.info("Aha moment detected: " + "; ".join(reasons))
        return triggered

    # ------------------------------------------------------------------
    @staticmethod
    def _to_float(value: Any) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().float().reshape(-1)[0].item())
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
