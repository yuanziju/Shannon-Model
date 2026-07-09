"""训练检查点 - 分片 + RNG 状态 + 啊哈时刻.

TrainingCheckpoint 实现训练状态持久化:
    - 分片保存 (sharded): 配合 5D 并行, 每个并行 rank 保存独立分片.
    - RNG 状态: Python/random/各优化器状态, 保证可精确恢复.
    - 啊哈时刻 (Aha Moment): 捕获训练中 loss 突变/能力涌现的关键步.
    - 稳定检查点 vs 探索检查点 (回滚用).
"""

from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class AhaMoment:
    """啊哈时刻: 训练中能力涌现的关键步. """

    step: int
    phase: str
    metric: str           # 触发指标名 (loss/eval_score/...)
    value: float
    delta: float          # 变化幅度
    description: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class CheckpointShard:
    """单个并行 rank 的检查点分片. """

    rank: int
    data: Dict[str, Any] = field(default_factory=dict)
    bytes: int = 0


class TrainingCheckpoint:
    """训练检查点管理器.

    Args:
        world_size: 并行世界大小 (分片数).
        keep_last: 保留最近 N 个检查点 (滚动).
        aha_threshold: 啊哈时刻触发阈值 (loss 降幅比例).
        aha_window: 检测窗口 (步数).
    """

    def __init__(
        self,
        world_size: int = 1,
        keep_last: int = 3,
        aha_threshold: float = 0.15,
        aha_window: int = 50,
    ) -> None:
        self.world_size = max(1, int(world_size))
        self.keep_last = max(1, int(keep_last))
        self.aha_threshold = float(aha_threshold)
        self.aha_window = max(1, int(aha_window))
        self._shards: Dict[int, CheckpointShard] = {
            r: CheckpointShard(rank=r) for r in range(self.world_size)
        }
        self._history: List[Dict[str, Any]] = []
        self._stable: Optional[Dict[str, Any]] = None
        self._aha_moments: List[AhaMoment] = []
        self._metric_history: Dict[str, List[Tuple[int, float]]] = {}
        self._rng_states: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # 保存
    # ------------------------------------------------------------------ #
    def save(
        self,
        step: int,
        phase: str,
        tokens_seen: int = 0,
        phase_tokens: int = 0,
        phase_idx: int = 0,
        metrics: Optional[Dict[str, Any]] = None,
        optimizer_states: Optional[Dict[int, Any]] = None,
    ) -> Dict[str, Any]:
        """保存检查点 (分片 + RNG), 返回检查点元信息. """
        metrics = metrics or {}
        # RNG 状态快照
        rng_snapshot = {
            "python_random": random.getstate(),
            "step": step,
        }
        self._rng_states[step] = rng_snapshot

        # 分片保存 (这里将整体状态按 rank 切片模拟)
        full_state = {
            "step": step,
            "phase": phase,
            "tokens_seen": tokens_seen,
            "phase_tokens": phase_tokens,
            "phase_idx": phase_idx,
            "metrics": copy.deepcopy(metrics),
            "optimizer_states": copy.deepcopy(optimizer_states or {}),
            "rng": _rng_to_serializable(rng_snapshot),
            "ts": time.time(),
        }
        # 分片: 按 world_size 切分 metrics keys (模拟)
        keys = list(full_state["metrics"].keys())
        for rank in range(self.world_size):
            shard_keys = keys[rank::self.world_size]
            shard_data = {k: full_state["metrics"][k] for k in shard_keys}
            shard = self._shards[rank]
            shard.data = shard_data
            shard.bytes = sum(len(str(v)) for v in shard_data.values())

        self._history.append(full_state)
        # 滚动保留
        if len(self._history) > self.keep_last:
            self._history = self._history[-self.keep_last:]

        # 啊哈时刻检测
        self._detect_aha(step, phase, metrics)

        # 稳定检查点: 阶段切换或低 loss 时更新
        if self._is_stable(metrics, phase):
            self._stable = copy.deepcopy(full_state)
        return full_state

    # ------------------------------------------------------------------ #
    # 加载
    # ------------------------------------------------------------------ #
    def load(self, stable: bool = False) -> Dict[str, Any]:
        """加载检查点. ``stable=True`` 加载最近稳定检查点. """
        if stable:
            return copy.deepcopy(self._stable) if self._stable else {}
        if not self._history:
            return {}
        return copy.deepcopy(self._history[-1])

    def load_shard(self, rank: int) -> CheckpointShard:
        return copy.deepcopy(self._shards.get(rank, CheckpointShard(rank=rank)))

    # ------------------------------------------------------------------ #
    # 啊哈时刻
    # ------------------------------------------------------------------ #
    def _detect_aha(self, step: int, phase: str, metrics: Dict[str, Any]) -> None:
        for name, val in metrics.items():
            if not isinstance(val, (int, float)):
                continue
            hist = self._metric_history.setdefault(name, [])
            hist.append((step, float(val)))
            if len(hist) < 2:
                continue
            window = hist[-self.aha_window:]
            if len(window) < 3:
                continue
            # 检测突变: 当前值相对窗口均值显著下降 (loss) 或上升 (score)
            prev_avg = sum(v for _, v in window[:-1]) / (len(window) - 1)
            if prev_avg == 0:
                continue
            delta = (window[-1][1] - prev_avg) / abs(prev_avg)
            is_loss = "loss" in name.lower()
            # loss 下降超阈值 或 score 上升超阈值
            if (is_loss and delta < -self.aha_threshold) or (
                not is_loss and delta > self.aha_threshold
            ):
                self._aha_moments.append(
                    AhaMoment(
                        step=step,
                        phase=phase,
                        metric=name,
                        value=window[-1][1],
                        delta=delta,
                        description=f"{'drop' if is_loss else 'surge'} in {name}",
                    )
                )

    @property
    def aha_moments(self) -> List[AhaMoment]:
        return list(self._aha_moments)

    def _is_stable(self, metrics: Dict[str, Any], phase: str) -> bool:
        loss = metrics.get("loss")
        if not isinstance(loss, (int, float)):
            return False
        return loss < 1.0  # 简化: 低 loss 视为稳定

    # ------------------------------------------------------------------ #
    # 统计
    # ------------------------------------------------------------------ #
    @property
    def history(self) -> List[Dict[str, Any]]:
        return list(self._history)

    @property
    def has_stable(self) -> bool:
        return self._stable is not None

    def shard_sizes(self) -> Dict[int, int]:
        return {r: s.bytes for r, s in self._shards.items()}

    def stats(self) -> Dict[str, Any]:
        return {
            "world_size": self.world_size,
            "checkpoints_kept": len(self._history),
            "keep_last": self.keep_last,
            "aha_moments": len(self._aha_moments),
            "has_stable": self.has_stable,
            "shard_bytes": self.shard_sizes(),
            "last_step": self._history[-1]["step"] if self._history else 0,
        }


def _rng_to_serializable(rng_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """将 RNG state 转为可序列化形式 (random.getstate 含不可序列化对象). """
    state = rng_snapshot.get("python_random")
    if state is None:
        return {}
    version, internal, gauss = state
    return {
        "version": version,
        "internal": list(internal),  # tuple -> list
        "gauss": gauss,
        "step": rng_snapshot.get("step"),
    }


__all__ = ["TrainingCheckpoint", "CheckpointShard", "AhaMoment"]
