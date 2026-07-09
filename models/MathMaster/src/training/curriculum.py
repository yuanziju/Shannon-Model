"""课程学习 - basic -> intermediate -> advanced -> frontier.

MathCurriculum 实现数学训练的课程调度:

    - 4 级难度: basic / intermediate / advanced / frontier
    - 各级课程权重 (mix_ratio): 课程进度中各级占比
    - 各级领域分布 (domain_dist): 不同难度侧重不同数学领域
    - 阶段推进: 基于掌握度 (mastery) 与样本量自动升级

课程策略 (螺旋上升):
    basic (40%)      算术/代数为主, 建立运算与符号直觉
    intermediate(30%) 九领域均衡, 引入分析与概率
    advanced (20%)   分析/数论/抽象代数加重, 引入竞赛题
    frontier (10%)   前沿研究问题 + 跨领域综合
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


class CurriculumLevel(Enum):
    """4 级课程难度. """

    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    FRONTIER = "frontier"


# 各级在课程整体中的权重 (和为 1.0)
LEVEL_MIX_RATIO: Dict[CurriculumLevel, float] = {
    CurriculumLevel.BASIC: 0.40,
    CurriculumLevel.INTERMEDIATE: 0.30,
    CurriculumLevel.ADVANCED: 0.20,
    CurriculumLevel.FRONTIER: 0.10,
}

# 各级领域分布 (每级内部和为 1.0)
LEVEL_DOMAIN_DIST: Dict[CurriculumLevel, Dict[str, float]] = {
    CurriculumLevel.BASIC: {
        "arithmetic": 0.35,
        "algebra": 0.25,
        "geometry": 0.15,
        "number_theory": 0.10,
        "probability": 0.10,
        "discrete_math": 0.05,
    },
    CurriculumLevel.INTERMEDIATE: {
        "arithmetic": 0.10,
        "algebra": 0.20,
        "geometry": 0.15,
        "analysis": 0.15,
        "number_theory": 0.10,
        "probability": 0.10,
        "discrete_math": 0.10,
        "abstract_algebra": 0.05,
        "topology": 0.05,
    },
    CurriculumLevel.ADVANCED: {
        "algebra": 0.15,
        "geometry": 0.10,
        "analysis": 0.20,
        "number_theory": 0.15,
        "probability": 0.10,
        "discrete_math": 0.10,
        "abstract_algebra": 0.10,
        "topology": 0.05,
        "competition": 0.05,
    },
    CurriculumLevel.FRONTIER: {
        "analysis": 0.20,
        "number_theory": 0.15,
        "abstract_algebra": 0.15,
        "topology": 0.15,
        "geometry": 0.10,
        "probability": 0.05,
        "discrete_math": 0.05,
        "frontier": 0.15,
    },
}

# 各级升级掌握度阈值 (当前级别评估正确率达到阈值方可升级)
MASTERY_THRESHOLD: Dict[CurriculumLevel, float] = {
    CurriculumLevel.BASIC: 0.85,
    CurriculumLevel.INTERMEDIATE: 0.75,
    CurriculumLevel.ADVANCED: 0.60,
    CurriculumLevel.FRONTIER: 0.50,  # frontier 为最高级, 仅作参考
}

# 各级最低样本量 (防止过早升级)
LEVEL_MIN_SAMPLES: Dict[CurriculumLevel, int] = {
    CurriculumLevel.BASIC: 50_000,
    CurriculumLevel.INTERMEDIATE: 80_000,
    CurriculumLevel.ADVANCED: 60_000,
    CurriculumLevel.FRONTIER: 30_000,
}

# 课程级别顺序 (升级路径)
LEVEL_ORDER: Tuple[CurriculumLevel, ...] = (
    CurriculumLevel.BASIC,
    CurriculumLevel.INTERMEDIATE,
    CurriculumLevel.ADVANCED,
    CurriculumLevel.FRONTIER,
)


@dataclass
class CurriculumState:
    """课程学习状态. """

    level: CurriculumLevel = CurriculumLevel.BASIC
    progress: float = 0.0           # 当前级别内进度 [0, 1]
    mastery: float = 0.0            # 当前级别掌握度 [0, 1]
    samples_seen: int = 0           # 当前级别已见样本数
    total_samples: int = 0          # 累计样本数
    upgrades: int = 0               # 升级次数
    history: List[Tuple[CurriculumLevel, float, float]] = field(default_factory=list)


def _coerce_level(level: Union[str, CurriculumLevel]) -> CurriculumLevel:
    """将字符串或 CurriculumLevel 统一为 CurriculumLevel. """
    if isinstance(level, CurriculumLevel):
        return level
    key = str(level).strip().lower()
    try:
        return CurriculumLevel(key)
    except ValueError as exc:
        raise ValueError(
            f"unknown curriculum level: {level!r}; "
            f"expected one of {[lv.value for lv in CurriculumLevel]}"
        ) from exc


class MathCurriculum:
    """数学课程学习调度器.

    Args:
        init_level: 起始级别 (默认 basic).
        mastery_thresholds: 各级升级掌握度覆盖 (可选).
        min_samples: 各级最低样本量覆盖 (可选).
        mix_ratio_overrides: 各级课程权重覆盖 (可选, 会重新归一).
    """

    def __init__(
        self,
        init_level: Union[str, CurriculumLevel] = CurriculumLevel.BASIC,
        mastery_thresholds: Optional[Dict[Union[str, CurriculumLevel], float]] = None,
        min_samples: Optional[Dict[Union[str, CurriculumLevel], int]] = None,
        mix_ratio_overrides: Optional[Dict[Union[str, CurriculumLevel], float]] = None,
    ) -> None:
        self.state = CurriculumState(level=_coerce_level(init_level))
        self._mastery_thresholds: Dict[CurriculumLevel, float] = dict(MASTERY_THRESHOLD)
        self._min_samples: Dict[CurriculumLevel, int] = dict(LEVEL_MIN_SAMPLES)
        self._mix_ratio: Dict[CurriculumLevel, float] = dict(LEVEL_MIX_RATIO)

        if mastery_thresholds:
            for k, v in mastery_thresholds.items():
                self._mastery_thresholds[_coerce_level(k)] = float(v)
        if min_samples:
            for k, v in min_samples.items():
                self._min_samples[_coerce_level(k)] = int(v)
        if mix_ratio_overrides:
            for k, v in mix_ratio_overrides.items():
                self._mix_ratio[_coerce_level(k)] = float(v)
            self._normalize_mix()

    # ------------------------------------------------------------------ #
    # 配比查询
    # ------------------------------------------------------------------ #
    def _normalize_mix(self) -> None:
        s = sum(self._mix_ratio.values())
        if s <= 0:
            raise ValueError("sum of mix ratios must be positive")
        for k in self._mix_ratio:
            self._mix_ratio[k] = self._mix_ratio[k] / s

    def mix_ratio(self, level: Union[str, CurriculumLevel]) -> float:
        """返回 ``level`` 在课程整体中的权重占比 [0, 1]. """
        lv = _coerce_level(level)
        return self._mix_ratio.get(lv, 0.0)

    def domain_dist(self, level: Union[str, CurriculumLevel]) -> Dict[str, float]:
        """返回 ``level`` 的领域分布字典 (和为 1). """
        lv = _coerce_level(level)
        return dict(LEVEL_DOMAIN_DIST.get(lv, {}))

    def current_domain_dist(self) -> Dict[str, float]:
        """返回当前级别的领域分布. """
        return self.domain_dist(self.state.level)

    def all_mix_ratios(self) -> Dict[str, float]:
        """返回所有级别的课程权重 (按级别名). """
        return {lv.value: self._mix_ratio[lv] for lv in CurriculumLevel}

    # ------------------------------------------------------------------ #
    # 采样配比: 当前阶段混合各级别的比例 (螺旋上升, 当前级别占主导)
    # ------------------------------------------------------------------ #
    def sampling_weights(self) -> Dict[str, float]:
        """返回当前阶段下各级别的采样权重.

        当前级别占 70%, 下一级占 20% (预演), 上一级占 10% (复习),
        形成螺旋上升的课程节奏.
        """
        idx = LEVEL_ORDER.index(self.state.level)
        weights: Dict[str, float] = {lv.value: 0.0 for lv in CurriculumLevel}
        weights[self.state.level.value] = 0.70
        if idx + 1 < len(LEVEL_ORDER):
            weights[LEVEL_ORDER[idx + 1].value] = 0.20
        if idx - 1 >= 0:
            weights[LEVEL_ORDER[idx - 1].value] = 0.10
        # frontier 时无下一级, 把余量补给当前级
        if idx + 1 >= len(LEVEL_ORDER):
            weights[self.state.level.value] += 0.20
        # basic 时无上一级, 余量补给当前级
        if idx - 1 < 0:
            weights[self.state.level.value] += 0.10
        return weights

    # ------------------------------------------------------------------ #
    # 推进与升级
    # ------------------------------------------------------------------ #
    def update(self, mastery: float, samples: int = 1) -> bool:
        """更新当前级别掌握度与样本量, 返回是否触发升级.

        Args:
            mastery: 本批次评估正确率 [0, 1].
            samples: 本批次样本数.
        """
        mastery = max(0.0, min(1.0, float(mastery)))
        # 滑动平均更新掌握度
        n = max(1, self.state.samples_seen)
        self.state.mastery = (self.state.mastery * n + mastery * samples) / (n + samples)
        self.state.samples_seen += int(samples)
        self.state.total_samples += int(samples)
        # 进度 = min(1, samples / min_samples)
        min_s = self._min_samples.get(self.state.level, 1)
        self.state.progress = min(1.0, self.state.samples_seen / max(1, min_s))
        return self.maybe_advance()

    def maybe_advance(self) -> bool:
        """检查是否满足升级条件 (掌握度 + 最小样本量), 满足则升级. """
        lv = self.state.level
        if lv == CurriculumLevel.FRONTIER:
            return False  # 已达最高级
        threshold = self._mastery_thresholds.get(lv, 1.0)
        min_s = self._min_samples.get(lv, 0)
        if self.state.mastery >= threshold and self.state.samples_seen >= min_s:
            self._advance()
            return True
        return False

    def _advance(self) -> None:
        idx = LEVEL_ORDER.index(self.state.level)
        next_lv = LEVEL_ORDER[min(len(LEVEL_ORDER) - 1, idx + 1)]
        self.state.history.append(
            (self.state.level, self.state.mastery, self.state.samples_seen)
        )
        self.state.level = next_lv
        self.state.mastery = 0.0
        self.state.progress = 0.0
        self.state.samples_seen = 0
        self.state.upgrades += 1

    def force_advance(self) -> bool:
        """强制升级到下一级 (用于人工干预 / 课程跳级). """
        if self.state.level == CurriculumLevel.FRONTIER:
            return False
        self._advance()
        return True

    def reset(self, level: Union[str, CurriculumLevel] = CurriculumLevel.BASIC) -> None:
        """重置课程到指定级别. """
        self.state = CurriculumState(level=_coerce_level(level))

    # ------------------------------------------------------------------ #
    # 统计
    # ------------------------------------------------------------------ #
    @property
    def level(self) -> CurriculumLevel:
        return self.state.level

    @property
    def is_finished(self) -> bool:
        return self.state.level == CurriculumLevel.FRONTIER

    def summary(self) -> Dict[str, Any]:
        return {
            "level": self.state.level.value,
            "progress": round(self.state.progress, 4),
            "mastery": round(self.state.mastery, 4),
            "samples_seen": self.state.samples_seen,
            "total_samples": self.state.total_samples,
            "upgrades": self.state.upgrades,
            "mix_ratios": self.all_mix_ratios(),
            "sampling_weights": self.sampling_weights(),
            "current_domain_dist": self.current_domain_dist(),
            "mastery_threshold": self._mastery_thresholds.get(self.state.level),
        }


__all__ = [
    "CurriculumLevel",
    "CurriculumState",
    "MathCurriculum",
    "LEVEL_MIX_RATIO",
    "LEVEL_DOMAIN_DIST",
    "MASTERY_THRESHOLD",
    "LEVEL_MIN_SAMPLES",
    "LEVEL_ORDER",
]
