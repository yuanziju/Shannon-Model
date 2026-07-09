"""数据工具 (Shannon / MathMaster 共享基础设施).

- :class:`DataRatioScheduler`: 按类别配比调度训练数据
  (代码 30% / 理科 30% / 中文 25% / 英文 10% / 多语言 5%).
- :func:`collate_multimodal`: 多模态 batch 整理 (文本 / 图像 / 张量 / 标量混合).
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

import torch

logger = logging.getLogger(__name__)

#: 默认数据配比 (与 spec.md 数据策略一致)
DEFAULT_RATIOS: Dict[str, float] = {
    "code": 0.30,          # 代码数据 >= 30%
    "science": 0.30,       # 理科 (数学/物理/化学/生物) >= 30%
    "chinese": 0.25,       # 中文通用
    "english": 0.10,       # 英文通用
    "multilingual": 0.05,  # 多语言
}


class DataRatioScheduler:
    """按类别配比采样数据, 支持动态调整与已消耗量追踪.

    ``ratios`` 不归一时会自动归一. ``adjust_ratio`` 在调整某一类时会
    按其它类当前占比等比例缩减, 保证总和恒为 1.
    """

    def __init__(
        self,
        ratios: Optional[Dict[str, float]] = None,
        total_tokens: int = 1_000_000,
        seed: Optional[int] = None,
    ) -> None:
        self.ratios: Dict[str, float] = dict(ratios) if ratios else dict(DEFAULT_RATIOS)
        self.total_tokens = total_tokens
        self.consumed: Dict[str, int] = defaultdict(int)
        self._rng = random.Random(seed)
        self._normalize()

    # ------------------------------------------------------------------
    def _normalize(self) -> None:
        s = sum(self.ratios.values())
        if s <= 0:
            raise ValueError("sum of ratios must be positive")
        if abs(s - 1.0) > 1e-9:
            for k in self.ratios:
                self.ratios[k] = self.ratios[k] / s

    def get_ratio(self, category: Optional[str] = None) -> Dict[str, float] | float:
        """返回全量配比字典, 或指定类别的配比 (不存在返回 0.0)."""
        if category is None:
            return dict(self.ratios)
        return self.ratios.get(category, 0.0)

    # ------------------------------------------------------------------
    def adjust_ratio(self, category: str, new_ratio: float) -> None:
        """调整 ``category`` 的配比为 ``new_ratio``, 从其它类按当前占比等比例缩减."""
        if new_ratio < 0 or new_ratio > 1:
            raise ValueError("new_ratio must be in [0, 1]")
        old = self.ratios.get(category, 0.0)
        diff = new_ratio - old
        others = {k: v for k, v in self.ratios.items() if k != category and v > 0}
        others_sum = sum(others.values())
        if diff > 0 and others_sum <= 0:
            raise ValueError("cannot increase ratio: no other categories to shrink")
        for k in others:
            self.ratios[k] = max(0.0, self.ratios[k] - diff * (self.ratios[k] / others_sum))
        if category not in self.ratios:
            self.ratios[category] = 0.0
        self.ratios[category] = new_ratio
        self._normalize()
        logger.info("Adjusted ratio: %s -> %.4f (current=%s)", category, new_ratio, self.ratios)

    # ------------------------------------------------------------------
    def sample_data(
        self,
        data_pool: Dict[str, Sequence[Any]],
        n: int,
        rng: Optional[random.Random] = None,
    ) -> List[Any]:
        """从 ``data_pool`` 按当前配比采样 ``n`` 条.

        ``data_pool``: ``{category: [item, ...]}``. 每类采样数 = ``round(n * ratio)``,
        不足时取全部并从余量最大的类补足, 最终打乱返回.
        """
        rng = rng or self._rng
        sampled: List[Any] = []
        remaining = n

        # 先按配比分配
        targets: Dict[str, int] = {}
        for cat, ratio in self.ratios.items():
            if cat not in data_pool or not data_pool[cat]:
                continue
            k = int(round(n * ratio))
            targets[cat] = k

        # 裁剪到池子大小
        for cat in list(targets):
            pool = data_pool[cat]
            targets[cat] = min(targets[cat], len(pool))

        total_target = sum(targets.values())
        # 若因池子太小导致不足, 按当前配比从有富余的类补足
        if total_target < n:
            for cat, ratio in sorted(self.ratios.items(), key=lambda kv: -kv[1]):
                if total_target >= n:
                    break
                pool = data_pool.get(cat, [])
                slack = len(pool) - targets.get(cat, 0)
                if slack > 0:
                    add = min(slack, n - total_target)
                    targets[cat] = targets.get(cat, 0) + add
                    total_target += add

        for cat, k in targets.items():
            pool = list(data_pool[cat])
            picks = rng.sample(pool, min(k, len(pool)))
            sampled.extend(picks)
            self.consumed[cat] += len(picks)

        rng.shuffle(sampled)
        return sampled[:n]

    # ------------------------------------------------------------------
    def consumed_ratio(self) -> Dict[str, float]:
        """返回各类已消耗量的占比."""
        total = sum(self.consumed.values())
        if total <= 0:
            return {k: 0.0 for k in self.ratios}
        return {k: self.consumed[k] / total for k in self.ratios}


def collate_multimodal(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """多模态 batch 整理.

    输入: 若干样本字典, 各样本键可不一致 (缺失键填 None).
    输出: 同键聚合字典. 规则:
        - 全为同形状 Tensor -> ``torch.stack``;
        - 全为 int -> LongTensor;
        - 全为 float -> FloatTensor;
        - 其余 -> 原始 list.
    缺失位置: Tensor 用 0 填充并附带 ``<key>_mask`` 标记; 标量用 -1 占位.
    """
    if not batch:
        return {}
    keys: set = set()
    for item in batch:
        keys.update(item.keys())

    out: Dict[str, Any] = {}
    for key in keys:
        values = [item.get(key) for item in batch]
        present = [v for v in values if v is not None]
        if not present:
            out[key] = None
            continue

        all_tensor = all(isinstance(v, torch.Tensor) for v in present)
        all_int = all(isinstance(v, (int,)) and not isinstance(v, bool) for v in present)
        all_float = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in present)

        if all_tensor:
            shapes = {tuple(v.shape) for v in present}
            if len(shapes) == 1 and len(present) == len(values):
                out[key] = torch.stack(values)
            else:
                # 形状不一或缺字段: padding 到最大形状
                out[key] = _pad_stack_tensors(values)
                out[key + "_mask"] = torch.tensor(
                    [v is not None for v in values], dtype=torch.bool
                )
        elif all_int and len(present) == len(values):
            out[key] = torch.tensor(values, dtype=torch.long)
        elif all_float and len(present) == len(values):
            out[key] = torch.tensor(values, dtype=torch.float32)
        else:
            out[key] = values
    return out


def _pad_stack_tensors(values: List[Any]) -> torch.Tensor:
    """把含 None 的 Tensor 列表按最大形状 0-padding 后 stack."""
    tensors = [v if v is not None else None for v in values]
    present = [t for t in tensors if t is not None]
    if not present:
        return torch.empty(0)
    ref = present[0]
    max_shape = list(ref.shape)
    for t in present[1:]:
        for i, s in enumerate(t.shape):
            if s > max_shape[i]:
                max_shape[i] = s
    padded = []
    for t in tensors:
        if t is None:
            padded.append(torch.zeros(max_shape, dtype=ref.dtype))
        else:
            pad_t = torch.zeros(max_shape, dtype=t.dtype)
            slices = tuple(slice(0, s) for s in t.shape)
            pad_t[slices] = t
            padded.append(pad_t)
    return torch.stack(padded)
