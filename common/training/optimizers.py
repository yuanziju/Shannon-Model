"""多优化器 - SAGE / Muon / AdEMAMix / SCALE 实现.

MultiOptimizer 为不同参数组分配不同优化器 (spec 多优化器组合):
    - SAGE:      隐式梯度压缩, 适合大 batch 训练 (注意力/嵌入)
    - Muon:      正交化动量, 适合 2D 权重矩阵 (FFN/专家)
    - AdEMAMix:  双 EMAM 混合, 长期记忆加速收敛 (MoE 路由)
    - SCALE:     自适应学习率缩放 (大 batch 稳定性)

注意: 纯 Python 参考实现 (无 torch 依赖), 用于接口与策略验证.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


class OptimizerKind(Enum):
    SAGE = "SAGE"
    MUON = "Muon"
    ADEMAMIX = "AdEMAMix"
    SCALE = "SCALE"


@dataclass
class ParamGroup:
    """参数组 (绑定一个优化器). """

    name: str
    kind: OptimizerKind
    params: List[str]                                   # 参数名 (占位)
    lr: float = 1e-3
    weight_decay: float = 0.0
    state: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------- #
# 单优化器策略实现 (纯 Python, 操作标量状态字典)
# ---------------------------------------------------------------------- #
class _SAGEState:
    """SAGE: 隐式梯度压缩 (K-bit 量化压缩梯度通信). """

    def __init__(self, compress_bits: int = 8) -> None:
        self.compress_bits = int(compress_bits)
        self.error_buf: Dict[str, float] = {}

    def step(self, group: ParamGroup, grads: Dict[str, float]) -> Dict[str, float]:
        updates: Dict[str, float] = {}
        for p in group.params:
            g = grads.get(p, 0.0) + self.error_buf.get(p, 0.0)
            # 模拟量化压缩 -> 误差反馈
            levels = 2 ** (self.compress_bits - 1) - 1
            scale = max(abs(g), 1e-8) / levels
            q = round(g / scale) * scale if scale > 0 else 0.0
            self.error_buf[p] = g - q
            updates[p] = -group.lr * (q + group.weight_decay * grads.get(p, 0.0))
        return updates


class _MuonState:
    """Muon: 动量 + Newton-Schulz 正交化 (2D 权重). """

    def __init__(self, momentum: float = 0.95, nesterov: bool = True) -> None:
        self.momentum = float(momentum)
        self.nesterov = bool(nesterov)
        self.buf: Dict[str, float] = {}

    def step(self, group: ParamGroup, grads: Dict[str, float]) -> Dict[str, float]:
        updates: Dict[str, float] = {}
        for p in group.params:
            g = grads.get(p, 0.0)
            m = self.buf.get(p, 0.0)
            m = self.momentum * m + g
            self.buf[p] = m
            # 模拟正交化: 对标量取 sign (2D 时为矩阵正交, 标量退化为符号)
            ortho = math.copysign(1.0, m) if m != 0 else 0.0
            if self.nesterov:
                ortho = math.copysign(1.0, self.momentum * m + g)
            updates[p] = -group.lr * ortho
        return updates


class _AdEMAMixState:
    """AdEMAMix: 双 EMAM 混合 (快速+慢速). """

    def __init__(self, beta_fast: float = 0.9, beta_slow: float = 0.9999, alpha: float = 8.0) -> None:
        self.beta_fast = float(beta_fast)
        self.beta_slow = float(beta_slow)
        self.alpha = float(alpha)
        self.m_fast: Dict[str, float] = {}
        self.m_slow: Dict[str, float] = {}

    def step(self, group: ParamGroup, grads: Dict[str, float]) -> Dict[str, float]:
        updates: Dict[str, float] = {}
        for p in group.params:
            g = grads.get(p, 0.0)
            mf = self.m_fast.get(p, 0.0)
            ms = self.m_slow.get(p, 0.0)
            mf = self.beta_fast * mf + (1 - self.beta_fast) * g
            ms = self.beta_slow * ms + (1 - self.beta_slow) * g
            self.m_fast[p] = mf
            self.m_slow[p] = ms
            combined = mf + self.alpha * ms
            updates[p] = -group.lr * (combined + group.weight_decay * g)
        return updates


class _SCALEState:
    """SCALE: 自适应学习率缩放 (按 batch size 缩放, 保持大 batch 稳定). """

    def __init__(self, base_batch: int = 512, warmup_steps: int = 1000) -> None:
        self.base_batch = int(base_batch)
        self.warmup_steps = int(warmup_steps)
        self.step_count: Dict[str, int] = {}

    def step(self, group: ParamGroup, grads: Dict[str, float], current_batch: int = 512) -> Dict[str, float]:
        updates: Dict[str, float] = {}
        # 学习率缩放: sqrt(batch/base), warmup 平滑
        scale = math.sqrt(max(1, current_batch) / max(1, self.base_batch))
        for p in group.params:
            self.step_count[p] = self.step_count.get(p, 0) + 1
            warmup = min(1.0, self.step_count[p] / max(1, self.warmup_steps))
            eff_lr = group.lr * scale * warmup
            updates[p] = -eff_lr * (grads.get(p, 0.0) + group.weight_decay * grads.get(p, 0.0))
        return updates


# 默认参数组 -> 优化器 映射 (spec)
DEFAULT_GROUP_MAPPING: Dict[str, OptimizerKind] = {
    "attention": OptimizerKind.SAGE,
    "embedding": OptimizerKind.SAGE,
    "expert_ffn": OptimizerKind.MUON,
    "router": OptimizerKind.ADEMAMIX,
    "layernorm": OptimizerKind.SCALE,
}


class MultiOptimizer:
    """多优化器组合管理器.

    Args:
        groups: 参数组列表.
        mapping: 参数组名 -> 优化器类型 映射 (默认 spec 策略).
    """

    def __init__(
        self,
        groups: Optional[Sequence[ParamGroup]] = None,
        mapping: Optional[Dict[str, OptimizerKind]] = None,
    ) -> None:
        self.mapping = dict(mapping or DEFAULT_GROUP_MAPPING)
        self._groups: Dict[str, ParamGroup] = {}
        self._impls: Dict[str, Any] = {}
        if groups:
            for g in groups:
                self.add_group(g)
        self._global_step = 0

    # ------------------------------------------------------------------ #
    # 参数组管理
    # ------------------------------------------------------------------ #
    def add_group(self, group: ParamGroup) -> None:
        self._groups[group.name] = group
        kind = group.kind
        if kind == OptimizerKind.SAGE:
            self._impls[group.name] = _SAGEState()
        elif kind == OptimizerKind.MUON:
            self._impls[group.name] = _MuonState()
        elif kind == OptimizerKind.ADEMAMIX:
            self._impls[group.name] = _AdEMAMixState()
        elif kind == OptimizerKind.SCALE:
            self._impls[group.name] = _SCALEState()

    def auto_group(self, param_names: Sequence[str]) -> None:
        """按 spec 默认映射自动分组. """
        for name in param_names:
            kind = None
            for key, k in self.mapping.items():
                if key in name.lower():
                    kind = k
                    break
            if kind is None:
                kind = OptimizerKind.SCALE
            group_name = kind.value
            if group_name not in self._groups:
                self.add_group(ParamGroup(name=group_name, kind=kind, params=[name]))
            else:
                self._groups[group_name].params.append(name)

    # ------------------------------------------------------------------ #
    # 优化步
    # ------------------------------------------------------------------ #
    def step(self, grads: Dict[str, float], current_batch: int = 512) -> Dict[str, float]:
        """执行一次优化步, 返回各参数更新量. """
        all_updates: Dict[str, float] = {}
        for name, group in self._groups.items():
            impl = self._impls.get(name)
            if impl is None:
                continue
            if isinstance(impl, _SCALEState):
                updates = impl.step(group, grads, current_batch=current_batch)
            else:
                updates = impl.step(group, grads)
            all_updates.update(updates)
        self._global_step += 1
        return all_updates

    def zero_grad(self) -> None:
        pass  # 纯 Python 实现无需显式清零

    # ------------------------------------------------------------------ #
    # 学习率调度
    # ------------------------------------------------------------------ #
    def set_lr(self, lr: float) -> None:
        for g in self._groups.values():
            g.lr = float(lr)

    def scale_lr(self, factor: float) -> None:
        for g in self._groups.values():
            g.lr *= float(factor)

    # ------------------------------------------------------------------ #
    # 状态
    # ------------------------------------------------------------------ #
    def state_dict(self) -> Dict[str, Any]:
        return {
            "global_step": self._global_step,
            "groups": {
                name: {"kind": g.kind.value, "lr": g.lr, "params": list(g.params)}
                for name, g in self._groups.items()
            },
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self._global_step = state.get("global_step", 0)

    @property
    def groups(self) -> Dict[str, ParamGroup]:
        return dict(self._groups)

    @property
    def global_step(self) -> int:
        return self._global_step

    def summary(self) -> Dict[str, Any]:
        return {
            "global_step": self._global_step,
            "num_groups": len(self._groups),
            "kinds": {n: g.kind.value for n, g in self._groups.items()},
            "param_counts": {n: len(g.params) for n, g in self._groups.items()},
        }


__all__ = [
    "MultiOptimizer",
    "ParamGroup",
    "OptimizerKind",
    "DEFAULT_GROUP_MAPPING",
]
