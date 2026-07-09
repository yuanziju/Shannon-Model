"""量化器 - 逐组件量化 + 动态切换.

Quantizer 按组件类型施加不同量化精度 (spec 逐组件量化策略):
    - 大专家 FFN:        W8A16 (权重 INT8, 激活 FP16)
    - KV Cache:          FP8
    - 其余 (注意力/小专家): FP16
    - NVFP4:             仅 Blackwell 当前不可用 (不支持)

支持运行时动态切换精度 (精度/显存权衡).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


class Precision(Enum):
    """支持的计算精度. """

    FP16 = "FP16"
    W8A16 = "W8A16"   # 权重 INT8, 激活 FP16
    FP8 = "FP8"
    INT8 = "INT8"
    BF16 = "BF16"
    # NVFP4 仅 Blackwell 当前不可用
    # NVFP4 = "NVFP4"


class ComponentType(Enum):
    """模型组件类型 (决定默认量化策略). """

    BIG_EXPERT_FFN = "BIG_EXPERT_FFN"     # 大专家 FFN
    SMALL_EXPERT_FFN = "SMALL_EXPERT_FFN"  # 小专家 FFN
    ATTENTION = "ATTENTION"               # 注意力 (MLA/KDA/...)
    KV_CACHE = "KV_CACHE"                 # KV Cache
    EMBEDDING = "EMBEDDING"               # 嵌入层
    LAYERNORM = "LAYERNORM"               # 归一化 (不量化)


# spec 默认逐组件量化策略
DEFAULT_POLICY: Dict[ComponentType, Precision] = {
    ComponentType.BIG_EXPERT_FFN: Precision.W8A16,
    ComponentType.SMALL_EXPERT_FFN: Precision.FP16,
    ComponentType.ATTENTION: Precision.FP16,
    ComponentType.KV_CACHE: Precision.FP8,
    ComponentType.EMBEDDING: Precision.FP16,
    ComponentType.LAYERNORM: Precision.FP16,
}

# 各精度相对 FP16 的显存占用倍率与精度损失估计
PRECISION_PROFILE: Dict[Precision, Dict[str, float]] = {
    Precision.FP16: {"mem_ratio": 1.0, "loss": 0.0},
    Precision.BF16: {"mem_ratio": 1.0, "loss": 0.0},
    Precision.W8A16: {"mem_ratio": 0.55, "loss": 0.005},
    Precision.FP8: {"mem_ratio": 0.5, "loss": 0.008},
    Precision.INT8: {"mem_ratio": 0.5, "loss": 0.01},
}


@dataclass
class Component:
    """单个量化组件. """

    name: str
    ctype: ComponentType
    precision: Precision
    params: int                       # 参数量
    enabled: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def mem_ratio(self) -> float:
        return PRECISION_PROFILE[self.precision]["mem_ratio"]

    @property
    def loss(self) -> float:
        return PRECISION_PROFILE[self.precision]["loss"]

    def fp16_bytes(self) -> float:
        return self.params * 2.0  # FP16 = 2 bytes/param

    def quantized_bytes(self) -> float:
        return self.fp16_bytes() * self.mem_ratio


class Quantizer:
    """逐组件量化器, 支持动态切换.

    Args:
        policy: 组件类型 -> 默认精度 映射.
        max_loss: 容许的总体精度损失上限 (默认 <2%, spec 约束).
    """

    def __init__(
        self,
        policy: Optional[Dict[ComponentType, Precision]] = None,
        max_loss: float = 0.02,
    ) -> None:
        self.policy = dict(policy or DEFAULT_POLICY)
        self.max_loss = float(max_loss)
        self._components: Dict[str, Component] = {}
        self._switch_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # 注册组件
    # ------------------------------------------------------------------ #
    def register(self, name: str, ctype: ComponentType, params: int) -> Component:
        comp = Component(
            name=name,
            ctype=ctype,
            precision=self.policy.get(ctype, Precision.FP16),
            params=int(params),
        )
        self._components[name] = comp
        return comp

    def get(self, name: str) -> Optional[Component]:
        return self._components.get(name)

    # ------------------------------------------------------------------ #
    # 动态切换精度
    # ------------------------------------------------------------------ #
    def switch(self, name: str, precision: Precision, reason: str = "") -> bool:
        """动态切换某组件精度, 校验整体精度损失不超阈值. """
        comp = self._components.get(name)
        if comp is None:
            return False
        old = comp.precision
        comp.precision = precision
        # 校验总体损失
        total_loss = self.total_loss()
        if total_loss > self.max_loss:
            comp.precision = old  # 回滚
            self._switch_log.append(
                {"name": name, "from": old.value, "to": precision.value, "ok": False, "reason": "loss_exceeded"}
            )
            return False
        self._switch_log.append(
            {"name": name, "from": old.value, "to": precision.value, "ok": True, "reason": reason}
        )
        return True

    def switch_batch(self, mapping: Dict[str, Precision]) -> Dict[str, bool]:
        """批量切换, 返回各组件切换结果. """
        results: Dict[str, bool] = {}
        # 备份
        backup = {n: c.precision for n, c in self._components.items()}
        for name, prec in mapping.items():
            comp = self._components.get(name)
            if comp is not None:
                comp.precision = prec
        if self.total_loss() > self.max_loss:
            # 整体回滚
            for n, p in backup.items():
                self._components[n].precision = p
            return {n: False for n in mapping}
        for n in mapping:
            results[n] = True
        return results

    # ------------------------------------------------------------------ #
    # 量化模拟 (伪量化: 对张量值做量化-反量化, 估计误差)
    # ------------------------------------------------------------------ """
    def fake_quantize(self, values: Sequence[float], precision: Precision) -> List[float]:
        """对一组数值做伪量化 (symmetric), 返回反量化后的近似值. """
        if precision in (Precision.FP16, Precision.BF16):
            return list(values)
        levels = {Precision.W8A16: 127, Precision.INT8: 127, Precision.FP8: 127}[precision]
        if not values:
            return []
        abs_max = max(abs(v) for v in values) or 1.0
        scale = abs_max / levels
        out = []
        for v in values:
            q = round(v / scale) if scale > 0 else 0.0
            q = max(-levels, min(levels, q))
            out.append(q * scale)
        return out

    # ------------------------------------------------------------------ #
    # 统计
    # ------------------------------------------------------------------ #
    def total_params(self) -> int:
        return sum(c.params for c in self._components.values() if c.enabled)

    def total_fp16_bytes(self) -> float:
        return sum(c.fp16_bytes() for c in self._components.values() if c.enabled)

    def total_quantized_bytes(self) -> float:
        return sum(c.quantized_bytes() for c in self._components.values() if c.enabled)

    def total_loss(self) -> float:
        """加权平均精度损失. """
        total = self.total_params()
        if total == 0:
            return 0.0
        weighted = sum(c.params * c.loss for c in self._components.values() if c.enabled)
        return weighted / total

    @property
    def compression_ratio(self) -> float:
        q = self.total_quantized_bytes()
        if q <= 0:
            return 1.0
        return self.total_fp16_bytes() / q

    def summary(self) -> Dict[str, Any]:
        return {
            "components": {
                n: {"type": c.ctype.value, "precision": c.precision.value, "params": c.params}
                for n, c in self._components.items()
            },
            "total_params": self.total_params(),
            "compression_ratio": round(self.compression_ratio, 3),
            "total_loss": round(self.total_loss(), 4),
            "max_loss": self.max_loss,
            "switches": len(self._switch_log),
        }

    @property
    def switch_log(self) -> List[Dict[str, Any]]:
        return list(self._switch_log)


__all__ = [
    "Quantizer",
    "Component",
    "Precision",
    "ComponentType",
    "DEFAULT_POLICY",
    "PRECISION_PROFILE",
]
