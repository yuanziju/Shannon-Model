"""损失函数 - 动态权重 / MoE 负载均衡 / MTP.

    - DynamicLossWeighter: 多任务损失动态加权 (spec 动态损失)
    - MoEBalanceLoss:      双层 MoE 负载均衡损失 (Top-2~4 路由)
    - MTPLoss:             多 token 预测损失 (k=2-4, 仅训练)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------- #
# DynamicLossWeighter
# ---------------------------------------------------------------------- #
class DynamicLossWeighter:
    """多任务损失动态加权器.

    采用 GradNorm 风格的自适应加权: 根据各任务 loss 的相对量级与训练速度
    动态调整权重, 避免某任务主导训练.

    Args:
        task_names: 任务名列表.
        init_weights: 初始权重 (默认均匀).
        alpha: 权重更新平滑系数 (0-1).
        temperature: 温度 (控制权重锐度).
    """

    def __init__(
        self,
        task_names: Sequence[str],
        init_weights: Optional[Dict[str, float]] = None,
        alpha: float = 0.9,
        temperature: float = 1.0,
    ) -> None:
        self.task_names = list(task_names)
        n = len(self.task_names)
        self.weights: Dict[str, float] = {
            t: (init_weights or {}).get(t, 1.0 / n) for t in self.task_names
        }
        self.alpha = float(alpha)
        self.temperature = float(temperature)
        # 各任务初始 loss (用于归一化)
        self._init_loss: Dict[str, Optional[float]] = {t: None for t in self.task_names}
        self._history: Dict[str, Deque[float]] = {t: deque(maxlen=100) for t in self.task_names}

    def update(self, losses: Dict[str, float]) -> Dict[str, float]:
        """根据当前各任务 loss 更新权重, 返回新权重. """
        # 记录初始 loss
        for t, v in losses.items():
            if t in self._init_loss and self._init_loss[t] is None:
                self._init_loss[t] = max(v, 1e-8)
            self._history[t].append(v)

        # 计算各任务相对训练速度 (loss/初始loss 越大 -> 训练越慢 -> 权重越高)
        rates: Dict[str, float] = {}
        for t in self.task_names:
            init = self._init_loss.get(t) or 1e-8
            cur = losses.get(t, init)
            rates[t] = max(1e-8, cur / init)

        # softmax 归一化 (温度控制锐度)
        import math as _m
        logits = {t: _m.log(r / self.temperature) for t, r in rates.items()}
        max_logit = max(logits.values())
        exps = {t: _m.exp(l - max_logit) for t, l in logits.items()}
        total = sum(exps.values()) or 1.0
        new_weights = {t: exps[t] / total for t in self.task_names}

        # 平滑更新
        for t in self.task_names:
            self.weights[t] = self.alpha * self.weights[t] + (1 - self.alpha) * new_weights[t]
        return dict(self.weights)

    def total(self, losses: Dict[str, float]) -> float:
        """加权求和总损失. """
        return sum(self.weights.get(t, 0.0) * losses.get(t, 0.0) for t in self.task_names)

    def __call__(self, losses: Dict[str, float]) -> float:
        w = self.update(losses)
        return self.total(losses)

    def stats(self) -> Dict[str, Any]:
        return {"weights": dict(self.weights), "init_loss": dict(self._init_loss)}


# ---------------------------------------------------------------------- #
# MoEBalanceLoss
# ---------------------------------------------------------------------- #
@dataclass
class ExpertLoad:
    """单层专家负载统计. """

    num_experts: int
    topk: int                       # Top-k 路由
    # 每个专家的 token 计数与平均路由概率
    token_counts: List[int] = field(default_factory=list)
    route_probs: List[float] = field(default_factory=list)


class MoEBalanceLoss:
    """双层 MoE 负载均衡损失.

    实现 DeepSeek 风格的辅助损失:
        L_balance = alpha * num_experts * sum(f_i * P_i)
    其中 f_i = 分配给专家 i 的 token 比例, P_i = 专家 i 的平均路由概率.
    适用于双层 MoE (16 大 x 16 小, Top-2~4).

    Args:
        alpha: 辅助损失系数.
        router_z_loss: 是否附加 router z-loss (防止 logits 过大).
        z_loss_coeff: z-loss 系数.
    """

    def __init__(
        self,
        alpha: float = 0.01,
        router_z_loss: bool = True,
        z_loss_coeff: float = 1e-3,
    ) -> None:
        self.alpha = float(alpha)
        self.router_z_loss = bool(router_z_loss)
        self.z_loss_coeff = float(z_loss_coeff)

    def compute(self, loads: Sequence[ExpertLoad]) -> float:
        """计算双层 MoE 总负载均衡损失.

        Args:
            loads: 各层 (大专家层 + 小专家层) 的负载统计.
        """
        total = 0.0
        for load in loads:
            total += self._single_layer(load)
        return self.alpha * total

    def _single_layer(self, load: ExpertLoad) -> float:
        n = load.num_experts
        if n == 0:
            return 0.0
        counts = load.token_counts or [0] * n
        probs = load.route_probs or [0.0] * n
        total_tokens = sum(counts) or 1
        f = [c / total_tokens for c in counts]           # token 比例
        P = [max(0.0, p) for p in probs]                 # 平均路由概率
        # 标准辅助损失
        balance = n * sum(fi * pi for fi, pi in zip(f, P))
        loss = balance
        # router z-loss
        if self.router_z_loss:
            # 模拟: 惩罚路由概率方差过大
            mean_p = sum(P) / n if n else 0.0
            z = sum((p - mean_p) ** 2 for p in P) / n
            loss += self.z_loss_coeff * z
        return loss

    def __call__(self, loads: Sequence[ExpertLoad]) -> float:
        return self.compute(loads)


# ---------------------------------------------------------------------- #
# MTPLoss
# ---------------------------------------------------------------------- #
class MTPLoss:
    """多 Token 预测 (MTP) 损失, k=2-4, 仅训练.

    DeepSeek-V3 风格: 同时预测未来 k 个 token, 用独立预测头 + 深度加权.
    推理时可选用于投机解码 draft.

    Args:
        k: 预测未来 token 数 (2-4).
        depth_weights: 各预测深度权重 (越深权重越小), 默认指数衰减.
    """

    def __init__(
        self,
        k: int = 2,
        depth_weights: Optional[Sequence[float]] = None,
    ) -> None:
        self.k = max(2, min(4, int(k)))
        if depth_weights is not None and len(depth_weights) >= self.k:
            self.depth_weights = list(depth_weights[:self.k])
        else:
            # 指数衰减: 1, 0.5, 0.25, ...
            self.depth_weights = [0.5 ** i for i in range(self.k)]

    def compute(self, main_loss: float, mtp_losses: Sequence[float]) -> float:
        """组合 MTP 总损失.

        Args:
            main_loss: 主任务 (深度0, 下一 token) 损失.
            mtp_losses: 深度 1..k 的预测损失.
        """
        total = main_loss
        for i, mtp_loss in enumerate(mtp_losses[:self.k]):
            total += self.depth_weights[i] * float(mtp_loss)
        return total

    def __call__(self, main_loss: float, mtp_losses: Sequence[float]) -> float:
        return self.compute(main_loss, mtp_losses)

    @property
    def num_heads(self) -> int:
        return self.k

    def stats(self) -> Dict[str, Any]:
        return {"k": self.k, "depth_weights": list(self.depth_weights)}


__all__ = ["DynamicLossWeighter", "MoEBalanceLoss", "MTPLoss", "ExpertLoad"]
