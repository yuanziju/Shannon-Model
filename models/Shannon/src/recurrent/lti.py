"""LTI 稳定性约束 (Linear Time-Invariant Stability).

对循环主体的状态转移矩阵施加谱半径 < 1 约束, 保证循环动力学稳定
(spec: LTI稳定性约束, 谱半径<1).

实现两种方式:
  1. 谱归一化 (power iteration 估计谱范数, 收缩为 spectral_radius)
  2. 残差收缩门控 (sigmoid 门控控制状态更新幅度)

参考: AGENTS.md Agent 1 (ArchAgent), spec §4.x LTI稳定性.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LTIStability(nn.Module):
    """LTI 稳定性约束模块.

    对线性时不变状态转移 x_{t+1} = A x_t + B u_t 施加谱半径约束:
      spectral_radius(A) < target (默认 0.99)

    通过 power iteration 估计 A 的最大奇异值 (谱范数的上界),
    并将 A 缩放为 A / max(sigma/sigma_max, 1.0), 保证收缩性.

    对于非线性情况, 额外使用一个 sigmoid 门控控制状态更新幅度.
    """

    def __init__(
        self,
        hidden_dim: int,
        spectral_radius: float = 0.99,
        n_power_iters: int = 1,
        gate_init: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.spectral_radius = float(spectral_radius)
        assert 0.0 < self.spectral_radius < 1.0, (
            f"spectral_radius 须在 (0, 1), got {self.spectral_radius}"
        )
        self.n_power_iters = max(1, int(n_power_iters))

        # 状态转移矩阵 A (hidden_dim x hidden_dim)
        # 初始化为接近正交的矩阵, 保证初始稳定
        self.A = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.orthogonal_(self.A.weight)

        # 输入投影 B
        self.B = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # 门控: 控制状态更新幅度 (sigmoid -> (0,1))
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

        # power iteration 向量 (buffer, 不持久化以避免设备问题)
        u = torch.randn(hidden_dim)
        u = u / (u.norm() + 1e-8)
        self.register_buffer("_u", u, persistent=False)

    # ------------------------------------------------------------------
    def _spectral_norm(self) -> torch.Tensor:
        """通过 power iteration 估计 A 的谱范数 (最大奇异值)."""
        w = self.A.weight  # [H, H]
        u = self._u
        with torch.no_grad():
            for _ in range(self.n_power_iters):
                v = F.normalize(u @ w.t(), dim=0, eps=1e-8)
                u_new = F.normalize(v @ w, dim=0, eps=1e-8)
                u.copy_(u_new)
        sigma = (u @ w @ v).clamp_min(1e-8)
        return sigma

    def stable_A(self) -> torch.Tensor:
        """返回满足谱半径约束的 A 矩阵."""
        w = self.A.weight
        sigma = self._spectral_norm()
        # 缩放: 使谱范数 <= spectral_radius
        scale = self.spectral_radius / sigma.clamp_min(1e-8)
        scale = torch.clamp(scale, max=1.0)  # 只收缩, 不放大
        return w * scale

    # ------------------------------------------------------------------
    def forward(
        self,
        state: torch.Tensor,
        update: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """一步 LTI 稳定状态更新.

        x_{t+1} = g * (A_stable @ x_t) + (1 - g) * (B @ u_t)

        其中 g = sigmoid(gate) 为状态保留比例, A_stable 满足谱半径约束.

        Args:
            state: [B, S, H] 或 [N, H] 当前状态.
            update: 同形状, 外部输入 u_t (None 则无输入).

        Returns:
            new_state: 同形状.
        """
        g = torch.sigmoid(self.gate)
        # 稳定状态转移
        A_stable = self.stable_A()
        new_state = F.linear(state, A_stable)
        if update is not None:
            bu = self.B(update)
            new_state = g * new_state + (1.0 - g) * bu
        else:
            new_state = g * new_state
        return new_state

    def spectral_radius_estimate(self) -> float:
        """返回当前 A 的谱范数估计 (用于监控)."""
        with torch.no_grad():
            return float(self._spectral_norm().item())

    def is_stable(self) -> bool:
        """检查当前是否满足谱半径约束."""
        return self.spectral_radius_estimate() <= self.spectral_radius + 1e-6

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, "
            f"spectral_radius={self.spectral_radius}, "
            f"n_power_iters={self.n_power_iters}"
        )


class ResidualStabilizer(nn.Module):
    """残差稳定器: 对循环残差施加收缩约束.

    用于循环主体中: h_{t+1} = h_t + alpha * f(h_t)
    其中 alpha 由谱半径约束自动调节, 保证整体动力学稳定.

    计算: alpha = spectral_radius * sigmoid(gate) / (1 + ||f(h_t)||)
    """

    def __init__(
        self,
        hidden_dim: int,
        spectral_radius: float = 0.99,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.spectral_radius = float(spectral_radius)
        self.eps = eps
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        h: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        """稳定残差更新.

        Args:
            h: [B, S, H] 当前隐状态.
            delta: [B, S, H] 残差增量 f(h).

        Returns:
            new_h: [B, S, H].
        """
        alpha_base = self.spectral_radius * torch.sigmoid(self.gate)
        # 按 token 归一化 delta 范数, 防止爆炸
        delta_norm = delta.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        scale = alpha_base / (1.0 + delta_norm)
        return h + scale * delta

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, "
            f"spectral_radius={self.spectral_radius}"
        )
