"""流匹配全局规划 (FlowPlanner).

方案 A (Flow Matching, 可选): 在隐空间全局规划阶段使用 rectified flow
(整流流) 进行全局一致性建模. 通过 Euler ODE 求解器从噪声隐变量积分到
目标隐空间表示, 为下游 NAR / 掩码精化提供高质量全局上下文.

Rectified Flow 核心:
    - 速度场 v_θ(z_t, t) 学习从 z_0 (噪声) 到 z_1 (数据) 的位移.
    - 训练目标: 沿直线插值 z_t = (1-t) z_0 + t z_1, 回归 v = z_1 - z_0.
    - 推理: Euler 积分 z_{t+dt} = z_t + v_θ(z_t, t) * dt.

本模块作为可选的全局规划器, 输出整合到 HierarchicalNAR / MaskRefinement
的条件中.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FlowPlannerConfig:
    """流匹配规划器配置."""

    latent_dim: int = 1024
    hidden_dim: int = 1024
    num_heads: int = 16
    num_layers: int = 4
    # ODE 求解
    num_euler_steps: int = 50          # Euler 积分步数
    solver: str = "euler"              # "euler" | "heun" (改进 Euler)
    # 时间条件
    time_embed_dim: int = 256
    # 训练
    sigma_min: float = 1e-3            # 数据端最小噪声
    use_rectified: bool = True         # 是否使用 rectified (直线) 路径


class SinusoidalTimeEmbedding(nn.Module):
    """正弦时间嵌入 (扩散/流匹配通用)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        half = dim // 2
        freqs = torch.exp(
            -torch.arange(half, dtype=torch.float32)
            * (torch.log(torch.tensor(10000.0)) / max(half - 1, 1))
        )
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [..., 1] or scalar
        t = t.float().reshape(-1, 1)
        args = t * self.freqs.unsqueeze(0) * 2 * torch.pi
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class VelocityNet(nn.Module):
    """速度场网络 v_θ(z_t, t): Transformer 主干 + 时间条件."""

    def __init__(self, cfg: FlowPlannerConfig):
        super().__init__()
        self.cfg = cfg
        self.time_embed = SinusoidalTimeEmbedding(cfg.time_embed_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(cfg.time_embed_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        # AdaLN-zero 风格的时间调制
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim * 2),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.in_proj = nn.Linear(cfg.latent_dim, cfg.hidden_dim)
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """v_θ(z_t, t).

        Args:
            z_t: [B, T, latent_dim].
            t: [B] 时间 ∈ [0, 1].
        """
        h = self.in_proj(z_t)
        # 时间条件
        t_emb = self.time_embed(t)         # [B, time_embed_dim]
        t_cond = self.time_proj(t_emb)     # [B, hidden_dim]
        # AdaLN-zero 调制
        scale_shift = self.adaLN(t_cond)   # [B, 2*hidden_dim]
        scale, shift = scale_shift.chunk(2, dim=-1)
        h = h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        # Transformer
        h = self.transformer(h)
        v = self.out_proj(h)
        return v


class FlowPlanner(nn.Module):
    """Rectified Flow 全局规划器 (Euler ODE 求解)."""

    def __init__(self, config: FlowPlannerConfig | None = None, **kwargs):
        super().__init__()
        cfg = config or FlowPlannerConfig(**kwargs)
        self.cfg = cfg
        self.velocity_net = VelocityNet(cfg)

    # ------------------------------------------------------------------
    # 训练: rectified flow 回归损失
    # ------------------------------------------------------------------
    def compute_loss(
        self,
        z_1: torch.Tensor,
        z_0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """rectified flow 训练损失.

        Args:
            z_1: [B, T, latent_dim] 数据端隐变量.
            z_0: [B, T, latent_dim] 噪声端. None 则随机采样.
        """
        B = z_1.shape[0]
        if z_0 is None:
            z_0 = torch.randn_like(z_1)
        # 随机时间 t ~ U[0, 1]
        t = torch.rand(B, device=z_1.device)
        # 直线插值 (rectified): z_t = (1-t) z_0 + t z_1
        t_expand = t.reshape(B, 1, 1)
        z_t = (1 - t_expand) * z_0 + t_expand * z_1
        # 目标速度: v = z_1 - z_0 (直线)
        v_target = z_1 - z_0
        v_pred = self.velocity_net(z_t, t)
        return F.mse_loss(v_pred, v_target)

    # ------------------------------------------------------------------
    # 推理: Euler ODE 积分 z_0 -> z_1
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        z_0: torch.Tensor | None = None,
        shape: tuple[int, ...] | None = None,
        num_steps: int | None = None,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """从噪声 z_0 积分到数据 z_1.

        Args:
            z_0: [B, T, latent_dim] 初始噪声. None 则按 shape 随机采样.
            shape: (B, T, latent_dim), 当 z_0 为 None 时使用.
            num_steps: Euler 步数, None 用配置默认值.
            return_trajectory: 是否返回中间轨迹.
        """
        if z_0 is None:
            assert shape is not None, "shape required when z_0 is None"
            z_0 = torch.randn(*shape, device=self._param_device())
        num_steps = num_steps or self.cfg.num_euler_steps
        dt = 1.0 / num_steps
        z = z_0.clone()
        trajectory = [z.clone()]
        B = z.shape[0]
        for i in range(num_steps):
            t = torch.full((B,), i * dt, device=z.device)
            v = self.velocity_net(z, t)
            if self.cfg.solver == "heun":
                # Heun (改进 Euler): 先 Euler 预测, 再取平均斜率
                z_pred = z + v * dt
                t_next = torch.full((B,), (i + 1) * dt, device=z.device)
                v_next = self.velocity_net(z_pred, t_next)
                z = z + 0.5 * (v + v_next) * dt
            else:
                # 标准 Euler
                z = z + v * dt
            trajectory.append(z.clone())
        if return_trajectory:
            return z, trajectory
        return z

    # ------------------------------------------------------------------
    def _param_device(self) -> torch.device:
        return next(self.parameters()).device

    def extra_repr(self) -> str:
        return (
            f"latent_dim={self.cfg.latent_dim}, "
            f"solver={self.cfg.solver}, "
            f"num_euler_steps={self.cfg.num_euler_steps}, "
            f"rectified={self.cfg.use_rectified}"
        )
