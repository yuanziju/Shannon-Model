"""ExpertPool — 共享专家池 (6 常驻 + 16x16 双层 MoE).

提取 Shannon 与 MathMaster 共享的专家池组件:

  * ``ExpertFFN``            — SwiGLU + down-proj 标准专家 FFN
  * ``EmptyExpert``          — 零初始化空专家 (Shannon 风格, 逐步填充)
  * ``ResidentExpertPool``   — 6 常驻专家 (4 固定 + 2 可学习), 始终开启
  * ``DualMoEPool``          — 16 大 x 16 小双层 MoE (Top-k 路由)
  * ``ExpertPool``           — 完整专家池 = ResidentExpertPool + DualMoEPool

Shannon 用 ResidentExpertPool (与现有 NestedMoE 并行, 结果相加).
MathMaster 用完整 ExpertPool.

参考: AGENTS.md Agent 9 (MoEAgent), MathMaster ExpertPool 设计.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.layers import RMSNorm, SwiGLU
from common.ctm import NLMLayer, CTMRouter

from .base_config import BaseConfig


# =====================================================================
# 基础专家
# =====================================================================

class ExpertFFN(nn.Module):
    """专家 FFN: SwiGLU 扩展 + 线性降维回 hidden_dim.

    标准 LLaMA 风格 FFN: w_down(silu(w_gate(x)) * w_up(x)).
    """

    def __init__(self, d_model: int, inter_dim: int, bias: bool = False,
                 dropout: float = 0.0):
        super().__init__()
        self.glu = SwiGLU(d_model, inter_dim, bias=bias)
        self.down = nn.Linear(inter_dim, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(self.glu(x)))


class EmptyExpert(nn.Module):
    """零初始化空专家 (参考 Shannon EmptyExpert 设计).

    down 投影零初始化且输出乘以零初始化标量门控, 初始贡献为 0,
    在持续学习阶段逐步吸收新能力.
    """

    def __init__(self, d_model: int, inter_dim: int, eps: float = 1e-6,
                 dropout: float = 0.0):
        super().__init__()
        self.glu = SwiGLU(d_model, inter_dim)
        self.down = nn.Linear(inter_dim, d_model, bias=False)
        nn.init.zeros_(self.down.weight)  # 零初始化 down 投影
        self.gate = nn.Parameter(torch.zeros(1))  # 零初始化门控
        self.norm = RMSNorm(d_model, eps=eps)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.gate * self.dropout(self.down(self.glu(x))))


# =====================================================================
# 常驻专家池 (6 常驻: 4 固定 + 2 可学习)
# =====================================================================

class ResidentExpertPool(nn.Module):
    """常驻专家池: 4 固定 + 2 可学习 (参考 MathMaster 设计).

    组件:
      * 4 固定常驻专家 (ExpertFFN, 密集, 不受路由, 始终开启)
      * 2 可学习常驻专家 (EmptyExpert 零初始化, 可选 NLM 增强)
      * NLM 增强 (可选, CTM 决策 C10: 仅实体专家使用 NLM)

    常驻专家与双层 MoE 并行计算, 结果相加.
    """

    def __init__(self, cfg: BaseConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim
        rc = cfg.resident_expert_config
        big_inter = rc.resident_ffn_dim

        # 4 固定常驻专家 (StandardExpert, 始终开启)
        self.fixed_experts = nn.ModuleList([
            ExpertFFN(d, big_inter, dropout=cfg.dropout)
            for _ in range(rc.num_fixed_resident_experts)
        ])

        # 2 可学习常驻专家 (EmptyExpert 零初始化, 逐步填充)
        self.learnable_experts = nn.ModuleList([
            EmptyExpert(d, big_inter, eps=cfg.rms_eps, dropout=cfg.dropout)
            for _ in range(rc.num_learnable_resident_experts)
        ])

        # NLM 增强 (可选, 增强可学习专家的激活)
        self.use_nlm = rc.learnable_nlm_enhanced and cfg.ctm_enabled
        if self.use_nlm:
            self.nlm_layers = nn.ModuleList([
                NLMLayer(
                    d, num_neurons=cfg.nlm_num_neurons,
                    d_state=cfg.nlm_d_state,
                    warmup_freeze=cfg.nlm_warmup_freeze,
                )
                for _ in range(rc.num_learnable_resident_experts)
            ])
        else:
            self.nlm_layers = None

        self.resident_norm = RMSNorm(d, eps=cfg.rms_eps)
        self.num_resident = rc.num_resident_experts
        self.num_fixed = rc.num_fixed_resident_experts
        self.num_learnable = rc.num_learnable_resident_experts

    def forward(
        self,
        x: torch.Tensor,
        use_nlm: bool = False,
    ) -> torch.Tensor:
        """常驻专家前向 (密集, 始终开启).

        Args:
            x: [b, s, d] 输入.
            use_nlm: 是否启用 NLM 增强 (可学习专家).

        Returns:
            output [b, s, d].
        """
        b, s, d = x.shape
        resident = x.new_zeros(b, s, d)

        # 固定常驻专家 (始终开启)
        for exp in self.fixed_experts:
            resident = resident + exp(x)

        # 可学习常驻专家 (EmptyExpert, 零门控逐步填充) + 可选 NLM
        for i, exp in enumerate(self.learnable_experts):
            exp_out = exp(x)
            if use_nlm and self.nlm_layers is not None:
                x_flat = x.reshape(b * s, d)
                nlm_out, _ = self.nlm_layers[i](x_flat)
                nlm_out = nlm_out.reshape(b, s, d)
                exp_out = exp_out + nlm_out
            resident = resident + exp_out

        # 均值聚合
        resident = resident / max(self.num_resident, 1)
        return self.resident_norm(resident)

    def extra_repr(self) -> str:
        return (
            f"fixed={self.num_fixed}, learnable={self.num_learnable}, "
            f"nlm={self.use_nlm}"
        )


# =====================================================================
# 双层 MoE 池 (16 大 x 16 小, Top-k 路由)
# =====================================================================

class DualMoEPool(nn.Module):
    """双层 MoE 池: 16 大专家 + 16 小专家 (Top-k 路由).

    大专家 (粗粒度) 与小专家 (细粒度) 独立路由, 输出加权合并.
    """

    def __init__(self, cfg: BaseConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim
        big_inter = cfg.moe_inter_dim
        small_inter = max(d, int(big_inter * cfg.small_expert_inter_ratio))

        self.big_experts = nn.ModuleList([
            ExpertFFN(d, big_inter, dropout=cfg.dropout)
            for _ in range(cfg.num_big_experts)
        ])
        self.small_experts = nn.ModuleList([
            ExpertFFN(d, small_inter, dropout=cfg.dropout)
            for _ in range(cfg.num_small_experts)
        ])
        self.big_router = nn.Linear(d, cfg.num_big_experts, bias=False)
        self.small_router = nn.Linear(d, cfg.num_small_experts, bias=False)
        nn.init.normal_(self.big_router.weight, std=0.02)
        nn.init.normal_(self.small_router.weight, std=0.02)
        self.moe_norm = RMSNorm(d, eps=cfg.rms_eps)

        self.top_k_big = cfg.top_k_big
        self.top_k_small = cfg.top_k_small
        self.num_big = cfg.num_big_experts
        self.num_small = cfg.num_small_experts
        self.router_noise_std = cfg.router_noise_std

    def _route_topk(
        self,
        x_flat: torch.Tensor,
        router: nn.Linear,
        num_experts: int,
        top_k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Top-k 路由: 返回 (indices, weights, scores)."""
        logits = router(x_flat)
        if self.training and self.router_noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.router_noise_std
        scores = F.softmax(logits, dim=-1)
        k = min(top_k, num_experts)
        topk_scores, topk_idx = scores.topk(k, dim=-1)
        topk_scores = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-9)
        return topk_idx, topk_scores, scores

    def _gather_moe(
        self,
        x_flat: torch.Tensor,
        experts: nn.ModuleList,
        topk_idx: torch.Tensor,
        topk_scores: torch.Tensor,
    ) -> torch.Tensor:
        """聚集 top-k 专家输出."""
        N, k = topk_idx.shape
        out = torch.zeros_like(x_flat)
        for ki in range(k):
            idx_ki = topk_idx[:, ki]
            w_ki = topk_scores[:, ki]
            for ei in range(len(experts)):
                mask = idx_ki == ei
                if not mask.any():
                    continue
                x_sel = x_flat[mask]
                out_sel = experts[ei](x_sel)
                out[mask] += w_ki[mask].unsqueeze(-1) * out_sel
        return out

    def _load_balance_loss(
        self, scores: torch.Tensor, topk_idx: torch.Tensor, num_experts: int
    ) -> torch.Tensor:
        """标准 MoE 负载均衡损失."""
        N = scores.shape[0]
        flat_idx = topk_idx.reshape(-1)
        tokens_per_expert = torch.bincount(flat_idx, minlength=num_experts).float()
        frac_tokens = tokens_per_expert / max(N, 1)
        mean_prob = scores.mean(dim=0)
        return num_experts * (frac_tokens * mean_prob).sum()

    def forward(
        self,
        x: torch.Tensor,
        top_k_big_override: Optional[int] = None,
        top_k_small_override: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """双层 MoE 前向.

        Args:
            x: [b, s, d] 输入.
        Returns:
            (output [b, s, d], aux_loss scalar)
        """
        b, s, d = x.shape
        aux = x.new_zeros(())
        x_flat = x.reshape(b * s, d)

        k_big = top_k_big_override or self.top_k_big
        k_small = top_k_small_override or self.top_k_small

        big_idx, big_w, big_scores = self._route_topk(
            x_flat, self.big_router, self.num_big, k_big)
        big_out = self._gather_moe(x_flat, self.big_experts, big_idx, big_w)
        aux = aux + self._load_balance_loss(big_scores, big_idx, self.num_big)

        small_idx, small_w, small_scores = self._route_topk(
            x_flat, self.small_router, self.num_small, k_small)
        small_out = self._gather_moe(x_flat, self.small_experts, small_idx, small_w)
        aux = aux + self._load_balance_loss(small_scores, small_idx, self.num_small)

        moe_out = (big_out + small_out).reshape(b, s, d)
        return self.moe_norm(moe_out), aux


# =====================================================================
# 完整专家池 (6 常驻 + 16x16 双层 MoE)
# =====================================================================

class ExpertPool(nn.Module):
    """完整专家池: 6 常驻 (4 固定 + 2 可学习) + 16 大 x 16 小 双层 MoE.

    常驻专家 (密集) 与双层 MoE (稀疏) 并行计算, 结果相加.

    组件:
      * ResidentExpertPool: 6 常驻专家
      * DualMoEPool: 16 大 + 16 小 双层 MoE
      * CTMRouter (可选): 复杂度驱动 NLM 增强
    """

    def __init__(self, cfg: BaseConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim

        self.resident_pool = ResidentExpertPool(cfg)
        self.moe_pool = DualMoEPool(cfg)

        # CTMRouter (可选): 复杂度驱动 NLM 增强开关
        self.use_ctm_router = cfg.ctm_enabled
        if self.use_ctm_router:
            self.ctm_router = CTMRouter(
                d_model=d,
                num_nlm=cfg.num_learnable_resident_experts,
                num_standard=cfg.num_big_experts,
                num_shared=cfg.num_fixed_resident_experts,
                top_k=min(cfg.top_k_big, 4),
                complexity_threshold=cfg.ctm_complexity_threshold,
                router_dropout=cfg.attention_dropout,
                noise_std=cfg.router_noise_std,
            )
        else:
            self.ctm_router = None

    def forward(
        self,
        x: torch.Tensor,
        top_k_big_override: Optional[int] = None,
        top_k_small_override: Optional[int] = None,
        use_nlm: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """完整专家池前向.

        Args:
            x: [b, s, d] 输入.
            top_k_big_override / top_k_small_override: 覆盖 top-k.
            use_nlm: 是否启用 NLM 增强.

        Returns:
            (output [b, s, d], aux_loss scalar)
        """
        # 常驻专家 (密集, 始终开启)
        resident_out = self.resident_pool(x, use_nlm=use_nlm)

        # 双层 MoE (稀疏, top-k 路由)
        moe_out, aux = self.moe_pool(
            x,
            top_k_big_override=top_k_big_override,
            top_k_small_override=top_k_small_override,
        )

        # 并行相加
        out = resident_out + moe_out
        return out, aux

    def extra_repr(self) -> str:
        return (
            f"resident={self.resident_pool.num_resident}, "
            f"moe={self.moe_pool.num_big}x{self.moe_pool.num_small}"
        )
