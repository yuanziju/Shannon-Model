"""双层MoE包 — 双层路由 + 专家 + 空专家 + 负载均衡 + All-to-All.

公开API:
    DualLayerRouter       - 16 Top-4 + 16 Top-4 双层路由器
    StandardExpert        - 标准SwiGLU FFN专家
    BigExpert             - 粗粒度大专家 (可选NLM增强)
    SmallExpert           - 细粒度小专家
    SharedExpert          - 常驻共享专家 (DeepSeek模式)
    EmptyExpert           - 零初始化空专家
    EmptyExpertFramework  - 空专家框架 (能力吸收)
    ExpertAbsorber        - 专家能力吸收器
    LoadBalancer          - 负载均衡器
    MoEAllToAll           - EP All-to-All 通信
    NestedMoE             - 双层MoE完整模块 (路由+专家+共享+空专家)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .router import DualLayerRouter, RouterOutput
from .experts import StandardExpert, BigExpert, SmallExpert, SharedExpert
from .empty_expert import EmptyExpert, EmptyExpertFramework
from .expert_absorb import ExpertAbsorber
from .load_balancer import LoadBalancer
from .all2all import MoEAllToAll


class NestedMoE(nn.Module):
    """双层MoE完整模块: 路由 + 大专家 + 小专家 + 共享专家 + 空专家.

    架构 (spec 双层MoE):
      - 16 BigExpert (Top-4 路由, 可选NLM增强)
      - 16 SmallExpert (Top-4 路由, 无NLM)
      - num_shared SharedExpert (常驻, DeepSeek模式)
      - num_empty EmptyExpert (零初始化, 能力吸收)
      - DualLayerRouter 双层路由
      - LoadBalancer 负载均衡

    前向输出 = 大专家加权 + 小专家加权 + 共享专家 + 空专家
    """

    def __init__(
        self,
        hidden_dim: int,
        num_big_experts: int = 16,
        num_small_experts: int = 16,
        top_k_big: int = 4,
        top_k_small: int = 4,
        expert_ffn_dim: int = 1024,
        small_expert_ffn_dim: int = 512,
        num_shared_experts: int = 2,
        num_empty_experts: int = 4,
        use_nlm: bool = False,
        nlm_num_neurons: int = 8,
        nlm_d_state: int = 16,
        nlm_warmup_freeze: bool = True,
        noise_std: float = 1.0,
        load_balance_alpha: float = 0.01,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_big_experts = num_big_experts
        self.num_small_experts = num_small_experts

        # 双层路由器
        self.router = DualLayerRouter(
            hidden_dim=hidden_dim,
            num_big_experts=num_big_experts,
            num_small_experts=num_small_experts,
            top_k_big=top_k_big,
            top_k_small=top_k_small,
            noise_std=noise_std,
            load_balance_alpha=load_balance_alpha,
        )

        # 大专家 (粗粒度, 可选NLM)
        self.big_experts = nn.ModuleList([
            BigExpert(
                hidden_dim, expert_ffn_dim,
                use_nlm=use_nlm,
                nlm_num_neurons=nlm_num_neurons,
                nlm_d_state=nlm_d_state,
                nlm_warmup_freeze=nlm_warmup_freeze,
                dropout=dropout,
            )
            for _ in range(num_big_experts)
        ])

        # 小专家 (细粒度, 无NLM)
        self.small_experts = nn.ModuleList([
            SmallExpert(hidden_dim, small_expert_ffn_dim, dropout=dropout)
            for _ in range(num_small_experts)
        ])

        # 常驻共享专家
        shared_ffn = expert_ffn_dim
        self.shared_experts = nn.ModuleList([
            SharedExpert(hidden_dim, shared_ffn, dropout=dropout)
            for _ in range(num_shared_experts)
        ])

        # 空专家框架
        self.empty_expert_framework = EmptyExpertFramework(
            hidden_dim, small_expert_ffn_dim,
            num_empty_experts=num_empty_experts,
        )

        # 负载均衡器 (用于大专家和小专家)
        self.big_balancer = LoadBalancer(num_big_experts, alpha=load_balance_alpha)
        self.small_balancer = LoadBalancer(num_small_experts, alpha=load_balance_alpha)

        # 输出归一化
        from common.layers import RMSNorm
        self.norm = RMSNorm(hidden_dim)

    def forward(
        self, x: torch.Tensor, nlm_states: Optional[List] = None
    ) -> Dict[str, torch.Tensor]:
        """双层MoE前向.

        Args:
            x: [B, S, H] 或 [N, H] 输入.
            nlm_states: 上一 tick 的 NLM 状态 (用于大专家CTM).

        Returns:
            dict 含 output, aux_loss, router_info.
        """
        orig_shape = x.shape
        if x.dim() == 3:
            B, S, H = x.shape
            x_flat = x.reshape(B * S, H)
        else:
            B, S = 1, x.shape[0]
            x_flat = x
        N = x_flat.shape[0]

        # 路由
        routing = self.router(x_flat)

        # ---- 大专家前向 ----
        big_out = torch.zeros_like(x_flat)
        new_nlm_states: Dict[int, list] = {}
        # NLM 状态维护: 全 token [N, d_state] per expert per neuron.
        # 路由每迭代变化, 需按 sel 选择/散射状态行 (functional, autograd-safe).
        for k in range(routing.big_indices.shape[1]):
            for ei in range(self.num_big_experts):
                sel = routing.big_indices[:, k] == ei
                if not sel.any():
                    continue
                x_sel = x_flat[sel]
                w = routing.big_weights[sel, k].unsqueeze(-1)
                # 选择 NLM 状态: 优先用当前迭代已更新的 (running), 其次上一迭代
                running = new_nlm_states.get(ei)
                prev = nlm_states.get(ei) if nlm_states else None
                if running is not None:
                    states_sel = [s[sel] for s in running]
                elif prev is not None:
                    states_sel = [s[sel] for s in prev]
                else:
                    states_sel = None
                out, ns = self.big_experts[ei](x_sel, states_sel)
                big_out[sel] += w * out
                # 散射回全 token 状态 (functional index_copy, autograd-safe)
                if ns is not None:
                    sel_idx = sel.nonzero(as_tuple=True)[0]
                    d_state = ns[0].shape[-1]
                    if running is not None:
                        base = running
                    elif prev is not None:
                        base = prev
                    else:
                        base = [
                            torch.zeros(
                                N, d_state,
                                device=x_flat.device, dtype=x_flat.dtype,
                            )
                            for _ in ns
                        ]
                    new_nlm_states[ei] = [
                        base[j].index_copy(0, sel_idx, ns[j])
                        for j in range(len(ns))
                    ]

        # ---- 小专家前向 ----
        small_out = torch.zeros_like(x_flat)
        for k in range(routing.small_indices.shape[1]):
            for ei in range(self.num_small_experts):
                sel = routing.small_indices[:, k] == ei
                if not sel.any():
                    continue
                x_sel = x_flat[sel]
                w = routing.small_weights[sel, k].unsqueeze(-1)
                out = self.small_experts[ei](x_sel)
                small_out[sel] += w * out

        # ---- 共享专家 (始终激活) ----
        shared_out = torch.zeros_like(x_flat)
        for se in self.shared_experts:
            shared_out = shared_out + se(x_flat) / max(len(self.shared_experts), 1)

        # ---- 空专家 (仅已吸收的) ----
        empty_out, _ = self.empty_expert_framework(x_flat, only_absorbed=True)

        # ---- 合并 ----
        combined = big_out + small_out + shared_out + empty_out
        combined = self.norm(combined)

        # 恢复原始形状
        if len(orig_shape) == 3:
            combined = combined.reshape(orig_shape)

        # 辅助损失
        aux_loss = routing.aux_loss

        return {
            "output": combined,
            "aux_loss": aux_loss,
            "big_out": big_out,
            "small_out": small_out,
            "shared_out": shared_out,
            "empty_out": empty_out,
            "new_nlm_states": new_nlm_states,
            "routing": routing,
        }

    def extra_repr(self) -> str:
        return (
            f"big={self.num_big_experts}, small={self.num_small_experts}, "
            f"shared={len(self.shared_experts)}, "
            f"empty={self.empty_expert_framework.num_empty_experts}"
        )


__all__ = [
    "DualLayerRouter",
    "RouterOutput",
    "StandardExpert",
    "BigExpert",
    "SmallExpert",
    "SharedExpert",
    "EmptyExpert",
    "EmptyExpertFramework",
    "ExpertAbsorber",
    "LoadBalancer",
    "MoEAllToAll",
    "NestedMoE",
]
