"""ABStack — 共享 AB 堆叠 (Shannon 简化版 + MathMaster 完整版).

提取 Shannon 与 MathMaster 共享的 AB 堆叠组件:

  * ``SimplifiedABStack`` — Shannon 简化版: 顺序 AB 块, 无 MetaRouter/SubAgent
  * ``MetaRouter``        — 1对1置换路由 (Sinkhorn 双随机矩阵)
  * ``SubAgent``          — 子agent (不同路由策略, 共享 ExpertPool)
  * ``FivePathAttention`` — 五路注意力 (Hybrid-M3 5 种注意力并行)
  * ``ABBlock``           — 完整 AB 块 (5路注意力 + 元路由器 + 子agent)
  * ``ABStack``           — 统一入口: ab_simplified=True 用简化版, False 用完整版

参考: MathMaster ABStack 设计, common.attention Hybrid-M3.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.layers import RMSNorm, GatedRMSNorm, SwiGLU
from common.attention import (
    AttentionConfig,
    AttentionOutput,
    MLAAttention,
    KDAAttention,
    LightningAttention,
    SlidingWindowAttention,
    MMAAttention,
    MoHAttention,
    GatedAttention,
)

from .base_config import BaseConfig
from .expert_pool import ExpertFFN, ExpertPool


# =====================================================================
# 辅助函数
# =====================================================================

def _sinkhorn_normalize(log_matrix: torch.Tensor, num_iters: int = 10) -> torch.Tensor:
    """对 [..., n, m] 的 log 概率矩阵做 Sinkhorn 归一化, 返回双随机矩阵."""
    z = log_matrix
    for _ in range(num_iters):
        z = z - torch.logsumexp(z, dim=-1, keepdim=True)  # 行归一化
        z = z - torch.logsumexp(z, dim=-2, keepdim=True)  # 列归一化
    return z.exp()


def _hungarian_hard_perm(soft_perm: torch.Tensor) -> torch.Tensor:
    """从软置换矩阵 [..., n, n] 提取硬置换 (贪心近似匈牙利匹配).

    返回 one-hot 置换矩阵, 保持可微性 (straight-through).
    """
    n = soft_perm.shape[-1]
    with torch.no_grad():
        hard = torch.zeros_like(soft_perm)
        cost = soft_perm.clone()
        for _ in range(n):
            idx = cost.argmax(dim=-1)
            row = torch.arange(n, device=soft_perm.device)
            flat_idx = idx + row * n
            flat_cost = cost.view(*cost.shape[:-2], n * n)
            hard_flat = hard.view(*hard.shape[:-2], n * n)
            hard_flat.scatter_(-1, flat_idx.unsqueeze(-1), 1.0)
            mask = torch.zeros_like(cost)
            mask.scatter_(
                -1, idx.unsqueeze(-1).unsqueeze(-2).expand_as(cost), 1.0,
            )
            cost = cost.masked_fill(mask.bool(), float("-inf"))
    return hard + soft_perm - soft_perm.detach()


# =====================================================================
# 简化版 AB 块 (Shannon 用)
# =====================================================================

class SimplifiedABBlock(nn.Module):
    """简化 AB 块 (Shannon 用): 单路注意力 + ExpertPool, 无 MetaRouter.

    结构 (pre-norm):
      h = norm(x)
      a = attention(h)
      m = moe_norm(h + a)
      out = x + ExpertPool(m)
    """

    def __init__(self, cfg: BaseConfig, layer_idx: int = 0):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim

        # 单路注意力 (MLA, 从 common.attention 导入)
        attn_cfg = cfg.build_attention_config(layer_idx=layer_idx)
        self.attn_norm = RMSNorm(d, eps=cfg.rms_eps)
        self.attention = MLAAttention(attn_cfg)

        # ExpertPool (6 常驻 + 16x16 MoE)
        self.moe_norm = RMSNorm(d, eps=cfg.rms_eps)
        self.expert_pool = ExpertPool(cfg)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """简化 AB 块前向.

        Returns:
            (output [b, s, d], aux_loss)
        """
        # 注意力 (pre-norm + residual)
        attn_in = self.attn_norm(x)
        attn_out = self.attention(attn_in, position_ids=position_ids)
        if isinstance(attn_out, AttentionOutput):
            attn_out = attn_out.output
        h = x + attn_out

        # ExpertPool (pre-norm + residual)
        moe_in = self.moe_norm(h)
        moe_out, aux = self.expert_pool(moe_in)
        out = h + moe_out
        return out, aux


class SimplifiedABStack(nn.Module):
    """简化 AB 堆叠 (Shannon 用): num_ab_blocks 个 SimplifiedABBlock 顺序堆叠.

    无 MetaRouter/SubAgent, 无五路注意力, 适合 Shannon 循环主体内的轻量 AB.
    """

    def __init__(self, cfg: BaseConfig, base_layer_idx: int = 0):
        super().__init__()
        self.cfg = cfg
        self.ab_blocks = nn.ModuleList([
            SimplifiedABBlock(cfg, layer_idx=base_layer_idx + i)
            for i in range(cfg.num_ab_blocks)
        ])
        self.output_norm = RMSNorm(cfg.hidden_dim, eps=cfg.rms_eps)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """简化 AB 堆叠前向.

        Returns:
            (output [b, s, d], aux_loss)
        """
        aux = x.new_zeros(())
        h = x
        for block in self.ab_blocks:
            h, a = block(h, position_ids=position_ids)
            aux = aux + a
        return self.output_norm(h), aux


# =====================================================================
# 完整版 AB 组件 (MathMaster 用)
# =====================================================================

class MetaRouter(nn.Module):
    """元路由器: 1对1置换路由 ("电线盒", 类似专家路由).

    将 num_paths 条注意力路径 1对1 置换到 num_sub_agents 个子agent.
    使用 Sinkhorn 归一化产生双随机矩阵 (软置换, 可微);
    推理时可选匈牙利硬置换.
    """

    def __init__(self, cfg: BaseConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim
        n = cfg.num_attention_paths
        self.path_proj = nn.Linear(d, n, bias=False)
        self.agent_keys = nn.Parameter(torch.randn(n, n) * 0.02)
        self.sinkhorn_iters = cfg.meta_router_sinkhorn_iters
        self.hard_perm = cfg.meta_router_hard_perm

    def forward(self, path_features: List[torch.Tensor]) -> torch.Tensor:
        """计算置换矩阵.

        Args:
            path_features: list of [b, s, d], 长度 num_paths.
        Returns:
            perm: [b, num_paths, num_paths] 双随机置换矩阵.
        """
        n = len(path_features)
        pooled = torch.stack([p.mean(dim=1) for p in path_features], dim=1)  # [b, n, d]
        proj = self.path_proj(pooled)                                        # [b, n, n]
        cost = torch.matmul(proj, self.agent_keys.t())                       # [b, n, n]
        perm = _sinkhorn_normalize(cost, self.sinkhorn_iters)
        if self.hard_perm and not self.training:
            perm = _hungarian_hard_perm(perm)
        return perm


class SubAgent(nn.Module):
    """子agent: 不同的路由策略, 共享 ExpertPool.

    5 个子agent (G1-G5) 各有不同的路由策略:
      * G1: top-1 big + top-1 small (激进/最小路由)
      * G2: top-2 big + top-2 small
      * G3: top-3 big + top-3 small
      * G4: top-4 big + top-4 small (默认/全路由)
      * G5: top-4 big + top-4 small + NLM 增强 (CTM)
    """

    STRATEGIES = [
        (1, 1, False),
        (2, 2, False),
        (3, 3, False),
        (4, 4, False),
        (4, 4, True),
    ]

    def __init__(self, cfg: BaseConfig, agent_id: int):
        super().__init__()
        self.cfg = cfg
        self.agent_id = agent_id
        self.norm = RMSNorm(cfg.hidden_dim, eps=cfg.rms_eps)
        strategy_idx = agent_id % len(self.STRATEGIES)
        self.top_k_big, self.top_k_small, self.use_nlm = self.STRATEGIES[strategy_idx]
        self.top_k_big = min(self.top_k_big, cfg.top_k_big)
        self.top_k_small = min(self.top_k_small, cfg.top_k_small)

    def forward(
        self, x: torch.Tensor, expert_pool: ExpertPool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """子agent 处理一条路径.

        Returns:
            (output [b, s, d], aux_loss)
        """
        normed = self.norm(x)
        out, aux = expert_pool(
            normed,
            top_k_big_override=self.top_k_big,
            top_k_small_override=self.top_k_small,
            use_nlm=self.use_nlm,
        )
        return x + out, aux


class FivePathAttention(nn.Module):
    """五路注意力 (A1-A5): 复用 Hybrid-M3 的 5 种注意力类型.

    对同一输入并行计算, 产生 num_attention_paths 条路径输出.
    """

    _ATTENTION_BUILDERS = {
        "mla": MLAAttention,
        "kda": KDAAttention,
        "lightning": LightningAttention,
        "sliding": SlidingWindowAttention,
        "mma": MMAAttention,
        "moh": MoHAttention,
        "gated": GatedAttention,
    }

    def __init__(self, cfg: BaseConfig, layer_idx: int = 0):
        super().__init__()
        self.cfg = cfg
        n = cfg.num_attention_paths
        attn_cfg = cfg.build_attention_config(layer_idx=layer_idx)

        available = [t for t in cfg.attention_types if t in self._ATTENTION_BUILDERS]
        selected = (available * ((n // len(available)) + 1))[:n] if available else []
        if not selected:
            selected = ["mla"] * n

        self.attentions = nn.ModuleList()
        for i, attn_type in enumerate(selected):
            builder = self._ATTENTION_BUILDERS[attn_type]
            self.attentions.append(builder(AttentionConfig(
                d_model=attn_cfg.d_model, n_heads=attn_cfg.n_heads,
                n_kv_heads=attn_cfg.n_kv_heads, d_kv=attn_cfg.d_kv,
                d_c=attn_cfg.d_c, max_seq_len=attn_cfg.max_seq_len,
                rope_theta=attn_cfg.rope_theta, dropout=attn_cfg.dropout,
                layer_idx=layer_idx * n + i, rms_eps=attn_cfg.rms_eps,
            )))
        self.norm = RMSNorm(cfg.hidden_dim, eps=cfg.rms_eps)

    def forward(
        self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """对同一输入计算 num_paths 路注意力输出.

        Returns:
            list of [b, s, d], 长度 num_paths.
        """
        normed = self.norm(x)
        outputs: List[torch.Tensor] = []
        for attn in self.attentions:
            out = attn(normed, position_ids=position_ids)
            if isinstance(out, AttentionOutput):
                out = out.output
            outputs.append(out)
        return outputs


class ABBlock(nn.Module):
    """完整 AB 块: 5路注意力 -> 元路由器置换 -> 子agent处理 -> 逆置换 -> 输出.

    内部结构:
      1. FivePathAttention: 对输入计算 num_paths 路注意力输出
      2. MetaRouter: 1对1置换 (路径 -> 子agent)
      3. SubAgents: 各子agent 用不同路由策略处理 (共享 ExpertPool)
      4. 逆置换: 子agent输出 -> 原路径顺序
      5. 聚合: 多路输出融合为单路
    """

    def __init__(self, cfg: BaseConfig, layer_idx: int = 0):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim
        n = cfg.num_attention_paths

        self.five_path_attn = FivePathAttention(cfg, layer_idx=layer_idx)
        self.meta_router = MetaRouter(cfg)
        self.sub_agents = nn.ModuleList([
            SubAgent(cfg, agent_id=i) for i in range(cfg.num_sub_agents)
        ])
        self.expert_pool = ExpertPool(cfg)

        self.input_norm = RMSNorm(d, eps=cfg.rms_eps)
        self.output_norm = RMSNorm(d, eps=cfg.rms_eps)
        self.path_gate = nn.Linear(d, n, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        paths: List[torch.Tensor],
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """AB 块前向.

        Args:
            x: [b, s, d] 原始输入 (用于五路注意力).
            paths: list of [b, s, d], 长度 num_paths.
        Returns:
            (new_paths, aux_loss)
        """
        n = self.cfg.num_attention_paths
        aux = x.new_zeros(())

        # 1. 五路注意力
        attn_outputs = self.five_path_attn(x, position_ids=position_ids)
        attended: List[torch.Tensor] = []
        for i in range(n):
            attended.append(self.input_norm(paths[i] + attn_outputs[i]))

        # 2. 元路由器
        perm = self.meta_router(attended)

        # 3. 应用置换
        stacked = torch.stack(attended, dim=1)                       # [b, n, s, d]
        permuted = torch.einsum("bpq,bqsd->bpsd", perm, stacked)    # [b, n, s, d]
        permuted_list = [permuted[:, i] for i in range(n)]

        # 4. 子agent 处理 (共享 ExpertPool)
        processed: List[torch.Tensor] = []
        for i in range(self.cfg.num_sub_agents):
            out, a = self.sub_agents[i](permuted_list[i], self.expert_pool)
            processed.append(out)
            aux = aux + a

        # 5. 逆置换
        processed_stack = torch.stack(processed, dim=1)              # [b, n, s, d]
        inv_perm = perm.transpose(-1, -2)
        output_paths_stack = torch.einsum(
            "bpq,bqsd->bpsd", inv_perm, processed_stack,
        )
        new_paths = [self.output_norm(output_paths_stack[:, i]) for i in range(n)]

        return new_paths, aux


# =====================================================================
# 统一 ABStack 入口
# =====================================================================

class ABStack(nn.Module):
    """AB 堆叠统一入口: 根据 ab_simplified 选择简化版或完整版.

    Shannon 用简化版 (ab_simplified=True): SimplifiedABStack
    MathMaster 用完整版 (ab_simplified=False): 完整 ABStack (num_ab_blocks 个 ABBlock)
    """

    def __init__(self, cfg: BaseConfig, base_layer_idx: int = 0):
        super().__init__()
        self.cfg = cfg
        self.simplified = cfg.ab_simplified

        if self.simplified:
            self.stack = SimplifiedABStack(cfg, base_layer_idx=base_layer_idx)
        else:
            d = cfg.hidden_dim
            n = cfg.num_attention_paths
            self.ab_blocks = nn.ModuleList([
                ABBlock(cfg, layer_idx=base_layer_idx + i)
                for i in range(cfg.num_ab_blocks)
            ])
            self.path_inits = nn.ModuleList([
                nn.Linear(d, d, bias=False) for _ in range(n)
            ])
            self.aggregate_norm = RMSNorm(d, eps=cfg.rms_eps)
            self.aggregate_gate = nn.Linear(d, n, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """AB 堆叠前向.

        Args:
            x: [b, s, d].
            position_ids: optional.
        Returns:
            (output [b, s, d], aux_loss)
        """
        if self.simplified:
            return self.stack(x, position_ids=position_ids)

        # 完整版
        n = self.cfg.num_attention_paths
        aux = x.new_zeros(())

        # 初始化 n 条路径
        paths: List[torch.Tensor] = [self.path_inits[i](x) for i in range(n)]

        # 依次通过各 AB 块
        for ab_block in self.ab_blocks:
            paths, a = ab_block(x, paths, position_ids=position_ids)
            aux = aux + a

        # 聚合 n 条路径
        stacked = torch.stack(paths, dim=-2)                         # [b, s, n, d]
        gate = torch.softmax(self.aggregate_gate(x), dim=-1)         # [b, s, n]
        out = torch.einsum("bsnd,bsn->bsd", stacked, gate)           # [b, s, d]
        out = self.aggregate_norm(out)
        return out, aux

    def extra_repr(self) -> str:
        mode = "simplified" if self.simplified else "full"
        return f"mode={mode}, blocks={self.cfg.num_ab_blocks}"
