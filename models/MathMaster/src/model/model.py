"""MathMaster 模型实现 (新底子架构).

架构总览:
    输入 → MathEncoder (文本嵌入 + NSL符号嵌入 + 位置编码 + 1层Transformer)
         → MathRecurrentBody (1-32次动态迭代, 每轮含4部分:
              1. ResidualPool  — AttnRes+mHC + attention检索 + 删除压缩 + 每3轮top-k
              2. IntuitionLayer — 轻量MLP快通道 + 隐变量采样 (基础版)
              3. ABStack       — 10个AB堆叠 (5路注意力 + 元路由器 + 子agent + 共享专家池)
              4. LoopControl   — AB输出传递 + 残差池管理 + 深度嵌入
           )
         → MathDecoder (多任务输出头: text/lean4/sympy/conjecture/proof_step/confidence)

所有公共组件 (注意力 / 层 / CTM / NSL) 从 common/ 导入复用, 不重新实现.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# === 从 common/ 导入复用 (不重新实现) ====================================
from common.attention import (
    MLAAttention,
    KDAAttention,
    LightningAttention,
    SlidingWindowAttention,
    MMAAttention,
    MoHAttention,
    GatedAttention,
    DynamicAttentionController,
    AttentionConfig,
    AttentionOutput,
)
from common.layers import (
    RMSNorm,
    GatedRMSNorm,
    SwiGLU,
    RoPE,
    YaRN,
    LongRoPE2,
    AttnRes,
    mHC,
)
from common.ctm import (
    NLMLayer,
    MLASync,
    CTMDynamicLoss,
    CTMRouter,
)
from common.nsl import (
    SymbolNeuralBridge,
    NSLGrammar,
    FormalParser,
    NSLDecoder,
)

from ..config.config import MathConfig


# =====================================================================
# 辅助函数
# =====================================================================

def _build_attention_config(cfg: MathConfig, layer_idx: int = 0) -> AttentionConfig:
    """从 MathConfig 构建 AttentionConfig (Hybrid-M3 共享配置)."""
    return AttentionConfig(
        d_model=cfg.hidden_dim,
        n_heads=cfg.num_heads,
        n_kv_heads=cfg.num_kv_heads,
        d_kv=cfg.head_dim,
        d_c=cfg.hidden_dim // 4,
        max_seq_len=cfg.max_seq_len,
        rope_theta=cfg.rope_theta,
        dropout=cfg.attention_dropout,
        layer_idx=layer_idx,
        rms_eps=cfg.rms_eps,
    )


def _sinkhorn_normalize(log_matrix: torch.Tensor, num_iters: int = 10) -> torch.Tensor:
    """对 [..., n, m] 的 log 概率矩阵做 Sinkhorn 归一化, 返回双随机矩阵.

    使用 log 空间计算以保证数值稳定性.
    """
    z = log_matrix
    for _ in range(num_iters):
        z = z - torch.logsumexp(z, dim=-1, keepdim=True)   # 行归一化
        z = z - torch.logsumexp(z, dim=-2, keepdim=True)   # 列归一化
    return z.exp()


def _hungarian_hard_perm(soft_perm: torch.Tensor) -> torch.Tensor:
    """从软置换矩阵 [..., n, n] 提取硬置换 (贪心近似匈牙利匹配).

    返回 one-hot 置换矩阵, 保持可微性 (straight-through: 前向用硬, 反向用软).
    """
    n = soft_perm.shape[-1]
    with torch.no_grad():
        hard = torch.zeros_like(soft_perm)
        cost = soft_perm.clone()
        for _ in range(n):
            idx = cost.argmax(dim=-1)             # [..., n] 每行最大列
            row = torch.arange(n, device=soft_perm.device)
            # 展平以便用 scatter
            flat_idx = idx + row * n
            flat_cost = cost.view(*cost.shape[:-2], n * n)
            hard_flat = hard.view(*hard.shape[:-2], n * n)
            hard_flat.scatter_(-1, flat_idx.unsqueeze(-1), 1.0)
            # 置零已选列, 防止重复
            mask = torch.zeros_like(cost)
            mask.scatter_(-1, idx.unsqueeze(-1).unsqueeze(-2).expand_as(cost), 1.0)
            cost = cost.masked_fill(mask.bool(), float("-inf"))
    # straight-through 估计器
    return hard + soft_perm - soft_perm.detach()


# =====================================================================
# ExpertFFN (专家前馈网络, 带 down-projection)
# =====================================================================

class ExpertFFN(nn.Module):
    """专家 FFN: SwiGLU 扩展 + 线性降维回 hidden_dim.

    SwiGLU(in, inter) 产生 inter 维输出, 再经 down_proj 降回 in 维.
    这是标准 LLaMA 风格 FFN: w_down(silu(w_gate(x)) * w_up(x)).
    """

    def __init__(self, d_model: int, inter_dim: int, bias: bool = False):
        super().__init__()
        self.glu = SwiGLU(d_model, inter_dim, bias=bias)
        self.down = nn.Linear(inter_dim, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.glu(x))


# =====================================================================
# MathEncoder
# =====================================================================

class MathEncoder(nn.Module):
    """数学编码器: 文本嵌入 + NSL符号嵌入 + 位置编码 + 1层Transformer.

    组件:
      * token embedding (vocab_size → hidden_dim)
      * NSL 符号 embedding (nsl_vocab_size → hidden_dim, 门控融合)
      * 位置编码 (LongRoPE2, 支持 1M-10M 上下文)
      * 1 层 Transformer (DynamicAttentionController 路由 MLA+Gated + SwiGLU FFN)
      * SymbolNeuralBridge (符号-神经对齐, 可选 AST 输入时启用)
    """

    def __init__(self, cfg: MathConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim

        # 文本嵌入 + NSL 符号嵌入 (门控融合)
        self.token_embed = nn.Embedding(cfg.vocab_size, d)
        self.symbol_embed = nn.Embedding(cfg.nsl_vocab_size, d)
        self.symbol_gate = nn.Linear(d, d)
        nn.init.zeros_(self.symbol_gate.weight)  # 初始门控为0, 逐步引入符号信息

        # 位置编码 (LongRoPE2 for 1M-10M; 同时保留 RoPE/YaRN 作为备选)
        self.pos_encoding = LongRoPE2(
            dim=d,
            base=cfg.rope_theta,
            max_seq_len=cfg.max_seq_len,
            original_max_position_embeddings=cfg.pos_original_max,
        )

        # NSL 组件 (导入复用)
        self.grammar = NSLGrammar()
        self.parser = FormalParser(self.grammar)
        self.symbol_bridge = SymbolNeuralBridge(
            d_model=d,
            grammar=self.grammar,
            num_heads=cfg.nsl_num_heads,
            num_layers=cfg.nsl_num_layers,
            vocab_size=cfg.nsl_vocab_size,
            temperature=cfg.nsl_temperature,
            max_nodes=cfg.nsl_max_nodes,
        )

        # 1 层 Transformer: DynamicAttentionController 路由 [MLA, Gated]
        attn_cfg = _build_attention_config(cfg, layer_idx=0)
        self.attn_norm = RMSNorm(d, eps=cfg.rms_eps)
        self.attn_pool = nn.ModuleList([
            MLAAttention(attn_cfg),
            GatedAttention(attn_cfg),
        ])
        self.attn_controller = DynamicAttentionController(attn_cfg, list(self.attn_pool))
        self.ffn_norm = RMSNorm(d, eps=cfg.rms_eps)
        self.ffn = ExpertFFN(d, cfg.moe_inter_dim)

        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        asts: Optional[List] = None,
    ) -> Dict[str, torch.Tensor]:
        b, s = input_ids.shape
        d = self.cfg.hidden_dim

        # 文本嵌入
        text_emb = self.token_embed(input_ids)                       # [b, s, d]
        # NSL 符号嵌入 (将 input_ids 映射到符号词表空间, 门控融合)
        sym_ids = input_ids % self.cfg.nsl_vocab_size
        sym_emb = self.symbol_embed(sym_ids)                          # [b, s, d]
        gate = torch.sigmoid(self.symbol_gate(text_emb))              # [b, s, d]
        hidden = text_emb + gate * sym_emb

        # 位置编码 (LongRoPE2)
        hidden = self.pos_encoding(hidden)
        hidden = self.dropout(hidden)

        # 1 层 Transformer (pre-norm)
        normed = self.attn_norm(hidden)
        attn_out = self.attn_controller(normed)
        if isinstance(attn_out, AttentionOutput):
            attn_out = attn_out.output
        hidden = hidden + self.dropout(attn_out)

        normed = self.ffn_norm(hidden)
        ffn_out = self.ffn(normed)
        hidden = hidden + self.dropout(ffn_out)

        out: Dict[str, torch.Tensor] = {"hidden": hidden, "text_embedding": text_emb}
        # 可选: 符号-神经对齐 (当提供 AST 时)
        if asts is not None:
            pooled = hidden.mean(dim=1)                               # [b, d]
            bridge_out = self.symbol_bridge(asts, pooled, device=hidden.device)
            out["symbol_embedding"] = bridge_out["symbol_embedding"]
            out["infonce_loss"] = bridge_out["infonce_loss"]
        return out


# =====================================================================
# ResidualPool (残差池)
# =====================================================================

class ResidualPool(nn.Module):
    """残差池: AttnRes+mHC约束 + attention检索"有用笔记" + 删除非AB残差+压缩 + 每3轮top-k.

    每轮迭代:
      1. 将当前 hidden 与池中残差 (仅AB残差) 堆叠
      2. AttnRes 注意力聚合 + mHC 流形约束聚合
      3. attention 检索: 用 hidden 查询池中残差, 取最相关的"有用笔记"
      4. 删除非AB残差 + 压缩 (线性投影降维)
      5. 每 pool_compress_every 轮: 注意力索引 + top-k 筛选
    """

    def __init__(self, cfg: MathConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim

        # AttnRes + mHC (从 common.layers 导入复用)
        self.attn_res = AttnRes(d, num_blocks=cfg.attn_res_num_blocks, eps=cfg.rms_eps) \
            if cfg.use_attn_res else None
        self.mhc = mHC(d, num_iters=cfg.mhc_num_iters) \
            if cfg.use_mhc else None

        # attention 检索 "有用笔记" (简易分类器: hidden × pool → 相关性分数)
        self.note_query = nn.Linear(d, d, bias=False)
        self.note_key = nn.Linear(d, d, bias=False)
        self.note_scale = d ** -0.5

        # 压缩投影 (删除非AB残差后压缩)
        self.compress_proj = nn.Linear(d, d, bias=False)
        self.compress_norm = RMSNorm(d, eps=cfg.rms_eps)

        # top-k 筛选的可学习门控
        self.topk_gate = nn.Linear(d, 1, bias=False)

        self.topk = cfg.pool_topk
        self.compress_every = cfg.pool_compress_every

    def forward(
        self,
        hidden: torch.Tensor,
        ab_residuals: List[torch.Tensor],
        iteration: int,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """处理残差池.

        Args:
            hidden: [b, s, d] 当前隐藏状态.
            ab_residuals: 池中 AB 残差列表 (每项 [b, s, d]); 非AB残差已被删除.
            iteration: 当前迭代索引.
        Returns:
            (pooled_hidden, new_ab_residuals, aux_loss)
        """
        b, s, d = hidden.shape
        aux_loss = hidden.new_zeros(())

        # --- 1. AttnRes + mHC 聚合残差 ---
        if len(ab_residuals) > 0 and self.attn_res is not None:
            stacked = torch.stack(ab_residuals, dim=-2)          # [b, s, L, d]
            attn_agg = self.attn_res(stacked)                     # [b, s, d]
            hidden = hidden + attn_agg
        if len(ab_residuals) > 0 and self.mhc is not None:
            stacked = torch.stack(ab_residuals, dim=-2)          # [b, s, L, d]
            mhc_agg = self.mhc(stacked)                           # [b, s, L, d]
            # 取最后一层 (最近残差) 融合
            hidden = hidden + mhc_agg[..., -1, :]

        # --- 2. attention 检索 "有用笔记" ---
        if len(ab_residuals) > 0:
            q = self.note_query(hidden)                           # [b, s, d]
            keys = torch.stack([self.note_key(r) for r in ab_residuals], dim=-2)  # [b, s, L, d]
            scores = torch.einsum("bsd,bsld->bsl", q, keys) * self.note_scale  # [b, s, L]
            attn_weights = F.softmax(scores, dim=-1)              # [b, s, L]
            retrieved = torch.einsum("bsl,bsld->bsd", attn_weights, torch.stack(ab_residuals, dim=-2))
            hidden = hidden + retrieved

        # --- 3. 压缩 (投影 + 归一化) ---
        compressed = self.compress_norm(self.compress_proj(hidden))

        # --- 4. 每 compress_every 轮: top-k 筛选 ---
        if (iteration + 1) % self.compress_every == 0 and len(ab_residuals) > self.topk:
            # 按 topk_gate 分数筛选最有用的残差
            gate_scores = torch.stack(
                [self.topk_gate(r).squeeze(-1).mean(dim=-1) for r in ab_residuals],
                dim=-1,
            )                                                      # [b, L]
            k = min(self.topk, len(ab_residuals))
            _, top_idx = gate_scores.topk(k, dim=-1)              # [b, k]
            # 收集每个 batch 的 top-k 残差 (批处理: 用 gather)
            new_residuals: List[torch.Tensor] = []
            stacked_res = torch.stack(ab_residuals, dim=1)        # [b, L, s, d]
            for ki in range(k):
                idx_ki = top_idx[:, ki]                            # [b]
                gathered = stacked_res[torch.arange(b), idx_ki]   # [b, s, d]
                new_residuals.append(gathered)
            ab_residuals = new_residuals

        return hidden, ab_residuals, aux_loss


# =====================================================================
# IntuitionLayer (直觉层, 基础版)
# =====================================================================

class IntuitionLayer(nn.Module):
    """直觉层 (基础版): 轻量MLP快通道 + 隐变量采样.

    待后续完善的完整版将引入更复杂的直觉机制. 当前基础版:
      * 快通道: GatedRMSNorm + 多层 SwiGLU MLP (提供快速直觉响应)
      * 隐变量采样: VAE 风格 (mean + logvar → 重参数化采样), 引入随机直觉
      * 融合: hidden + fast_channel + latent_sample
    """

    def __init__(self, cfg: MathConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim
        h = cfg.intuition_hidden_dim

        # 快通道: GatedRMSNorm + 多层 ExpertFFN (SwiGLU + down-proj)
        self.norm = GatedRMSNorm(d, eps=cfg.rms_eps)
        self.fast_layers = nn.ModuleList([
            ExpertFFN(d, h) for _ in range(cfg.intuition_num_layers)
        ])
        self.fast_proj = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.fast_proj.weight)  # 初始零, 逐步引入快通道

        # 隐变量采样 (VAE 风格)
        self.latent_mean = nn.Linear(d, cfg.intuition_latent_dim)
        self.latent_logvar = nn.Linear(d, cfg.intuition_latent_dim)
        self.latent_decode = nn.Linear(cfg.intuition_latent_dim, d)
        nn.init.zeros_(self.latent_decode.weight)  # 初始零, 逐步引入隐变量

        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (output, kl_loss)."""
        # 快通道
        x = self.norm(hidden)
        for layer in self.fast_layers:
            x = layer(x)
        fast = self.fast_proj(x)

        # 隐变量采样 (重参数化)
        mean = self.latent_mean(hidden)
        logvar = self.latent_logvar(hidden).clamp(-4, 4)
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        z = mean + eps * std
        latent = self.latent_decode(z)

        out = hidden + self.dropout(fast) + self.dropout(latent)

        # KL 散度 (标准正态先验)
        kl = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp()).mean()
        return out, kl


# =====================================================================
# ExpertPool (共享专家池)
# =====================================================================

class EmptyExpert(nn.Module):
    """空专家 (参考 Shannon EmptyExpert 设计): 零初始化门控, 逐步填充.

    专家本体为 ExpertFFN (SwiGLU + down-proj), 但 down 投影零初始化且输出乘以
    零初始化标量门控, 使得初始贡献为 0, 在持续学习阶段逐步吸收新能力.
    """

    def __init__(self, d_model: int, inter_dim: int, eps: float = 1e-6):
        super().__init__()
        self.glu = SwiGLU(d_model, inter_dim)
        self.down = nn.Linear(inter_dim, d_model, bias=False)
        nn.init.zeros_(self.down.weight)  # 零初始化 down 投影
        self.gate = nn.Parameter(torch.zeros(1))  # 零初始化门控
        self.norm = RMSNorm(d_model, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.gate * self.down(self.glu(x)))


class ExpertPool(nn.Module):
    """共享专家池: 6常驻 (4固定+2可学习) + 16大×16小双层MoE.

    组件:
      * 4 固定常驻专家 (SwiGLU, 密集, 不受路由, 始终开启)
      * 2 可学习常驻专家 (EmptyExpert 零初始化, NLM增强, 密集)
      * 16 大专家 (SwiGLU 粗粒度, top-k 路由, 稀疏)
      * 16 小专家 (SwiGLU 细粒度, top-k 路由, 稀疏)
      * CTMRouter (复杂度驱动, 控制 NLM 增强开关)
      * NLMLayer (神经元级模型, 增强 2 可学习专家的激活)
    """

    def __init__(self, cfg: MathConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim
        big_inter = cfg.moe_inter_dim
        small_inter = max(d, int(big_inter * cfg.small_expert_inter_ratio))

        # --- 常驻专家: 4 固定 + 2 可学习 ---
        self.fixed_experts = nn.ModuleList([
            ExpertFFN(d, big_inter) for _ in range(cfg.num_fixed_resident_experts)
        ])
        self.learnable_experts = nn.ModuleList([
            EmptyExpert(d, big_inter, eps=cfg.rms_eps)
            for _ in range(cfg.num_learnable_resident_experts)
        ])
        # NLM 增强 (从 common.ctm 导入复用)
        self.nlm_layers = nn.ModuleList([
            NLMLayer(d, num_neurons=cfg.ctm_num_neurons, d_state=cfg.ctm_d_state,
                     warmup_freeze=cfg.ctm_warmup_freeze)
            for _ in range(cfg.num_learnable_resident_experts)
        ])
        self.resident_norm = RMSNorm(d, eps=cfg.rms_eps)

        # --- 双层 MoE: 16 大 + 16 小 ---
        self.big_experts = nn.ModuleList([
            ExpertFFN(d, big_inter) for _ in range(cfg.num_big_experts)
        ])
        self.small_experts = nn.ModuleList([
            ExpertFFN(d, small_inter) for _ in range(cfg.num_small_experts)
        ])
        self.big_router = nn.Linear(d, cfg.num_big_experts, bias=False)
        self.small_router = nn.Linear(d, cfg.num_small_experts, bias=False)
        self.moe_norm = RMSNorm(d, eps=cfg.rms_eps)

        # CTMRouter (从 common.ctm 导入复用): 复杂度驱动 NLM 增强
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
        """Top-k 路由: 返回 (output, aux_loss, scores)."""
        logits = router(x_flat)                                    # [N, E]
        if self.training and self.router_noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.router_noise_std
        scores = F.softmax(logits, dim=-1)                         # [N, E]
        k = min(top_k, num_experts)
        topk_scores, topk_idx = scores.topk(k, dim=-1)            # [N, k]
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
        d = x_flat.shape[-1]
        out = torch.zeros_like(x_flat)
        for ki in range(k):
            idx_ki = topk_idx[:, ki]                              # [N]
            w_ki = topk_scores[:, ki]                             # [N]
            # 按专家分组处理
            for ei in range(len(experts)):
                mask = idx_ki == ei
                if not mask.any():
                    continue
                x_sel = x_flat[mask]                              # [M, d]
                out_sel = experts[ei](x_sel)                      # [M, d]
                out[mask] += w_ki[mask].unsqueeze(-1) * out_sel
        return out

    def _load_balance_loss(self, scores: torch.Tensor, topk_idx: torch.Tensor, num_experts: int) -> torch.Tensor:
        """标准 MoE 负载均衡损失."""
        N = scores.shape[0]
        # 每个专家收到的 token 比例
        flat_idx = topk_idx.reshape(-1)
        tokens_per_expert = torch.bincount(flat_idx, minlength=num_experts).float()
        frac_tokens = tokens_per_expert / max(N, 1)
        # 每个专家的平均路由概率
        mean_prob = scores.mean(dim=0)
        return num_experts * (frac_tokens * mean_prob).sum()

    def forward(
        self,
        x: torch.Tensor,
        top_k_big_override: Optional[int] = None,
        top_k_small_override: Optional[int] = None,
        use_nlm: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """专家池前向.

        Args:
            x: [b, s, d] 输入.
            top_k_big_override / top_k_small_override: 子agent 可覆盖 top-k (不同路由策略).
            use_nlm: 是否启用 NLM 增强 (G5 子agent).
        Returns:
            (output [b, s, d], aux_loss scalar)
        """
        b, s, d = x.shape
        aux = x.new_zeros(())

        # --- 常驻专家 (密集, 始终开启) ---
        resident = x.new_zeros(b, s, d)
        for exp in self.fixed_experts:
            resident = resident + exp(x)
        # 可学习专家 (EmptyExpert, 零门控逐步填充) + 可选 NLM 增强
        for i, (exp, nlm) in enumerate(zip(self.learnable_experts, self.nlm_layers)):
            exp_out = exp(x)
            if use_nlm:
                # NLM 增强: 对 flatten 的 token 跑一个 tick
                x_flat = x.reshape(b * s, d)
                nlm_out, _ = nlm(x_flat)
                nlm_out = nlm_out.reshape(b, s, d)
                exp_out = exp_out + nlm_out
            resident = resident + exp_out
        resident = resident / max(self.cfg.num_resident_experts, 1)

        # --- 双层 MoE (稀疏, top-k 路由) ---
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
        out = self.resident_norm(resident) + self.moe_norm(moe_out)
        return out, aux


# =====================================================================
# MetaRouter (元路由器, 1对1置换)
# =====================================================================

class MetaRouter(nn.Module):
    """元路由器: 1对1置换路由 ("电线盒", 类似专家路由).

    将 num_paths 条注意力路径 1对1 置换到 num_sub_agents 个子agent.
    使用 Sinkhorn 归一化产生双随机矩阵 (软置换, 可微); 推理时可选匈牙利硬置换.

    每条路径的 pooled 特征 → 投影 → 构建代价矩阵 → Sinkhorn → 置换矩阵.
    """

    def __init__(self, cfg: MathConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim
        n = cfg.num_attention_paths
        # 路径特征投影到 n 维 (构建 n×n 代价矩阵)
        self.path_proj = nn.Linear(d, n, bias=False)
        # 可学习的子agent键 (n × n)
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
        b = path_features[0].shape[0]
        n = len(path_features)
        # 池化每条路径: [b, d]
        pooled = torch.stack([p.mean(dim=1) for p in path_features], dim=1)  # [b, n, d]
        # 投影: [b, n, n]
        proj = self.path_proj(pooled)                                  # [b, n, n]
        # 与子agent键做点积构建代价矩阵: [b, n, n]
        cost = torch.matmul(proj, self.agent_keys.t())                 # [b, n, n]
        # Sinkhorn 归一化 → 双随机矩阵
        perm = _sinkhorn_normalize(cost, self.sinkhorn_iters)         # [b, n, n]
        if self.hard_perm and not self.training:
            perm = _hungarian_hard_perm(perm)
        return perm


# =====================================================================
# SubAgent (子agent, 路由策略)
# =====================================================================

class SubAgent(nn.Module):
    """子agent: 不同的路由策略, 共享 ExpertPool.

    5 个子agent (G1-G5) 各有不同的路由策略:
      * G1: top-1 big + top-1 small (激进/最小路由)
      * G2: top-2 big + top-2 small
      * G3: top-3 big + top-3 small
      * G4: top-4 big + top-4 small (默认/全路由)
      * G5: top-4 big + top-4 small + NLM 增强 (CTM)
    """

    # 路由策略表: (top_k_big, top_k_small, use_nlm)
    STRATEGIES = [
        (1, 1, False),
        (2, 2, False),
        (3, 3, False),
        (4, 4, False),
        (4, 4, True),
    ]

    def __init__(self, cfg: MathConfig, agent_id: int):
        super().__init__()
        self.cfg = cfg
        self.agent_id = agent_id
        self.norm = RMSNorm(cfg.hidden_dim, eps=cfg.rms_eps)
        # 选择策略 (循环使用 STRATEGIES 表)
        strategy_idx = agent_id % len(self.STRATEGIES)
        self.top_k_big, self.top_k_small, self.use_nlm = self.STRATEGIES[strategy_idx]
        # 限制在配置范围内
        self.top_k_big = min(self.top_k_big, cfg.top_k_big)
        self.top_k_small = min(self.top_k_small, cfg.top_k_small)

    def forward(self, x: torch.Tensor, expert_pool: ExpertPool) -> Tuple[torch.Tensor, torch.Tensor]:
        """子agent 处理一条路径.

        Args:
            x: [b, s, d] 输入路径.
            expert_pool: 共享专家池.
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


# =====================================================================
# FivePathAttention (五路注意力)
# =====================================================================

class FivePathAttention(nn.Module):
    """五路注意力 (A1-A5): 复用 Hybrid-M3 的 5 种注意力类型.

    从 8 种 Hybrid-M3 注意力中选择 num_attention_paths 种,
    对同一输入并行计算, 产生 num_attention_paths 条路径输出.
    """

    # 8 种注意力类型的构造器映射
    _ATTENTION_BUILDERS = {
        "mla": MLAAttention,
        "kda": KDAAttention,
        "lightning": LightningAttention,
        "sliding": SlidingWindowAttention,
        "mma": MMAAttention,
        "moh": MoHAttention,
        "gated": GatedAttention,
    }

    def __init__(self, cfg: MathConfig, layer_idx: int = 0):
        super().__init__()
        self.cfg = cfg
        n = cfg.num_attention_paths
        attn_cfg = _build_attention_config(cfg, layer_idx=layer_idx)

        # 选择前 n 种注意力类型
        available = [t for t in cfg.attention_types if t in self._ATTENTION_BUILDERS]
        selected = (available * ((n // len(available)) + 1))[:n] if available else []
        if not selected:
            # 回退到 MLA
            selected = ["mla"] * n

        self.attentions = nn.ModuleList()
        for i, attn_type in enumerate(selected):
            builder = self._ATTENTION_BUILDERS[attn_type]
            # 每路使用独立 layer_idx 以区分
            self.attentions.append(builder(AttentionConfig(
                d_model=attn_cfg.d_model, n_heads=attn_cfg.n_heads,
                n_kv_heads=attn_cfg.n_kv_heads, d_kv=attn_cfg.d_kv,
                d_c=attn_cfg.d_c, max_seq_len=attn_cfg.max_seq_len,
                rope_theta=attn_cfg.rope_theta, dropout=attn_cfg.dropout,
                layer_idx=layer_idx * n + i, rms_eps=attn_cfg.rms_eps,
            )))
        self.norm = RMSNorm(cfg.hidden_dim, eps=cfg.rms_eps)

    def forward(self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """对同一输入计算 num_paths 路注意力输出.

        Args:
            x: [b, s, d].
            position_ids: optional [b, s].
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


# =====================================================================
# ABBlock (单个AB块)
# =====================================================================

class ABBlock(nn.Module):
    """单个 AB 块: 5路注意力 → 元路由器置换 → 子agent处理 → 逆置换 → 输出.

    内部结构:
      1. FivePathAttention: 对输入计算 num_paths 路注意力输出
      2. MetaRouter: 1对1置换 (路径 → 子agent)
      3. SubAgents: 各子agent 用不同路由策略处理 (共享 ExpertPool)
      4. 逆置换: 子agent输出 → 原路径顺序
      5. 聚合: 多路输出融合为单路
    """

    def __init__(self, cfg: MathConfig, layer_idx: int = 0):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim
        n = cfg.num_attention_paths

        self.five_path_attn = FivePathAttention(cfg, layer_idx=layer_idx)
        self.meta_router = MetaRouter(cfg)
        self.sub_agents = nn.ModuleList([
            SubAgent(cfg, agent_id=i) for i in range(cfg.num_sub_agents)
        ])
        # ExpertPool 在所有子agent间共享 (实例化一次)
        self.expert_pool = ExpertPool(cfg)

        # 输入/输出归一化
        self.input_norm = RMSNorm(d, eps=cfg.rms_eps)
        self.output_norm = RMSNorm(d, eps=cfg.rms_eps)
        # 多路聚合: 学习各路径权重
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
            paths: list of [b, s, d], 长度 num_paths (来自上一 AB 的路径输出).
            position_ids: optional.
        Returns:
            (new_paths, aux_loss)
        """
        b, s, d = x.shape
        n = self.cfg.num_attention_paths
        aux = x.new_zeros(())

        # 1. 五路注意力: 对 x 计算注意力, 与输入路径残差相加
        attn_outputs = self.five_path_attn(x, position_ids=position_ids)  # list of [b,s,d]
        # 与传入路径残差融合
        attended: List[torch.Tensor] = []
        for i in range(n):
            attended.append(self.input_norm(paths[i] + attn_outputs[i]))

        # 2. 元路由器: 计算置换矩阵 [b, n, n]
        perm = self.meta_router(attended)

        # 3. 应用置换: attended → permuted (子agent顺序)
        stacked = torch.stack(attended, dim=1)                      # [b, n, s, d]
        permuted = torch.einsum("bpq,bqsd->bpsd", perm, stacked)   # [b, n, s, d]
        permuted_list = [permuted[:, i] for i in range(n)]          # list of [b,s,d]

        # 4. 子agent 处理 (共享 ExpertPool)
        processed: List[torch.Tensor] = []
        for i in range(self.cfg.num_sub_agents):
            out, a = self.sub_agents[i](permuted_list[i], self.expert_pool)
            processed.append(out)
            aux = aux + a

        # 5. 逆置换: processed → 原路径顺序
        processed_stack = torch.stack(processed, dim=1)             # [b, n, s, d]
        inv_perm = perm.transpose(-1, -2)                           # 逆置换 = 转置 (双随机)
        output_paths_stack = torch.einsum("bpq,bqsd->bpsd", inv_perm, processed_stack)
        new_paths = [self.output_norm(output_paths_stack[:, i]) for i in range(n)]

        return new_paths, aux


# =====================================================================
# ABStack (10个AB固定堆叠)
# =====================================================================

class ABStack(nn.Module):
    """AB 堆叠: num_ab_blocks 个 AB 块固定 1对1 堆叠 (编号对编号, 不做路由).

    输入 → 初始化 num_paths 条路径 (从输入复制) → 依次通过各 AB 块
    → 聚合 num_paths 条路径 → 单路输出.
    """

    def __init__(self, cfg: MathConfig, base_layer_idx: int = 0):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim
        n = cfg.num_attention_paths

        self.ab_blocks = nn.ModuleList([
            ABBlock(cfg, layer_idx=base_layer_idx + i)
            for i in range(cfg.num_ab_blocks)
        ])
        # 路径初始化投影 (输入 → 各路径的初始变换)
        self.path_inits = nn.ModuleList([
            nn.Linear(d, d, bias=False) for _ in range(n)
        ])
        # 路径聚合: 学习各路径权重
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
        b, s, d = x.shape
        n = self.cfg.num_attention_paths
        aux = x.new_zeros(())

        # 初始化 n 条路径 (从输入做不同投影)
        paths: List[torch.Tensor] = [self.path_inits[i](x) for i in range(n)]

        # 依次通过各 AB 块 (固定 1对1: 路径 i → AB[j] → 路径 i → AB[j+1])
        for ab_block in self.ab_blocks:
            paths, a = ab_block(x, paths, position_ids=position_ids)
            aux = aux + a

        # 聚合 n 条路径 → 单路输出 (门控加权)
        stacked = torch.stack(paths, dim=-2)                        # [b, s, n, d]
        gate = torch.softmax(self.aggregate_gate(x), dim=-1)        # [b, s, n]
        out = torch.einsum("bsnd,bsn->bsd", stacked, gate)         # [b, s, d]
        out = self.aggregate_norm(out)
        return out, aux


# =====================================================================
# LoopControl (循环控制)
# =====================================================================

class LoopControl(nn.Module):
    """循环控制: AB输出传递 + 残差池管理 + 深度嵌入.

    每轮迭代:
      * 深度嵌入: 根据当前迭代索引注入位置信号
      * AB 输出作为新的 hidden 传递到下一轮
      * 将 AB 输出存入残差池 (作为 AB 残差)
      * ACT 停止概率计算
    """

    def __init__(self, cfg: MathConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim
        max_iter = cfg.dynamic_iterations[1]
        # 深度嵌入 (迭代索引)
        self.depth_embed = nn.Embedding(max_iter + 1, d)
        nn.init.normal_(self.depth_embed.weight, std=0.02)
        # 输出归一化
        self.out_norm = RMSNorm(d, eps=cfg.rms_eps)
        # ACT 停止概率预测头
        self.halt_proj = nn.Linear(d, 1, bias=False)
        nn.init.zeros_(self.halt_proj.weight)  # 初始倾向于不停 (sigmoid(0)=0.5)

    def forward(
        self,
        ab_out: torch.Tensor,
        hidden: torch.Tensor,
        iteration: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """循环控制前向.

        Args:
            ab_out: [b, s, d] AB 堆叠输出.
            hidden: [b, s, d] 当前隐藏状态.
            iteration: 当前迭代索引.
        Returns:
            (new_hidden, halt_prob)
        """
        b, s, d = hidden.shape
        # 深度嵌入
        depth_id = torch.tensor(iteration, device=hidden.device).clamp(
            max=self.cfg.dynamic_iterations[1])
        depth_emb = self.depth_embed(depth_id)                     # [d]
        # 融合: AB输出 + 深度嵌入 + 残差
        new_hidden = self.out_norm(ab_out + depth_emb + hidden)
        # ACT 停止概率
        halt_logit = self.halt_proj(new_hidden).mean()             # scalar
        halt_prob = torch.sigmoid(halt_logit)
        return new_hidden, halt_prob


# =====================================================================
# MathRecurrentBody (循环主体)
# =====================================================================

class MathRecurrentBody(nn.Module):
    """Looped 循环主体: 1-32 次动态迭代, 每轮含 4 部分.

    每轮迭代:
      1. ResidualPool: 残差池 (AttnRes+mHC + attention检索 + 删除压缩 + 每3轮top-k)
      2. IntuitionLayer: 直觉层 (轻量MLP快通道 + 隐变量采样)
      3. ABStack: 10个AB堆叠 (5路注意力 + 元路由器 + 子agent + 共享专家池)
      4. LoopControl: 循环控制 (AB输出传递 + 残差池管理 + 深度嵌入 + ACT停止)

    CTM 集成: MLASync 在迭代间同步潜变量; CTMDynamicLoss 在多 tick 上计算.
    """

    def __init__(self, cfg: MathConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim

        self.residual_pool = ResidualPool(cfg)
        self.intuition_layer = IntuitionLayer(cfg)
        self.ab_stack = ABStack(cfg, base_layer_idx=1)
        self.loop_control = LoopControl(cfg)

        # MLASync (从 common.ctm 导入复用): 迭代间潜变量同步
        self.c_kv_proj = nn.Linear(d, d // 4)  # 投影到 MLA 潜空间
        self.mla_sync = MLASync(d_c=d // 4, num_neurons=cfg.ctm_num_neurons)

        # CTM 动态损失 (从 common.ctm 导入复用)
        self.ctm_loss = CTMDynamicLoss()

        self.min_iter, self.max_iter = cfg.dynamic_iterations

    def forward(
        self,
        hidden: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        decoder: Optional[nn.Module] = None,
    ) -> Dict[str, torch.Tensor]:
        """循环主体前向 (动态迭代).

        Args:
            hidden: [b, s, d] 编码器输出.
            position_ids: optional.
            labels: optional [b, s] 目标 (用于 CTM 动态损失).
            decoder: optional, 用于计算每 tick 的 logits (CTM 损失).
        Returns:
            dict with hidden, aux_loss, kl_loss, ponder_loss, num_iterations, (ctm_loss).
        """
        b, s, d = hidden.shape
        ab_residuals: List[torch.Tensor] = []
        total_aux = hidden.new_zeros(())
        total_kl = hidden.new_zeros(())
        ponder_loss = hidden.new_zeros(())
        accum_halt = 0.0
        logits_per_tick: List[torch.Tensor] = []

        min_iter = max(self.min_iter, 1)
        max_iter = self.max_iter

        for it in range(max_iter):
            # 1. 残差池
            pooled, ab_residuals, pool_aux = self.residual_pool(
                hidden, ab_residuals, it)
            total_aux = total_aux + pool_aux

            # 2. 直觉层
            intuited, kl = self.intuition_layer(pooled)
            total_kl = total_kl + kl

            # 3. AB 堆叠
            ab_out, ab_aux = self.ab_stack(intuited, position_ids=position_ids)
            total_aux = total_aux + ab_aux

            # 4. 循环控制
            new_hidden, halt_prob = self.loop_control(ab_out, hidden, it)

            # CTM: MLASync 同步潜变量 (每轮) — 用 sync_matrix 直接同步 hidden
            c_kv = self.c_kv_proj(new_hidden)                       # [b, s, d_c]
            sync_matrix = self.mla_sync.sync_matrix(c_kv)           # [b, s, s]
            synced = torch.matmul(sync_matrix, new_hidden)          # [b, s, d]
            new_hidden = new_hidden + 0.1 * synced

            # 将 AB 输出存入残差池 (AB 残差)
            ab_residuals.append(ab_out.detach() if not self.training else ab_out)

            # ACT 停止决策
            accum_halt = accum_halt + float(halt_prob.item())
            ponder_loss = ponder_loss + halt_prob * float(it + 1)

            # 收集每 tick 的 logits (用于 CTM 损失, 需 decoder)
            if decoder is not None and labels is not None:
                tick_out = decoder(new_hidden)
                logits_per_tick.append(tick_out.get("logits", tick_out.get("text_logits")))

            hidden = new_hidden

            # 停止条件: 超过最小迭代且累积停止概率超过阈值
            if (it + 1) >= min_iter and accum_halt >= self.cfg.act_halting_threshold:
                break

        out: Dict[str, torch.Tensor] = {
            "hidden": hidden,
            "aux_loss": total_aux,
            "kl_loss": total_kl,
            "ponder_loss": ponder_loss,
            "num_iterations": float(it + 1),
        }

        # CTM 动态损失 (多 tick)
        if labels is not None and decoder is not None and len(logits_per_tick) > 1:
            # 展平 labels 和 logits 用于 CTM 损失
            tick_logits = torch.stack(logits_per_tick, dim=0)       # [T, b, s, V]
            T, B, S, V = tick_logits.shape
            flat_logits = tick_logits.reshape(T, B * S, V)
            flat_labels = labels.reshape(B * S)
            # 只对有效标签 (>=0) 计算
            valid = flat_labels >= 0
            if valid.any():
                ctm_out = self.ctm_loss(
                    flat_logits[:, valid], flat_labels[valid])
                out["ctm_loss"] = ctm_out["loss"]

        return out


# =====================================================================
# MathDecoder (多任务解码器)
# =====================================================================

class MathDecoder(nn.Module):
    """多任务解码器: text / lean4 / sympy / conjecture / proof_step / confidence.

    组件:
      * text_head: 主文本输出 (vocab_size)
      * lean4_head: Lean4 形式化输出 (nsl_vocab_size)
      * sympy_head: SymPy 符号输出 (nsl_vocab_size)
      * conjecture_head: 猜想生成 (vocab_size)
      * proof_step_head: 证明步骤 (vocab_size)
      * confidence_head: 置信度标量 (per token)
      * NSLDecoder: 树结构符号解码 (从 common.nsl 导入复用)
    """

    def __init__(self, cfg: MathConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim

        self.norm = RMSNorm(d, eps=cfg.rms_eps)
        # 多任务输出头
        self.text_head = nn.Linear(d, cfg.vocab_size)
        self.lean4_head = nn.Linear(d, cfg.nsl_vocab_size)
        self.sympy_head = nn.Linear(d, cfg.nsl_vocab_size)
        self.conjecture_head = nn.Linear(d, cfg.vocab_size)
        self.proof_step_head = nn.Linear(d, cfg.vocab_size)
        self.confidence_head = nn.Linear(d, 1)

        # NSLDecoder (从 common.nsl 导入复用): 树结构符号解码
        self.nsl_decoder = NSLDecoder(
            d_model=d,
            num_heads=cfg.nsl_num_heads,
            num_layers=cfg.nsl_num_layers,
            vocab_size=cfg.nsl_vocab_size,
            max_nodes=cfg.nsl_max_nodes,
        )

    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        """多任务解码.

        Args:
            hidden: [b, s, d].
        Returns:
            dict with logits (主文本) 及各任务头输出 + confidence.
        """
        x = self.norm(hidden)
        text_logits = self.text_head(x)
        confidence = self.confidence_head(x).squeeze(-1)           # [b, s]

        return {
            "logits": text_logits,
            "text_logits": text_logits,
            "lean4_logits": self.lean4_head(x),
            "sympy_logits": self.sympy_head(x),
            "conjecture_logits": self.conjecture_head(x),
            "proof_step_logits": self.proof_step_head(x),
            "confidence": confidence,
        }


# =====================================================================
# MathModel (完整模型)
# =====================================================================

class MathModel(nn.Module):
    """MathMaster 完整模型: 编码器 → 循环主体 → 解码器.

    forward 返回 dict, 包含:
      * logits: 主文本 logits [b, s, vocab]
      * text_logits / lean4_logits / sympy_logits / conjecture_logits / proof_step_logits
      * confidence: [b, s]
      * aux_loss: 标量 (MoE 负载均衡 + 残差池)
      * kl_loss: 直觉层 KL 散度
      * ponder_loss: ACT 停止正则
      * num_iterations: 实际迭代次数
      * (可选) ctm_loss / infonce_loss / symbol_embedding
    """

    def __init__(self, cfg: MathConfig):
        super().__init__()
        self.cfg = cfg
        # 1对1置换要求 num_attention_paths == num_sub_agents
        assert cfg.num_attention_paths == cfg.num_sub_agents, (
            f"MetaRouter 1对1置换要求 num_attention_paths({cfg.num_attention_paths}) "
            f"== num_sub_agents({cfg.num_sub_agents})"
        )
        self.encoder = MathEncoder(cfg)
        self.recurrent_body = MathRecurrentBody(cfg)
        self.decoder = MathDecoder(cfg)

        # 输入位置 ids 缓冲
        self.register_buffer(
            "_pos_buf", torch.arange(cfg.max_seq_len), persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        asts: Optional[List] = None,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """完整前向.

        Args:
            input_ids: [b, s] 输入 token ids.
            asts: optional NSL AST 列表 (符号-神经对齐).
            labels: optional [b, s] 目标 (启用 CTM 动态损失).
            position_ids: optional [b, s].
        Returns:
            输出字典 (见类文档).
        """
        b, s = input_ids.shape

        # 位置 ids
        if position_ids is None:
            position_ids = self._pos_buf[:s].unsqueeze(0).expand(b, -1).contiguous()

        # 1. 编码器
        enc_out = self.encoder(input_ids, asts=asts)
        hidden = enc_out["hidden"]

        # 2. 循环主体 (动态迭代)
        body_out = self.recurrent_body(
            hidden, position_ids=position_ids,
            labels=labels, decoder=self.decoder if labels is not None else None,
        )
        hidden = body_out["hidden"]

        # 3. 解码器
        dec_out = self.decoder(hidden)

        # 4. 汇总输出
        out: Dict[str, torch.Tensor] = {
            **dec_out,
            "aux_loss": body_out["aux_loss"]
                       + self.cfg.moe_aux_loss_weight * body_out["aux_loss"],
            "kl_loss": body_out["kl_loss"],
            "ponder_loss": body_out["ponder_loss"],
            "num_iterations": body_out["num_iterations"],
        }
        # 合并 aux_loss (总辅助损失)
        total_aux = (
            body_out["aux_loss"]
            + self.cfg.kl_loss_weight * body_out["kl_loss"]
            + self.cfg.ponder_loss_weight * body_out["ponder_loss"]
        )
        if "ctm_loss" in body_out:
            out["ctm_loss"] = body_out["ctm_loss"]
            total_aux = total_aux + body_out["ctm_loss"]
        if "infonce_loss" in enc_out:
            out["infonce_loss"] = enc_out["infonce_loss"]
            out["symbol_embedding"] = enc_out["symbol_embedding"]
            total_aux = total_aux + enc_out["infonce_loss"]
        out["aux_loss"] = total_aux

        return out
