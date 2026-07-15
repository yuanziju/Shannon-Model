"""BaseConfig — Shannon 与 MathMaster 共享的配置基类.

定义 150B MoE 目标规格下的共享字段 (模型维度 / 常驻专家 / 双层 MoE /
残差池 / CTM / AB 堆叠), 并提供 ``from_shannon`` / ``from_math`` 工厂方法
从各自的专有配置中提取共享字段, 供 common_base 组件统一消费.

参考: AGENTS.md 多 Agent 协作 (Shannon 融合决策), spec §14 设计决策索引.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Tuple


@dataclass
class ResidentExpertConfig:
    """常驻专家子配置 (6 常驻: 4 固定 + 2 可学习).

    参考 MathMaster 多 MoE 结构与 Shannon EmptyExpert 零初始化设计.
    """

    num_resident_experts: int = 6
    num_fixed_resident_experts: int = 4
    num_learnable_resident_experts: int = 2
    # 固定常驻专家始终开启, 不受路由影响
    fixed_always_active: bool = True
    # 可学习常驻专家零初始化 (EmptyExpert 风格), 逐步填充
    learnable_zero_init: bool = True
    # 可学习专家可选 NLM 增强 (CTM 决策 C10: 仅实体专家使用 NLM)
    learnable_nlm_enhanced: bool = True
    # 常驻专家 FFN 维度 (默认与大专家一致)
    resident_ffn_dim: int = 0  # 0 -> __post_init__ 设为 expert_ffn_dim

    def __post_init__(self):
        if self.resident_ffn_dim <= 0:
            self.resident_ffn_dim = 2048
        assert (
            self.num_fixed_resident_experts + self.num_learnable_resident_experts
            == self.num_resident_experts
        ), (
            f"常驻专家数不匹配: 固定{self.num_fixed_resident_experts} + "
            f"可学习{self.num_learnable_resident_experts} != "
            f"总数{self.num_resident_experts}"
        )


@dataclass
class BaseConfig:
    """共享配置基类 — Shannon 150B MoE 与 MathMaster 共用的底层字段.

    目标规格: 150B 总参数 (MoE), 激活参数约 20-40B.
    所有 common_base 组件 (ExpertPool / ResidualPool / ABStack) 均消费此配置.
    """

    # ---- 模型维度 (150B 目标) ----
    hidden_dim: int = 8192
    num_layers: int = 48
    num_heads: int = 64
    head_dim: int = 128
    num_kv_heads: int = 16
    vocab_size: int = 128000
    max_seq_len: int = 524288  # 512K (从 5M 降低到更现实)
    rms_eps: float = 1e-6
    dropout: float = 0.0
    bias: bool = False

    # ---- 循环深度 ----
    dynamic_iterations: Tuple[int, int] = (1, 32)

    # ---- 常驻专家 (6: 4 固定 + 2 可学习) ----
    num_resident_experts: int = 6
    num_fixed_resident_experts: int = 4
    num_learnable_resident_experts: int = 2

    # ---- 双层 MoE (16 大 x 16 小) ----
    num_big_experts: int = 16
    num_small_experts: int = 16
    top_k_big: int = 4
    top_k_small: int = 4
    expert_ffn_dim: int = 2048
    small_expert_ffn_dim: int = 1024
    moe_inter_dim: int = 0  # 0 -> __post_init__ 设为 expert_ffn_dim
    small_expert_inter_ratio: float = 0.5
    router_noise_std: float = 1.0
    load_balance_alpha: float = 0.01

    # ---- 残差池 ----
    pool_topk: int = 32
    pool_compress_every: int = 3
    use_attn_res: bool = True
    use_mhc: bool = True
    attn_res_num_blocks: int = 8
    mhc_num_iters: int = 20

    # ---- CTM / NLM ----
    ctm_enabled: bool = True
    nlm_num_neurons: int = 8
    nlm_d_state: int = 16
    nlm_warmup_freeze: bool = True
    ctm_complexity_threshold: float = 0.7

    # ---- AB 堆叠 ----
    num_ab_blocks: int = 10
    num_attention_paths: int = 5
    num_sub_agents: int = 5
    ab_simplified: bool = False  # Shannon 用简化版, MathMaster 用完整版
    rope_theta: float = 10000.0
    attention_dropout: float = 0.0
    attention_types: Tuple[str, ...] = (
        "mla", "kda", "lightning", "sliding",
        "mma", "moh", "gated", "controller",
    )

    # ---- MetaRouter ----
    meta_router_sinkhorn_iters: int = 10
    meta_router_hard_perm: bool = False

    # ---- 常驻专家子配置 ----
    resident_expert_config: ResidentExpertConfig = field(
        default_factory=ResidentExpertConfig
    )

    # ------------------------------------------------------------------
    def __post_init__(self):
        """派生默认值与一致性校验."""
        if self.moe_inter_dim <= 0:
            self.moe_inter_dim = self.expert_ffn_dim
        # 常驻专家一致性
        if (
            self.num_fixed_resident_experts + self.num_learnable_resident_experts
            != self.num_resident_experts
        ):
            self.num_fixed_resident_experts = (
                self.num_resident_experts - self.num_learnable_resident_experts
            )
        # 同步常驻专家子配置
        rc = self.resident_expert_config
        rc.num_resident_experts = self.num_resident_experts
        rc.num_fixed_resident_experts = self.num_fixed_resident_experts
        rc.num_learnable_resident_experts = self.num_learnable_resident_experts
        if rc.resident_ffn_dim <= 0:
            rc.resident_ffn_dim = self.expert_ffn_dim

    # ------------------------------------------------------------------
    @classmethod
    def from_shannon(cls, config: Any) -> "BaseConfig":
        """从 ShannonConfig 提取共享字段.

        ShannonConfig 拥有顶层快捷字段与子配置对象 (moe / recurrent / ctm / attention).
        此方法将相关字段映射到 BaseConfig.
        """
        moe = getattr(config, "moe", None)
        rec = getattr(config, "recurrent", None)
        ctm = getattr(config, "ctm", None)
        attn = getattr(config, "attention", None)
        pe = getattr(config, "positional_encoding", None)

        rc = ResidentExpertConfig(
            num_resident_experts=getattr(config, "num_resident_experts", 6),
            num_fixed_resident_experts=getattr(config, "num_fixed_resident_experts", 4),
            num_learnable_resident_experts=getattr(config, "num_learnable_resident_experts", 2),
            learnable_nlm_enhanced=getattr(config, "ctm_enabled", True),
            resident_ffn_dim=getattr(config, "expert_ffn_dim", 2048),
        )

        return cls(
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            num_kv_heads=config.num_kv_heads,
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            rms_eps=config.rms_eps,
            dropout=config.dropout,
            bias=config.bias,
            dynamic_iterations=config.dynamic_iterations,
            num_resident_experts=getattr(config, "num_resident_experts", 6),
            num_fixed_resident_experts=getattr(config, "num_fixed_resident_experts", 4),
            num_learnable_resident_experts=getattr(config, "num_learnable_resident_experts", 2),
            num_big_experts=config.num_big_experts,
            num_small_experts=config.num_small_experts,
            top_k_big=config.top_k_big,
            top_k_small=config.top_k_small,
            expert_ffn_dim=config.expert_ffn_dim,
            small_expert_ffn_dim=config.small_expert_ffn_dim,
            moe_inter_dim=config.expert_ffn_dim,
            small_expert_inter_ratio=(
                config.small_expert_ffn_dim / max(config.expert_ffn_dim, 1)
            ),
            router_noise_std=getattr(moe, "router_noise_std", 1.0),
            load_balance_alpha=getattr(moe, "load_balance_alpha", 0.01),
            use_attn_res=getattr(rec, "use_attn_res", True),
            use_mhc=getattr(rec, "use_mhc", True),
            attn_res_num_blocks=getattr(rec, "attn_res_num_blocks", 8),
            mhc_num_iters=getattr(rec, "mhc_num_iters", 20),
            ctm_enabled=config.ctm_enabled,
            nlm_num_neurons=config.nlm_num_neurons,
            nlm_d_state=config.nlm_d_state,
            nlm_warmup_freeze=getattr(ctm, "nlm_warmup_freeze", True),
            ctm_complexity_threshold=getattr(ctm, "ctm_complexity_threshold", 0.7),
            ab_simplified=True,  # Shannon 用简化版 ABStack
            rope_theta=config.rope_theta,
            attention_dropout=getattr(attn, "attn_dropout", 0.0),
            resident_expert_config=rc,
        )

    # ------------------------------------------------------------------
    @classmethod
    def from_math(cls, config: Any) -> "BaseConfig":
        """从 MathConfig 提取共享字段.

        MathConfig 是扁平 dataclass, 字段直接映射.
        """
        small_inter = max(
            config.hidden_dim,
            int(config.moe_inter_dim * config.small_expert_inter_ratio),
        )
        rc = ResidentExpertConfig(
            num_resident_experts=config.num_resident_experts,
            num_fixed_resident_experts=config.num_fixed_resident_experts,
            num_learnable_resident_experts=config.num_learnable_resident_experts,
            learnable_nlm_enhanced=True,
            resident_ffn_dim=config.moe_inter_dim,
        )

        return cls(
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            num_kv_heads=config.num_kv_heads,
            vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
            rms_eps=config.rms_eps,
            dropout=config.dropout,
            bias=getattr(config, "bias", False),
            dynamic_iterations=config.dynamic_iterations,
            num_resident_experts=config.num_resident_experts,
            num_fixed_resident_experts=config.num_fixed_resident_experts,
            num_learnable_resident_experts=config.num_learnable_resident_experts,
            num_big_experts=config.num_big_experts,
            num_small_experts=config.num_small_experts,
            top_k_big=config.top_k_big,
            top_k_small=config.top_k_small,
            expert_ffn_dim=config.moe_inter_dim,
            small_expert_ffn_dim=small_inter,
            moe_inter_dim=config.moe_inter_dim,
            small_expert_inter_ratio=config.small_expert_inter_ratio,
            router_noise_std=config.router_noise_std,
            load_balance_alpha=config.moe_aux_loss_weight,
            pool_topk=config.pool_topk,
            pool_compress_every=config.pool_compress_every,
            use_attn_res=config.use_attn_res,
            use_mhc=config.use_mhc,
            attn_res_num_blocks=config.attn_res_num_blocks,
            mhc_num_iters=config.mhc_num_iters,
            ctm_enabled=True,
            nlm_num_neurons=config.ctm_num_neurons,
            nlm_d_state=config.ctm_d_state,
            nlm_warmup_freeze=config.ctm_warmup_freeze,
            ctm_complexity_threshold=config.ctm_complexity_threshold,
            num_ab_blocks=config.num_ab_blocks,
            num_attention_paths=config.num_attention_paths,
            num_sub_agents=config.num_sub_agents,
            ab_simplified=False,  # MathMaster 用完整版 ABStack
            rope_theta=config.rope_theta,
            attention_dropout=config.attention_dropout,
            attention_types=config.attention_types,
            meta_router_sinkhorn_iters=config.meta_router_sinkhorn_iters,
            meta_router_hard_perm=config.meta_router_hard_perm,
            resident_expert_config=rc,
        )

    # ------------------------------------------------------------------
    def build_attention_config(self, layer_idx: int = 0):
        """构建 common.attention.AttentionConfig (供 ABStack 使用)."""
        from common.attention import AttentionConfig

        return AttentionConfig(
            d_model=self.hidden_dim,
            n_heads=self.num_heads,
            n_kv_heads=self.num_kv_heads,
            d_kv=self.head_dim,
            d_c=max(self.hidden_dim // 4, 8),
            max_seq_len=self.max_seq_len,
            rope_theta=self.rope_theta,
            dropout=self.attention_dropout,
            layer_idx=layer_idx,
            bias=self.bias,
            rms_eps=self.rms_eps,
        )

    def extra_repr(self) -> str:
        return (
            f"hidden={self.hidden_dim}, layers={self.num_layers}, "
            f"heads={self.num_heads}x{self.head_dim}, "
            f"kv_heads={self.num_kv_heads}, "
            f"resident={self.num_resident_experts}"
            f"({self.num_fixed_resident_experts}+{self.num_learnable_resident_experts}), "
            f"moe={self.num_big_experts}x{self.num_small_experts}"
        )
