"""MathMaster 模型配置 (MathConfig).

定义 MathMaster 30-70B MoE 数学专精模型的全部超参数. 基于"新底子"架构:
输入 → 神经语编码(NSL) → Looped 循环主体(1-32次动态迭代, 每轮含4部分:
残差池 / 直觉层 / AB堆叠 / 循环控制) → 多任务解码器.

所有字段均有合理默认值 (目标 70B 配置), 可通过构造参数覆盖以支持小规模测试.
``__post_init__`` 执行一致性校验; ``to_dict`` / ``from_dict`` 支持序列化.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Tuple


@dataclass
class MathConfig:
    """MathMaster 模型配置.

    属性分组:
      * 模型主体: vocab_size / hidden_dim / num_layers / num_heads / ...
      * 循环深度: dynamic_iterations / silent_thinking
      * AB堆叠: num_ab_blocks / num_attention_paths / num_sub_agents
      * 专家池: num_resident_experts / num_big_experts / num_small_experts / top_k_*
      * 残差池: pool_topk / pool_compress_every / use_attn_res / use_mhc
      * 直觉层: intuition_hidden_dim / intuition_num_layers
      * 注意力: attention_types (Hybrid-M3 8种)
      * NSL / CTM / 位置编码 / 形式化 / 训练 / 评估 / 领域权重
    """

    # ------------------------------------------------------------------
    # 模型主体
    # ------------------------------------------------------------------
    model_name: str = "MathMaster-70B"
    vocab_size: int = 128000
    hidden_dim: int = 8192              # 70B 参考
    num_layers: int = 40
    num_heads: int = 64
    head_dim: int = 128
    num_kv_heads: int = 16              # GQA
    max_seq_len: int = 1_000_000        # 1M-10M 上下文

    # ------------------------------------------------------------------
    # 循环深度 (Looped 循环主体)
    # ------------------------------------------------------------------
    dynamic_iterations: Tuple[int, int] = (1, 32)
    silent_thinking: bool = True
    act_halting_threshold: float = 0.95  # ACT 自适应停止阈值

    # ------------------------------------------------------------------
    # AB堆叠 (10个AB固定堆叠)
    # ------------------------------------------------------------------
    num_ab_blocks: int = 10
    num_attention_paths: int = 5         # A1-A5 五路注意力
    num_sub_agents: int = 5              # G1-G5 五个子agent

    # ------------------------------------------------------------------
    # 专家池 (共享)
    # ------------------------------------------------------------------
    num_resident_experts: int = 6        # 常驻专家总数
    num_fixed_resident_experts: int = 4  # 固定 (不可学习初始化, 正常训练)
    num_learnable_resident_experts: int = 2  # 可学习 (EmptyExpert 零初始化)
    num_big_experts: int = 16            # 大专家 (粗粒度)
    num_small_experts: int = 16          # 小专家 (细粒度)
    top_k_big: int = 4
    top_k_small: int = 4
    moe_inter_dim: int = 0              # 0 -> __post_init__ 设为 hidden_dim * 4
    small_expert_inter_ratio: float = 0.25  # 小专家 inter_dim = moe_inter_dim * ratio
    expert_capacity_factor: float = 1.25
    router_noise_std: float = 1.0
    moe_aux_loss_weight: float = 0.01

    # ------------------------------------------------------------------
    # 残差池 (ResidualPool)
    # ------------------------------------------------------------------
    pool_topk: int = 32
    pool_compress_every: int = 3         # 每3轮注意力索引 + top-k
    use_attn_res: bool = True
    use_mhc: bool = True
    attn_res_num_blocks: int = 8
    mhc_num_iters: int = 20

    # ------------------------------------------------------------------
    # 直觉层 (IntuitionLayer, 基础版)
    # ------------------------------------------------------------------
    intuition_hidden_dim: int = 0       # 0 -> __post_init__ 设为 hidden_dim * 4
    intuition_num_layers: int = 2
    intuition_latent_dim: int = 0       # 0 -> __post_init__ 设为 hidden_dim

    # ------------------------------------------------------------------
    # 注意力 (Hybrid-M3 8种)
    # ------------------------------------------------------------------
    attention_types: Tuple[str, ...] = (
        "mla", "kda", "lightning", "sliding",
        "mma", "moh", "gated", "controller",
    )
    rope_theta: float = 10000.0
    attention_dropout: float = 0.0

    # ------------------------------------------------------------------
    # NSL (神经语系统)
    # ------------------------------------------------------------------
    nsl_vocab_size: int = 8192
    nsl_num_heads: int = 4
    nsl_num_layers: int = 2
    nsl_max_nodes: int = 256
    nsl_temperature: float = 0.07

    # ------------------------------------------------------------------
    # CTM (Continuous Thought Machine)
    # ------------------------------------------------------------------
    ctm_num_neurons: int = 8
    ctm_d_state: int = 16
    ctm_warmup_freeze: bool = True
    ctm_complexity_threshold: float = 0.7

    # ------------------------------------------------------------------
    # 位置编码 (1M-10M)
    # ------------------------------------------------------------------
    pos_encoding: str = "longrope2"     # longrope2 / yarn / rope
    pos_original_max: int = 8192

    # ------------------------------------------------------------------
    # 形式化
    # ------------------------------------------------------------------
    formal_backend: str = "lean4"       # lean4 / sympy

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------
    dropout: float = 0.0
    rms_eps: float = 1e-6
    aux_loss_weight: float = 0.01
    kl_loss_weight: float = 0.001       # 直觉层 KL 损失权重
    ponder_loss_weight: float = 0.01    # ACT ponder 损失权重

    # ------------------------------------------------------------------
    # 评估
    # ------------------------------------------------------------------
    eval_confidence_threshold: float = 0.5
    eval_benchmarks: Tuple[str, ...] = (
        "MMLU-STEM", "GSM8K", "MATH", "HumanEval", "Lean4-Proof",
    )

    # ------------------------------------------------------------------
    # 输出头 (多任务)
    # ------------------------------------------------------------------
    output_tasks: Tuple[str, ...] = (
        "text", "lean4", "sympy", "conjecture", "proof_step", "confidence",
    )

    # ------------------------------------------------------------------
    # 领域权重 (数学子领域)
    # ------------------------------------------------------------------
    domain_weights: Dict[str, float] = field(default_factory=lambda: {
        "algebra": 1.0,
        "analysis": 1.0,
        "geometry": 1.0,
        "number_theory": 1.0,
        "topology": 1.0,
        "logic": 1.0,
        "probability": 1.0,
        "combinatorics": 1.0,
    })

    # ------------------------------------------------------------------
    # 元路由器 (MetaRouter, 1对1置换)
    # ------------------------------------------------------------------
    meta_router_sinkhorn_iters: int = 10
    meta_router_hard_perm: bool = False  # 推理时是否用匈牙利硬置换

    # ------------------------------------------------------------------
    # 校验与后处理
    # ------------------------------------------------------------------
    def __post_init__(self):
        # --- 派生默认值 ---
        if self.moe_inter_dim <= 0:
            self.moe_inter_dim = self.hidden_dim * 4
        if self.intuition_hidden_dim <= 0:
            self.intuition_hidden_dim = self.hidden_dim * 4
        if self.intuition_latent_dim <= 0:
            self.intuition_latent_dim = self.hidden_dim

        # --- 一致性校验 ---
        assert self.hidden_dim % self.num_heads == 0, (
            f"hidden_dim={self.hidden_dim} 必须能被 num_heads={self.num_heads} 整除"
        )
        assert self.num_heads % self.num_kv_heads == 0, (
            f"num_heads={self.num_heads} 必须能被 num_kv_heads={self.num_kv_heads} 整除"
        )
        assert self.head_dim == self.hidden_dim // self.num_heads, (
            f"head_dim={self.head_dim} 与 hidden_dim/num_heads="
            f"{self.hidden_dim // self.num_heads} 不一致"
        )
        assert self.dynamic_iterations[0] >= 1, (
            f"dynamic_iterations 最小值必须 >= 1, 得到 {self.dynamic_iterations[0]}"
        )
        assert self.dynamic_iterations[0] <= self.dynamic_iterations[1], (
            f"dynamic_iterations 区间非法: {self.dynamic_iterations}"
        )
        assert self.num_ab_blocks >= 1, "num_ab_blocks 必须 >= 1"
        assert self.num_attention_paths >= 1, "num_attention_paths 必须 >= 1"
        assert self.num_sub_agents >= 1, "num_sub_agents 必须 >= 1"
        assert (
            self.num_fixed_resident_experts + self.num_learnable_resident_experts
            == self.num_resident_experts
        ), (
            f"常驻专家数不匹配: 固定{self.num_fixed_resident_experts} + "
            f"可学习{self.num_learnable_resident_experts} != "
            f"总数{self.num_resident_experts}"
        )
        assert 1 <= self.top_k_big <= self.num_big_experts, (
            f"top_k_big={self.top_k_big} 超出 [1, {self.num_big_experts}]"
        )
        assert 1 <= self.top_k_small <= self.num_small_experts, (
            f"top_k_small={self.top_k_small} 超出 [1, {self.num_small_experts}]"
        )
        assert self.pool_compress_every >= 1, "pool_compress_every 必须 >= 1"
        assert 0 < self.act_halting_threshold <= 1.0, (
            "act_halting_threshold 必须在 (0, 1]"
        )
        assert self.pos_encoding in ("longrope2", "yarn", "rope"), (
            f"未知 pos_encoding={self.pos_encoding}"
        )
        assert self.formal_backend in ("lean4", "sympy"), (
            f"未知 formal_backend={self.formal_backend}"
        )

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化的字典 (嵌套 dataclass / tuple 均展开)."""
        d = asdict(self)
        # asdict 已将 tuple 转为 list, dict 保持; 恢复 tuple 语义不必要
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MathConfig":
        """从字典构造配置, 忽略未知字段以兼容旧版本."""
        import inspect
        valid = set(inspect.signature(cls).parameters.keys())
        filtered = {k: v for k, v in d.items() if k in valid}
        # list -> tuple (针对 tuple 类型字段)
        if "dynamic_iterations" in filtered and isinstance(
            filtered["dynamic_iterations"], list
        ):
            filtered["dynamic_iterations"] = tuple(filtered["dynamic_iterations"])
        if "attention_types" in filtered and isinstance(
            filtered["attention_types"], list
        ):
            filtered["attention_types"] = tuple(filtered["attention_types"])
        if "output_tasks" in filtered and isinstance(
            filtered["output_tasks"], list
        ):
            filtered["output_tasks"] = tuple(filtered["output_tasks"])
        if "eval_benchmarks" in filtered and isinstance(
            filtered["eval_benchmarks"], list
        ):
            filtered["eval_benchmarks"] = tuple(filtered["eval_benchmarks"])
        return cls(**filtered)
