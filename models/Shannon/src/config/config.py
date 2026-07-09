"""ShannonConfig — Shannon 15B MoE 模型统一配置.

定义编码器(3%)-循环主体(94%)-解码器(3%)三层架构的全部超参数,
包括循环深度、双层MoE、Hybrid-M3注意力、NSL、CTM、位置编码、形式化验证、
训练与评估配置.

参考: AGENTS.md 项目结构全景, spec.md §14 设计决策索引.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Tuple


# ---------------------------------------------------------------------------
# 子配置 dataclass
# ---------------------------------------------------------------------------

@dataclass
class PositionalEncodingConfig:
    """多模态位置编码配置 (RoPE/YaRN/RoPE-2D/1D+时序衰减/3D+LongRoPE2)."""

    rope_theta: float = 10000.0
    rope_base_scale: float = 1.0
    yarn_original_max: int = 8192
    yarn_beta_fast: int = 32
    yarn_beta_slow: int = 1
    yarn_attn_factor: float = 1.0
    longrope_original_max: int = 8192
    longrope_learnable: bool = True
    temporal_decay_init: float = 0.01
    sliding_window: int = 512


@dataclass
class AttentionConfig:
    """Hybrid-M3 8种注意力配置 (MLA/KDA/Lightning/Sliding/MMA/MoH/Gated/Dynamic)."""

    moh_top_k: int = 4
    moh_n_shared: int = 2
    gated_rank: int = 64
    lightning_block_size: int = 64
    kda_chunk_size: int = 64
    mma_n_modalities: int = 2
    rms_eps: float = 1e-6
    attn_dropout: float = 0.0
    use_qk_norm: bool = True
    # 4层周期: 4k+1=KDA, 4k+2=KDA+MoH, 4k+3=KDA, 4k+4=MLA+QKNorm+MMA
    cycle_period: int = 4


@dataclass
class MoEConfig:
    """双层MoE配置 (16大专家×16小专家, Top-4×Top-4, 空专家)."""

    num_big_experts: int = 16
    num_small_experts: int = 16
    top_k_big: int = 4
    top_k_small: int = 4
    expert_ffn_dim: int = 1024
    small_expert_ffn_dim: int = 512
    num_shared_experts: int = 2
    num_empty_experts: int = 4
    # 分层: 浅8/中16/深24
    shallow_layers: int = 8
    middle_layers: int = 16
    deep_layers: int = 24
    # 路由
    router_noise_std: float = 1.0
    load_balance_alpha: float = 0.01
    # 空专家
    empty_expert_zero_init: bool = True
    empty_expert_absorb_threshold: float = 0.1
    # NLM增强 (CTM决策C10: 仅实体专家使用NLM)
    nlm_enhanced: bool = True
    nlm_num_neurons: int = 8
    nlm_d_state: int = 16
    nlm_warmup_freeze: bool = True


@dataclass
class RecurrentConfig:
    """循环主体配置 (RDT循环块, 1-32次动态迭代)."""

    dynamic_iterations: Tuple[int, int] = (1, 32)
    silent_thinking: bool = True
    # ACT 自适应停止
    act_enabled: bool = True
    act_threshold: float = 0.99
    act_penalty_weight: float = 0.01
    # LTI 稳定性 (谱半径<1)
    lti_enabled: bool = True
    lti_spectral_radius: float = 0.99
    # 深度嵌入
    depth_embed_dim: int = 64
    # 深度LoRA
    depth_lora_rank: int = 32
    depth_lora_alpha: float = 32.0
    depth_lora_dropout: float = 0.0
    # 残差
    use_attn_res: bool = True
    attn_res_num_blocks: int = 8
    use_mhc: bool = True
    mhc_num_iters: int = 20
    # 梯度检查点
    use_gradient_checkpoint: bool = True


@dataclass
class NSLConfig:
    """神经语系统配置 (符号-神经双向翻译, 形式化解析)."""

    enabled: bool = True
    nsl_vocab_size: int = 1024
    nsl_num_heads: int = 4
    nsl_num_layers: int = 2
    nsl_max_nodes: int = 128
    nsl_max_depth: int = 16
    nsl_max_children: int = 8
    nsl_temperature: float = 0.07
    # 形式化解析
    formal_parser_enabled: bool = True
    lean_interop: bool = True
    sympy_interop: bool = True


@dataclass
class CTMConfig:
    """CTM集成配置 (NLM神经元级模型 + MLA同步 + 动态损失)."""

    enabled: bool = True
    # NLM
    nlm_num_neurons: int = 8
    nlm_d_state: int = 16
    nlm_warmup_freeze: bool = True
    # MLA同步矩阵 (c_kv . c_kv^T)
    mla_sync_dropout: float = 0.0
    # CTM动态损失
    ctm_lambda_certainty: float = 0.5
    ctm_lambda_tick: float = 0.1
    ctm_lambda_monotone: float = 0.1
    # 路由器
    ctm_complexity_threshold: float = 0.7
    ctm_router_top_k: int = 2
    ctm_router_noise_std: float = 1.0


@dataclass
class LatentDecodeConfig:
    """隐空间解码配置 (B+C融合: 层次化NAR + 掩码精化 + 流匹配 + AR保底)."""

    # 方案B: 层次化NAR
    nar_hidden_dim: int = 1024
    nar_num_heads: int = 16
    nar_num_layers_per_level: int = 2
    nar_max_paragraphs: int = 64
    nar_max_sentences: int = 16
    nar_max_tokens: int = 64
    nar_mask_token_id: int = 4
    # 方案C: 掩码精化 (复用RDT)
    mask_refine_max_iters: int = 8
    mask_refine_confidence: float = 0.9
    mask_refine_schedule: str = "cosine"
    # 方案A: 流匹配 (可选)
    flow_enabled: bool = True
    flow_latent_dim: int = 1024
    flow_num_heads: int = 16
    flow_num_layers: int = 4
    flow_num_euler_steps: int = 50
    flow_solver: str = "euler"
    # AR保底
    ar_token_threshold: float = 0.55
    ar_block_threshold: float = 0.70
    ar_global_threshold: float = 0.75
    ar_max_new_tokens: int = 2048
    ar_force_proof: bool = True
    # 模式切换 (reasoning/decoding LoRA)
    mode_lora_rank: int = 32
    mode_lora_alpha: float = 32.0
    mode_gating: str = "soft"
    # 拟人流式
    human_stream_enabled: bool = True
    human_revision_cap: float = 0.15
    # 神经码本
    codebook_size: int = 8192
    codebook_ema_decay: float = 0.99
    # Lean验证
    lean_enabled: bool = True
    lean_timeout_sec: float = 30.0


@dataclass
class EncoderConfig:
    """编码器配置 (文本嵌入 + 视觉双通道 + 视频 + 文档 + SVG)."""

    num_encoder_layers: int = 1
    # 文本
    num_special_tokens: int = 9
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    mask_token_id: int = 4
    # 视觉
    vit_patch_size: int = 16
    vit_num_layers: int = 12
    vit_num_heads: int = 12
    qformer_num_queries: int = 32
    qformer_num_layers: int = 4
    vae_latent_dim: int = 256
    vae_downsample: int = 8
    # 视频
    video_fps: int = 4
    video_max_frames: int = 32
    video_ssm_state_dim: int = 64
    video_memory_tokens: int = 16
    # 文档解析
    doc_max_pages: int = 128
    doc_pdf_dpi: int = 150
    # SVG
    svg_max_paths: int = 256
    svg_coord_precision: int = 4


@dataclass
class DecoderOutputConfig:
    """解码器多任务输出头配置 (文本/SVG/工具/TTS)."""

    num_output_heads: int = 4
    # 文本
    text_head_tied: bool = True
    # SVG
    svg_vocab_size: int = 8192
    svg_max_paths: int = 256
    # 工具
    tool_vocab_size: int = 512
    # TTS
    tts_sample_rate: int = 24000
    tts_hop_length: int = 256
    # 图像编辑
    image_edit_enabled: bool = True
    image_edit_latent_dim: int = 256


@dataclass
class TrainingConfig:
    """训练引擎配置 (6阶段, 5D并行, 多优化器)."""

    # 6阶段
    phase1_pretrain_steps: int = 100000
    phase2_intermediate_steps: int = 50000
    phase3_sft_steps: int = 20000
    phase4_alignment_steps: int = 10000
    phase5_continual_steps: int = 50000
    phase6_evolution_steps: int = 50000
    # 5D并行
    tp_size: int = 4
    pp_size: int = 4
    dp_size: int = 2
    sp_size: int = 1
    ep_size: int = 4
    # 优化器
    optimizer: str = "sage"
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    # 精度
    precision: str = "bf16"
    # 梯度裁剪
    grad_clip: float = 1.0
    # MTP训练增强
    mtp_k: int = 4
    mtp_enabled: bool = True
    # 持续学习
    cls_hot_line: bool = True
    cls_warm_line: bool = True
    cls_cold_line: bool = True
    # 知识编辑
    knowledge_edit_method: str = "rome"


@dataclass
class EvaluationConfig:
    """评估配置 (多模态基准 + 安全测试)."""

    # 通用基准
    mmlu: bool = True
    gsm8k: bool = True
    humaneval: bool = True
    swe_bench: bool = True
    livecodebench: bool = True
    # 代码生成目标
    humaneval_target: float = 0.85
    swe_bench_target: float = 0.30
    livecodebench_target: float = 0.60
    bugfix_target: float = 0.70
    # 全库理解
    needle_in_codebase_target: float = 0.90
    # 安全
    safety_red_team: bool = True
    safety_hallucination: bool = True
    safety_privacy: bool = True
    # 社交图灵测试
    turing_test_enabled: bool = True


# ---------------------------------------------------------------------------
# 主配置
# ---------------------------------------------------------------------------

@dataclass
class ShannonConfig:
    """Shannon 15B MoE 模型统一配置.

    架构: 编码器(3%) + 循环主体(94%) + 解码器(3%)
    参数: 15B总参数(MoE), 激活参数约2-4B
    循环深度: 1-32次动态迭代
    """

    # ---- 基础模型参数 ----
    vocab_size: int = 128000
    hidden_dim: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    head_dim: int = 128
    num_kv_heads: int = 8
    max_seq_len: int = 32768
    rms_eps: float = 1e-6
    dropout: float = 0.0
    bias: bool = False

    # ---- 循环深度 ----
    dynamic_iterations: Tuple[int, int] = (1, 32)
    silent_thinking: bool = True

    # ---- MoE ----
    num_big_experts: int = 16
    num_small_experts: int = 16
    top_k_big: int = 4
    top_k_small: int = 4
    expert_ffn_dim: int = 1024
    small_expert_ffn_dim: int = 512
    num_shared_experts: int = 2
    num_empty_experts: int = 4

    # ---- 注意力 ----
    moh_top_k: int = 4
    moh_n_shared: int = 2
    gated_rank: int = 64
    use_qk_norm: bool = True

    # ---- 位置编码 ----
    rope_theta: float = 10000.0
    yarn_original_max: int = 8192
    longrope_max_seq_len: int = 5_000_000

    # ---- NSL ----
    nsl_enabled: bool = True
    nsl_vocab_size: int = 1024

    # ---- CTM ----
    ctm_enabled: bool = True
    nlm_num_neurons: int = 8
    nlm_d_state: int = 16

    # ---- 隐空间解码 ----
    latent_decode_enabled: bool = True
    nar_hidden_dim: int = 1024
    mask_refine_max_iters: int = 8
    flow_enabled: bool = True
    ar_global_threshold: float = 0.75

    # ---- 编码器 ----
    num_encoder_layers: int = 1
    vit_patch_size: int = 16
    vae_latent_dim: int = 256

    # ---- 解码器输出 ----
    svg_vocab_size: int = 8192
    tool_vocab_size: int = 512
    image_edit_enabled: bool = True

    # ---- 训练 ----
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    precision: str = "bf16"
    mtp_enabled: bool = True
    mtp_k: int = 4

    # ---- 评估 ----
    humaneval_target: float = 0.85
    swe_bench_target: float = 0.30

    # ---- 子配置对象 ----
    positional_encoding: PositionalEncodingConfig = field(
        default_factory=PositionalEncodingConfig
    )
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    recurrent: RecurrentConfig = field(default_factory=RecurrentConfig)
    nsl: NSLConfig = field(default_factory=NSLConfig)
    ctm: CTMConfig = field(default_factory=CTMConfig)
    latent_decode: LatentDecodeConfig = field(default_factory=LatentDecodeConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder_output: DecoderOutputConfig = field(default_factory=DecoderOutputConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # ------------------------------------------------------------------
    def __post_init__(self):
        """验证配置一致性与约束."""
        # 基础维度
        assert self.hidden_dim > 0, "hidden_dim 必须为正"
        assert self.num_layers > 0, "num_layers 必须为正"
        assert self.num_heads > 0, "num_heads 必须为正"
        assert self.head_dim > 0, "head_dim 必须为正"
        assert self.vocab_size > 0, "vocab_size 必须为正"
        # hidden_dim 应可被 num_heads 整除 (使用 head_dim)
        expected_dim = self.num_heads * self.head_dim
        assert self.hidden_dim == expected_dim, (
            f"hidden_dim({self.hidden_dim}) 必须等于 num_heads({self.num_heads}) "
            f"* head_dim({self.head_dim}) = {expected_dim}"
        )
        # GQA
        assert self.num_heads % self.num_kv_heads == 0, (
            f"num_heads({self.num_heads}) 必须能被 num_kv_heads({self.num_kv_heads}) 整除"
        )
        # 循环深度
        iters = self.dynamic_iterations
        assert isinstance(iters, (tuple, list)) and len(iters) == 2, (
            "dynamic_iterations 必须是 (min, max) 元组"
        )
        assert 1 <= iters[0] <= iters[1] <= 32, (
            f"dynamic_iterations({iters}) 须满足 1 <= min <= max <= 32"
        )
        # MoE
        assert 0 < self.top_k_big <= self.num_big_experts, (
            f"top_k_big({self.top_k_big}) 须在 (0, {self.num_big_experts}]"
        )
        assert 0 < self.top_k_small <= self.num_small_experts, (
            f"top_k_small({self.top_k_small}) 须在 (0, {self.num_small_experts}]"
        )
        # 同步子配置 (顶层快捷字段覆盖子配置默认值)
        self._sync_subconfigs()

    def _sync_subconfigs(self):
        """将顶层快捷字段同步到子配置对象, 保持一致性."""
        # 位置编码
        pe = self.positional_encoding
        pe.rope_theta = self.rope_theta
        pe.yarn_original_max = self.yarn_original_max
        pe.longrope_original_max = self.yarn_original_max
        # 注意力
        attn = self.attention
        attn.moh_top_k = self.moh_top_k
        attn.moh_n_shared = self.moh_n_shared
        attn.gated_rank = self.gated_rank
        # MoE
        moe = self.moe
        moe.num_big_experts = self.num_big_experts
        moe.num_small_experts = self.num_small_experts
        moe.top_k_big = self.top_k_big
        moe.top_k_small = self.top_k_small
        moe.expert_ffn_dim = self.expert_ffn_dim
        moe.small_expert_ffn_dim = self.small_expert_ffn_dim
        moe.num_shared_experts = self.num_shared_experts
        moe.num_empty_experts = self.num_empty_experts
        moe.nlm_num_neurons = self.nlm_num_neurons
        moe.nlm_d_state = self.nlm_d_state
        moe.nlm_warmup_freeze = self.ctm_enabled
        # 循环
        rec = self.recurrent
        rec.dynamic_iterations = self.dynamic_iterations
        rec.silent_thinking = self.silent_thinking
        # NSL
        self.nsl.enabled = self.nsl_enabled
        self.nsl.nsl_vocab_size = self.nsl_vocab_size
        # CTM
        ctm = self.ctm
        ctm.enabled = self.ctm_enabled
        ctm.nlm_num_neurons = self.nlm_num_neurons
        ctm.nlm_d_state = self.nlm_d_state
        # 隐空间解码
        ld = self.latent_decode
        ld.nar_hidden_dim = self.nar_hidden_dim
        ld.mask_refine_max_iters = self.mask_refine_max_iters
        ld.flow_enabled = self.flow_enabled
        ld.ar_global_threshold = self.ar_global_threshold
        # 编码器
        self.encoder.num_encoder_layers = self.num_encoder_layers
        self.encoder.vit_patch_size = self.vit_patch_size
        self.encoder.vae_latent_dim = self.vae_latent_dim
        # 解码器输出
        do = self.decoder_output
        do.svg_vocab_size = self.svg_vocab_size
        do.tool_vocab_size = self.tool_vocab_size
        do.image_edit_enabled = self.image_edit_enabled
        # 训练
        self.training.learning_rate = self.learning_rate
        self.training.weight_decay = self.weight_decay
        self.training.precision = self.precision
        self.training.mtp_enabled = self.mtp_enabled
        self.training.mtp_k = self.mtp_k
        # 评估
        self.evaluation.humaneval_target = self.humaneval_target
        self.evaluation.swe_bench_target = self.swe_bench_target

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """序列化为可JSON化的字典."""
        d = asdict(self)
        # dynamic_iterations 转为 list (JSON兼容)
        if "dynamic_iterations" in d:
            d["dynamic_iterations"] = list(self.dynamic_iterations)
        if "recurrent" in d and "dynamic_iterations" in d["recurrent"]:
            d["recurrent"]["dynamic_iterations"] = list(self.dynamic_iterations)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ShannonConfig":
        """从字典反序列化."""
        # 深拷贝避免修改输入
        d = dict(d)
        # 处理子配置: 递归构造
        sub_configs = {
            "positional_encoding": PositionalEncodingConfig,
            "attention": AttentionConfig,
            "moe": MoEConfig,
            "recurrent": RecurrentConfig,
            "nsl": NSLConfig,
            "ctm": CTMConfig,
            "latent_decode": LatentDecodeConfig,
            "encoder": EncoderConfig,
            "decoder_output": DecoderOutputConfig,
            "training": TrainingConfig,
            "evaluation": EvaluationConfig,
        }
        for key, cls_sub in sub_configs.items():
            if key in d and isinstance(d[key], dict):
                d[key] = cls_sub(**d[key])
        # dynamic_iterations 转回 tuple
        if "dynamic_iterations" in d and isinstance(d["dynamic_iterations"], (list, tuple)):
            d["dynamic_iterations"] = tuple(d["dynamic_iterations"])
        return cls(**d)

    # ------------------------------------------------------------------
    def safe_moh_config(self) -> Tuple[int, int]:
        """返回与当前 num_heads 兼容的 (moh_n_shared, moh_top_k).

        确保 n_dynamic = n_heads - n_shared >= top_k.
        """
        n = self.num_heads
        n_shared = min(self.moh_n_shared, max(1, n // 4))
        n_shared = min(n_shared, n - 1)
        n_dynamic = n - n_shared
        top_k = min(self.moh_top_k, max(1, n_dynamic))
        return n_shared, top_k

    def num_iterations(self) -> int:
        """返回默认迭代次数 (取 dynamic_iterations 的最大值)."""
        return self.dynamic_iterations[1]

    def extra_repr(self) -> str:
        return (
            f"vocab_size={self.vocab_size}, hidden_dim={self.hidden_dim}, "
            f"num_layers={self.num_layers}, heads={self.num_heads}x{self.head_dim}, "
            f"kv_heads={self.num_kv_heads}, iters={self.dynamic_iterations}, "
            f"experts={self.num_big_experts}x{self.num_small_experts}"
        )
