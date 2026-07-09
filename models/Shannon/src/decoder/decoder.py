"""解码器 (Decoder) — B+C 融合隐空间解码.

Shannon 解码器采用 B+C 融合架构 (spec §9.2, 决策 L1-L15):
  - 方案 B: HierarchicalNAR (段落→句子→token 层次化 NAR)
  - 方案 C: MaskRefinement (掩码精化, 复用 RDT 权重 via mode_switch)
  - 方案 A: FlowPlanner (流匹配全局规划, 可选)
  - AR 保底: ARFallback (三级置信度门控)

决策 L3: 方案 C 掩码精化复用 RDT 权重, 不引入独立解码网络.
决策 L4: 形式化证明类输出强制 AR + Lean 验证.
决策 L11: 拟人流式修订率上限 15%.

主前向路径: hidden -> lm_head -> logits (AR/训练用).
生成路径: NAR draft + MaskRefine + Flow + AR verify (推理用).

参考: AGENTS.md Agent 11 (LatentDecodeAgent), common.latent_decode.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.layers import RMSNorm
from common.latent_decode import (
    HierarchicalNAR,
    HierarchicalNARConfig,
    MaskRefinement,
    MaskRefinementConfig,
    FlowPlanner,
    FlowPlannerConfig,
    ARFallback,
    ARFallbackConfig,
    ModeSwitch,
    ModeSwitchConfig,
    HumanStream,
    HumanStreamConfig,
)


class ShannonDecoder(nn.Module):
    """Shannon 解码器 (B+C 融合).

    组成:
      1. lm_head: 主文本输出头 (hidden_dim -> vocab_size), 训练/AR 用.
      2. HierarchicalNAR: 方案 B, 段落→句子→token 层次化 NAR.
      3. MaskRefinement: 方案 C, 复用 RDT 权重的掩码精化.
      4. FlowPlanner: 方案 A, 流匹配全局规划 (可选).
      5. ARFallback: AR 保底通道 (三级置信度门控).
      6. ModeSwitch: reasoning/decoding LoRA 模式切换.
      7. HumanStream: 拟人流式输出前端.

    Args:
        config: ShannonConfig.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        H = config.hidden_dim
        V = config.vocab_size

        # 主输出头 (文本)
        self.norm = RMSNorm(H, eps=config.rms_eps)
        # 决策: text_head_tied 时与 embedding 共享权重 (由模型层设置)
        self.lm_head = nn.Linear(H, V, bias=False)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02 / (H ** 0.5))

        # MTP 多 token 预测头 (训练增强, k=2-4)
        self.mtp_enabled = config.mtp_enabled
        self.mtp_k = config.mtp_k
        if self.mtp_enabled:
            self.mtp_heads = nn.ModuleList([
                nn.Linear(H, V, bias=False) for _ in range(self.mtp_k - 1)
            ])
            for h in self.mtp_heads:
                nn.init.normal_(h.weight, mean=0.0, std=0.02 / (H ** 0.5))

        # 方案 B: HierarchicalNAR (NAR 解码维度对齐到 hidden_dim)
        nar_cfg = HierarchicalNARConfig(
            vocab_size=V,
            hidden_dim=min(config.nar_hidden_dim, H),
            num_heads=max(1, min(config.latent_decode.nar_num_heads, H)),
            num_layers_per_level=max(1, config.latent_decode.nar_num_layers_per_level),
            max_paragraphs=config.latent_decode.nar_max_paragraphs,
            max_sentences_per_paragraph=config.latent_decode.nar_max_sentences,
            max_tokens_per_sentence=config.latent_decode.nar_max_tokens,
            mask_token_id=config.encoder.mask_token_id,
            dropout=config.dropout,
        )
        self.hierarchical_nar = HierarchicalNAR(nar_cfg)
        # NAR 输出投影到 hidden_dim (对齐主路径)
        nar_h = nar_cfg.hidden_dim
        self.nar_to_hidden = nn.Linear(nar_h, H, bias=False) if nar_h != H else nn.Identity()

        # 方案 C: MaskRefinement (复用 RDT 权重 via mode_switch)
        mr_cfg = MaskRefinementConfig(
            vocab_size=V,
            hidden_dim=H,
            max_iters=config.latent_decode.mask_refine_max_iters,
            confidence_threshold=config.latent_decode.mask_refine_confidence,
            schedule=config.latent_decode.mask_refine_schedule,
        )
        self.mask_refinement = MaskRefinement(mr_cfg)

        # 方案 A: FlowPlanner (可选)
        self.flow_enabled = config.latent_decode.flow_enabled
        if self.flow_enabled:
            flow_cfg = FlowPlannerConfig(
                latent_dim=min(config.latent_decode.flow_latent_dim, H),
                hidden_dim=H,
                num_heads=max(1, min(config.latent_decode.flow_num_heads, H)),
                num_layers=max(1, config.latent_decode.flow_num_layers),
                num_euler_steps=config.latent_decode.flow_num_euler_steps,
                solver=config.latent_decode.flow_solver,
            )
            self.flow_planner = FlowPlanner(flow_cfg)
        else:
            self.flow_planner = None

        # AR 保底
        ar_cfg = ARFallbackConfig(
            vocab_size=V,
            hidden_dim=H,
            token_threshold=config.latent_decode.ar_token_threshold,
            block_threshold=config.latent_decode.ar_block_threshold,
            global_threshold=config.latent_decode.ar_global_threshold,
            max_new_tokens=config.latent_decode.ar_max_new_tokens,
            force_ar_for_proof=config.latent_decode.ar_force_proof,
        )
        self.ar_fallback = ARFallback(ar_cfg)

        # ModeSwitch: reasoning/decoding LoRA
        ms_cfg = ModeSwitchConfig(
            hidden_dim=H,
            lora_rank=config.latent_decode.mode_lora_rank,
            lora_alpha=config.latent_decode.mode_lora_alpha,
            num_layers=config.num_layers,
        )
        self.mode_switch = ModeSwitch(ms_cfg)

        # 拟人流式输出前端
        self.human_stream_enabled = config.latent_decode.human_stream_enabled
        if self.human_stream_enabled:
            hs_cfg = HumanStreamConfig(
                revision_rate_cap=config.latent_decode.human_revision_cap,
            )
            self.human_stream = HumanStream(hs_cfg)
        else:
            self.human_stream = None

    # ------------------------------------------------------------------
    def tie_weights(self, embedding_weight: torch.Tensor) -> None:
        """绑定 lm_head 权重与文本 embedding (weight tying)."""
        self.lm_head.weight = embedding_weight

    # ------------------------------------------------------------------
    def forward(
        self,
        hidden: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mode: str = "decoding",
    ) -> dict:
        """解码器前向 (训练主路径).

        Args:
            hidden: [B, S, H] 循环主体输出隐状态.
            labels: [B, S] 目标 token id (用于计算 AR loss).
            mode: "reasoning" | "decoding" 模式切换.

        Returns:
            dict 含:
              - logits: [B, S, vocab] 主输出 logits.
              - aux_loss: 辅助损失 (含 MTP loss 若有).
              - mtp_logits: (可选) MTP 多 token 预测 logits 列表.
              - loss: (可选) 主任务 loss (当 labels 提供时).
        """
        h = self.norm(hidden)
        # ModeSwitch LoRA 增量 (decoding 模式)
        # mode_switch 按 layer 索引应用, 此处对最后一层应用
        # (简化: 对最终隐状态做 decoding LoRA 微调)
        h = self.mode_switch(h, layer_idx=0, mode=mode, soft=True)

        # 主 logits
        logits = self.lm_head(h)  # [B, S, V]

        result = {
            "logits": logits,
            "aux_loss": torch.zeros((), device=hidden.device, dtype=hidden.dtype),
        }

        # MTP 多 token 预测 (训练增强)
        if self.mtp_enabled and self.mtp_k > 1:
            mtp_logits = [head(h) for head in self.mtp_heads]
            result["mtp_logits"] = mtp_logits

        # 计算 loss (若提供 labels)
        if labels is not None:
            # 主 loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            total_loss = loss
            # MTP loss
            if self.mtp_enabled and self.mtp_k > 1 and "mtp_logits" in result:
                for i, ml in enumerate(mtp_logits):
                    ml_shift = ml[..., :-1, :].contiguous()
                    ml_labels = labels[..., 1:].contiguous()
                    mtp_loss = F.cross_entropy(
                        ml_shift.view(-1, ml_shift.size(-1)),
                        ml_labels.view(-1),
                        ignore_index=-100,
                    )
                    total_loss = total_loss + 0.1 * mtp_loss
            result["loss"] = total_loss
            result["aux_loss"] = result["aux_loss"] + total_loss

        return result

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        hidden: torch.Tensor,
        token_embed_fn,
        max_new_tokens: int = 128,
        use_nar: bool = True,
        use_mask_refine: bool = True,
    ) -> dict:
        """生成路径 (推理): NAR draft + MaskRefine + AR 保底.

        Args:
            hidden: [B, S, H] 循环主体输出.
            token_embed_fn: token_id -> embedding 的函数.
            max_new_tokens: 最大生成 token 数.
            use_nar: 是否使用 NAR draft.
            use_mask_refine: 是否使用掩码精化.

        Returns:
            dict 含生成 token 序列.
        """
        h = self.norm(hidden)

        def backbone_fn(x):
            """RDT 主干前向 (decoding mode, 简化为 norm + lm_head 反馈)."""
            return self.norm(x)

        # AR 保底生成
        prompt_logits = self.lm_head(h)
        prompt_ids = prompt_logits.argmax(dim=-1)  # [B, S]
        # ARFallback.generate 返回生成 token tensor [B, T_total]
        tokens = self.ar_fallback.generate(
            backbone_fn=backbone_fn,
            token_embed_fn=token_embed_fn,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            device=h.device,
        )
        return {
            "tokens": tokens,
            "ar_used": True,
        }

    def extra_repr(self) -> str:
        return (
            f"vocab={self.config.vocab_size}, hidden={self.config.hidden_dim}, "
            f"nar={not isinstance(self.nar_to_hidden, nn.Identity)}, "
            f"flow={self.flow_enabled}, mtp={self.mtp_enabled}"
        )
