"""循环主体 (RecurrentBody) — RDT 循环块, 1-32 次动态迭代.

Shannon 架构核心: 编码器(3%) + 循环主体(94%) + 解码器(3%).
循环主体复用同一组循环块权重, 迭代 1-32 次 (动态深度), 每次:

  1. 注入深度位置嵌入 (DepthEmbedding)
  2. 应用深度 LoRA 适配 (DepthLoRA)
  3. Hybrid-M3 注意力 (UnifiedAttentionScheduler: 4层周期)
  4. 双层 MoE (NestedMoE: 16大×16小 + 共享 + 空专家)
  5. AttnRes + mHC 残差聚合
  6. LTI 稳定性约束 (谱半径<1)
  7. ACT 自适应停止 + CTM 动态损失

Silent Thinking (决策): 仅最终迭代步计算 loss, 中间步不计 loss.

参考: AGENTS.md Agent 1 (ArchAgent), spec §4 循环主体.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.layers import (
    RMSNorm,
    AttnRes,
    mHC,
    GradientCheckpoint,
)
from common.attention import (
    AttentionConfig,
    AttentionOutput,
    KDAAttention,
    MLAAttention,
    MoHAttention,
    MMAAttention,
    apply_rope,
)
from common.ctm import MLASync, CTMDynamicLoss

from ..moe import NestedMoE
from .depth_embed import DepthEmbedding
from .lti import LTIStability, ResidualStabilizer
from .lora_adapter import DepthLoRAAdapter
from .act import ACTStop


class HybridM3AttentionLayer(nn.Module):
    """Hybrid-M3 注意力层 (4层周期, 安全配置).

    周期映射 (layer_idx % 4):
      - phase 0 (4k+1): KDA
      - phase 1 (4k+2): KDA + MoH (安全配置, cross-project)
      - phase 2 (4k+3): KDA
      - phase 3 (4k+4): MLA + MMA (QK-Norm)

    使用 safe_moh_config 确保小 num_heads 也能工作.
    """

    def __init__(
        self,
        attn_cfg: AttentionConfig,
        n_shared: int,
        top_k: int,
    ):
        super().__init__()
        self.config = attn_cfg
        self.phase = attn_cfg.layer_idx % 4

        # KDA (phase 0/1/2 共享)
        self.kda = KDAAttention(attn_cfg)

        # MoH (phase 1, 安全配置)
        moh_cfg = dataclasses.replace(attn_cfg, moh_n_shared=n_shared, moh_top_k=top_k)
        self.moh = MoHAttention(moh_cfg)
        # cross-projection (KDA + MoH 合并)
        self.cross_proj = nn.Linear(attn_cfg.d_model * 2, attn_cfg.d_model, bias=False)

        # MLA + MMA (phase 3)
        mla_cfg = dataclasses.replace(attn_cfg)
        self.mla = MLAAttention(mla_cfg)
        self.mma = MMAAttention(mla_cfg, inner=self.mla)

        # 输出归一化
        self.norm = RMSNorm(attn_cfg.d_model, eps=attn_cfg.rms_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv: Any = None,
        use_cache: bool = False,
        iteration_step: int = 0,
    ) -> torch.Tensor:
        """前向: 根据 phase 路由注意力计算.

        Args:
            iteration_step: 循环迭代步 (用于动态计算 phase,
                使权重共享时 4 层周期注意力真正轮转).
        """
        # 动态 phase: (layer_idx + iteration_step) % 4
        # 解决 layer_idx 硬编码为 0 导致永远只走 KDA 的问题.
        phase = (self.config.layer_idx + int(iteration_step)) % 4

        if phase in (0, 2):
            out = self.kda(
                hidden_states,
                position_ids=position_ids,
                past_kv=past_kv,
                attention_mask=attention_mask,
                use_cache=False,
            )
            return self.norm(out.output)

        if phase == 1:
            out_kda = self.kda(
                hidden_states,
                position_ids=position_ids,
                past_kv=past_kv,
                attention_mask=attention_mask,
                use_cache=False,
            )
            out_moh = self.moh(
                hidden_states,
                position_ids=position_ids,
                past_kv=past_kv,
                attention_mask=attention_mask,
                use_cache=False,
            )
            combined = self.cross_proj(
                torch.cat([out_kda.output, out_moh.output], dim=-1)
            )
            return self.norm(combined)

        # phase 3
        out = self.mma(
            hidden_states,
            position_ids=position_ids,
            past_kv=past_kv,
            attention_mask=attention_mask,
            use_cache=use_cache,
        )
        return self.norm(out.output)


class RecurrentBlock(nn.Module):
    """单次循环迭代块.

    结构 (pre-norm):
      h = x + depth_embed(d) + depth_lora(x, d)
      a = AttentionNorm(h); h = h + Attn(a)
      m = MoENorm(h); h = h + MoE(m)
      h = LTI(h) 或 ResidualStabilizer(h, delta)

    Args:
        config: ShannonConfig.
        layer_idx: 层索引 (决定注意力 phase).
    """

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        H = config.hidden_dim

        # 深度嵌入
        self.depth_embed = DepthEmbedding(
            hidden_dim=H,
            max_iterations=config.dynamic_iterations[1],
            embed_dim=config.recurrent.depth_embed_dim,
        )

        # 深度 LoRA 适配器 (Q/O 投影适配)
        self.depth_lora = DepthLoRAAdapter(
            hidden_dim=H,
            max_depths=config.dynamic_iterations[1],
            rank=config.recurrent.depth_lora_rank,
            alpha=config.recurrent.depth_lora_alpha,
            dropout=config.recurrent.depth_lora_dropout,
            num_adapted=2,
        )

        # 注意力配置
        n_shared, top_k = config.safe_moh_config()
        attn_cfg = AttentionConfig(
            d_model=H,
            n_heads=config.num_heads,
            n_kv_heads=config.num_kv_heads,
            d_kv=config.head_dim,
            d_c=max(H // 4, 8),
            max_seq_len=config.max_seq_len,
            rope_theta=config.rope_theta,
            rope_base_scale=config.positional_encoding.rope_base_scale,
            layer_idx=layer_idx,
            bias=config.bias,
            rms_eps=config.rms_eps,
            moh_top_k=top_k,
            moh_n_shared=n_shared,
            kda_chunk_size=config.attention.kda_chunk_size,
            n_modalities=config.attention.mma_n_modalities,
            window_size=config.positional_encoding.sliding_window,
        )
        self.attn_norm = RMSNorm(H, eps=config.rms_eps)
        self.attention = HybridM3AttentionLayer(attn_cfg, n_shared, top_k)

        # 双层 MoE
        self.moe_norm = RMSNorm(H, eps=config.rms_eps)
        self.moe = NestedMoE(
            hidden_dim=H,
            num_big_experts=config.num_big_experts,
            num_small_experts=config.num_small_experts,
            top_k_big=config.top_k_big,
            top_k_small=config.top_k_small,
            expert_ffn_dim=config.expert_ffn_dim,
            small_expert_ffn_dim=config.small_expert_ffn_dim,
            num_shared_experts=config.num_shared_experts,
            num_empty_experts=config.num_empty_experts,
            use_nlm=config.ctm_enabled,
            nlm_num_neurons=config.nlm_num_neurons,
            nlm_d_state=config.nlm_d_state,
            nlm_warmup_freeze=config.ctm.nlm_warmup_freeze,
            noise_std=config.moe.router_noise_std,
            load_balance_alpha=config.moe.load_balance_alpha,
            dropout=config.dropout,
        )

        # 残差: AttnRes + mHC
        self.use_attn_res = config.recurrent.use_attn_res
        self.use_mhc = config.recurrent.use_mhc
        if self.use_attn_res:
            self.attn_res = AttnRes(
                H,
                num_blocks=config.recurrent.attn_res_num_blocks,
                eps=config.rms_eps,
            )
        if self.use_mhc:
            self.mhc = mHC(
                H,
                num_iters=config.recurrent.mhc_num_iters,
            )

        # LTI 稳定性
        self.lti_enabled = config.recurrent.lti_enabled
        if self.lti_enabled:
            self.residual_stabilizer = ResidualStabilizer(
                H,
                spectral_radius=config.recurrent.lti_spectral_radius,
            )

        # 梯度检查点
        self.use_gradient_checkpoint = config.recurrent.use_gradient_checkpoint
        if self.use_gradient_checkpoint:
            self._attn_ckpt = GradientCheckpoint(self.attention)
            self._moe_ckpt = GradientCheckpoint(self.moe)
        else:
            self._attn_ckpt = self.attention
            self._moe_ckpt = self.moe

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        depth: int,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        nlm_states: Optional[Dict[int, list]] = None,
        iteration_step: int = 0,
        attn_history: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[int, list], List[torch.Tensor]]:
        """单次循环迭代前向.

        Args:
            x: [B, S, H] 输入隐状态.
            depth: 循环深度 (0-indexed).
            position_ids: [B, S].
            attention_mask: [B, S] 或 [B, 1, S, S].
            nlm_states: 上一 tick 的 NLM 状态.
            iteration_step: 循环迭代步 (用于 Hybrid-M3 phase 轮转).
            attn_history: 前序迭代的注意力输出列表 (用于 AttnRes 聚合).

        Returns:
            out: [B, S, H].
            aux: 辅助信息 (aux_loss 等).
            new_nlm_states: 更新后的 NLM 状态.
            new_attn_history: 更新后的注意力输出历史 (含本轮).
        """
        B, S, H = x.shape
        aux: Dict[str, torch.Tensor] = {}
        if attn_history is None:
            attn_history = []

        # 1. 深度位置嵌入
        d_emb = self.depth_embed(depth, B, S)
        h = x + d_emb

        # 2. 深度 LoRA 适配 (对注意力输入做适配)
        h = self.depth_lora.apply("adapter_0", h, depth)

        # 3. 注意力 (pre-norm + residual, 传入 iteration_step 让 phase 轮转)
        attn_in = self.attn_norm(h)
        attn_out = self._attn_ckpt(
            attn_in,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=False,
            iteration_step=iteration_step,
        )

        # 3b. AttnRes 块级残差聚合 (Bug 3 修复):
        #     将当前注意力输出与前序迭代输出通过 AttnRes 加权聚合,
        #     替代简单残差 h = h + attn_out.
        if self.use_attn_res and self.attn_res is not None and len(attn_history) > 0:
            all_attn = list(attn_history) + [attn_out]
            # block_ids: 将各迭代步分配到 num_blocks 个块
            L_attn = len(all_attn)
            block_ids = torch.arange(
                L_attn, device=attn_out.device
            ) % max(self.attn_res.num_blocks, 1)
            aggregated = self.attn_res(all_attn, block_ids=block_ids)
            h = h + aggregated
        else:
            h = h + attn_out

        new_attn_history = attn_history + [attn_out]

        # 4. 双层 MoE (pre-norm + residual)
        moe_in = self.moe_norm(h)
        moe_result = self._moe_ckpt(moe_in, nlm_states=nlm_states)
        moe_out = moe_result["output"]
        aux["moe_aux_loss"] = moe_result["aux_loss"]
        h = h + moe_out

        # 5. LTI 稳定性 (对残差增量收缩)
        if self.lti_enabled:
            delta = h - x
            h = self.residual_stabilizer(x, delta)

        return h, aux, moe_result.get("new_nlm_states", {}), new_attn_history


class RecurrentBody(nn.Module):
    """循环主体: 管理 1-32 次动态迭代.

    复用同一组 RecurrentBlock 权重 (权重共享), 每次迭代:
      - 注入深度信号
      - 注意力 + MoE
      - ACT 自适应停止
      - CTM 动态损失

    Silent Thinking: 仅最终步计算 loss, 中间步 silent.

    Args:
        config: ShannonConfig.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.min_iters, self.max_iters = config.dynamic_iterations
        self.silent_thinking = config.silent_thinking

        # 共享的循环块 (权重共享, 单实例)
        self.block = RecurrentBlock(config, layer_idx=0)

        # ACT 自适应停止
        self.act_enabled = config.recurrent.act_enabled
        if self.act_enabled:
            self.act = ACTStop(
                hidden_dim=config.hidden_dim,
                threshold=config.recurrent.act_threshold,
                penalty_weight=config.recurrent.act_penalty_weight,
                max_iters=self.max_iters,
            )
        else:
            self.act = None

        # CTM: MLA 同步矩阵 + 动态损失
        self.ctm_enabled = config.ctm_enabled
        if self.ctm_enabled:
            d_c = max(config.hidden_dim // 4, 8)
            self.mla_sync = MLASync(
                d_c=d_c,
                num_neurons=config.nlm_num_neurons,
            )
            self.ctm_loss = CTMDynamicLoss(
                lambda_certainty=config.ctm.ctm_lambda_certainty,
                lambda_tick=config.ctm.ctm_lambda_tick,
                lambda_monotone=config.ctm.ctm_lambda_monotone,
            )
            # CTM MLA 同步投影层 (Bug 2 修复):
            #   ctm_proj: H -> d_c (投影到 MLA 压缩维度)
            #   ctm_back_proj: d_c -> H (同步结果投影回 hidden_dim, 残差连接)
            #   ctm_logits_proj: H -> d_c (每 tick 的 "logits", 用于 CTM 动态损失)
            self.ctm_d_c = d_c
            self.ctm_proj = nn.Linear(config.hidden_dim, d_c, bias=False)
            self.ctm_back_proj = nn.Linear(d_c, config.hidden_dim, bias=False)
            self.ctm_logits_proj = nn.Linear(config.hidden_dim, d_c, bias=False)
        else:
            self.mla_sync = None
            self.ctm_loss = None
            self.ctm_d_c = 0
            self.ctm_proj = None
            self.ctm_back_proj = None
            self.ctm_logits_proj = None

        # 输出归一化
        self.norm = RMSNorm(config.hidden_dim, eps=config.rms_eps)

        # 迭代历史 (用于 AttnRes/mHC 聚合)
        self.use_attn_res = config.recurrent.use_attn_res
        self.use_mhc = config.recurrent.use_mhc

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        num_iters: Optional[int] = None,
        return_all_layers: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """循环主体前向.

        Args:
            x: [B, S, H] 编码器输出.
            position_ids: [B, S].
            attention_mask: 注意力掩码.
            num_iters: 显式指定迭代次数 (None 用 max_iters 或 ACT 决定).
            return_all_layers: 是否返回各迭代步隐状态.

        Returns:
            dict 含:
              - hidden: [B, S, H] 最终隐状态.
              - aux_loss: 总辅助损失 (MoE + ACT ponder).
              - n_iters: 实际迭代次数.
              - all_layers: (可选) 各步隐状态列表.
              - act_state: ACT 状态.
              - n_updates: 每 token 更新次数.
        """
        B, S, H = x.shape
        device = x.device
        dtype = x.dtype

        # 决定迭代次数
        if num_iters is not None:
            target_iters = int(num_iters)
        else:
            target_iters = self.max_iters
        target_iters = max(self.min_iters, min(target_iters, self.max_iters))

        # ACT 状态
        if self.act_enabled:
            act_state = self.act.init_state(B, S, device, dtype)
        else:
            act_state = None

        # CTM NLM 状态 (跨迭代传递)
        nlm_states: Optional[Dict[int, list]] = None

        # AttnRes 注意力输出历史 (跨迭代传递, 用于块级残差聚合)
        attn_history: List[torch.Tensor] = []

        # CTM 每 tick 的 logits (用于动态损失, Bug 2 修复)
        ctm_tick_logits: List[torch.Tensor] = []

        h = x
        all_layers: List[torch.Tensor] = []
        total_aux = torch.zeros((), device=device, dtype=dtype)
        total_ponder = torch.zeros((), device=device, dtype=dtype)

        it = 0
        for it in range(target_iters):
            # ACT 检查: 是否还有 active token
            if self.act_enabled and act_state is not None:
                active = ~act_state["halted"]
                if not active.any():
                    break

            # 循环块前向 (detach NLM states: 截断 BPTT, 防止跨迭代图爆炸)
            # 传入 iteration_step 让 Hybrid-M3 phase 真正轮转 (Bug 1 修复)
            # 传入 attn_history 让 AttnRes 做块级残差聚合 (Bug 3 修复)
            h_new, block_aux, nlm_states, attn_history = self.block(
                h,
                depth=it,
                position_ids=position_ids,
                attention_mask=attention_mask,
                nlm_states=_detach_nlm_states(nlm_states),
                iteration_step=it,
                attn_history=attn_history,
            )
            # ACT: 对 active token 更新, halted token 保持
            if self.act_enabled and act_state is not None:
                active_mask = ~act_state["halted"]  # [B, S]
                m = active_mask.unsqueeze(-1).to(h_new.dtype)
                h = m * h_new + (1.0 - m) * h
                # ACT step
                _, act_state, ponder = self.act.step(h_new, act_state, it)
                total_ponder = total_ponder + ponder
            else:
                h = h_new

            # 累积辅助损失
            moe_aux = block_aux.get("moe_aux_loss")
            if moe_aux is not None:
                total_aux = total_aux + moe_aux

            # CTM MLA 同步 (每步同步, Bug 2 修复):
            #   1. 把当前隐状态 h 投影到 MLA 压缩维度 d_c
            #   2. 调用 mla_sync.sync_matrix 计算 c_kv·c_kv^T 同步矩阵 S
            #   3. 用 S 聚合 c_kv 后投影回 hidden_dim, 残差加到 h 上
            if self.ctm_enabled and self.mla_sync is not None:
                c_kv = self.ctm_proj(h)  # [B, S, d_c]
                S = self.mla_sync.sync_matrix(c_kv)  # [B, S, S]
                synced_c = torch.matmul(S, c_kv)  # [B, S, d_c] (潜变量同步)
                h_sync = self.ctm_back_proj(synced_c)  # [B, S, H]
                h = h + h_sync  # 残差连接

                # 收集当前 tick 的 "logits" (取最后 token, 用于 CTM 动态损失)
                tick_logits = self.ctm_logits_proj(h[:, -1, :])  # [B, d_c]
                ctm_tick_logits.append(tick_logits)

            # 收集各迭代步隐状态 (mHC 或 return_all_layers 需要)
            if self.use_mhc or return_all_layers:
                all_layers.append(h)

            # mHC 深度方向归一化约束 (Bug 3 修复):
            #   将所有迭代步的隐状态通过 Sinkhorn 双随机矩阵重新混合,
            #   保证深度方向信号不爆炸 (谱范数 <= 1).
            if self.use_mhc and hasattr(self.block, "mhc") and self.block.mhc is not None:
                remixed = self.block.mhc(all_layers)  # [B, S, L, H]
                h = remixed[..., -1, :]  # [B, S, H] 取最后一层作为当前隐状态
                all_layers[-1] = h

        n_iters = it + 1

        # CTM 动态损失 (Bug 2 修复): 基于各 tick 的 logits 计算动态损失
        if (
            self.ctm_enabled
            and self.ctm_loss is not None
            and len(ctm_tick_logits) >= 2
        ):
            logits_per_tick = torch.stack(ctm_tick_logits, dim=0)  # [T, B, d_c]
            # 自监督标签: 最终 tick 的 argmax (鼓励各 tick 收敛到一致结果)
            with torch.no_grad():
                labels = logits_per_tick[-1].argmax(dim=-1)  # [B]
            ctm_result = self.ctm_loss(logits_per_tick, labels)
            total_aux = total_aux + ctm_result["loss"]

        # 最终归一化
        h = self.norm(h)

        # 总辅助损失 = MoE aux + ACT ponder penalty
        total_aux = total_aux + total_ponder

        result: Dict[str, torch.Tensor] = {
            "hidden": h,
            "aux_loss": total_aux,
            "n_iters": torch.tensor(n_iters, device=device),
            "ponder_cost": total_ponder,
        }
        if return_all_layers:
            result["all_layers"] = all_layers
        if self.act_enabled and act_state is not None:
            result["act_state"] = act_state
            result["n_updates"] = act_state["n_updates"]
        return result

    def extra_repr(self) -> str:
        return (
            f"iters={self.config.dynamic_iterations}, "
            f"silent={self.silent_thinking}, "
            f"act={self.act_enabled}, "
            f"ctm={self.ctm_enabled}"
        )


def _detach_nlm_states(
    states: Optional[Dict[int, list]],
) -> Optional[Dict[int, list]]:
    """ detach NLM 状态, 截断跨迭代反向传播.

    NLM 神经元状态在迭代间传递, detach 后每步独立计算梯度,
    避免循环展开导致的梯度爆炸/显存溢出 (truncated BPTT).
    """
    if states is None:
        return None
    return {
        ei: [s.detach() if torch.is_tensor(s) else s for s in st]
        for ei, st in states.items()
    }
