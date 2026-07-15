"""Shannon 模型 (ShannonModel) — 编码器(3%) + 循环主体(94%) + 解码器(3%).

Shannon 150B MoE 模型顶层架构:
    input_ids ──→ ShannonEncoder (3%) ──→ ShannonRecurrentBody (94%) ──→ ShannonDecoder (3%) ──→ logits

  - ShannonEncoder: 文本嵌入 + 模态嵌入 + 轻量编码层 (3% 参数)
  - ShannonRecurrentBody: RDT 循环主体, 1-32 次动态迭代 + 6 常驻专家 (94% 参数)
  - ShannonDecoder: B+C 融合解码器 + 多任务输出头 (3% 参数)

常驻专家 (与 MathMaster 共用 common_base 底子):
  - 4 固定常驻专家 (ExpertFFN, 始终开启, 不受路由)
  - 2 可学习常驻专家 (EmptyExpert 零初始化, 逐步填充)
  - 与循环主体内双层 MoE 并行计算, 结果相加

权重共享:
  - 循环块权重在 1-32 次迭代间共享 (RecurrentBody 内部)
  - lm_head 与文本 token embedding 权重共享 (weight tying)

参考: AGENTS.md 项目结构全景, spec §2 三层架构, common_base 底子架构.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.layers import RMSNorm

from models.common_base import BaseConfig, ResidentExpertPool

from ..config.config import ShannonConfig
from ..encoder import (
    TextEmbedding,
    ModalityEmbedding,
    MODALITY_TEXT,
)
from ..recurrent.body import RecurrentBody
from ..decoder.decoder import ShannonDecoder
from ..decoder.svg_decoder import SVGDecoder
from ..decoder.structured import StructuredOutput
from ..decoder.image_edit import ImageEditRouter


# ---------------------------------------------------------------------------
# 编码器
# ---------------------------------------------------------------------------
class ShannonEncoder(nn.Module):
    """Shannon 编码器 (3% 参数).

    组成:
      1. TextEmbedding: token embedding + 位置编码 (含 9 特殊 token)
      2. ModalityEmbedding: 模态类型嵌入 (文本为默认模态)
      3. 轻量编码层: num_encoder_layers 层 pre-norm Transformer (自注意力 + FFN)

    Args:
        config: ShannonConfig.
    """

    def __init__(self, config: ShannonConfig):
        super().__init__()
        self.config = config
        H = config.hidden_dim

        # 文本嵌入 (token + 位置)
        self.text_embed = TextEmbedding(
            vocab_size=config.vocab_size,
            hidden_dim=H,
            max_seq_len=config.max_seq_len,
            padding_idx=config.encoder.pad_token_id,
            dropout=config.dropout,
        )

        # 模态嵌入 (文本默认模态)
        self.modality_embed = ModalityEmbedding(
            hidden_dim=H,
            num_modalities=5,
            dropout=config.dropout,
        )

        # 轻量编码层 (3% 参数: 少量 Transformer 层)
        num_enc = max(1, config.num_encoder_layers)
        self.encoder_layers = nn.ModuleList([
            _EncoderLayer(H, config.num_heads, config.rms_eps, config.dropout)
            for _ in range(num_enc)
        ])

        self.norm = RMSNorm(H, eps=config.rms_eps)

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        modality_id: int = MODALITY_TEXT,
    ) -> torch.Tensor:
        """编码器前向.

        Args:
            input_ids: [B, S] token id.
            position_ids: [B, S] 位置 id.
            modality_id: 模态类型 (默认文本).

        Returns:
            [B, S, H] 编码后隐状态.
        """
        # 文本嵌入
        h = self.text_embed(input_ids, position_ids=position_ids)
        # 模态类型嵌入
        h = self.modality_embed(h, modality_id=modality_id)
        # 编码层
        for layer in self.encoder_layers:
            h = layer(h)
        h = self.norm(h)
        return h


class _EncoderLayer(nn.Module):
    """轻量编码层: pre-norm 自注意力 + FFN."""

    def __init__(self, hidden_dim: int, num_heads: int, eps: float, dropout: float):
        super().__init__()
        n = max(1, min(num_heads, hidden_dim // 32))
        self.attn_norm = RMSNorm(hidden_dim, eps=eps)
        self.attn = nn.MultiheadAttention(
            hidden_dim, n, dropout=dropout, batch_first=True,
        )
        self.ffn_norm = RMSNorm(hidden_dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.attn_norm(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ---------------------------------------------------------------------------
# 循环主体包装
# ---------------------------------------------------------------------------
class ShannonRecurrentBody(nn.Module):
    """Shannon 循环主体包装 (94% 参数).

    直接复用 recurrent.body.RecurrentBody, 提供与编码器/解码器对齐的接口.
    额外添加 6 常驻专家 (4 固定 + 2 可学习), 与循环主体内双层 MoE 并行计算,
    结果相加 (参考 MathMaster 多 MoE 结构, 共用 common_base 底子).

    常驻专家:
      * 4 固定常驻专家 (ExpertFFN, 始终开启, 不受路由)
      * 2 可学习常驻专家 (EmptyExpert 零初始化, 逐步填充, 可选 NLM 增强)
      * 常驻专家处理循环主体输出, 与双层 MoE 输出相加 (并行残差)

    Args:
        config: ShannonConfig.
    """

    def __init__(self, config: ShannonConfig):
        super().__init__()
        self.config = config
        self.body = RecurrentBody(config)

        # 6 常驻专家 (4 固定 + 2 可学习), 从 common_base 导入, 与 MathMaster 共用底子
        base_cfg = BaseConfig.from_shannon(config)
        self.resident_experts = ResidentExpertPool(base_cfg)
        # 残差缩放 (可学习, 初始化为小值, 避免常驻专家主导)
        self.resident_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        num_iters: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """循环主体前向.

        Args:
            x: [B, S, H] 编码器输出.
            position_ids: [B, S].
            attention_mask: 注意力掩码.
            num_iters: 显式迭代次数.

        Returns:
            dict 含 hidden, aux_loss, n_iters 等.
        """
        body_out = self.body(
            x,
            position_ids=position_ids,
            attention_mask=attention_mask,
            num_iters=num_iters,
        )
        hidden = body_out["hidden"]  # [B, S, H]

        # 常驻专家与双层 MoE 并行计算, 结果相加 (并行残差)
        # use_nlm: 训练时启用 NLM 增强 (CTM 决策 C10: 仅实体专家使用 NLM)
        resident_out = self.resident_experts(
            hidden, use_nlm=self.config.ctm_enabled and self.training
        )
        body_out["hidden"] = hidden + self.resident_scale * resident_out
        return body_out


# ---------------------------------------------------------------------------
# 解码器包装
# ---------------------------------------------------------------------------
class ShannonDecoderWrapper(nn.Module):
    """Shannon 解码器包装 (3% 参数).

    组合:
      1. 主解码器 (ShannonDecoder): B+C 融合, 主 logits + MTP
      2. SVGDecoder: SVG 矢量图输出
      3. StructuredOutput: JSON / 工具调用 / TTS
      4. ImageEditRouter: 图像编辑路由

    Args:
        config: ShannonConfig.
    """

    def __init__(self, config: ShannonConfig):
        super().__init__()
        self.config = config
        H = config.hidden_dim

        # 主解码器 (B+C 融合)
        self.main = ShannonDecoder(config)

        # SVG 解码器
        self.svg_decoder = SVGDecoder(
            hidden_dim=H,
            svg_hidden_dim=min(config.nar_hidden_dim, H),
            max_paths=config.encoder.svg_max_paths,
        )

        # 结构化输出 (JSON / 工具 / TTS)
        self.structured = StructuredOutput(
            hidden_dim=H,
            vocab_size=config.vocab_size,
            tool_vocab_size=config.decoder_output.tool_vocab_size,
        )

        # 图像编辑路由
        self.image_edit_enabled = config.decoder_output.image_edit_enabled
        if self.image_edit_enabled:
            self.image_edit = ImageEditRouter(
                hidden_dim=H,
                latent_dim=config.decoder_output.image_edit_latent_dim,
            )
        else:
            self.image_edit = None

    # ------------------------------------------------------------------
    def tie_weights(self, embedding_weight: torch.Tensor) -> None:
        """权重共享: lm_head 与文本 embedding."""
        self.main.tie_weights(embedding_weight)

    def forward(
        self,
        hidden: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mode: str = "decoding",
    ) -> Dict[str, torch.Tensor]:
        """解码器前向 (主路径)."""
        return self.main(hidden, labels=labels, mode=mode)


# ---------------------------------------------------------------------------
# 顶层模型
# ---------------------------------------------------------------------------
class ShannonModel(nn.Module):
    """Shannon 150B MoE 顶层模型.

    架构: 编码器(3%) + 循环主体(94%) + 解码器(3%).
    循环主体含 6 常驻专家 (4 固定 + 2 可学习), 与 MathMaster 共用 common_base 底子.

    Args:
        config: ShannonConfig. 若 None 用默认配置.
    """

    def __init__(self, config: Optional[ShannonConfig] = None):
        super().__init__()
        self.config = config or ShannonConfig()

        # 三层架构
        self.encoder = ShannonEncoder(self.config)
        self.recurrent_body = ShannonRecurrentBody(self.config)
        self.decoder = ShannonDecoderWrapper(self.config)

        # 权重共享: lm_head <-> text embedding
        if self.config.decoder_output.text_head_tied:
            self.decoder.tie_weights(self.encoder.text_embed.token_embed.weight)

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        num_iters: Optional[int] = None,
        modality_id: int = MODALITY_TEXT,
    ) -> Dict[str, torch.Tensor]:
        """模型前向.

        Args:
            input_ids: [B, S] token id.
            position_ids: [B, S] 位置 id (None 用 0..S-1).
            attention_mask: 注意力掩码.
            labels: [B, S] 目标 token id (训练用).
            num_iters: 显式循环迭代次数.
            modality_id: 模态类型.

        Returns:
            dict 含:
              - logits: [B, S, vocab] 主输出 logits.
              - aux_loss: 总辅助损失 (循环主体 MoE + ACT + 解码器).
              - loss: (当 labels 提供时) 主任务 loss.
              - n_iters: 实际循环迭代次数.
              - hidden: [B, S, H] 循环主体输出隐状态.
        """
        # 1. 编码器
        enc_out = self.encoder(
            input_ids,
            position_ids=position_ids,
            modality_id=modality_id,
        )  # [B, S, H]

        # 2. 循环主体 (1-32 次动态迭代)
        body_out = self.recurrent_body(
            enc_out,
            position_ids=position_ids,
            attention_mask=attention_mask,
            num_iters=num_iters,
        )
        hidden = body_out["hidden"]  # [B, S, H]
        body_aux = body_out["aux_loss"]  # scalar

        # 3. 解码器
        dec_out = self.decoder(hidden, labels=labels, mode="decoding")
        logits = dec_out["logits"]  # [B, S, V]
        dec_aux = dec_out.get("aux_loss", torch.zeros((), device=hidden.device))

        # 合并辅助损失
        total_aux = body_aux + dec_aux

        result: Dict[str, torch.Tensor] = {
            "logits": logits,
            "aux_loss": total_aux,
            "hidden": hidden,
            "n_iters": body_out["n_iters"],
        }

        # 主任务 loss
        if "loss" in dec_out:
            result["loss"] = dec_out["loss"]
        elif labels is not None:
            # 显式计算 loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            result["loss"] = loss

        # 透传循环主体信息
        if "ponder_cost" in body_out:
            result["ponder_cost"] = body_out["ponder_cost"]
        if "act_state" in body_out:
            result["act_state"] = body_out["act_state"]

        return result

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        num_iters: Optional[int] = None,
    ) -> torch.Tensor:
        """推理生成.

        维护 current_pos 追踪已生成位置数, 每步位置编码从 0..current_pos-1
        完整计算 (循环主体权重共享+动态深度, 不支持 KV cache, 故用完整序列
        重新前向, 但位置编码必须连续且正确).

        Args:
            input_ids: [B, S] prompt token id.
            max_new_tokens: 最大新生成 token 数.
            num_iters: 循环迭代次数.

        Returns:
            tokens: [B, S + max_new_tokens] 生成 token 序列 (含 prompt).
        """
        B = input_ids.shape[0]
        device = input_ids.device
        ids = input_ids.clone()
        # current_pos 记录已编码的位置数 (prompt 长度)
        current_pos = input_ids.shape[1]

        for _ in range(max_new_tokens):
            # 位置编码: 0 .. current_pos-1 (完整序列, 连续且正确)
            pos = torch.arange(current_pos, device=device).unsqueeze(0).expand(B, -1)

            # 完整模型前向 (无 KV cache, 重新计算完整序列)
            enc_out = self.encoder(
                ids,
                position_ids=pos,
                modality_id=MODALITY_TEXT,
            )
            body_out = self.recurrent_body(
                enc_out,
                position_ids=pos,
                num_iters=num_iters,
            )
            hidden = body_out["hidden"]  # [B, current_pos, H]

            # 取最后 token 的 hidden, 解码为下一 token
            h_last = self.decoder.main.norm(hidden[:, -1, :])  # [B, H]
            logits = self.decoder.main.lm_head(h_last)  # [B, vocab]
            next_token = logits.argmax(dim=-1, keepdim=True)  # [B, 1]

            ids = torch.cat([ids, next_token], dim=1)
            current_pos += 1

        return ids

    # ------------------------------------------------------------------
    def num_parameters(self, only_trainable: bool = False) -> int:
        """返回模型参数总数."""
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return self.config.extra_repr()
