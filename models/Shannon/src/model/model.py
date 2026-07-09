"""Shannon 模型 (ShannonModel) — 编码器(3%) + 循环主体(94%) + 解码器(3%).

Shannon 15B MoE 模型顶层架构:
    input_ids ──→ ShannonEncoder (3%) ──→ ShannonRecurrentBody (94%) ──→ ShannonDecoder (3%) ──→ logits

  - ShannonEncoder: 文本嵌入 + 模态嵌入 + 轻量编码层 (3% 参数)
  - ShannonRecurrentBody: RDT 循环主体, 1-32 次动态迭代 (94% 参数)
  - ShannonDecoder: B+C 融合解码器 + 多任务输出头 (3% 参数)

权重共享:
  - 循环块权重在 1-32 次迭代间共享 (RecurrentBody 内部)
  - lm_head 与文本 token embedding 权重共享 (weight tying)

参考: AGENTS.md 项目结构全景, spec §2 三层架构.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.layers import RMSNorm

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

    Args:
        config: ShannonConfig.
    """

    def __init__(self, config: ShannonConfig):
        super().__init__()
        self.config = config
        self.body = RecurrentBody(config)

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
        return self.body(
            x,
            position_ids=position_ids,
            attention_mask=attention_mask,
            num_iters=num_iters,
        )


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
    """Shannon 15B MoE 顶层模型.

    架构: 编码器(3%) + 循环主体(94%) + 解码器(3%).

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
    ) -> Dict[str, torch.Tensor]:
        """推理生成.

        Args:
            input_ids: [B, S] prompt token id.
            max_new_tokens: 最大新生成 token 数.
            num_iters: 循环迭代次数.

        Returns:
            dict 含 tokens (生成 token 序列).
        """
        def token_embed_fn(ids):
            pos = torch.arange(ids.shape[1], device=ids.device)
            return self.encoder.text_embed(ids, position_ids=pos.unsqueeze(0).expand_as(ids))

        # 编码 + 循环
        enc_out = self.encoder(input_ids, modality_id=MODALITY_TEXT)
        body_out = self.recurrent_body(enc_out, num_iters=num_iters)
        hidden = body_out["hidden"]

        # 解码生成
        gen = self.decoder.main.generate(
            hidden,
            token_embed_fn=token_embed_fn,
            max_new_tokens=max_new_tokens,
        )
        return gen

    # ------------------------------------------------------------------
    def num_parameters(self, only_trainable: bool = False) -> int:
        """返回模型参数总数."""
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        return self.config.extra_repr()
