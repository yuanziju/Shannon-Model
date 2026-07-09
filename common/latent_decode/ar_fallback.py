"""AR 保底通道 (ARFallback).

决策 L4: 形式化证明类输出必须强制 AR + Lean 验证器.
NAR / 掩码精化在低置信度场景可能产生不一致输出, 此时启用 AR (自回归)
保底通道, 逐 token 强制生成, 保证质量.

三级置信度门控 (决策):
    - token 级: 单 token 置信度 < 0.55  → 该 token 触发 AR 重写
    - block 级: 块平均置信度 < 0.70     → 整块触发 AR 重写
    - global 级: 全局平均置信度 < 0.75  → 整个输出退化为纯 AR 生成

置信度 = softmax(logits).max(), 越高表示模型越确定.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# 三级置信度阈值 (与决策一致)
TOKEN_THRESHOLD = 0.55
BLOCK_THRESHOLD = 0.70
GLOBAL_THRESHOLD = 0.75

# 触发级别
LEVEL_NONE = 0
LEVEL_TOKEN = 1
LEVEL_BLOCK = 2
LEVEL_GLOBAL = 3


@dataclass
class ARFallbackConfig:
    """AR 保底配置."""

    vocab_size: int = 128000
    hidden_dim: int = 1024
    token_threshold: float = TOKEN_THRESHOLD
    block_threshold: float = BLOCK_THRESHOLD
    global_threshold: float = GLOBAL_THRESHOLD
    max_new_tokens: int = 2048
    # AR 采样
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    # 形式化证明强制 AR (决策 L4)
    force_ar_for_proof: bool = True
    proof_keywords: tuple = (
        "theorem", "lemma", "proof", "Qed", "Lean", "Coq",
        "by", "exact", "induction", "rw",
    )


class ARFallback(nn.Module):
    """三级置信度门控的 AR 保底通道."""

    def __init__(self, config: ARFallbackConfig | None = None, **kwargs):
        super().__init__()
        cfg = config or ARFallbackConfig(**kwargs)
        self.cfg = cfg
        # AR 解码头 (可复用主 logits head, 此处独立以便差异训练)
        self.ln_f = nn.LayerNorm(cfg.hidden_dim)
        self.lm_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)
        nn.init.normal_(self.lm_head.weight, std=0.02)

    # ------------------------------------------------------------------
    # 置信度评估
    # ------------------------------------------------------------------
    @staticmethod
    def compute_confidence(logits: torch.Tensor) -> torch.Tensor:
        """计算每位置最大 softmax 概率作为置信度.

        Args:
            logits: [B, T, vocab].

        Returns:
            confidence: [B, T] 每位置置信度.
        """
        probs = F.softmax(logits, dim=-1)
        return probs.max(dim=-1).values

    def assess(
        self,
        logits: torch.Tensor,
        block_boundaries: torch.Tensor | None = None,
        prompt: str | None = None,
    ) -> dict:
        """评估输出置信度, 决定触发级别.

        Args:
            logits: [B, T, vocab] NAR/掩码精化的输出 logits.
            block_boundaries: [B, T] 每位置所属块 id. None 视为单一块.
            prompt: 原始提示 (用于检测形式化证明, 强制 AR).

        Returns:
            dict 含 level / token_conf / block_conf / global_conf /
                 trigger_mask / block_trigger / proof_detected.
        """
        B, T, V = logits.shape
        conf = self.compute_confidence(logits)  # [B, T]

        # token 级
        token_trigger = conf < self.cfg.token_threshold  # [B, T]

        # block 级
        if block_boundaries is None:
            block_boundaries = torch.zeros(B, T, dtype=torch.long, device=logits.device)
        block_trigger = torch.zeros(B, dtype=torch.bool, device=logits.device)
        block_conf_per_block = {}
        for b in range(B):
            unique_blocks = torch.unique(block_boundaries[b])
            for blk_id in unique_blocks:
                mask_b = block_boundaries[b] == blk_id
                blk_conf = conf[b][mask_b].mean()
                block_conf_per_block[(b, int(blk_id))] = float(blk_conf)
                if blk_conf < self.cfg.block_threshold:
                    block_trigger[b] = True

        # global 级
        global_conf = conf.mean(dim=-1)  # [B]
        global_trigger = global_conf < self.cfg.global_threshold  # [B]

        # 综合判定级别
        level = torch.zeros(B, dtype=torch.long, device=logits.device)
        level = torch.where(token_trigger.any(dim=-1), level.maximum(torch.tensor(LEVEL_TOKEN)), level)
        level = torch.where(block_trigger, level.maximum(torch.tensor(LEVEL_BLOCK)), level)
        level = torch.where(global_trigger, level.maximum(torch.tensor(LEVEL_GLOBAL)), level)

        # 形式化证明强制 AR (决策 L4)
        proof_detected = False
        if self.cfg.force_ar_for_proof and prompt is not None:
            proof_detected = any(
                kw in prompt for kw in self.cfg.proof_keywords
            )
            if proof_detected:
                level = level.maximum(torch.tensor(LEVEL_GLOBAL))

        return {
            "level": level,
            "token_confidence": conf,
            "global_confidence": global_conf,
            "token_trigger_mask": token_trigger,
            "block_trigger": block_trigger,
            "block_conf_per_block": block_conf_per_block,
            "proof_detected": proof_detected,
        }

    # ------------------------------------------------------------------
    # AR 生成
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        backbone_fn: Callable[[torch.Tensor], torch.Tensor],
        token_embed_fn: Callable[[torch.Tensor], torch.Tensor],
        prompt_ids: torch.Tensor,
        max_new_tokens: int | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """自回归生成.

        Args:
            backbone_fn: 主干前向 [B,T,H] -> [B,T,H].
            token_embed_fn: token id -> embedding.
            prompt_ids: [B, T_prompt] 提示 token.
            max_new_tokens: 最大新生成 token 数.
        """
        max_new = max_new_tokens or self.cfg.max_new_tokens
        device = device or prompt_ids.device
        ids = prompt_ids.clone()
        for _ in range(max_new):
            emb = token_embed_fn(ids)
            h = backbone_fn(emb)
            h = self.ln_f(h)
            logits = self.lm_head(h[:, -1:, :])  # [B, 1, vocab]
            logits = logits[:, -1, :] / max(self.cfg.temperature, 1e-6)
            # top-k + top-p 采样
            if self.cfg.top_k > 0:
                topk_vals, _ = logits.topk(
                    min(self.cfg.top_k, logits.size(-1)), dim=-1
                )
                logits = torch.where(
                    logits < topk_vals[..., -1:], 
                    torch.full_like(logits, float("-inf")), 
                    logits,
                )
            if self.cfg.top_p < 1.0:
                sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
                cumprobs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                remove = cumprobs > self.cfg.top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                logits = torch.full_like(logits, float("-inf")).scatter(
                    -1, sorted_idx, sorted_logits
                )
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_tok], dim=-1)
        return ids

    # ------------------------------------------------------------------
    # 保底调度入口
    # ------------------------------------------------------------------
    def maybe_fallback(
        self,
        logits: torch.Tensor,
        nar_tokens: torch.Tensor,
        backbone_fn: Callable[[torch.Tensor], torch.Tensor],
        token_embed_fn: Callable[[torch.Tensor], torch.Tensor],
        prompt_ids: torch.Tensor,
        block_boundaries: torch.Tensor | None = None,
        prompt: str | None = None,
        device: torch.device | None = None,
    ) -> dict:
        """根据置信度评估决定是否触发 AR 保底.

        Returns:
            dict 含 final_tokens / level / assessment / ar_used.
        """
        device = device or logits.device
        assessment = self.assess(
            logits, block_boundaries=block_boundaries, prompt=prompt
        )
        level = assessment["level"]
        # 逐样本处理
        B = logits.shape[0]
        final_tokens = []
        ar_used = []
        for b in range(B):
            lvl = int(level[b].item())
            if lvl >= LEVEL_GLOBAL:
                # 整体重写
                prompt_b = prompt_ids[b:b + 1]
                ar_ids = self.generate(
                    backbone_fn, token_embed_fn, prompt_b,
                    device=device,
                )
                # 去掉 prompt 部分, 保留新生成
                new_part = ar_ids[:, prompt_b.shape[1]:]
                final_tokens.append(new_part[0])
                ar_used.append(True)
            elif lvl >= LEVEL_BLOCK and block_boundaries is not None:
                # 重写低置信块 (简化: 全部重写)
                prompt_b = prompt_ids[b:b + 1]
                ar_ids = self.generate(
                    backbone_fn, token_embed_fn, prompt_b,
                    max_new_tokens=nar_tokens.shape[1],
                    device=device,
                )
                new_part = ar_ids[:, prompt_b.shape[1]:]
                final_tokens.append(new_part[0])
                ar_used.append(True)
            elif lvl >= LEVEL_TOKEN:
                # 仅重写低置信 token
                tokens_b = nar_tokens[b].clone()
                trig = assessment["token_trigger_mask"][b]
                if trig.any():
                    # 用 logits argmax 重写 (低成本) + 对连续段用 AR
                    pred = logits[b].argmax(dim=-1)
                    tokens_b[trig] = pred[trig]
                final_tokens.append(tokens_b)
                ar_used.append(False)
            else:
                # 无需保底
                final_tokens.append(nar_tokens[b])
                ar_used.append(False)

        # 对齐长度
        max_len = max(t.shape[0] for t in final_tokens)
        padded = torch.full(
            (B, max_len), 0, dtype=torch.long, device=device
        )
        for b, t in enumerate(final_tokens):
            padded[b, : t.shape[0]] = t
        return {
            "final_tokens": padded,
            "level": level,
            "assessment": assessment,
            "ar_used": ar_used,
        }

    def extra_repr(self) -> str:
        return (
            f"token_threshold={self.cfg.token_threshold}, "
            f"block_threshold={self.cfg.block_threshold}, "
            f"global_threshold={self.cfg.global_threshold}, "
            f"force_ar_for_proof={self.cfg.force_ar_for_proof}"
        )
