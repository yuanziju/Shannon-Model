"""投机解码 (SpeculativeDecoder).

NAR draft + AR verify 模式: 利用 NAR (层次化 / 掩码精化) 快速并行生成
草稿 token, 再由 AR 模型一次性验证整段草稿, 接受匹配部分, 拒绝处用 AR
重新采样. 兼顾 NAR 的速度与 AR 的质量.

核心流程 (Leviathan 2023):
    1. Draft 模型并行生成 k 个候选 token: x_1, ..., x_k.
    2. AR 模型对 [prompt, x_1..x_k] 前向, 得到每位置 p_AR(.|prefix).
    3. 对每个 i: 若 x_i ~ q_draft 与 p_AR 一致 (按 r = min(1, p_AR/p_draft)
       概率接受), 否则拒绝并从 (p_AR - p_draft)_+ 重采样.
    4. 接受的 token 直接保留, 第一个拒绝位置之后丢弃.

加速比取决于 draft 命中率. NAR draft 因并行, draft 阶段近乎零成本.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SpeculativeDecoderConfig:
    """投机解码配置."""

    vocab_size: int = 128000
    hidden_dim: int = 1024
    max_draft_tokens: int = 8         # 单轮 draft token 数 k
    max_iters: int = 64               # 最大 verify 轮数
    temperature: float = 1.0
    # 接受策略: "strict" (逐 token) | "block" (块级全接受/拒绝)
    accept_strategy: str = "strict"
    # draft 模型置信度门控: 过低则放弃 draft 直接 AR
    draft_min_confidence: float = 0.3


class SpeculativeDecoder(nn.Module):
    """NAR draft + AR verify 投机解码器."""

    def __init__(self, config: SpeculativeDecoderConfig | None = None, **kwargs):
        super().__init__()
        cfg = config or SpeculativeDecoderConfig(**kwargs)
        self.cfg = cfg
        # AR 验证用 logits 头 (复用主模型 head)
        self.ar_ln = nn.LayerNorm(cfg.hidden_dim)
        self.ar_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)
        nn.init.normal_(self.ar_head.weight, std=0.02)

    # ------------------------------------------------------------------
    # 单轮 draft + verify
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _verify_round(
        self,
        ar_backbone_fn: Callable[[torch.Tensor], torch.Tensor],
        ar_token_embed_fn: Callable[[torch.Tensor], torch.Tensor],
        prompt_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        draft_logits: torch.Tensor | None = None,
        device: torch.device | None = None,
    ) -> dict:
        """对一批 draft token 执行 AR 验证.

        Args:
            ar_backbone_fn: AR 主干前向.
            ar_token_embed_fn: token -> embed.
            prompt_ids: [B, P] 已确认的 prompt.
            draft_tokens: [B, k] draft 候选.
            draft_logits: [B, k, vocab] draft 模型 logits (用于拒绝采样).
        """
        device = device or prompt_ids.device
        B, P = prompt_ids.shape
        k = draft_tokens.shape[1]

        # 拼接: [prompt; draft]
        full_ids = torch.cat([prompt_ids, draft_tokens], dim=1)
        emb = ar_token_embed_fn(full_ids)
        h = ar_backbone_fn(emb)
        h = self.ar_ln(h)
        ar_logits = self.ar_head(h)  # [B, P+k, vocab]

        # 取 prompt 末尾位置的 logits 作为第 1 个 draft 的验证分布
        # ar_logits[:, P-1] 验证 draft[0], ar_logits[:, P] 验证 draft[1], ...
        ar_verify_logits = ar_logits[:, P - 1: P - 1 + k, :]  # [B, k, vocab]
        ar_probs = F.softmax(
            ar_verify_logits / max(self.cfg.temperature, 1e-6), dim=-1
        )  # [B, k, vocab]

        # draft 概率
        if draft_logits is not None:
            draft_probs = F.softmax(
                draft_logits / max(self.cfg.temperature, 1e-6), dim=-1
            )
        else:
            # 无 draft logits 时, 假设 draft 是确定性 argmax (one-hot)
            draft_probs = F.one_hot(
                draft_tokens, num_classes=self.cfg.vocab_size
            ).float()

        # 逐 token 接受/拒绝
        accepted_mask = torch.zeros(B, k, dtype=torch.bool, device=device)
        resample_tokens = torch.full(
            (B, k), -1, dtype=torch.long, device=device
        )
        first_reject = torch.full((B,), k, dtype=torch.long, device=device)

        for b in range(B):
            reject_seen = False
            for i in range(k):
                if reject_seen:
                    break
                tok = draft_tokens[b, i].item()
                p_ar = ar_probs[b, i, tok].item()
                p_draft = draft_probs[b, i, tok].item()
                # 接受概率 r = min(1, p_ar / p_draft)
                if p_draft <= 0:
                    r = 1.0
                else:
                    r = min(1.0, p_ar / max(p_draft, 1e-8))
                # 按 r 接受
                if torch.rand(1, device=device).item() < r:
                    accepted_mask[b, i] = True
                else:
                    reject_seen = True
                    first_reject[b] = i
                    # 从 (p_ar - p_draft)_+ 重采样
                    diff = (ar_probs[b, i] - draft_probs[b, i]).clamp(min=0)
                    if diff.sum() > 0:
                        diff_norm = diff / diff.sum()
                        resample = torch.multinomial(diff_norm, num_samples=1)
                        resample_tokens[b, i] = resample[0]
                    else:
                        # 退化: 直接取 AR argmax
                        resample_tokens[b, i] = ar_probs[b, i].argmax()

        # bonus token: 若全部接受, 用 AR 在最后位置生成 1 个 bonus
        bonus_tokens = torch.full((B,), -1, dtype=torch.long, device=device)
        for b in range(B):
            if first_reject[b] == k:
                # 全部接受, bonus = AR 在末尾位置的采样
                bonus_logits = ar_logits[b, P - 1 + k - 1] / max(
                    self.cfg.temperature, 1e-6
                )
                bonus_tokens[b] = torch.multinomial(
                    F.softmax(bonus_logits, dim=-1), num_samples=1
                )[0]

        return {
            "ar_verify_logits": ar_verify_logits,
            "accepted_mask": accepted_mask,
            "first_reject_idx": first_reject,
            "resample_tokens": resample_tokens,
            "bonus_tokens": bonus_tokens,
        }

    # ------------------------------------------------------------------
    # 完整投机解码 (推理入口)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        ar_backbone_fn: Callable[[torch.Tensor], torch.Tensor],
        ar_token_embed_fn: Callable[[torch.Tensor], torch.Tensor],
        draft_fn: Callable[[torch.Tensor, int], tuple[torch.Tensor, torch.Tensor]],
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 128,
        device: torch.device | None = None,
    ) -> dict:
        """投机解码主循环.

        Args:
            ar_backbone_fn: AR 主干.
            ar_token_embed_fn: token embed.
            draft_fn: (prompt_ids, k) -> (draft_tokens, draft_logits)
                      NAR 草稿生成函数.
            prompt_ids: [B, P].
            max_new_tokens: 目标新生成 token 数.
        """
        device = device or prompt_ids.device
        B = prompt_ids.shape[0]
        k = self.cfg.max_draft_tokens
        all_new = [[] for _ in range(B)]
        total = 0
        iters = 0
        stats = {"accepted": 0, "rejected": 0, "iters": 0}

        cur_prompt = prompt_ids.clone()
        while total < max_new_tokens and iters < self.cfg.max_iters:
            iters += 1
            # 1. NAR draft
            draft_tokens, draft_logits = draft_fn(cur_prompt, k)
            # 2. AR verify
            result = self._verify_round(
                ar_backbone_fn, ar_token_embed_fn,
                cur_prompt, draft_tokens, draft_logits, device,
            )
            accepted = result["accepted_mask"]
            first_reject = result["first_reject_idx"]
            resample = result["resample_tokens"]
            bonus = result["bonus_tokens"]

            # 3. 组装本轮接受的 token
            for b in range(B):
                fr = int(first_reject[b].item())
                # 接受 0..fr-1
                accepted_tokens = draft_tokens[b, :fr].tolist()
                # 拒绝位置用 resample
                if fr < k:
                    rt = resample[b, fr].item()
                    if rt >= 0:
                        accepted_tokens.append(rt)
                # 全部接受则加 bonus
                if fr == k:
                    bt = bonus[b].item()
                    if bt >= 0:
                        accepted_tokens.append(bt)
                all_new[b].extend(accepted_tokens)
                # 更新统计
                stats["accepted"] += fr
                stats["rejected"] += (k - fr) if fr < k else 0

            # 4. 更新 prompt (附加本轮新 token)
            max_round = max(len(all_new[b]) for b in range(B))
            total = max_round
            # 构造下一轮 prompt: prompt + 已生成
            new_ids = torch.full(
                (B, max_round), 0, dtype=torch.long, device=device
            )
            for b in range(B):
                nt = all_new[b][:max_round]
                if nt:
                    new_ids[b, : len(nt)] = torch.tensor(nt, device=device)
            cur_prompt = torch.cat([prompt_ids, new_ids], dim=1)

        stats["iters"] = iters
        # 截断到 max_new_tokens
        final_new = []
        for b in range(B):
            final_new.append(all_new[b][:max_new_tokens])
        max_len = max(len(t) for t in final_new)
        out = torch.full((B, max_len), 0, dtype=torch.long, device=device)
        for b, t in enumerate(final_new):
            out[b, : len(t)] = torch.tensor(t, device=device)

        return {
            "tokens": out,
            "stats": stats,
            "accept_rate": (
                stats["accepted"] / max(stats["accepted"] + stats["rejected"], 1)
            ),
        }

    def extra_repr(self) -> str:
        return (
            f"max_draft_tokens={self.cfg.max_draft_tokens}, "
            f"accept_strategy={self.cfg.accept_strategy}, "
            f"draft_min_confidence={self.cfg.draft_min_confidence}"
        )
