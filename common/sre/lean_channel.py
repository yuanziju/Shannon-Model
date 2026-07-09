"""Lean 4 通道 (LeanChannel).

spec §5.6 / §7.1: Lean/Coq 通道使用 Goal-Context Transformer 编码证明状态.
Lean 的证明状态 (Proof State) 包含:

    - 当前目标 (Goal): 待证明的命题类型.
    - 上下文 (Context / Local Hypotheses): 已有的局部假设列表.
    - 证明历史: 已应用的策略 (tactic) 序列.

每个证明状态编码为:
    [HYP_1] [HYP_2] ... [HYP_n] [SEP] [GOAL] [SEP] [TACTIC_HISTORY]

通过 Goal-Context Transformer 编码为证明状态向量, 供:
    - Cross-Attention Fusion 注入主干.
    - 奖励计算 (证明闭合度).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# Lean tactic 词表 (主要策略)
LEAN_TACTICS = (
    "intro", "apply", "exact", "rw", "simp", "induction", "cases",
    "refl", "symm", "trans", "have", "show", "by_contra", "contradiction",
    "left", "right", "constructor", "exists", "use", "fun", "decide",
    "ring", "linarith", "nlinarith", "norm_num", "tauto", "push_neg",
    "sorry", "admit", "by",
)
TACTIC_TO_ID = {t: i for i, t in enumerate(LEAN_TACTICS)}
NUM_TACTICS = len(LEAN_TACTICS)

# 证明状态段类型
SEG_HYP = 0       # 假设
SEG_GOAL = 1      # 目标
SEG_TACTIC = 2    # 策略历史
SEG_SEP = 3       # 分隔
NUM_SEG_TYPES = 4


@dataclass
class LeanChannelConfig:
    """Lean 通道配置."""

    hidden_dim: int = 1024
    num_heads: int = 16
    num_layers: int = 4
    token_vocab_size: int = 32000   # Lean 词表 (子词)
    max_hyps: int = 32              # 最大假设数
    max_goal_length: int = 256      # 目标最大 token 数
    max_tactic_history: int = 64    # 策略历史最大长度
    max_seq_len: int = 1024
    dropout: float = 0.1
    output_dim: int = 1024


class GoalContextTransformer(nn.Module):
    """Goal-Context Transformer: 编码证明状态序列."""

    def __init__(self, cfg: LeanChannelConfig):
        super().__init__()
        self.cfg = cfg
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if mask is not None:
            kpm = ~mask
            return self.transformer(x, src_key_padding_mask=kpm)
        return self.transformer(x)


class LeanChannel(nn.Module):
    """Lean4 通道: Goal-Context Transformer 编码证明状态."""

    def __init__(self, config: LeanChannelConfig | None = None, **kwargs):
        super().__init__()
        cfg = config or LeanChannelConfig(**kwargs)
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.token_vocab_size, cfg.hidden_dim)
        self.seg_embed = nn.Embedding(NUM_SEG_TYPES, cfg.hidden_dim)
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.hidden_dim)
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.transformer = GoalContextTransformer(cfg)
        # 策略分类头 (预测下一步策略)
        self.tactic_head = nn.Linear(cfg.hidden_dim, NUM_TACTICS)
        # 证明闭合度预测 (sigmoid, 1=证明完成)
        self.closure_head = nn.Linear(cfg.hidden_dim, 1)
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.output_dim),
        )
        self.pool_weight = nn.Linear(cfg.hidden_dim, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    # ------------------------------------------------------------------
    # 证明状态序列化
    # ------------------------------------------------------------------
    @staticmethod
    def serialize_state(
        hyps: list[str],
        goal: str,
        tactic_history: list[str],
        tokenizer: Optional[callable] = None,
    ) -> dict:
        """将证明状态序列化为 token id + 段类型序列.

        Args:
            hyps: 局部假设文本列表.
            goal: 当前目标文本.
            tactic_history: 已应用的策略列表.
            tokenizer: 文本 -> token id 列表 函数. None 用 char 级.

        Returns:
            dict 含 token_ids / seg_ids.
        """
        tok = tokenizer or (lambda s: [ord(c) % 32000 for c in s[:256]])
        token_ids = []
        seg_ids = []
        for h in hyps:
            tids = tok(h)
            token_ids.extend(tids)
            seg_ids.extend([SEG_HYP] * len(tids))
            token_ids.append(0)
            seg_ids.append(SEG_SEP)
        # goal
        gids = tok(goal)
        token_ids.extend(gids)
        seg_ids.extend([SEG_GOAL] * len(gids))
        token_ids.append(0)
        seg_ids.append(SEG_SEP)
        # tactic history
        for t in tactic_history:
            tids = tok(t)
            token_ids.extend(tids)
            seg_ids.extend([SEG_TACTIC] * len(tids))
            token_ids.append(0)
            seg_ids.append(SEG_SEP)
        return {"token_ids": token_ids, "seg_ids": seg_ids}

    # ------------------------------------------------------------------
    # 前向
    # ------------------------------------------------------------------
    def forward(
        self,
        token_ids: torch.Tensor,  # [B, N]
        seg_ids: torch.Tensor,    # [B, N]
        mask: torch.Tensor | None = None,
    ) -> dict:
        """编码证明状态.

        Returns:
            dict 含 state_vector / tactic_logits / closure_logit /
                 hidden / attention.
        """
        B, N = token_ids.shape
        positions = torch.arange(N, device=token_ids.device).unsqueeze(0).expand(B, N)
        h = (
            self.token_embed(token_ids)
            + self.seg_embed(seg_ids)
            + self.pos_embed(positions)
        )
        h = self.norm(h)
        h = self.transformer(h, mask)

        # Attention pooling (聚焦 goal 段)
        if mask is not None:
            scores = self.pool_weight(h).squeeze(-1).masked_fill(
                ~mask, float("-inf")
            )
        else:
            scores = self.pool_weight(h).squeeze(-1)
        attn = F.softmax(scores, dim=-1).unsqueeze(-1)
        pooled = (h * attn).sum(dim=1)
        state_vec = self.output_proj(pooled)

        # 辅助头
        tactic_logits = self.tactic_head(pooled)        # [B, NUM_TACTICS]
        closure_logit = self.closure_head(pooled).squeeze(-1)  # [B]

        return {
            "state_vector": state_vec,
            "tactic_logits": tactic_logits,
            "closure_logit": closure_logit,
            "hidden": h,
            "attention": attn.squeeze(-1),
        }

    # ------------------------------------------------------------------
    # 便捷接口
    # ------------------------------------------------------------------
    def encode_state(
        self,
        hyps: list[str],
        goal: str,
        tactic_history: list[str],
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """从证明状态文本直接编码为状态向量 [1, output_dim]."""
        device = device or next(self.parameters()).device
        ser = self.serialize_state(hyps, goal, tactic_history)
        token_ids = torch.tensor([ser["token_ids"]], device=device, dtype=torch.long)
        seg_ids = torch.tensor([ser["seg_ids"]], device=device, dtype=torch.long)
        mask = torch.ones(1, token_ids.shape[1], dtype=torch.bool, device=device)
        out = self.forward(token_ids, seg_ids, mask)
        return out["state_vector"]

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.cfg.hidden_dim}, "
            f"num_layers={self.cfg.num_layers}, "
            f"max_hyps={self.cfg.max_hyps}, "
            f"output_dim={self.cfg.output_dim}"
        )
