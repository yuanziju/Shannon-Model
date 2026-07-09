"""ACT 自适应停止 (Adaptive Computation Time).

实现 Universal Transformers 的 ACT 机制, 允许循环主体在每个 token
位置动态决定何时停止迭代 (spec: ACT自适应停止 + CTM动态损失).

核心机制:
  - 每个 token 维护一个"计算预算" halting_probability
  - 每次迭代产生一个 halt_score ∈ (0, 1)
  - 累计 halting_probability, 达到阈值 (默认 0.99) 则该 token 停止
  - 停止后的迭代不再更新该 token
  - 训练时引入 ponder penalty 鼓励早停

参考: ACT 原论文 (Graves 2016), spec §4.x ACT停止.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ACTStop(nn.Module):
    """ACT 自适应停止模块.

    为循环主体的每次迭代产生 halt_score, 并跟踪每个 token 的累计
    halting probability, 决定何时停止迭代.

    Args:
        hidden_dim: 隐维度.
        threshold: 停止阈值 (默认 0.99).
        penalty_weight: ponder penalty 权重 (鼓励早停).
        max_iters: 最大迭代次数上限 (防止无限循环).
    """

    def __init__(
        self,
        hidden_dim: int,
        threshold: float = 0.99,
        penalty_weight: float = 0.01,
        max_iters: int = 32,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.threshold = float(threshold)
        assert 0.0 < self.threshold <= 1.0, (
            f"threshold 须在 (0, 1], got {self.threshold}"
        )
        self.penalty_weight = float(penalty_weight)
        self.max_iters = max(1, int(max_iters))

        # halt 投影: hidden -> 标量 halt_score
        self.halt_proj = nn.Linear(hidden_dim, 1, bias=True)
        nn.init.zeros_(self.halt_proj.weight)
        nn.init.zeros_(self.halt_proj.bias)

    # ------------------------------------------------------------------
    def compute_halt_score(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """计算每个 token 的 halt_score ∈ (0, 1).

        Args:
            hidden_states: [B, S, H].

        Returns:
            halt_score: [B, S] ∈ (0, 1).
        """
        # sigmoid 保证 ∈ (0, 1)
        return torch.sigmoid(self.halt_proj(hidden_states).squeeze(-1))

    # ------------------------------------------------------------------
    def init_state(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        """初始化 ACT 状态.

        Returns:
            dict 含:
              - halted: [B, S] bool, 已停止的 token.
              - cum_halt: [B, S] 累计 halting probability.
              - remainders: [B, S] 剩余预算 (1 - cum_halt, 用于最后一步补齐).
              - n_updates: [B, S] 已更新次数.
              - pond_cost: [B, S] ponder cost (累计).
        """
        return {
            "halted": torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device),
            "cum_halt": torch.zeros(batch_size, seq_len, dtype=dtype, device=device),
            "remainders": torch.zeros(batch_size, seq_len, dtype=dtype, device=device),
            "n_updates": torch.zeros(batch_size, seq_len, dtype=torch.long, device=device),
            "pond_cost": torch.zeros(batch_size, seq_len, dtype=dtype, device=device),
        }

    # ------------------------------------------------------------------
    def step(
        self,
        hidden_states: torch.Tensor,
        state: Dict[str, torch.Tensor],
        iter_idx: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """执行一次 ACT 步骤.

        Args:
            hidden_states: [B, S, H] 当前隐状态.
            state: ACT 状态 (init_state 返回).
            iter_idx: 当前迭代索引.

        Returns:
            active_mask: [B, S] bool, 本轮仍需继续迭代的 token.
            new_state: 更新后的 ACT 状态.
            ponder_penalty: 标量, 本轮 ponder penalty.
        """
        B, S, H = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        halt_score = self.compute_halt_score(hidden_states)  # [B, S]
        # 已停止的 token 不再增加 halt
        halted = state["halted"]
        cum_halt = state["cum_halt"]
        n_updates = state["n_updates"]
        pond_cost = state["pond_cost"]

        # 新增的 halt (仅对未停止的 token)
        new_halt = torch.where(halted, torch.zeros_like(halt_score), halt_score)
        # 防止累计超过 1
        new_cum = cum_halt + new_halt

        # 判断本轮是否达到阈值
        newly_halted = (new_cum >= self.threshold) & (~halted)
        # 对刚停止的 token, remainder = 1 - cum_halt (补齐到 1)
        remainder = torch.where(
            newly_halted,
            1.0 - cum_halt,
            torch.zeros_like(cum_halt),
        )
        # 更新累计 (停止的 token 补齐到 1)
        final_cum = torch.where(newly_halted, torch.ones_like(new_cum), new_cum)

        # 更新次数 (未停止 + 本轮新增 halt)
        still_active_before = ~halted
        n_updates = n_updates + still_active_before.long()

        # ponder cost: halt_score + remainder (ACT 原始定义)
        step_cost = new_halt + remainder
        pond_cost = pond_cost + step_cost

        # 新的 halted 集合
        new_halted = halted | newly_halted
        # 强制: 最后一轮全部停止
        if iter_idx >= self.max_iters - 1:
            # 未停止的 remainder 补齐
            extra_remainder = torch.where(
                ~new_halted, 1.0 - final_cum, torch.zeros_like(final_cum)
            )
            remainder = remainder + extra_remainder
            final_cum = torch.where(
                ~new_halted, torch.ones_like(final_cum), final_cum
            )
            pond_cost = pond_cost + extra_remainder
            new_halted = torch.ones_like(new_halted)

        new_state = {
            "halted": new_halted,
            "cum_halt": final_cum,
            "remainders": state["remainders"] + remainder,
            "n_updates": n_updates,
            "pond_cost": pond_cost,
        }
        active_mask = ~new_halted
        # ponder penalty (均值)
        ponder_penalty = self.penalty_weight * step_cost.mean()
        return active_mask, new_state, ponder_penalty

    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
        iter_idx: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """便捷前向: 等同于 step."""
        B, S, H = hidden_states.shape
        if state is None:
            state = self.init_state(B, S, hidden_states.device, hidden_states.dtype)
        return self.step(hidden_states, state, iter_idx)

    def get_ponder_cost(
        self, state: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """返回累计 ponder cost (用于训练损失)."""
        return state["pond_cost"]

    def get_n_updates(
        self, state: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """返回每个 token 的更新次数."""
        return state["n_updates"]

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, "
            f"threshold={self.threshold}, "
            f"penalty_weight={self.penalty_weight}, "
            f"max_iters={self.max_iters}"
        )
