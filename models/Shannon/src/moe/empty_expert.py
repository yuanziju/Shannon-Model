"""自学习空专家 (EmptyExpert) + 空专家框架.

实现零初始化的空专家, 支持无需重训的新能力注入:
  - EmptyExpert: 零初始化, 输出为0, 不影响已有专家
  - EmptyExpertFramework: 管理多个空专家, 支持能力吸收与填充

参考: AGENTS.md Agent 9, spec 自学习空专家 (零初始化逐步填充).
决策C10: 空专家保持标准设计, 不使用NLM.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .experts import StandardExpert


class EmptyExpert(nn.Module):
    """零初始化空专家.

    初始时输出为0 (down_proj 权重全零), 不影响已有专家的输出.
    通过能力吸收机制逐步填充, 实现无需重训的新能力注入.
    """

    def __init__(self, hidden_dim: int, ffn_dim: int, expert_id: int = 0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.expert_id = expert_id
        self.ffn = StandardExpert(hidden_dim, ffn_dim)
        # 强制零初始化: down_proj 权重全零, 保证初始输出为0
        nn.init.zeros_(self.ffn.down_proj.weight)
        # 吸收状态
        self.register_buffer("absorbed", torch.tensor(False))
        self.register_buffer("absorb_count", torch.tensor(0, dtype=torch.long))
        # 路由门控 (吸收后启用)
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向计算 (初始输出为0, 吸收后逐步生效)."""
        return torch.sigmoid(self.gate) * self.ffn(x)

    def is_absorbed(self) -> bool:
        return bool(self.absorbed.item())

    def mark_absorbed(self):
        self.absorbed.fill_(True)

    def extra_repr(self) -> str:
        return (
            f"id={self.expert_id}, hidden={self.hidden_dim}, "
            f"ffn={self.ffn_dim}, absorbed={self.is_absorbed()}"
        )


class EmptyExpertFramework(nn.Module):
    """空专家框架: 管理多个空专家, 支持能力吸收.

    功能:
      1. 持有 num_empty_experts 个空专家槽位
      2. 提供能力吸收接口 (从已有专家蒸馏知识到空专家)
      3. 路由时可选使用已吸收的空专家
      4. 防止能力污染 (吸收阈值验证)
    """

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        num_empty_experts: int = 4,
        absorb_threshold: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.num_empty_experts = num_empty_experts
        self.absorb_threshold = absorb_threshold

        self.empty_experts = nn.ModuleList([
            EmptyExpert(hidden_dim, ffn_dim, expert_id=i)
            for i in range(num_empty_experts)
        ])
        # 空专家路由 (仅对已吸收的空专家生效)
        self.empty_router = nn.Linear(hidden_dim, num_empty_experts, bias=False)
        nn.init.normal_(self.empty_router.weight, std=0.02)

    def forward(
        self, x: torch.Tensor, only_absorbed: bool = True
    ) -> tuple:
        """前向: 对已吸收的空专家进行路由.

        Args:
            x: [N, H] token 特征.
            only_absorbed: 是否仅使用已吸收的空专家.

        Returns:
            (output [N, H], aux_loss).
        """
        N, H = x.shape
        output = torch.zeros_like(x)
        aux_loss = torch.tensor(0.0, device=x.device)

        # 检查是否有已吸收的空专家
        absorbed_mask = torch.tensor(
            [e.is_absorbed() for e in self.empty_experts],
            device=x.device, dtype=torch.bool,
        )
        if only_absorbed and not absorbed_mask.any():
            return output, aux_loss

        # 路由
        logits = self.empty_router(x)  # [N, num_empty]
        if only_absorbed:
            # 屏蔽未吸收的空专家
            mask = absorbed_mask.float().unsqueeze(0)  # [1, E]
            logits = logits.masked_fill(~absorbed_mask.unsqueeze(0), float("-inf"))
        scores = F.softmax(logits, dim=-1)  # [N, E]
        top_k = min(2, self.num_empty_experts)
        topk_scores, topk_idx = scores.topk(top_k, dim=-1)
        topk_weights = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-9)

        # 分发到空专家
        for k in range(top_k):
            for ei in range(self.num_empty_experts):
                sel = topk_idx[:, k] == ei
                if not sel.any():
                    continue
                x_sel = x[sel]
                w = topk_weights[sel, k].unsqueeze(-1)
                out_sel = self.empty_experts[ei](x_sel)
                output[sel] += w * out_sel

        return output, aux_loss

    def absorb_capability(
        self,
        source_expert: nn.Module,
        empty_idx: int,
        num_samples: int = 100,
        hidden_dim: int = None,
    ) -> Dict[str, float]:
        """从源专家吸收能力到空专家 (知识蒸馏).

        Args:
            source_expert: 源专家模块 (如 BigExpert 或 StandardExpert).
            empty_idx: 目标空专家索引.
            num_samples: 蒸馏样本数.
            hidden_dim: 输入维度 (None 则用 self.hidden_dim).

        Returns:
            吸收统计 (loss, threshold_met).
        """
        assert 0 <= empty_idx < self.num_empty_experts
        hd = hidden_dim or self.hidden_dim
        target = self.empty_experts[empty_idx]

        # 生成随机输入进行蒸馏
        x = torch.randn(num_samples, hd, device=target.ffn.gate_proj.weight.device)
        with torch.no_grad():
            if hasattr(source_expert, "ffn"):
                target_out = source_expert.ffn(x)
            else:
                target_out = source_expert(x)

        # 蒸馏: 让空专家的FFN逼近源专家输出
        opt = torch.optim.Adam(target.ffn.parameters(), lr=1e-3)
        total_loss = 0.0
        num_steps = 10
        for _ in range(num_steps):
            opt.zero_grad()
            pred = target.ffn(x)
            loss = F.mse_loss(pred, target_out)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())

        avg_loss = total_loss / num_steps
        # 验证吸收质量 (低于阈值则标记为已吸收)
        if avg_loss < self.absorb_threshold:
            target.mark_absorbed()
            target.absorb_count += 1
            # 启用门控
            with torch.no_grad():
                target.gate.fill_(0.5)

        return {
            "avg_loss": avg_loss,
            "threshold_met": avg_loss < self.absorb_threshold,
            "absorbed": target.is_absorbed(),
        }

    def num_absorbed(self) -> int:
        return sum(1 for e in self.empty_experts if e.is_absorbed())

    def extra_repr(self) -> str:
        return (
            f"num_empty={self.num_empty_experts}, "
            f"absorbed={self.num_absorbed()}, "
            f"threshold={self.absorb_threshold}"
        )
