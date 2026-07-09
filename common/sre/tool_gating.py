"""工具门控 (ToolGating).

spec §7.1: 在 Cross-Attention Fusion 之后使用动态门控, 决定工具知识
注入主干的程度. 采用 sigmoid 门, 输入为当前隐状态 + 任务类型 + 工具置信度,
输出每层 / 每位置的门值 g ∈ [0, 1].

设计要点:
    - 动态门: 不同位置、不同工具通道有独立门值.
    - 任务感知: 任务类型 (数学/证明/代码/对话) 影响门值.
    - 工具置信度: 工具执行结果的置信度作为门控先验.
    - 稀疏正则: 鼓励门值稀疏 (避免过度依赖工具).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# 任务类型
TASK_TYPES = ("math", "proof", "code", "dialogue", "multimodal")
TASK_TYPE_TO_ID = {t: i for i, t in enumerate(TASK_TYPES)}
NUM_TASK_TYPES = len(TASK_TYPES)

# 工具通道
TOOL_CHANNELS = ("sympy", "lean", "python")
NUM_TOOL_CHANNELS = len(TOOL_CHANNELS)


@dataclass
class ToolGatingConfig:
    """工具门控配置."""

    hidden_dim: int = 1024
    num_heads: int = 16
    # 门控粒度: "token" (每位置) | "block" (每块) | "layer" (每层标量)
    granularity: str = "token"
    # 是否任务感知
    task_aware: bool = True
    # 是否工具置信度感知
    confidence_aware: bool = True
    # 稀疏正则权重
    l1_lambda: float = 0.01
    # 初始偏置 (负值 → 初始接近 0, 鼓励稀疏)
    init_bias: float = -2.0
    # dropout
    dropout: float = 0.1


class ToolGating(nn.Module):
    """动态 sigmoid 工具门控."""

    def __init__(self, config: ToolGatingConfig | None = None, **kwargs):
        super().__init__()
        cfg = config or ToolGatingConfig(**kwargs)
        self.cfg = cfg
        # 门控网络: (hidden + task + confidence) -> gate_logit
        input_dim = cfg.hidden_dim
        if cfg.task_aware:
            input_dim += NUM_TASK_TYPES
        if cfg.confidence_aware:
            input_dim += NUM_TOOL_CHANNELS

        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 1),
        )
        # 初始化最后一层 bias 为负, 初始门值小
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.constant_(self.gate_net[-1].bias, cfg.init_bias)

        # 任务 / 通道嵌入
        if cfg.task_aware:
            self.task_embed = nn.Embedding(NUM_TASK_TYPES, NUM_TASK_TYPES)
            nn.init.eye_(self.task_embed.weight)
        if cfg.confidence_aware:
            self.channel_confidence_proj = nn.Linear(NUM_TOOL_CHANNELS, NUM_TOOL_CHANNELS)
            nn.init.eye_(self.channel_confidence_proj.weight)
            nn.init.zeros_(self.channel_confidence_proj.bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        hidden: torch.Tensor,                       # [B, T, H]
        task_type: torch.Tensor | None = None,      # [B] 任务类型 id
        tool_confidence: torch.Tensor | None = None,  # [B, NUM_TOOL_CHANNELS]
        mask: torch.Tensor | None = None,           # [B, T] valid
        return_logit: bool = False,
    ) -> dict:
        """计算门控值.

        Returns:
            dict 含 gate [B, T, 1] / gate_logit / l1_loss.
        """
        B, T, H = hidden.shape
        features = [hidden]  # [B, T, H]

        if self.cfg.task_aware:
            if task_type is None:
                task_type = torch.zeros(B, dtype=torch.long, device=hidden.device)
            task_emb = self.task_embed(task_type)  # [B, NUM_TASK_TYPES]
            task_emb = task_emb.unsqueeze(1).expand(-1, T, -1)
            features.append(task_emb)

        if self.cfg.confidence_aware:
            if tool_confidence is None:
                tool_confidence = torch.zeros(
                    B, NUM_TOOL_CHANNELS, device=hidden.device
                )
            conf = self.channel_confidence_proj(tool_confidence)  # [B, C]
            conf = conf.unsqueeze(1).expand(-1, T, -1)
            features.append(conf)

        x = torch.cat(features, dim=-1)  # [B, T, input_dim]
        gate_logit = self.gate_net(x)    # [B, T, 1]
        gate = torch.sigmoid(gate_logit)

        if mask is not None:
            gate = gate * mask.unsqueeze(-1).float()

        # L1 稀疏正则
        l1_loss = gate.abs().mean() * self.cfg.l1_lambda

        result = {"gate": gate, "gate_logit": gate_logit, "l1_loss": l1_loss}
        if not return_logit:
            result.pop("gate_logit", None)
        return result

    # ------------------------------------------------------------------
    # 层级门控: 为每个融合层生成门
    # ------------------------------------------------------------------
    def layer_gates(
        self,
        hidden_per_layer: list[torch.Tensor],
        task_type: torch.Tensor | None = None,
        tool_confidence: torch.Tensor | None = None,
        fusion_layers: tuple = (8, 16, 24, 32),
        masks: dict | None = None,
    ) -> dict:
        """为每个融合层生成门控.

        Args:
            hidden_per_layer: 每层隐状态列表 [B, T, H].
            fusion_layers: 融合层索引.

        Returns:
            {layer_idx: gate_tensor [B, T, 1]}
        """
        masks = masks or {}
        gates = {}
        for layer_idx in fusion_layers:
            idx = layer_idx - 1  # 0-indexed
            if idx < 0 or idx >= len(hidden_per_layer):
                continue
            h = hidden_per_layer[idx]
            m = masks.get(layer_idx)
            out = self.forward(h, task_type, tool_confidence, m)
            gates[layer_idx] = out["gate"]
        return gates

    # ------------------------------------------------------------------
    # 推理时强制开关 (调试用)
    # ------------------------------------------------------------------
    def force_gate(
        self,
        hidden: torch.Tensor,
        value: float,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """强制门值 (1.0 全开 / 0.0 全关), 调试用."""
        gate = torch.full(
            hidden.shape[:-1] + (1,), value, device=hidden.device
        )
        if mask is not None:
            gate = gate * mask.unsqueeze(-1).float()
        return gate

    def extra_repr(self) -> str:
        return (
            f"granularity={self.cfg.granularity}, "
            f"task_aware={self.cfg.task_aware}, "
            f"confidence_aware={self.cfg.confidence_aware}, "
            f"l1_lambda={self.cfg.l1_lambda}, "
            f"init_bias={self.cfg.init_bias}"
        )
