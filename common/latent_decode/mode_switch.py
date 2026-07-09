"""模式切换 (ModeSwitch).

隐空间解码与 RDT 循环主体共享权重, 通过两套 LoRA 适配器区分
"reasoning" (内部推理/思考) 与 "decoding" (输出解码) 两种模式.
模式切换通过一个可学习的标量门完成, 支持:

- 硬切换: 直接选择某一模式的 LoRA.
- 软插值: 门控 g ∈ [0,1] 加权融合两套 LoRA.

决策 L3 约束: 方案 C 掩码精化复用 RDT 权重, 不引入独立解码网络,
本模块负责为复用的 RDT 主干提供模式相关的低秩增量.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


MODES = ("reasoning", "decoding")
REASONING = 0
DECODING = 1


@dataclass
class ModeSwitchConfig:
    """模式切换配置."""

    hidden_dim: int = 1024
    lora_rank: int = 32
    lora_alpha: float = 32.0
    lora_dropout: float = 0.0
    gating: str = "soft"   # "soft" | "hard"
    gate_init: float = 0.5  # 初始门控值 (软插值时)
    num_layers: int = 32    # 与 RDT 主干层数对齐


class LoRALinear(nn.Module):
    """单套 LoRA 适配器: W + (alpha/r) * B @ A."""

    def __init__(
        self,
        hidden_dim: int,
        rank: int,
        alpha: float = 32.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / max(rank, 1)

        self.lora_A = nn.Parameter(torch.empty(rank, hidden_dim))
        self.lora_B = nn.Parameter(torch.zeros(hidden_dim, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5.0 / 3)
        # B 初始化为 0, 保证训练初期增量为 0
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回 LoRA 增量 delta = x @ A^T @ B^T * scaling."""
        # x: [..., H]
        d = self.dropout(x)
        # A: [r, H], B: [H, r]
        delta = F.linear(F.linear(d, self.lora_A), self.lora_B)
        return delta * self.scaling


class ModeSwitch(nn.Module):
    """reasoning / decoding 双 LoRA 模式切换.

    为每一层 RDT 主干维护两套 LoRA, 前向时根据当前模式选择或插值.
    """

    def __init__(self, config: ModeSwitchConfig | None = None, **kwargs):
        super().__init__()
        cfg = config or ModeSwitchConfig(**kwargs)
        self.cfg = cfg

        # 每层两套 LoRA: reasoning / decoding
        self.reasoning_loras = nn.ModuleList(
            [
                LoRALinear(
                    cfg.hidden_dim, cfg.lora_rank,
                    cfg.lora_alpha, cfg.lora_dropout,
                )
                for _ in range(cfg.num_layers)
            ]
        )
        self.decoding_loras = nn.ModuleList(
            [
                LoRALinear(
                    cfg.hidden_dim, cfg.lora_rank,
                    cfg.lora_alpha, cfg.lora_dropout,
                )
                for _ in range(cfg.num_layers)
            ]
        )

        # 模式门控 logit (可学习, sigmoid 后表示 decoding 模式权重)
        # gate ∈ (0,1): 0 -> reasoning, 1 -> decoding
        self.gate_logit = nn.Parameter(
            torch.tensor(float(cfg.gate_init))
        )

        # 当前活动模式 (推理时硬切换用), 默认 reasoning
        self.register_buffer(
            "active_mode", torch.tensor(REASONING, dtype=torch.long)
        )

    # ------------------------------------------------------------------
    # 模式管理
    # ------------------------------------------------------------------
    def set_mode(self, mode: str | int) -> None:
        """设置当前活动模式 (硬切换)."""
        if isinstance(mode, str):
            mode_idx = MODES.index(mode)
        else:
            mode_idx = int(mode)
        self.active_mode.fill_(mode_idx)

    def get_mode(self) -> str:
        return MODES[int(self.active_mode.item())]

    def gate(self) -> torch.Tensor:
        """返回当前 decoding 模式权重 g ∈ [0, 1]."""
        return torch.sigmoid(self.gate_logit)

    # ------------------------------------------------------------------
    # 前向: 应用 LoRA 增量
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        layer_idx: int,
        mode: Optional[str | int] = None,
        soft: Optional[bool] = None,
    ) -> torch.Tensor:
        """为指定层计算 LoRA 增量并叠加到 x.

        Args:
            x: [..., hidden_dim] RDT 主干输出.
            layer_idx: RDT 层索引.
            mode: 显式指定模式, 否则使用 active_mode.
            soft: 是否软插值. None 时按 cfg.gating 决定.

        Returns:
            x + lora_delta, 形状不变.
        """
        if layer_idx < 0 or layer_idx >= self.cfg.num_layers:
            # 越界则不施加增量 (保持主干预不变)
            return x

        if soft is None:
            soft = self.cfg.gating == "soft"

        if mode is not None:
            if isinstance(mode, str):
                mode_idx = MODES.index(mode)
            else:
                mode_idx = int(mode)
        else:
            mode_idx = int(self.active_mode.item())

        delta_r = self.reasoning_loras[layer_idx](x)
        delta_d = self.decoding_loras[layer_idx](x)

        if soft:
            g = self.gate().to(x.dtype)
            # g 是 decoding 权重, (1-g) 是 reasoning 权重
            delta = (1 - g) * delta_r + g * delta_d
        else:
            # 硬切换: 根据当前模式选择
            if mode_idx == DECODING:
                delta = delta_d
            else:
                delta = delta_r
        return x + delta

    # ------------------------------------------------------------------
    # 批量切换所有层 (便于整段解码流程切换)
    # ------------------------------------------------------------------
    def apply_to_layers(
        self,
        hidden_states: list[torch.Tensor],
        mode: str | int | None = None,
        soft: Optional[bool] = None,
    ) -> list[torch.Tensor]:
        """对 RDT 主干各层隐状态统一应用模式增量."""
        out = []
        for i, h in enumerate(hidden_states):
            out.append(self.forward(h, layer_idx=i, mode=mode, soft=soft))
        return out

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.cfg.hidden_dim}, "
            f"lora_rank={self.cfg.lora_rank}, "
            f"gating={self.cfg.gating}, "
            f"num_layers={self.cfg.num_layers}"
        )
