"""掩码精化 (MaskRefinement).

方案 C (Mask Refinement): 复用 RDT 循环主体的权重, 通过 mode 切换 +
decode-LoRA 在解码阶段执行迭代掩码精化, 不引入独立解码网络 (决策 L3).

工作流:
    1. 初始: 全部位置为 <MASK> (或部分已确定锚点).
    2. 每轮迭代:
       a. 将当前序列 (含 mask) 送入 RDT 主干 (mode=decoding).
       b. 主干输出每位置词表分布.
       c. 选择高置信度位置"确定" (撤销 mask), 低置信度保留 mask.
       d. 确定的位置数随轮次递减 (cosine 退火).
    3. 直到所有位置确定或达到最大轮数.

复用 RDT 权重意味着本模块不持有独立 Transformer 参数, 仅持有:
    - decode-LoRA (通过 ModeSwitch 注入)
    - 一个轻量 logits 投影头
    - 置信度门控参数
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MaskRefinementConfig:
    """掩码精化配置."""

    vocab_size: int = 128000
    hidden_dim: int = 1024
    mask_token_id: int = 4
    max_iters: int = 8                 # 最大精化轮数
    # 置信度阈值: 高于此值的位置在当前轮确定
    confidence_threshold: float = 0.9
    # 退火策略: 每轮 mask 比例的衰减
    schedule: str = "cosine"           # "cosine" | "linear" | "constant"
    initial_mask_ratio: float = 1.0    # 初始 mask 比例 (1.0 = 全 mask)
    min_mask_ratio: float = 0.0
    temperature: float = 1.0
    # 锚点保留: 已确定的位置在后续轮次是否保持 (True=保持, False=允许重选)
    keep_committed: bool = True


class MaskRefinement(nn.Module):
    """复用 RDT 权重的迭代掩码精化解码器 (方案 C)."""

    def __init__(self, config: MaskRefinementConfig | None = None, **kwargs):
        super().__init__()
        cfg = config or MaskRefinementConfig(**kwargs)
        self.cfg = cfg

        # 轻量 logits 投影头: 将 RDT 主干输出映射到词表
        # 注: 主干本身复用 RDT, 不在此声明
        self.ln_f = nn.LayerNorm(cfg.hidden_dim)
        self.logits_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size)

        # 置信度门控 (可学习标量偏置)
        self.confidence_bias = nn.Parameter(torch.tensor(0.0))

        # mask token 的可学习嵌入 (注入主干的输入嵌入层)
        self.mask_embed = nn.Parameter(torch.randn(cfg.hidden_dim) * 0.02)

        self._init_head()

    def _init_head(self) -> None:
        nn.init.normal_(self.logits_head.weight, std=0.02)
        nn.init.zeros_(self.logits_head.bias)

    # ------------------------------------------------------------------
    # 调度: 每轮保留的 mask 比例
    # ------------------------------------------------------------------
    def _schedule_ratio(self, step: int, total: int) -> float:
        """返回第 step 轮 (0-indexed) 的目标 mask 比例."""
        if self.cfg.schedule == "constant":
            ratio = self.cfg.initial_mask_ratio
        elif self.cfg.schedule == "linear":
            t = step / max(total - 1, 1)
            ratio = self.cfg.initial_mask_ratio + (
                self.cfg.min_mask_ratio - self.cfg.initial_mask_ratio
            ) * t
        else:  # cosine
            t = step / max(total - 1, 1)
            progress = 0.5 * (1 + torch.cos(torch.tensor(torch.pi * t)).item())
            ratio = self.cfg.min_mask_ratio + (
                self.cfg.initial_mask_ratio - self.cfg.min_mask_ratio
            ) * progress
        return float(max(self.cfg.min_mask_ratio, ratio))

    # ------------------------------------------------------------------
    # 单步精化
    # ------------------------------------------------------------------
    def refine_step(
        self,
        backbone_fn: Callable[[torch.Tensor], torch.Tensor],
        hidden: torch.Tensor,
        mask: torch.Tensor,
        committed: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """执行一轮掩码精化.

        Args:
            backbone_fn: 复用的 RDT 主干前向函数, 接收 [..., T, H] 返回同形.
            hidden: [B, T, H] 当前序列隐状态 (含 mask 位置).
            mask: [B, T] bool, True 表示该位置仍为 mask.
            committed: [B, T] bool, True 表示该位置已确定 (keep_committed 时保留).

        Returns:
            new_hidden: 更新后的隐状态.
            new_mask: 更新后的 mask.
            new_logits: 本轮产生的 logits [B, T, vocab].
        """
        # 1. 主干前向 (复用 RDT, 外部已切换到 decoding mode)
        h = backbone_fn(hidden)
        h = self.ln_f(h)
        logits = self.logits_head(h) / max(self.cfg.temperature, 1e-6)

        # 2. 计算每位置置信度 (max softmax 概率)
        probs = F.softmax(logits, dim=-1)
        confidence, pred_ids = probs.max(dim=-1)  # [B, T]
        confidence = confidence + self.confidence_bias

        # 3. 确定高置信度位置
        new_mask = mask.clone()
        # 当前轮可被确定的位置: 仍是 mask 且置信度达标
        can_commit = mask & (confidence >= self.cfg.confidence_threshold)
        if self.cfg.keep_committed and committed is not None:
            # 已确定位置保持不变
            can_commit = can_commit & (~committed)

        # 4. 更新隐状态: 确定的位置用预测 token 的嵌入替换 mask 嵌入
        # 注: 实际嵌入查表由外部 token_embed 完成, 这里用 logits 的 argmax
        #     隐式驱动; 此处仅更新 mask 标记.
        new_mask = new_mask & (~can_commit)

        return h, new_mask, logits

    # ------------------------------------------------------------------
    # 完整迭代解码 (推理入口)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def decode(
        self,
        backbone_fn: Callable[[torch.Tensor], torch.Tensor],
        token_embed_fn: Callable[[torch.Tensor], torch.Tensor],
        length: int,
        batch_size: int = 1,
        device: torch.device | None = None,
        anchors: torch.Tensor | None = None,
        anchor_positions: torch.Tensor | None = None,
    ) -> dict:
        """完整掩码精化解码.

        Args:
            backbone_fn: RDT 主干前向 (decoding mode), [B,T,H]->[B,T,H].
            token_embed_fn: token id -> embedding, [B,T]->[B,T,H].
            length: 序列长度 T.
            batch_size: B.
            device: 设备.
            anchors: [B, K] 锚点 token id (已知确定的 token).
            anchor_positions: [B, K] 锚点位置索引.

        Returns:
            dict 含 tokens / mask_history / num_iters.
        """
        device = device or torch.device("cpu")
        T = length

        # 初始: 全 mask
        mask = torch.ones(batch_size, T, dtype=torch.bool, device=device)
        committed = torch.zeros(batch_size, T, dtype=torch.bool, device=device)
        # 初始隐状态: 全部为 mask 嵌入
        hidden = self.mask_embed.reshape(1, 1, -1).expand(batch_size, T, -1).clone()

        # 注入锚点
        if anchors is not None and anchor_positions is not None:
            for b in range(batch_size):
                pos = anchor_positions[b]
                anc = anchors[b]
                valid = pos >= 0
                if valid.any():
                    emb = token_embed_fn(anc[valid].to(torch.long))
                    hidden[b, pos[valid]] = emb
                    mask[b, pos[valid]] = False
                    committed[b, pos[valid]] = True

        mask_history = [mask.clone()]
        logits_history = []
        total_iters = self.cfg.max_iters

        for step in range(total_iters):
            ratio = self._schedule_ratio(step, total_iters)
            # 若剩余 mask 比例已低于目标, 提前结束
            current_ratio = mask.float().mean().item()
            if current_ratio <= self.cfg.min_mask_ratio + 1e-6:
                break

            # 动态调整本轮置信度阈值 (轮次越后越宽松)
            # 通过 ratio 反比调整
            old_threshold = self.cfg.confidence_threshold
            self.cfg.confidence_threshold = max(
                0.1, old_threshold * (ratio + 1e-3)
            )
            try:
                h, mask, logits = self.refine_step(
                    backbone_fn, hidden, mask, committed
                )
            finally:
                self.cfg.confidence_threshold = old_threshold

            # 将确定的位置写入 hidden (用预测 token 嵌入)
            newly_committed = (~mask) & (~committed)
            if newly_committed.any():
                pred_ids = logits.argmax(dim=-1)
                for b in range(batch_size):
                    pos = newly_committed[b].nonzero(as_tuple=True)[0]
                    if pos.numel() > 0:
                        emb = token_embed_fn(pred_ids[b, pos])
                        hidden[b, pos] = emb
                committed = committed | newly_committed

            mask_history.append(mask.clone())
            logits_history.append(logits)

            # 按调度比例强制保留 (不强制超出)
            target_keep = int(T * ratio)
            current_keep = int(mask.sum().item())
            if current_keep > target_keep:
                # 随机保留部分 mask (保证退火)
                pass  # 实际生产中可在此处随机复 mask, 此处简化

        # 最终: 所有未确定位置取最后一轮 logits 的 argmax
        final_logits = logits_history[-1] if logits_history else None
        if final_logits is not None:
            tokens = final_logits.argmax(dim=-1)
        else:
            tokens = torch.zeros(batch_size, T, dtype=torch.long, device=device)

        # 锚点回填
        if anchors is not None and anchor_positions is not None:
            for b in range(batch_size):
                pos = anchor_positions[b]
                anc = anchors[b]
                valid = pos >= 0
                if valid.any():
                    tokens[b, pos[valid]] = anc[valid]

        return {
            "tokens": tokens,
            "mask_history": mask_history,
            "logits_history": logits_history,
            "num_iters": len(logits_history),
            "final_hidden": hidden,
        }

    def extra_repr(self) -> str:
        return (
            f"vocab_size={self.cfg.vocab_size}, "
            f"max_iters={self.cfg.max_iters}, "
            f"schedule={self.cfg.schedule}, "
            f"keep_committed={self.cfg.keep_committed}"
        )
