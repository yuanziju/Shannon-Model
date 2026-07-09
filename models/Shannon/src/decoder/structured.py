"""结构化输出 (StructuredOutput) — JSON / 工具调用 / TTS 多任务头.

将循环主体输出的隐状态解码为结构化数据:
  - JSON: 结构化 JSON 对象 (键值对序列化为 token)
  - 工具调用: Function Calling 格式 (name + arguments)
  - TTS: 文本到语音的声学特征

每种结构化输出有独立的轻量投影头, 共享主模型隐状态.

参考: spec §9 多任务输出头 (文本/SVG/工具/TTS), ReAct+CRA Agent架构.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.layers import RMSNorm


# 结构化输出类型
STRUCT_TEXT = 0
STRUCT_JSON = 1
STRUCT_TOOL = 2
STRUCT_TTS = 3
NUM_STRUCT_TYPES = 4


class StructuredOutput(nn.Module):
    """结构化输出多任务头.

    在主 lm_head 之上, 提供额外的结构化输出投影:
      - json_head: JSON schema 约束输出
      - tool_head: 工具调用分类 (name) + 参数 JSON
      - tts_head: 声学特征回归

    Args:
        hidden_dim: 模型隐维度.
        vocab_size: 主词表大小 (JSON/工具名复用文本词表).
        tool_vocab_size: 工具名词表大小.
        tts_dim: TTS 声学特征维度.
        num_struct_types: 结构化类型数.
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        tool_vocab_size: int = 512,
        tts_dim: int = 80,
        num_struct_types: int = NUM_STRUCT_TYPES,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tool_vocab_size = tool_vocab_size
        self.tts_dim = tts_dim
        self.num_struct_types = num_struct_types

        # 类型分类器: 判断输出应为哪种结构化类型
        self.type_classifier = nn.Linear(hidden_dim, num_struct_types)
        nn.init.normal_(self.type_classifier.weight, std=0.02)

        # 共享归一化
        self.norm = RMSNorm(hidden_dim)

        # JSON 输出头: 复用主词表, 但有独立的 schema 约束投影
        self.json_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        nn.init.normal_(self.json_head.weight, std=0.02 / (hidden_dim ** 0.5))

        # 工具调用头: 工具名分类 + 参数 JSON (复用 json_head)
        self.tool_name_head = nn.Linear(hidden_dim, tool_vocab_size)
        nn.init.normal_(self.tool_name_head.weight, std=0.02)

        # TTS 声学特征回归头
        self.tts_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, tts_dim),
        )
        nn.init.normal_(self.tts_head[-1].weight, std=0.02)

        # 工具名词表占位 (运行时由外部填充)
        self.tool_names: List[str] = []

    # ------------------------------------------------------------------
    def register_tools(self, tool_names: List[str]) -> None:
        """注册可用工具名 (用于工具调用解码)."""
        self.tool_names = list(tool_names)

    def _tool_id_to_name(self, tid: int) -> str:
        if 0 <= tid < len(self.tool_names):
            return self.tool_names[tid]
        return f"<tool_{tid}>"

    # ------------------------------------------------------------------
    def forward(
        self,
        hidden: torch.Tensor,
        struct_type: Optional[int] = None,
        labels: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """结构化输出前向.

        Args:
            hidden: [B, S, H] 循环主体输出.
            struct_type: 显式指定结构化类型 (None 则自动分类).
            labels: 训练标签 dict, 可含:
              - "type": [B] 结构化类型标签.
              - "json_ids": [B, S] JSON token 标签.
              - "tool_name": [B] 工具名 id 标签.
              - "tts_target": [B, S, tts_dim] 声学特征标签.

        Returns:
            dict 含各头的 logits / 预测, 以及 loss (训练).
        """
        h = self.norm(hidden)
        B, S, H = h.shape

        # 类型分类
        type_logits = self.type_classifier(h.mean(dim=1))  # [B, num_types]
        if struct_type is not None:
            # 强制类型
            type_pred = torch.full(
                (B,), struct_type, dtype=torch.long, device=h.device
            )
        else:
            type_pred = type_logits.argmax(dim=-1)  # [B]

        result: Dict[str, torch.Tensor] = {
            "type_logits": type_logits,
            "type_pred": type_pred,
        }

        total_loss = torch.zeros((), device=h.device, dtype=h.dtype)

        # JSON 头 (始终计算, 用于文本/JSON token)
        json_logits = self.json_head(h)  # [B, S, vocab]
        result["json_logits"] = json_logits

        # 工具名头
        tool_logits = self.tool_name_head(h.mean(dim=1))  # [B, tool_vocab]
        result["tool_logits"] = tool_logits

        # TTS 头
        tts_feats = self.tts_head(h)  # [B, S, tts_dim]
        result["tts_feats"] = tts_feats

        # ---- 计算损失 ----
        if labels is not None:
            # 类型分类 loss
            if "type" in labels:
                type_loss = F.cross_entropy(type_logits, labels["type"])
                total_loss = total_loss + type_loss
                result["type_loss"] = type_loss

            # JSON token loss
            if "json_ids" in labels:
                json_labels = labels["json_ids"]
                shift_logits = json_logits[..., :-1, :].contiguous()
                shift_labels = json_labels[..., 1:].contiguous()
                json_loss = F.cross_entropy(
                    shift_logits.view(-1, self.vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
                total_loss = total_loss + json_loss
                result["json_loss"] = json_loss

            # 工具名 loss
            if "tool_name" in labels:
                tool_loss = F.cross_entropy(
                    tool_logits, labels["tool_name"]
                )
                total_loss = total_loss + tool_loss
                result["tool_loss"] = tool_loss

            # TTS 回归 loss
            if "tts_target" in labels:
                tts_loss = F.mse_loss(tts_feats, labels["tts_target"])
                total_loss = total_loss + tts_loss
                result["tts_loss"] = tts_loss

            result["loss"] = total_loss

        return result

    # ------------------------------------------------------------------
    @torch.no_grad()
    def decode_tool_call(
        self,
        hidden: torch.Tensor,
        token_detokenize_fn=None,
    ) -> List[Dict[str, Any]]:
        """解码工具调用.

        Args:
            hidden: [B, S, H].
            token_detokenize_fn: token id -> str 函数 (用于参数 JSON 解析).

        Returns:
            list of dict, 每个元素含 "name" 和 "arguments".
        """
        h = self.norm(hidden)
        B = h.shape[0]

        # 工具名
        tool_logits = self.tool_name_head(h.mean(dim=1))
        tool_ids = tool_logits.argmax(dim=-1)  # [B]

        # 参数 JSON (用 json_head 解码 token)
        json_logits = self.json_head(h)  # [B, S, vocab]
        arg_ids = json_logits.argmax(dim=-1)  # [B, S]

        results = []
        for b in range(B):
            name = self._tool_id_to_name(int(tool_ids[b].item()))
            if token_detokenize_fn is not None:
                arg_str = token_detokenize_fn(arg_ids[b])
                try:
                    arguments = json.loads(arg_str)
                except (json.JSONDecodeError, TypeError):
                    arguments = {"raw": arg_str}
            else:
                arguments = {}
            results.append({"name": name, "arguments": arguments})
        return results

    def extra_repr(self) -> str:
        return (
            f"hidden={self.hidden_dim}, vocab={self.vocab_size}, "
            f"tool_vocab={self.tool_vocab_size}, tts_dim={self.tts_dim}"
        )
