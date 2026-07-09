"""SVG 分词器 (SVGTokenizer) — 矢量图符号化.

将 SVG 路径数据离散化为 token 序列, 实现矢量图的序列化表示.
支持路径命令 (M/L/C/Q/Z 等) 与坐标量化.

参考: spec §3 编码器 SVG 模态, AGENTS.md 多模态位置编码.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.layers import RMSNorm


# SVG 路径命令 token
SVG_COMMANDS = (
    "M", "m",      # moveto
    "L", "l",      # lineto
    "H", "h",      # horizontal lineto
    "V", "v",      # vertical lineto
    "C", "c",      # cubic bezier
    "S", "s",      # smooth cubic
    "Q", "q",      # quadratic bezier
    "T", "t",      # smooth quadratic
    "A", "a",      # arc
    "Z", "z",      # closepath
)
NUM_COMMAND_TOKENS = len(SVG_COMMANDS)

# 特殊 token
SVG_PAD = 0
SVG_BOS = 1
SVG_EOS = 2
SVG_SEP = 3        # 路径分隔符
SVG_NUM_BASE = 4   # 数字 token 起始 id


class SVGTokenizer(nn.Module):
    """SVG 矢量图分词器.

    将 SVG path 字符串离散化为 token 序列, 并提供 embedding.

    token 体系:
      0-3: 特殊 token (PAD/BOS/EOS/SEP)
      4-27: 24 个路径命令 token
      28+: 坐标值 token (量化到 coord_bins 个 bin)

    Args:
        hidden_dim: embedding 输出维度.
        coord_bins: 坐标量化 bin 数 (默认 256, 覆盖 [-1, 1] 区间).
        coord_precision: 坐标小数精度 (用于解析).
        max_paths: 最大路径数.
    """

    def __init__(
        self,
        hidden_dim: int,
        coord_bins: int = 256,
        coord_precision: int = 4,
        max_paths: int = 256,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.coord_bins = max(16, int(coord_bins))
        self.coord_precision = int(coord_precision)
        self.max_paths = max(1, int(max_paths))

        # 词表大小 = 特殊 + 命令 + 坐标 bin
        self.vocab_size = SVG_NUM_BASE + NUM_COMMAND_TOKENS + self.coord_bins
        self.num_command_tokens = NUM_COMMAND_TOKENS

        # token embedding
        self.embed = nn.Embedding(self.vocab_size, hidden_dim, padding_idx=SVG_PAD)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.embed.weight[SVG_PAD].fill_(0.0)

        # 位置编码投影
        self.pos_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.normal_(self.pos_proj.weight, std=0.02)

        self.norm = RMSNorm(hidden_dim)

    # ------------------------------------------------------------------
    def _coord_to_bin(self, coord: float) -> int:
        """将坐标值 [-1, 1] 量化到 bin id."""
        # clip 到 [-1, 1]
        c = max(-1.0, min(1.0, float(coord)))
        # 映射到 [0, coord_bins-1]
        bin_id = int((c + 1.0) / 2.0 * (self.coord_bins - 1))
        return bin_id

    def _bin_to_coord(self, bin_id: int) -> float:
        """将 bin id 反量化为坐标值."""
        c = (float(bin_id) / (self.coord_bins - 1)) * 2.0 - 1.0
        return c

    # ------------------------------------------------------------------
    def tokenize(self, svg_path_str: str) -> List[int]:
        """将 SVG path 字符串解析为 token id 序列.

        Args:
            svg_path_str: SVG path d 属性字符串 (如 "M 0 0 L 1 1 Z").

        Returns:
            token id 列表 (含 BOS/EOS).
        """
        tokens = [SVG_BOS]
        # 正则提取命令与数字
        pattern = r"([MmLlHhVvCcSsQqTtAaZz])|(-?\d+\.?\d*)"
        matches = re.findall(pattern, svg_path_str)
        path_count = 0
        for cmd, num in matches:
            if cmd:
                if cmd in SVG_COMMANDS:
                    cmd_id = SVG_NUM_BASE + SVG_COMMANDS.index(cmd)
                    tokens.append(cmd_id)
                if cmd in ("Z", "z"):
                    path_count += 1
                    tokens.append(SVG_SEP)
                    if path_count >= self.max_paths:
                        break
            elif num:
                bin_id = self._coord_to_bin(float(num))
                tokens.append(SVG_NUM_BASE + self.num_command_tokens + bin_id)
        tokens.append(SVG_EOS)
        return tokens

    def detokenize(self, token_ids: List[int]) -> str:
        """将 token id 序列还原为 SVG path 字符串 (近似)."""
        parts = []
        for tid in token_ids:
            if tid == SVG_BOS or tid == SVG_PAD:
                continue
            if tid == SVG_EOS:
                break
            if tid == SVG_SEP:
                parts.append(" ")
                continue
            if SVG_NUM_BASE <= tid < SVG_NUM_BASE + self.num_command_tokens:
                parts.append(SVG_COMMANDS[tid - SVG_NUM_BASE])
            elif tid >= SVG_NUM_BASE + self.num_command_tokens:
                coord = self._bin_to_coord(tid - SVG_NUM_BASE - self.num_command_tokens)
                parts.append(f"{coord:.{self.coord_precision}f}")
        return " ".join(parts)

    # ------------------------------------------------------------------
    def forward(
        self,
        token_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """SVG token embedding 前向.

        Args:
            token_ids: [B, S] SVG token id.
            position_ids: [B, S] 位置 id.

        Returns:
            [B, S, hidden_dim] SVG embedding.
        """
        B, S = token_ids.shape
        h = self.embed(token_ids)  # [B, S, H]

        # sinusoidal 位置编码
        device = h.device
        dtype = h.dtype
        inv_freq = 1.0 / (
            10000.0 ** (torch.arange(0, self.hidden_dim, 2, device=device, dtype=torch.float32) / self.hidden_dim)
        )
        if position_ids is None:
            pos = torch.arange(S, device=device, dtype=torch.float32)
        else:
            pos = position_ids.float()
        freqs = torch.outer(pos.reshape(-1), inv_freq)  # [S, H/2]
        pos_emb = torch.cat([freqs.sin(), freqs.cos()], dim=-1).to(dtype)
        if position_ids is not None:
            pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1)
        else:
            pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1)
        h = h + self.pos_proj(pos_emb)
        h = self.norm(h)
        return h

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, "
            f"vocab_size={self.vocab_size}, "
            f"coord_bins={self.coord_bins}, "
            f"max_paths={self.max_paths}"
        )
