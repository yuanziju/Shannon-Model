"""Python 通道 (PythonChannel).

spec §5.6 / §7.1: Python 通道使用多类型编码器处理 stdout / stderr /
DataFrame / plot / exception 等多种执行输出, 统一编码为执行结果向量.

支持的输出类型:
    - stdout: 标准输出文本 (tokenize + Transformer).
    - stderr: 错误输出 (含 traceback).
    - DataFrame: 表格数据 (行/列 schema + 采样行).
    - plot: matplotlib 图像 (降采样为低分辨率 patch).
    - exception: 异常类型 + 消息.
    - value: 直接返回值 (repr).
    - None: 无输出.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# 输出类型枚举
OUTPUT_TYPES = (
    "stdout", "stderr", "dataframe", "plot",
    "exception", "value", "none",
)
OUTPUT_TYPE_TO_ID = {t: i for i, t in enumerate(OUTPUT_TYPES)}
NUM_OUTPUT_TYPES = len(OUTPUT_TYPES)


@dataclass
class PythonChannelConfig:
    """Python 通道配置."""

    hidden_dim: int = 1024
    num_heads: int = 16
    num_layers: int = 4
    token_vocab_size: int = 32000
    max_text_length: int = 512        # stdout/stderr 最大 token
    max_rows: int = 16                # DataFrame 采样行
    max_cols: int = 32                # DataFrame 最大列
    plot_patch_size: int = 16         # plot 图像 patch
    plot_num_patches: int = 16        # plot patch 数
    dropout: float = 0.1
    output_dim: int = 1024


class TextEncoder(nn.Module):
    """文本输出编码器 (stdout / stderr / value repr)."""

    def __init__(self, cfg: PythonChannelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.token_vocab_size, cfg.hidden_dim)
        self.pos_embed = nn.Embedding(cfg.max_text_length, cfg.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.norm = nn.LayerNorm(cfg.hidden_dim)

    def forward(
        self, token_ids: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, N = token_ids.shape
        pos = torch.arange(N, device=token_ids.device).unsqueeze(0).expand(B, N)
        h = self.token_embed(token_ids) + self.pos_embed(pos)
        h = self.norm(h)
        if mask is not None:
            h = self.transformer(h, src_key_padding_mask=~mask)
        else:
            h = self.transformer(h)
        return h


class DataFrameEncoder(nn.Module):
    """DataFrame 编码器: schema + 采样行."""

    def __init__(self, cfg: PythonChannelConfig):
        super().__init__()
        self.cfg = cfg
        self.col_embed = nn.Embedding(cfg.max_cols, cfg.hidden_dim)
        self.row_embed = nn.Embedding(cfg.max_rows, cfg.hidden_dim)
        self.cell_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)

    def forward(
        self, cells: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """cells: [B, R, C, H] 表格 cell 嵌入."""
        B, R, C, H = cells.shape
        # 加行列位置编码
        row_pos = self.row_embed(torch.arange(R, device=cells.device))  # [R, H]
        col_pos = self.col_embed(torch.arange(C, device=cells.device))  # [C, H]
        h = cells + row_pos.reshape(1, R, 1, H) + col_pos.reshape(1, 1, C, H)
        h = self.cell_proj(h)
        h = self.norm(h)
        # 展平为序列 [B, R*C, H]
        h = h.reshape(B, R * C, H)
        h = self.transformer(h)
        return h


class PlotEncoder(nn.Module):
    """Plot 图像编码器: patch embedding."""

    def __init__(self, cfg: PythonChannelConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_proj = nn.Linear(
            3 * cfg.plot_patch_size * cfg.plot_patch_size, cfg.hidden_dim
        )
        self.pos_embed = nn.Embedding(cfg.plot_num_patches, cfg.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """patches: [B, P, 3, ps, ps] 图像 patch."""
        B, P, C, ps, ps = patches.shape
        flat = patches.reshape(B, P, -1)
        h = self.patch_proj(flat)
        pos = torch.arange(P, device=patches.device).unsqueeze(0).expand(B, P)
        h = h + self.pos_embed(pos)
        h = self.transformer(h)
        return h


class PythonChannel(nn.Module):
    """Python 通道: 多类型编码器统一处理执行输出."""

    def __init__(self, config: PythonChannelConfig | None = None, **kwargs):
        super().__init__()
        cfg = config or PythonChannelConfig(**kwargs)
        self.cfg = cfg
        self.text_encoder = TextEncoder(cfg)
        self.dataframe_encoder = DataFrameEncoder(cfg)
        self.plot_encoder = PlotEncoder(cfg)
        # 类型嵌入 + 输出投影
        self.type_embed = nn.Embedding(NUM_OUTPUT_TYPES, cfg.hidden_dim)
        # 跨类型融合 (各类型 pooled + 类型嵌入 -> 统一)
        self.fusion = nn.Sequential(
            nn.Linear(cfg.hidden_dim * NUM_OUTPUT_TYPES, cfg.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim * 2, cfg.output_dim),
        )
        self.pool_weight = nn.Linear(cfg.hidden_dim, 1)
        # 异常严重度预测
        self.severity_head = nn.Linear(cfg.hidden_dim, 3)  # info/warning/error
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
    def _pool(self, h: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is not None:
            scores = self.pool_weight(h).squeeze(-1).masked_fill(
                ~mask, float("-inf")
            )
        else:
            scores = self.pool_weight(h).squeeze(-1)
        attn = F.softmax(scores, dim=-1).unsqueeze(-1)
        return (h * attn).sum(dim=1), attn.squeeze(-1)

    # ------------------------------------------------------------------
    # 多类型前向
    # ------------------------------------------------------------------
    def forward(
        self,
        stdout_ids: torch.Tensor | None = None,
        stderr_ids: torch.Tensor | None = None,
        value_ids: torch.Tensor | None = None,
        dataframe_cells: torch.Tensor | None = None,
        plot_patches: torch.Tensor | None = None,
        exception_type_id: torch.Tensor | None = None,
        masks: dict | None = None,
    ) -> dict:
        """多类型执行输出编码.

        每个参数可选, 缺省类型用零向量占位. masks: {"stdout":..., "stderr":...}.
        """
        device = next(self.parameters()).device
        B = 1
        if stdout_ids is not None:
            B = stdout_ids.shape[0]
        elif dataframe_cells is not None:
            B = dataframe_cells.shape[0]
        elif plot_patches is not None:
            B = plot_patches.shape[0]

        masks = masks or {}
        pooled_list = []

        # stdout
        if stdout_ids is not None:
            h = self.text_encoder(stdout_ids, masks.get("stdout"))
            p, _ = self._pool(h, masks.get("stdout"))
            pooled_list.append(p + self.type_embed(torch.tensor(OUTPUT_TYPE_TO_ID["stdout"], device=device)).expand(B, -1))
        else:
            pooled_list.append(self.type_embed(torch.tensor(OUTPUT_TYPE_TO_ID["none"], device=device)).expand(B, -1))

        # stderr
        if stderr_ids is not None:
            h = self.text_encoder(stderr_ids, masks.get("stderr"))
            p, _ = self._pool(h, masks.get("stderr"))
            pooled_list.append(p + self.type_embed(torch.tensor(OUTPUT_TYPE_TO_ID["stderr"], device=device)).expand(B, -1))
        else:
            pooled_list.append(self.type_embed(torch.tensor(OUTPUT_TYPE_TO_ID["none"], device=device)).expand(B, -1))

        # value
        if value_ids is not None:
            h = self.text_encoder(value_ids, masks.get("value"))
            p, _ = self._pool(h, masks.get("value"))
            pooled_list.append(p + self.type_embed(torch.tensor(OUTPUT_TYPE_TO_ID["value"], device=device)).expand(B, -1))
        else:
            pooled_list.append(self.type_embed(torch.tensor(OUTPUT_TYPE_TO_ID["none"], device=device)).expand(B, -1))

        # dataframe
        if dataframe_cells is not None:
            h = self.dataframe_encoder(dataframe_cells)
            p, _ = self._pool(h)
            pooled_list.append(p + self.type_embed(torch.tensor(OUTPUT_TYPE_TO_ID["dataframe"], device=device)).expand(B, -1))
        else:
            pooled_list.append(self.type_embed(torch.tensor(OUTPUT_TYPE_TO_ID["none"], device=device)).expand(B, -1))

        # plot
        if plot_patches is not None:
            h = self.plot_encoder(plot_patches)
            p, _ = self._pool(h)
            pooled_list.append(p + self.type_embed(torch.tensor(OUTPUT_TYPE_TO_ID["plot"], device=device)).expand(B, -1))
        else:
            pooled_list.append(self.type_embed(torch.tensor(OUTPUT_TYPE_TO_ID["none"], device=device)).expand(B, -1))

        # exception
        if exception_type_id is not None:
            # exception_type_id: [B] 异常类型 id
            exc_embed = self.type_embed(torch.tensor(OUTPUT_TYPE_TO_ID["exception"], device=device)).expand(B, -1)
            pooled_list.append(exc_embed + exception_type_id.float().unsqueeze(-1).expand(-1, self.cfg.hidden_dim) * 0.01)
        else:
            pooled_list.append(self.type_embed(torch.tensor(OUTPUT_TYPE_TO_ID["none"], device=device)).expand(B, -1))

        # None type
        pooled_list.append(self.type_embed(torch.tensor(OUTPUT_TYPE_TO_ID["none"], device=device)).expand(B, -1))

        # 融合: 拼接所有 pooled + 类型嵌入
        concat = torch.cat(pooled_list, dim=-1)  # [B, hidden_dim * NUM_OUTPUT_TYPES]
        result_vec = self.fusion(concat)
        severity_logits = self.severity_head(result_vec)
        return {
            "result_vector": result_vec,
            "severity_logits": severity_logits,
            "pooled_per_type": pooled_list,
        }

    # ------------------------------------------------------------------
    # 便捷接口: 从执行结果字典编码
    # ------------------------------------------------------------------
    def encode_result(
        self,
        result: dict,
        tokenizer: Optional[callable] = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """从执行结果字典编码.

        result 支持键: stdout / stderr / value / dataframe / plot / exception.
        """
        device = device or next(self.parameters()).device
        tok = tokenizer or (lambda s: [ord(c) % 32000 for c in str(s)[:512]])
        kwargs = {}
        masks = {}
        if "stdout" in result and result["stdout"]:
            ids = tok(result["stdout"])
            kwargs["stdout_ids"] = torch.tensor([ids], device=device, dtype=torch.long)
            masks["stdout"] = torch.ones(1, len(ids), dtype=torch.bool, device=device)
        if "stderr" in result and result["stderr"]:
            ids = tok(result["stderr"])
            kwargs["stderr_ids"] = torch.tensor([ids], device=device, dtype=torch.long)
            masks["stderr"] = torch.ones(1, len(ids), dtype=torch.bool, device=device)
        if "value" in result and result["value"] is not None:
            ids = tok(repr(result["value"]))
            kwargs["value_ids"] = torch.tensor([ids], device=device, dtype=torch.long)
            masks["value"] = torch.ones(1, len(ids), dtype=torch.bool, device=device)
        if "exception" in result and result["exception"]:
            kwargs["exception_type_id"] = torch.tensor(
                [abs(hash(result["exception"])) % 100], device=device
            )
        if masks:
            kwargs["masks"] = masks
        out = self.forward(**kwargs)
        return out["result_vector"]

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.cfg.hidden_dim}, "
            f"output_types={OUTPUT_TYPES}, "
            f"output_dim={self.cfg.output_dim}"
        )
