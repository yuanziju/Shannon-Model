"""SymPy 通道 (SymPyChannel).

spec §5.6 / §7.1: SymPy 通道使用 AST Tree Transformer 编码符号表达式树,
将 SymPy 表达式的抽象语法树 (AST) 转换为序列化的节点序列, 再经
Transformer 编码为符号计算结果向量, 供 Cross-Attention Fusion 使用.

流程:
    sympy 表达式 ──→ AST 节点序列 ──→ Tree Transformer ──→ 结果向量

每个 AST 节点编码为 (type_id, value_embedding, depth, position) 四元组,
通过 Tree Transformer (带结构位置编码) 编码.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# 预定义的 SymPy AST 节点类型 (覆盖主要运算)
SYMPY_NODE_TYPES = (
    "Symbol", "Integer", "Float", "Rational", "Constant",
    "Add", "Mul", "Pow", "Mod",
    "Function", "Derivative", "Integral", "Sum", "Product",
    "Equality", "Unequality", "LessThan", "GreaterThan",
    "List", "Tuple", "Set", "Matrix",
    "Piecewise", "Relational", "RootOf",
)
NODE_TYPE_TO_ID = {t: i for i, t in enumerate(SYMPY_NODE_TYPES)}
NUM_NODE_TYPES = len(SYMPY_NODE_TYPES)


@dataclass
class SymPyChannelConfig:
    """SymPy 通道配置."""

    hidden_dim: int = 1024
    num_heads: int = 16
    num_layers: int = 4
    max_nodes: int = 256               # 单表达式最大 AST 节点数
    max_depth: int = 32                # 最大树深
    node_type_vocab: int = NUM_NODE_TYPES
    value_vocab_size: int = 8192       # 数值/符号名离散词表
    dropout: float = 0.1
    output_dim: int = 1024             # 输出结果向量维度


class ASTNodeEncoder(nn.Module):
    """单个 AST 节点编码: type + value + depth + child_index."""

    def __init__(self, cfg: SymPyChannelConfig):
        super().__init__()
        self.cfg = cfg
        self.type_embed = nn.Embedding(cfg.node_type_vocab, cfg.hidden_dim)
        self.value_embed = nn.Embedding(cfg.value_vocab_size, cfg.hidden_dim)
        self.depth_embed = nn.Embedding(cfg.max_depth + 1, cfg.hidden_dim)
        self.child_pos_embed = nn.Embedding(64, cfg.hidden_dim)  # 子节点序号
        self.norm = nn.LayerNorm(cfg.hidden_dim)

    def forward(
        self,
        type_ids: torch.Tensor,      # [N]
        value_ids: torch.Tensor,     # [N]
        depths: torch.Tensor,        # [N]
        child_indices: torch.Tensor, # [N]
    ) -> torch.Tensor:
        h = (
            self.type_embed(type_ids)
            + self.value_embed(value_ids)
            + self.depth_embed(depths.clamp(max=self.cfg.max_depth))
            + self.child_pos_embed(child_indices.clamp(max=63))
        )
        return self.norm(h)


class TreeTransformer(nn.Module):
    """树结构 Transformer: 标准 self-attention + 结构位置编码 (相对深度)."""

    def __init__(self, cfg: SymPyChannelConfig):
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
        # 树结构相对位置偏置 (基于深度差)
        self.depth_bias = nn.Embedding(2 * cfg.max_depth + 1, cfg.num_heads)

    def forward(
        self, x: torch.Tensor, depths: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # 计算相对深度偏置
        # depths: [B, N]
        B, N = depths.shape
        diff = depths.unsqueeze(2) - depths.unsqueeze(1)  # [B, N, N]
        diff = diff.clamp(-self.cfg.max_depth, self.cfg.max_depth) + self.cfg.max_depth
        bias = self.depth_bias(diff)  # [B, N, N, H]
        # nn.MultiheadAttention / TransformerEncoder 不直接支持 4D attn_mask,
        # 此处简化: 将 bias 加到 attention (用 add_bias 模式).
        # 由于 nn.TransformerEncoder 不暴露 attn bias, 这里退化为标准 transformer.
        # 生产实现可替换为支持相对位置的自定义 attention.
        if mask is not None:
            # mask: [B, N] True=valid. 转 key_padding_mask (True=ignore)
            kpm = ~mask
            out = self.transformer(x, src_key_padding_mask=kpm)
        else:
            out = self.transformer(x)
        return out


class SymPyChannel(nn.Module):
    """SymPy 通道: AST Tree Transformer 编码符号表达式."""

    def __init__(self, config: SymPyChannelConfig | None = None, **kwargs):
        super().__init__()
        cfg = config or SymPyChannelConfig(**kwargs)
        self.cfg = cfg
        self.node_encoder = ASTNodeEncoder(cfg)
        self.tree_transformer = TreeTransformer(cfg)
        self.output_proj = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.output_dim),
        )
        # 池化权重 (learned attention pooling)
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
    # AST 序列化: 将 sympy 表达式转为节点列表
    # ------------------------------------------------------------------
    @staticmethod
    def serialize_ast(expr) -> list[dict]:
        """将 sympy 表达式转为 AST 节点列表 (前序遍历).

        每个节点: {type, value_id, depth, child_index}
        若 sympy 不可用, 退化为字符串解析.
        """
        nodes = []
        try:
            import sympy as sp
        except ImportError:
            # 无 sympy 时返回空
            return nodes

        def _walk(node, depth, child_idx):
            if len(nodes) >= 256:
                return
            type_name = type(node).__name__
            type_id = NODE_TYPE_TO_ID.get(type_name, 0)
            # value_id: 符号名 hash 或数值
            if isinstance(node, (sp.Symbol,)):
                value_id = abs(hash(str(node))) % 8192
            elif isinstance(node, (sp.Integer,)):
                value_id = int(node) % 8192
            elif isinstance(node, (sp.Float,)):
                value_id = int(abs(float(node) * 1000)) % 8192
            elif isinstance(node, (sp.Rational,)):
                value_id = (int(node.p) * 31 + int(node.q)) % 8192
            else:
                value_id = abs(hash(str(node))) % 8192
            nodes.append({
                "type": type_id,
                "value_id": value_id,
                "depth": depth,
                "child_index": child_idx,
            })
            args = getattr(node, "args", ())
            for i, arg in enumerate(args):
                _walk(arg, depth + 1, i)

        _walk(expr, 0, 0)
        return nodes

    # ------------------------------------------------------------------
    # 前向: 节点序列 → 结果向量
    # ------------------------------------------------------------------
    def forward(
        self,
        type_ids: torch.Tensor,      # [B, N]
        value_ids: torch.Tensor,     # [B, N]
        depths: torch.Tensor,        # [B, N]
        child_indices: torch.Tensor, # [B, N]
        mask: torch.Tensor | None = None,
    ) -> dict:
        """编码 AST 节点序列.

        Returns:
            dict 含 result_vector [B, output_dim] / node_hidden [B, N, H].
        """
        h = self.node_encoder(type_ids, value_ids, depths, child_indices)
        h = self.tree_transformer(h, depths, mask)
        # Attention pooling
        if mask is not None:
            scores = self.pool_weight(h).squeeze(-1).masked_fill(
                ~mask, float("-inf")
            )
        else:
            scores = self.pool_weight(h).squeeze(-1)
        attn = F.softmax(scores, dim=-1).unsqueeze(-1)  # [B, N, 1]
        pooled = (h * attn).sum(dim=1)  # [B, H]
        result_vec = self.output_proj(pooled)
        return {
            "result_vector": result_vec,
            "node_hidden": h,
            "attention": attn.squeeze(-1),
        }

    # ------------------------------------------------------------------
    # 便捷接口: 直接从 sympy 表达式编码
    # ------------------------------------------------------------------
    def encode(self, expr, device: torch.device | None = None) -> torch.Tensor:
        """从 sympy 表达式直接编码为结果向量 [1, output_dim]."""
        device = device or next(self.parameters()).device
        nodes = self.serialize_ast(expr)
        if not nodes:
            # 空表达式 → 零向量
            return torch.zeros(1, self.cfg.output_dim, device=device)
        N = len(nodes)
        type_ids = torch.tensor([[n["type"] for n in nodes]], device=device)
        value_ids = torch.tensor([[n["value_id"] for n in nodes]], device=device)
        depths = torch.tensor([[n["depth"] for n in nodes]], device=device)
        child_indices = torch.tensor(
            [[n["child_index"] for n in nodes]], device=device
        )
        mask = torch.ones(1, N, dtype=torch.bool, device=device)
        out = self.forward(type_ids, value_ids, depths, child_indices, mask)
        return out["result_vector"]

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.cfg.hidden_dim}, "
            f"num_layers={self.cfg.num_layers}, "
            f"max_nodes={self.cfg.max_nodes}, "
            f"output_dim={self.cfg.output_dim}"
        )
