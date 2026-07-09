"""Symbol-Neural bridge: AST encoder + neural decoder + InfoNCE (T2.5.1).

Aligns symbolic ASTs with the model's neural latent space using a
contrastive InfoNCE objective, realising the bidirectional symbol<-neural
translation layer.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from .grammar import ASTNode, NSLGrammar, SymbolType

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:  # torch optional at import / py_compile time
    torch = None
    nn = None
    F = None
    _HAS_TORCH = False

if _HAS_TORCH:
    _Module = nn.Module
else:
    class _Module:  # pragma: no cover - torch-free import path
        def __init__(self, *args, **kwargs):
            pass


def _require_torch():
    if not _HAS_TORCH:
        raise RuntimeError("torch is required to run the symbol-neural bridge")


class SymbolNeuralBridge(_Module):
    """Bidirectional symbol <-> neural translation layer.

    Components:
      * AST encoder: tree-position-aware transformer that encodes a symbolic
        AST into a sequence of node embeddings and a pooled vector.
      * Neural decoder: maps a neural vector to a (teacher-forced) symbol
        token logit sequence.
      * InfoNCE: contrastive loss aligning pooled AST embeddings with neural
        embeddings in a shared latent space.
    """

    def __init__(self, d_model: int, grammar: Optional[NSLGrammar] = None,
                 num_heads: int = 4, num_layers: int = 2,
                 vocab_size: int = 1024, temperature: float = 0.07,
                 max_nodes: int = 128):
        super().__init__()
        _require_torch()
        self.d_model = d_model
        self.grammar = grammar or NSLGrammar()
        self.temperature = temperature
        self.max_nodes = max_nodes
        self.vocab_size = vocab_size

        # Symbol embedding (by symbol id) + type embedding.
        self.symbol_embed = nn.Embedding(vocab_size, d_model)
        self.type_embed = nn.Embedding(len(SymbolType), d_model)
        # Tree positional encoding (depth + sibling index).
        self.depth_embed = nn.Embedding(max_nodes, d_model)
        self.sibling_embed = nn.Embedding(max_nodes, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, batch_first=True,
            dim_feedforward=d_model * 2)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.Linear(d_model, d_model)

        # Neural -> symbol decoder.
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads,
                                       batch_first=True,
                                       dim_feedforward=d_model * 2),
            num_layers=num_layers,
        )
        self.query_proj = nn.Linear(d_model, d_model)
        self.output_head = nn.Linear(d_model, vocab_size)

        # Projections for InfoNCE alignment in a shared latent space.
        self.neural_proj = nn.Linear(d_model, d_model)
        self.symbol_proj = nn.Linear(d_model, d_model)

        self._type_order = list(SymbolType)

    # ---- AST -> tensors ----
    def _flatten_ast(self, root: ASTNode) -> Tuple[List[ASTNode], List[Tuple[int, int]]]:
        """Pre-order flatten with (depth, sibling_index) per node."""
        nodes: List[ASTNode] = []
        order: List[Tuple[int, int]] = []

        def walk(node: ASTNode, depth: int, sib: int):
            nodes.append(node)
            order.append((depth, sib))
            for i, c in enumerate(node.children):
                walk(c, depth + 1, i)

        walk(root, 0, 0)
        return nodes, order

    def _encode_ast_batch(self, asts: List[ASTNode], device=None):
        _require_torch()
        batch = []
        for ast in asts:
            nodes, order = self._flatten_ast(ast)
            nodes = nodes[: self.max_nodes]
            order = order[: self.max_nodes]
            ids = [self.grammar.symbol_id(n.value) % self.vocab_size for n in nodes]
            type_idx = [self._type_order.index(n.type) for n in nodes]
            depths = [o[0] for o in order]
            sibs = [o[1] for o in order]
            batch.append((ids, type_idx, depths, sibs))

        maxlen = max((len(b[0]) for b in batch), default=1)
        B = len(batch)
        ids_t = torch.zeros(B, maxlen, dtype=torch.long, device=device)
        type_t = torch.zeros(B, maxlen, dtype=torch.long, device=device)
        depth_t = torch.zeros(B, maxlen, dtype=torch.long, device=device)
        sib_t = torch.zeros(B, maxlen, dtype=torch.long, device=device)
        mask = torch.zeros(B, maxlen, dtype=torch.bool, device=device)
        for i, (ids, ty, d, s) in enumerate(batch):
            L = len(ids)
            ids_t[i, :L] = torch.tensor(ids, dtype=torch.long, device=device)
            type_t[i, :L] = torch.tensor(ty, dtype=torch.long, device=device)
            depth_t[i, :L] = torch.tensor(d, dtype=torch.long, device=device)
            sib_t[i, :L] = torch.tensor(s, dtype=torch.long, device=device)
            mask[i, :L] = True

        emb = (self.symbol_embed(ids_t) + self.type_embed(type_t)
               + self.depth_embed(depth_t.clamp(max=self.max_nodes - 1))
               + self.sibling_embed(sib_t.clamp(max=self.max_nodes - 1)))
        # Transformer: key_padding_mask True == ignore padding.
        enc = self.encoder(emb, src_key_padding_mask=~mask)
        return enc, mask

    def encode_symbols(self, asts: List[ASTNode], device=None):
        """Encode ASTs to pooled symbol embeddings [B, d_model]."""
        _require_torch()
        enc, mask = self._encode_ast_batch(asts, device=device)
        m = mask.unsqueeze(-1).float()
        pooled = (enc * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)
        return self.symbol_proj(self.pool(pooled)), enc, mask

    def decode_to_symbols(self, neural, memory, memory_mask, seq_len: int):
        """Autoregressively decode a symbol token sequence from a neural vector.

        Args:
            neural: [B, d_model] neural latent (seed for decoder queries).
            memory: [B, M, d_model] encoded AST memory (teacher-forcing target).
            memory_mask: [B, M] bool valid mask.
            seq_len: number of decode steps.
        Returns:
            logits [B, seq_len, vocab].
        """
        _require_torch()
        B = neural.shape[0]
        queries = self.query_proj(neural).unsqueeze(1).expand(B, seq_len, -1)
        pos = torch.arange(seq_len, device=neural.device).unsqueeze(0).expand(B, -1)
        pos = pos.clamp(max=self.max_nodes - 1)
        queries = queries + self.depth_embed(pos)
        out = self.decoder(queries, memory, memory_key_padding_mask=~memory_mask)
        logits = self.output_head(out)
        return logits

    def info_nce(self, symbol_emb, neural_emb):
        """Symmetric InfoNCE contrastive loss between symbol and neural embeddings.

        Args:
            symbol_emb: [B, d] projected symbol embeddings.
            neural_emb: [B, d] projected neural embeddings.
        Returns:
            scalar loss.
        """
        _require_torch()
        s = F.normalize(symbol_emb, dim=-1)
        n = F.normalize(neural_emb, dim=-1)
        logits = (s @ n.t()) / self.temperature
        B = s.shape[0]
        labels = torch.arange(B, device=s.device)
        loss_s2n = F.cross_entropy(logits, labels)
        loss_n2s = F.cross_entropy(logits.t(), labels)
        return (loss_s2n + loss_n2s) / 2.0

    def forward(self, asts: List[ASTNode], neural_emb, device=None):
        """End-to-end: encode ASTs, decode, compute InfoNCE alignment.

        Args:
            asts: list of ASTNode (length B).
            neural_emb: [B, d_model] neural latent embeddings to align.
        Returns:
            dict with pooled symbol emb, projected neural emb, decode logits,
            and the InfoNCE loss.
        """
        _require_torch()
        sym_pooled, enc, mask = self.encode_symbols(asts, device=device)
        seq_len = enc.shape[1]
        logits = self.decode_to_symbols(neural_emb, enc, mask, seq_len=seq_len)
        n_proj = self.neural_proj(neural_emb)
        nce = self.info_nce(sym_pooled, n_proj)
        return {
            "symbol_embedding": sym_pooled,
            "neural_embedding": n_proj,
            "decode_logits": logits,
            "infonce_loss": nce,
        }
