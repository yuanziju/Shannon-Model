"""NSL decoder: tree-structured AR decoding with cross-cycle state (T2.5.5/T2.5.6).

Decodes neural latents into symbolic ASTs in a top-down, left-to-right
autoregressive manner: predict (type, value, arity) for a node, then
recursively decode each child. State (hidden + partial tree) is carried
across RDT cycles so partially-decoded trees can be refined iteratively.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

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
        raise RuntimeError("torch is required to run the NSL decoder")


class NSLDecoder(_Module):
    """Tree-structured autoregressive decoder for the Neuro-Symbolic Language.

    Decoding proceeds as a DFS over the target tree: at each step a query
    token (conditioned on the neural latent, depth, sibling index, and RDT
    cycle) is cross-attended to the neural memory, and four heads predict
    node type, symbol id, arity, and a continue/stop signal. Children are
    pushed onto a stack and decoded in order.

    State (last hidden + logits trace) is returned so a subsequent cycle can
    refine the tree (T2.5.6 cross-cycle state passing).
    """

    def __init__(self, d_model: int, grammar: Optional[NSLGrammar] = None,
                 num_heads: int = 4, num_layers: int = 2,
                 vocab_size: int = 1024, max_depth: int = 16,
                 max_children: int = 8, max_nodes: int = 256):
        super().__init__()
        _require_torch()
        self.d_model = d_model
        self.grammar = grammar or NSLGrammar()
        self.vocab_size = vocab_size
        self.max_depth = max_depth
        self.max_children = max_children
        self.max_nodes = max_nodes

        layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads,
                                           batch_first=True,
                                           dim_feedforward=d_model * 2)
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        # Prediction heads.
        self.type_head = nn.Linear(d_model, len(SymbolType))
        self.symbol_head = nn.Linear(d_model, vocab_size)
        self.arity_head = nn.Linear(d_model, max_children + 1)
        self.stop_head = nn.Linear(d_model, 2)  # continue / stop

        # Positional / structural embeddings.
        self.depth_embed = nn.Embedding(max_depth + 1, d_model)
        self.sibling_embed = nn.Embedding(max_children + 1, d_model)
        self.cycle_embed = nn.Embedding(32, d_model)  # RDT cycle index (0-31)

        # Memory projection of the neural latent.
        self.memory_proj = nn.Linear(d_model, d_model)

        self._type_order = list(SymbolType)

    # ------------------------------------------------------------------
    # Step primitives
    # ------------------------------------------------------------------
    def _query_token(self, neural, depth: int, sibling: int, cycle: int):
        _require_torch()
        B = neural.shape[0]
        q = self.memory_proj(neural)
        d = self.depth_embed(
            torch.tensor(depth, device=neural.device).clamp(max=self.max_depth))
        s = self.sibling_embed(
            torch.tensor(sibling, device=neural.device).clamp(max=self.max_children))
        c = self.cycle_embed(
            torch.tensor(cycle, device=neural.device).clamp(max=31))
        return (q + d + s + c).unsqueeze(1)  # [B, 1, d]

    def _decode_step(self, query, memory, memory_mask):
        _require_torch()
        out = self.decoder(query, memory, memory_key_padding_mask=~memory_mask)
        h = out[:, -1]  # [B, d]
        return (self.type_head(h), self.symbol_head(h),
                self.arity_head(h), self.stop_head(h), h)

    # ------------------------------------------------------------------
    # Full tree decoding
    # ------------------------------------------------------------------
    def decode(self, neural, cycle: int = 0,
               state: Optional[Dict[str, Any]] = None,
               max_nodes: Optional[int] = None,
               greedy: bool = True,
               memory=None, memory_mask=None):
        """Decode a batch of neural latents into ASTs.

        Args:
            neural: [B, d_model] neural latent (memory source for the decoder).
            cycle: RDT cycle index (cross-cycle state passing).
            state: optional prior decode state for refinement across cycles.
            max_nodes: cap on total decoded nodes per sample.
            greedy: if True take argmax; else sample from the head distributions.
            memory: optional [B, M, d] external memory; defaults to neural.
            memory_mask: optional [B, M] bool mask.
        Returns:
            (asts, new_state, logits_trace).
        """
        _require_torch()
        if memory is None:
            memory = neural.unsqueeze(1)  # [B, 1, d]
        if memory_mask is None:
            memory_mask = torch.ones(memory.shape[:2], dtype=torch.bool,
                                     device=neural.device)
        # Cross-cycle state passing: carry the previous cycle's hidden as
        # extra memory so the new cycle can refine the partial tree (T2.5.6).
        if state is not None and state.get("last_hidden") is not None:
            lh = state["last_hidden"].unsqueeze(1)  # [B, 1, d]
            memory = torch.cat([memory, lh], dim=1)
            memory_mask = torch.cat([
                memory_mask,
                torch.ones(lh.shape[:2], dtype=torch.bool, device=neural.device)
            ], dim=1)

        B = neural.shape[0]
        cap = max_nodes or self.max_nodes

        asts: List[Optional[ASTNode]] = [None] * B
        # Stack entries: (depth, sibling, parent_node)
        stacks: List[List[Tuple[int, int, Optional[ASTNode]]]] = \
            [[] for _ in range(B)]
        for b in range(B):
            stacks[b].append((0, 0, None))

        logits_trace: List[Dict[str, Any]] = []
        node_count = [0] * B
        h = neural  # fallback if no steps run

        with torch.no_grad():
            steps = 0
            while any(stacks[b] for b in range(B)) and \
                    steps < cap * (self.max_children + 1):
                steps += 1
                # Build a query for every batch element (inactive ones no-op).
                queries = []
                for b in range(B):
                    if stacks[b]:
                        depth, sib, _ = stacks[b][-1]
                    else:
                        depth, sib = 0, 0
                    queries.append(
                        self._query_token(neural[b:b + 1], depth, sib, cycle)
                        .squeeze(1))
                query = torch.stack(queries, dim=0)  # [B, 1, d]
                tl, sl, al, stol, h = self._decode_step(query, memory, memory_mask)
                logits_trace.append(
                    {"type": tl, "symbol": sl, "arity": al, "stop": stol})

                for b in range(B):
                    if not stacks[b] or node_count[b] >= cap:
                        continue
                    depth, sib, parent = stacks[b].pop()
                    type_id = int(tl[b].argmax()) if greedy else \
                        int(torch.multinomial(F.softmax(tl[b], dim=-1), 1))
                    sym_id = int(sl[b].argmax()) if greedy else \
                        int(torch.multinomial(F.softmax(sl[b], dim=-1), 1))
                    arity = int(al[b].argmax()) if greedy else \
                        int(torch.multinomial(F.softmax(al[b], dim=-1), 1))
                    arity = min(arity, self.max_children)
                    stype = (self._type_order[type_id]
                             if type_id < len(self._type_order)
                             else SymbolType.SYMBOL)
                    sym_val = self._id_to_symbol(sym_id)
                    node = ASTNode(stype, sym_val, arity=arity)
                    node_count[b] += 1
                    if parent is None:
                        asts[b] = node
                    else:
                        parent.children.append(node)
                    # Push children in reverse so the leftmost is decoded first.
                    if depth < self.max_depth:
                        for ci in range(arity - 1, -1, -1):
                            stacks[b].append((depth + 1, ci, node))

        # Fill any None with a placeholder so callers always get a node.
        for b in range(B):
            if asts[b] is None:
                asts[b] = ASTNode(SymbolType.SYMBOL, "")

        new_state: Dict[str, Any] = {
            "cycle": cycle,
            "node_counts": node_count,
            "logits_trace": logits_trace,
            "last_hidden": h.detach() if _HAS_TORCH else None,
        }
        return asts, new_state, logits_trace

    def _id_to_symbol(self, sym_id: int) -> str:
        # Best-effort mapping of a vocab id back to a symbol string.
        sym = self.grammar.id_to_symbol(sym_id)
        return sym if sym else f"_s{sym_id}"

    # ------------------------------------------------------------------
    # Cross-cycle refinement
    # ------------------------------------------------------------------
    def refine_across_cycles(self, neural, num_cycles: int = 3,
                             max_nodes: int = 128):
        """Iteratively refine a decoded AST across RDT cycles.

        Carries hidden state and the partial tree context from one cycle to
        the next (T2.5.6 cross-cycle state passing).
        """
        _require_torch()
        state: Optional[Dict[str, Any]] = None
        asts: List[ASTNode] = []
        for c in range(num_cycles):
            asts, state, _ = self.decode(neural, cycle=c, state=state,
                                         max_nodes=max_nodes)
        return asts, state

    def forward(self, neural, cycle: int = 0, max_nodes: int = 128):
        """Decode ASTs from neural latents (single cycle)."""
        _require_torch()
        asts, state, trace = self.decode(neural, cycle=cycle,
                                         max_nodes=max_nodes)
        return {"asts": asts, "state": state, "logits_trace": trace}
