"""CTM router: complexity-driven 3-way routing (decisions C7/C10).

Routes inputs to one of three expert categories based on a learned
complexity score:
  * NLM-enhanced experts      (high complexity, entity experts only - C10)
  * Standard experts          (medium complexity)
  * Shared (always-on) experts (low complexity / common patterns)

Per decision C10: only entity experts use NLM; empty experts keep the
standard design, so they are never routed to the NLM category.
"""
from __future__ import annotations

from typing import Dict, Optional

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
        raise RuntimeError("torch is required to run CTM routing")


class CTMRouter(_Module):
    """Complexity-driven 3-way router: NLM / standard / shared experts.

    A lightweight complexity head scores each token in [0, 1]; tokens above
    the complexity threshold are routed to NLM-enhanced entity experts,
    mid-range tokens to standard experts, and the rest to always-on shared
    experts. A standard MoE load-balancing auxiliary loss is produced.
    """

    CATEGORIES = ("nlm", "standard", "shared")

    def __init__(self,
                 d_model: int,
                 num_nlm: int = 8,
                 num_standard: int = 16,
                 num_shared: int = 2,
                 top_k: int = 2,
                 complexity_threshold: float = 0.7,
                 router_dropout: float = 0.0,
                 noise_std: float = 1.0):
        super().__init__()
        _require_torch()
        self.d_model = d_model
        self.num_nlm = num_nlm
        self.num_standard = num_standard
        self.num_shared = num_shared
        self.top_k = top_k
        self.complexity_threshold = complexity_threshold
        self.noise_std = noise_std

        # Complexity head: scalar complexity score in [0, 1] per token.
        self.complexity_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        # Per-category routing logits.
        self.nlm_router = nn.Linear(d_model, num_nlm, bias=False)
        self.standard_router = nn.Linear(d_model, num_standard, bias=False)
        self.shared_router = nn.Linear(d_model, num_shared, bias=False)
        self.dropout = nn.Dropout(router_dropout)

    def compute_complexity(self, x):
        """Return per-token complexity score in [0, 1]."""
        _require_torch()
        return torch.sigmoid(self.complexity_head(x).squeeze(-1))

    def _route_category(self, logits, top_k):
        # logits: [..., num_experts]
        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std
        scores = F.softmax(logits, dim=-1)
        k = min(top_k, scores.shape[-1])
        topk_scores, topk_idx = scores.topk(k, dim=-1)
        topk_scores = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-9)
        return topk_idx, topk_scores, scores

    def forward(self, x, return_aux: bool = True) -> Dict:
        """Route tokens to expert categories.

        Args:
            x: [batch, seq, d_model] (or [N, d_model]).
            return_aux: whether to compute the load-balancing aux loss.
        Returns:
            dict with per-category top-k indices, weights, masks, scores,
            the per-token complexity, and (optionally) an aux loss.
        """
        _require_torch()
        orig_shape = x.shape
        if x.dim() == 3:
            b, s, d = x.shape
            x_flat = x.reshape(b * s, d)
        else:
            x_flat = x
        x_flat = self.dropout(x_flat)

        complexity = self.compute_complexity(x_flat)  # [N]
        nlm_mask = complexity >= self.complexity_threshold
        shared_mask = complexity < (1.0 - self.complexity_threshold)
        standard_mask = ~(nlm_mask | shared_mask)

        # NLM experts (entity experts only; C10).
        nlm_idx, nlm_w, nlm_scores = self._route_category(
            self.nlm_router(x_flat), self.top_k)
        # Standard experts (also the category for empty experts - C10).
        std_idx, std_w, std_scores = self._route_category(
            self.standard_router(x_flat), self.top_k)
        # Shared experts (always-on; small top_k).
        sh_idx, sh_w, sh_scores = self._route_category(
            self.shared_router(x_flat), min(self.top_k, self.num_shared))

        routes: Dict = {
            "nlm": {"indices": nlm_idx, "weights": nlm_w, "mask": nlm_mask,
                    "scores": nlm_scores, "num_experts": self.num_nlm},
            "standard": {"indices": std_idx, "weights": std_w, "mask": std_mask,
                         "scores": std_scores, "num_experts": self.num_standard},
            "shared": {"indices": sh_idx, "weights": sh_w, "mask": shared_mask,
                       "scores": sh_scores, "num_experts": self.num_shared},
            "complexity": complexity,
            "orig_shape": orig_shape,
        }
        routes["aux_loss"] = self._load_balance_loss(routes, x_flat.shape[0]) \
            if return_aux else None
        return routes

    def _load_balance_loss(self, routes, num_tokens):
        """Load-balancing loss within and across categories."""
        _require_torch()
        parts = []
        for cat in self.CATEGORIES:
            scores = routes[cat]["scores"]                       # [N, E]
            mask = routes[cat]["mask"].float().unsqueeze(-1)     # [N, 1]
            frac_tokens = mask.sum(dim=0) / max(num_tokens, 1)   # [1]
            mean_prob = (scores * mask).sum(dim=0) / (mask.sum(dim=0) + 1e-9)  # [E]
            E = scores.shape[-1]
            parts.append(E * (frac_tokens.mean() * mean_prob.mean()))
        # Also balance across the 3 categories (avoid one dominating).
        cat_fracs = torch.stack(
            [routes[c]["mask"].float().mean() for c in self.CATEGORIES])
        cat_balance = (cat_fracs * cat_fracs).sum() * len(self.CATEGORIES)
        return sum(parts) + cat_balance

    def select_route(self, routes: Dict, category: str):
        """Helper: return (indices, weights, mask) for a single category."""
        r = routes[category]
        return r["indices"], r["weights"], r["mask"]
