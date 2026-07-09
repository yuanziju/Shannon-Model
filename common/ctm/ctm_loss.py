"""CTM dynamic loss: min-loss + max-certainty tick (decision C8).

The CTM "tick" mechanism runs the model for a variable number of ticks and
chooses when to emit/stop. This loss combines:
  * min-loss: per-sample minimum task loss across ticks.
  * max-certainty tick: rewards high certainty at the chosen (stopping)
    tick and monotonic certainty growth across ticks.
  * tick regulariser: encourages an efficient confident early stop.
"""
from __future__ import annotations

from typing import List, Optional

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
        raise RuntimeError("torch is required to compute CTM loss")


class CTMDynamicLoss(_Module):
    """min-loss + max-certainty tick.

    For each sample, evaluate the task loss at every CTM tick and pick the
    tick that jointly minimises loss and maximises certainty (the
    "max-certainty tick"). The total loss is the sum of:
      * min-loss: the per-sample minimum task loss across ticks.
      * certainty term: pushes the chosen tick to be high-confidence.
      * monotonicity term: certainty should not decrease across ticks.
      * tick cost: rewards an early confident stop (efficiency).
    """

    def __init__(self,
                 lambda_certainty: float = 0.5,
                 lambda_tick: float = 0.1,
                 lambda_monotone: float = 0.1,
                 ignore_index: int = -100,
                 label_smoothing: float = 0.0):
        super().__init__()
        _require_torch()
        self.lambda_certainty = lambda_certainty
        self.lambda_tick = lambda_tick
        self.lambda_monotone = lambda_monotone
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def _task_loss(self, logits, labels):
        # logits: [batch, vocab], labels: [batch] -> [batch]
        return F.cross_entropy(
            logits, labels,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )

    def forward(self, logits_per_tick, labels, certainty_per_tick=None,
                ticks_mask=None):
        """Compute the CTM dynamic loss.

        Args:
            logits_per_tick: [num_ticks, batch, vocab] tensor or list of
                [batch, vocab].
            labels: [batch] target indices.
            certainty_per_tick: optional [num_ticks, batch] certainty scores
                in [0, 1]. If None, derived from the max softmax prob.
            ticks_mask: optional [num_ticks, batch] bool mask of valid ticks
                (e.g. padding for variable tick counts).
        Returns:
            dict with the total loss and per-component diagnostics.
        """
        _require_torch()
        if isinstance(logits_per_tick, (list, tuple)):
            logits_per_tick = torch.stack(logits_per_tick, dim=0)
        if isinstance(certainty_per_tick, (list, tuple)):
            certainty_per_tick = torch.stack(certainty_per_tick, dim=0)

        T, B, V = logits_per_tick.shape

        # Per-tick task loss: [T, B]
        loss_per_tick = torch.stack(
            [self._task_loss(logits_per_tick[t], labels) for t in range(T)],
            dim=0,
        )

        # Certainty per tick: max softmax probability, [T, B]
        if certainty_per_tick is None:
            with torch.no_grad():
                probs = F.softmax(logits_per_tick, dim=-1)
                certainty_per_tick = probs.max(dim=-1).values  # [T, B]

        if ticks_mask is None:
            ticks_mask = torch.ones(T, B, dtype=torch.bool, device=labels.device)
        else:
            ticks_mask = ticks_mask.bool()

        # Mask invalid ticks with +inf loss so they are never selected.
        masked_loss = loss_per_tick.masked_fill(~ticks_mask, float("inf"))

        # min-loss: per-sample minimum loss across ticks.
        min_loss, best_tick = masked_loss.min(dim=0)  # [B], [B]

        # max-certainty tick: among the valid ticks, prefer high certainty.
        masked_cert = certainty_per_tick.masked_fill(~ticks_mask, -1.0)
        max_cert, _ = masked_cert.max(dim=0)  # [B]

        # Certainty term: (1 - max_cert) -> push the chosen tick to be confident.
        cert_term = (1.0 - max_cert).mean()

        # Monotonicity: certainty should not decrease across ticks.
        diffs = certainty_per_tick[1:] - certainty_per_tick[:-1]  # [T-1, B]
        mono_penalty = F.relu(-diffs).mean()

        # Tick regulariser: reward early confident stop (normalised tick index).
        denom = max(T - 1, 1)
        tick_idx = torch.arange(T, device=labels.device,
                                dtype=loss_per_tick.dtype) / denom
        tick_cost = tick_idx[best_tick].mean()

        total = (min_loss.mean()
                 + self.lambda_certainty * cert_term
                 + self.lambda_monotone * mono_penalty
                 + self.lambda_tick * tick_cost)
        return {
            "loss": total,
            "min_loss": min_loss.mean(),
            "certainty_term": cert_term,
            "monotone_penalty": mono_penalty,
            "tick_cost": tick_cost,
            "best_tick": best_tick,
            "max_certainty": max_cert.mean(),
        }
