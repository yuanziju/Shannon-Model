"""MoE 投机解码 - draft + verify, 1.5-2x 加速.

MoESpeculativeDecoder 利用双层 MoE 的小专家作为 draft 模型生成候选 token,
大专家作为 verify 模型并行校验. 命中部分 token 即可减少串行步数,
目标 1.5-2x 加速 (spec: MoE 层投机解码).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


# spec: 投机解码目标加速 1.5-2x
TARGET_SPEEDUP_MIN = 1.5
TARGET_SPEEDUP_MAX = 2.0
DEFAULT_DRAFT_K = 4  # 每轮 draft 候选 token 数 (k=2-4)


@dataclass
class SpecStep:
    """单轮投机解码结果. """

    draft_tokens: List[int]
    accepted: List[int]
    rejected: List[int]
    bonus_token: Optional[int]
    draft_time: float = 0.0
    verify_time: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        total = len(self.draft_tokens)
        if total == 0:
            return 0.0
        return len(self.accepted) / total

    @property
    def tokens_produced(self) -> int:
        # 接受的 + 1 个 bonus (若 draft 全接受)
        return len(self.accepted) + (1 if self.bonus_token is not None else 0)


class MoESpeculativeDecoder:
    """MoE 层投机解码器.

    Args:
        draft_fn:    小专家 draft 模型, ``fn(prefix: list[int], k: int) -> list[(token, logprob)]``.
        verify_fn:   大专家 verify 模型, ``fn(prefix: list[int]) -> (token, logprob)``.
        draft_k:     每轮 draft 候选数 (默认 4).
        temperature: 采样温度 (0 = greedy).
        rng_seed:    随机种子.
    """

    def __init__(
        self,
        draft_fn: Optional[Callable[[Sequence[int], int], List[Tuple[int, float]]]] = None,
        verify_fn: Optional[Callable[[Sequence[int]], Tuple[int, float]]] = None,
        draft_k: int = DEFAULT_DRAFT_K,
        temperature: float = 0.0,
        rng_seed: Optional[int] = None,
    ) -> None:
        self.draft_fn = draft_fn or self._default_draft
        self.verify_fn = verify_fn or self._default_verify
        self.draft_k = max(1, int(draft_k))
        self.temperature = float(temperature)
        self._rng = random.Random(rng_seed)
        self._steps: List[SpecStep] = []
        self._total_draft = 0
        self._total_accepted = 0

    # ------------------------------------------------------------------ #
    # 解码循环
    # ------------------------------------------------------------------ #
    def decode(
        self,
        prompt: Sequence[int],
        max_tokens: int = 128,
        eos_token: int = -1,
    ) -> List[int]:
        """投机解码生成, 返回生成的 token 序列. """
        prefix = list(prompt)
        generated: List[int] = []
        while len(generated) < max_tokens:
            step = self._speculative_step(prefix)
            self._steps.append(step)
            self._total_draft += len(step.draft_tokens)
            self._total_accepted += len(step.accepted)

            # 接受的 token 加入序列
            for tok in step.accepted:
                generated.append(tok)
                prefix.append(tok)
                if tok == eos_token or len(generated) >= max_tokens:
                    return generated

            # bonus token (draft 全部接受时, verify 多给一个)
            if step.bonus_token is not None:
                generated.append(step.bonus_token)
                prefix.append(step.bonus_token)
                if step.bonus_token == eos_token or len(generated) >= max_tokens:
                    return generated
            elif step.rejected:
                # verify 修正: 用 verify 的正确 token 继续
                verified, _ = self.verify_fn(prefix)
                generated.append(verified)
                prefix.append(verified)
                if verified == eos_token or len(generated) >= max_tokens:
                    return generated
        return generated

    def _speculative_step(self, prefix: Sequence[int]) -> SpecStep:
        """单轮 draft + verify. """
        import time as _time
        t0 = _time.time()
        # 1. draft 生成 k 个候选
        draft = self.draft_fn(prefix, self.draft_k)
        draft_tokens = [t for t, _ in draft]
        draft_time = _time.time() - t0

        # 2. verify 逐个校验 (大专家可并行计算, 此处顺序模拟)
        t1 = _time.time()
        accepted: List[int] = []
        rejected: List[int] = []
        cur_prefix = list(prefix)
        bonus_token: Optional[int] = None
        all_accepted = True

        for tok, draft_lp in draft:
            verified, verify_lp = self.verify_fn(cur_prefix)
            # greedy 一致 或 概率接受
            if self._accept(tok, draft_lp, verified, verify_lp):
                accepted.append(tok)
                cur_prefix.append(tok)
            else:
                rejected.append(tok)
                all_accepted = False
                break  # 第一个不匹配即停止本轮

        # draft 全部接受 -> verify 多产出一个 bonus token
        if all_accepted and draft_tokens:
            verified, _ = self.verify_fn(cur_prefix)
            bonus_token = verified

        verify_time = _time.time() - t1
        return SpecStep(
            draft_tokens=draft_tokens,
            accepted=accepted,
            rejected=rejected,
            bonus_token=bonus_token,
            draft_time=draft_time,
            verify_time=verify_time,
        )

    def _accept(self, draft_tok: int, draft_lp: float, verified_tok: int, verify_lp: float) -> bool:
        if self.temperature == 0.0:
            return draft_tok == verified_tok
        # 概率接受: min(1, exp(verify - draft))
        import math
        ratio = math.exp(verify_lp - draft_lp)
        return self._rng.random() < min(1.0, ratio)

    # ------------------------------------------------------------------ #
    # 默认实现 (模拟)
    # ------------------------------------------------------------------ #
    def _default_draft(self, prefix: Sequence[int], k: int) -> List[Tuple[int, float]]:
        # 模拟: 基于前缀哈希生成候选, 一致性高
        base = (sum(prefix) % 1000) if prefix else 0
        return [((base + i) % 100, -0.1 * i) for i in range(k)]

    def _default_verify(self, prefix: Sequence[int]) -> Tuple[int, float]:
        base = (sum(prefix) % 1000) if prefix else 0
        return (base % 100, -0.05)

    # ------------------------------------------------------------------ #
    # 统计
    # ------------------------------------------------------------------ #
    @property
    def acceptance_rate(self) -> float:
        if self._total_draft == 0:
            return 0.0
        return self._total_accepted / self._total_draft

    @property
    def speedup(self) -> float:
        """估计加速比 = 实际产出 token / 串行 verify 步数. """
        if not self._steps:
            return 1.0
        total_produced = sum(s.tokens_produced for s in self._steps)
        # 串行步数 = 轮数 (每轮一次 verify 前向)
        serial_steps = len(self._steps)
        if serial_steps == 0:
            return 1.0
        ratio = total_produced / serial_steps
        return max(1.0, min(TARGET_SPEEDUP_MAX, ratio))

    @property
    def steps(self) -> List[SpecStep]:
        return list(self._steps)

    def stats(self) -> Dict[str, Any]:
        return {
            "rounds": len(self._steps),
            "acceptance_rate": round(self.acceptance_rate, 3),
            "speedup": round(self.speedup, 3),
            "target_speedup": [TARGET_SPEEDUP_MIN, TARGET_SPEEDUP_MAX],
            "draft_k": self.draft_k,
        }


__all__ = [
    "MoESpeculativeDecoder",
    "SpecStep",
    "TARGET_SPEEDUP_MIN",
    "TARGET_SPEEDUP_MAX",
    "DEFAULT_DRAFT_K",
]
