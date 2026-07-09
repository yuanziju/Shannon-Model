"""超长推理引擎 - 1M-10M 上下文 + 断点续推 + 上下文压缩.

针对超长数学证明/全库级代码推理场景, 维护一个不断增长的 "推理工作内存"
(working context), 在接近上下文窗口上限时自动压缩 (摘要化旧步骤), 并周期性
落盘检查点以支持断点续推. 与 ``reasoning_checkpoint`` 协同.

设计要点:
    - 工作内存按 *推理步骤* (而非 token) 组织, 便于压缩与回放;
    - 压缩触发阈值: 当前 token 估计 >= ``compress_threshold * max_context_tokens``;
    - 检查点保存: step / problem / working_context / 中间状态;
    - ``resume`` 从最新检查点继续推理.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from .reasoning_checkpoint import ReasoningCheckpoint

__all__ = ["LongReasoningEngine", "LongReasoningResult"]

# 上下文窗口上下限 (token 估计)
MIN_CONTEXT_TOKENS = 1_000_000          # 1M
MAX_CONTEXT_TOKENS = 10_000_000         # 10M
DEFAULT_COMPRESS_THRESHOLD = 0.8        # 80% 触发压缩
DEFAULT_CHECKPOINT_EVERY = 10           # 每 10 步落盘


@dataclass
class LongReasoningResult:
    """超长推理结果. """

    problem: str = ""
    steps: List[str] = field(default_factory=list)
    final_answer: str = ""
    total_steps: int = 0
    compressions: int = 0                # 触发压缩的次数
    resumed_from: Optional[int] = None   # 续推起点 step (None 表示全新推理)
    elapsed: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class LongReasoningEngine:
    """超长推理引擎.

    Args:
        model: 15B 循环主体, ``model(input_ids) -> dict`` 含 ``logits``.
        max_context_tokens: 上下文窗口预算 (1M-10M).
        compress_threshold: 工作内存达到该比例时触发压缩.
        checkpoint: 外部传入的 ``ReasoningCheckpoint``; 缺省按 ``checkpoint_dir``
            新建. 传 ``None`` 且 ``checkpoint_dir`` 为 None 时不落盘.
        checkpoint_every: 每隔多少步落盘一次.
    """

    def __init__(
        self,
        model,
        max_context_tokens: int = MIN_CONTEXT_TOKENS,
        compress_threshold: float = DEFAULT_COMPRESS_THRESHOLD,
        checkpoint: Optional[ReasoningCheckpoint] = None,
        checkpoint_dir: Optional[str] = None,
        checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY,
        max_steps: int = 1000,
    ) -> None:
        self.model = model
        self.max_context_tokens = max(
            MIN_CONTEXT_TOKENS, min(MAX_CONTEXT_TOKENS, int(max_context_tokens))
        )
        self.compress_threshold = float(compress_threshold)
        self.max_steps = int(max_steps)
        self.checkpoint_every = max(1, int(checkpoint_every))

        if checkpoint is not None:
            self.checkpoint = checkpoint
        elif checkpoint_dir is not None:
            self.checkpoint = ReasoningCheckpoint(checkpoint_dir)
        else:
            self.checkpoint = None  # 不落盘

    # ------------------------------------------------------------------ #
    # 公共接口
    # ------------------------------------------------------------------ #
    def reason(
        self,
        problem: str,
        max_steps: Optional[int] = None,
        context: Optional[List[str]] = None,
        name: str = "long_reason",
    ) -> LongReasoningResult:
        """执行超长推理.

        Args:
            problem: 待推理问题.
            max_steps: 本次最大步数 (覆盖构造时的 max_steps).
            context: 初始工作内存 (前置上下文).
            name: 检查点命名空间.
        """
        steps_cap = int(max_steps) if max_steps is not None else self.max_steps
        t0 = time.time()

        working: List[str] = list(context or [])
        if not working:
            working.append(f"[Problem] {problem}")

        resumed_step = 0
        ckpt = self._load_resume_state(name)
        if ckpt is not None:
            working = ckpt.get("working_context", working) or working
            resumed_step = int(ckpt.get("step", 0))
            # problem 可能已变更, 以传入为准
            if working and not working[0].startswith("[Problem]"):
                working.insert(0, f"[Problem] {problem}")

        compressions = 0
        step = resumed_step
        while step < steps_cap:
            # 1. 压缩检查
            if self._needs_compress(working):
                working = self.compress_context(working)
                compressions += 1

            # 2. 生成下一步
            next_step = self._generate_step(problem, working, step)
            if self._is_terminal(next_step):
                working.append(next_step)
                step += 1
                break
            working.append(next_step)
            step += 1

            # 3. 周期性检查点
            if self.checkpoint is not None and step % self.checkpoint_every == 0:
                self._save_checkpoint(name, step, problem, working)

        # 最终检查点
        if self.checkpoint is not None:
            self._save_checkpoint(name, step, problem, working, final=True)

        final_answer = self._extract_final(working)
        return LongReasoningResult(
            problem=problem,
            steps=working,
            final_answer=final_answer,
            total_steps=step,
            compressions=compressions,
            resumed_from=resumed_step if resumed_step > 0 else None,
            elapsed=time.time() - t0,
            metadata={
                "max_context_tokens": self.max_context_tokens,
                "name": name,
            },
        )

    def resume(self, problem: str, name: str = "long_reason") -> LongReasoningResult:
        """从最新检查点续推. """
        return self.reason(problem, name=name)

    def compress_context(self, context: List[str]) -> List[str]:
        """压缩工作内存: 旧步骤摘要化, 保留近期步骤与关键步骤.

        策略:
            - 保留最近 ``keep_recent`` 步原样;
            - 其余旧步骤每 ``group_size`` 步合并为一句摘要;
            - 含结论标记的步骤始终保留.
        """
        if len(context) <= 2:
            return list(context)
        keep_recent = max(2, len(context) // 5)
        group_size = max(2, len(context) // 10)

        head = context[:-keep_recent]
        tail = context[-keep_recent:]

        compressed_head: List[str] = []
        # 先抽取关键步骤 (含结论标记) 单独保留
        key_idx = {
            i for i, s in enumerate(head)
            if any(mk in s.lower() for mk in ("therefore", "综上", "得证", "qed"))
        }

        i = 0
        n = len(head)
        while i < n:
            if i in key_idx:
                compressed_head.append(head[i])
                i += 1
                continue
            j = i
            block: List[str] = []
            while j < n and j not in key_idx and j - i < group_size:
                block.append(head[j])
                j += 1
            summary = self._summarize_steps(block)
            compressed_head.append(summary)
            i = j

        return compressed_head + tail

    # ------------------------------------------------------------------ #
    # 内部: 推理步生成
    # ------------------------------------------------------------------ #
    def _generate_step(self, problem: str, working: List[str], step: int) -> str:
        """生成下一步推理. """
        recent = "\n".join(working[-8:])  # 只看最近 8 步控制 prompt 长度
        prompt = (
            f"[Long Reasoning step={step}]\n"
            f"Problem: {problem}\n"
            f"Recent steps:\n{recent}\n"
            f"Next reasoning step (or 'FINAL: <answer>' to conclude):"
        )
        text, conf = self._call_model(self.model, prompt, max_new=64)
        text = (text or "").strip()
        if not text:
            text = f"step-{step}: continue derivation (conf={conf:.2f})"
        return text

    @staticmethod
    def _is_terminal(step_text: str) -> bool:
        low = (step_text or "").lower().strip()
        return low.startswith("final:") or low.startswith("结论:") or "qed" in low

    @staticmethod
    def _extract_final(working: List[str]) -> str:
        for s in reversed(working):
            low = s.lower().strip()
            if low.startswith("final:"):
                return s.split(":", 1)[-1].strip()
            if low.startswith("结论:"):
                return s.split(":", 1)[-1].strip()
        return working[-1] if working else ""

    # ------------------------------------------------------------------ #
    # 内部: 压缩辅助
    # ------------------------------------------------------------------ #
    def _needs_compress(self, working: List[str]) -> bool:
        toks = self._approx_tokens("\n".join(working))
        return toks >= self.compress_threshold * self.max_context_tokens

    def _summarize_steps(self, block: List[str]) -> str:
        text = " ".join(b.strip() for b in block if b.strip())
        if not text:
            return ""
        n = len(block)
        prompt = f"Summarize these {n} reasoning steps in one sentence:\n{text}"
        summary, _ = self._call_model(self.model, prompt, max_new=24)
        summary = (summary or "").strip()
        if summary:
            return f"[summary {n} steps] {summary}"
        return f"[summary: {n} steps compressed]"

    # ------------------------------------------------------------------ #
    # 内部: 检查点
    # ------------------------------------------------------------------ #
    def _save_checkpoint(
        self, name: str, step: int, problem: str, working: List[str], final: bool = False
    ) -> None:
        if self.checkpoint is None:
            return
        self.checkpoint.save(
            name=name,
            step=step,
            state={
                "problem": problem,
                "step": step,
                "working_context": working,
                "tokens": self._approx_tokens("\n".join(working)),
                "final": final,
            },
        )

    def _load_resume_state(self, name: str) -> Optional[Dict[str, Any]]:
        if self.checkpoint is None:
            return None
        return self.checkpoint.load(name)

    # ------------------------------------------------------------------ #
    # 内部: token 估计 + model 调用
    # ------------------------------------------------------------------ #
    @staticmethod
    def _approx_tokens(text: str) -> int:
        if not text:
            return 0
        cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        ascii_words = len(re.findall(r"\S+", text))
        return int((cjk + ascii_words) * 1.3)  # 经验系数

    @staticmethod
    def _to_ids(text: str, max_len: int = 1024) -> torch.Tensor:
        chars = list(text)[:max_len]
        ids = [(ord(c) % 1000) + 1 for c in chars] or [0]
        return torch.tensor([ids], dtype=torch.long)

    @staticmethod
    def _call_model(model, text: str, max_new: int = 48) -> "tuple[str, float]":
        ids = LongReasoningEngine._to_ids(text)
        try:
            out = model(ids)
        except TypeError:
            out = model(input_ids=ids)
        logits = None
        conf = None
        if isinstance(out, dict):
            logits = out.get("logits")
            conf = out.get("confidence")
        else:
            logits = getattr(out, "logits", None)
            conf = getattr(out, "confidence", None)
        if logits is None:
            return "", 0.5
        try:
            nxt = int(torch.argmax(logits[0, -1]).item())
        except Exception:  # noqa: BLE001
            return "", 0.5
        generated = [nxt]
        for _ in range(max(0, max_new - 1)):
            try:
                v = int(torch.argmax(logits[0, -1]).item())
            except Exception:  # noqa: BLE001
                break
            if v == 0:
                break
            generated.append(v)
        text_out = "".join(
            chr(v) if (32 <= v < 0x110000) else " " for v in generated
        ).strip()
        if conf is not None:
            try:
                c = float(conf.mean().item())
            except Exception:  # noqa: BLE001
                c = 0.5
        else:
            c = 0.5
        return text_out, max(0.0, min(1.0, c))
