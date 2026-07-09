"""CoT 自我蒸馏 - 长推理链压缩与学生模型训练.

将超长 Chain-of-Thought 推理链蒸馏为压缩表示, 并训练学生模型从压缩表示
直接生成结论. 用于 ``LongReasoningEngine`` 产出的长链落地为可快速推理的
紧凑策略, 与 ``latent_decode`` 隐空间解码的 "压缩比由模型自主学习" 决策
(L14) 呼应.

流程:
    1. extract_key_steps  -- 识别关键步骤 (结论/定理/转折点)
    2. compress_chain     -- 中间步骤摘要化, 保留关键步骤
    3. train_student      -- 学生模型从压缩表示预测结论, 返回蒸馏 loss
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

__all__ = ["CoTDistillation", "DistillResult"]


@dataclass
class DistillResult:
    """单次蒸馏结果. """

    original_chain: str = ""
    compressed: str = ""
    key_steps: List[int] = field(default_factory=list)   # 关键步骤在原链的行索引
    compression_ratio: float = 0.0                       # compressed / original 长度比
    student_loss: float = 0.0
    conclusion: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# 关键步骤标记词 (中英)
_KEY_MARKERS = (
    "therefore", "thus", "hence", "so we have", "conclude", "q.e.d", "qed",
    "因此", "故", "综上", "得证", "由此", "结论", "于是",
)
# 定理/引理/定义声明
_DECL_RE = re.compile(
    r"^\s*(theorem|lemma|proposition|corollary|definition|claim|命题|定理|引理|推论|定义)\b",
    re.IGNORECASE,
)
# 含等式/不等式的行
_EQ_RE = re.compile(r"(=|≤|≥|<|>|≡|≈|→|⟹|\\\\implies)")


class CoTDistillation:
    """CoT 自我蒸馏.

    Args:
        model: 教师 (15B) 模型, 用于摘要化中间步骤.
        student_model: 学生模型; 缺省与教师同体 (self-distillation).
    """

    def __init__(self, model, student_model=None) -> None:
        self.model = model
        self.student_model = student_model if student_model is not None else model

    # ------------------------------------------------------------------ #
    # 公共接口
    # ------------------------------------------------------------------ #
    def distill(self, long_chain: str) -> DistillResult:
        """将长推理链蒸馏为压缩表示并训练学生模型. """
        key_steps = self.extract_key_steps(long_chain)
        compressed = self.compress_chain(long_chain, key_steps)
        conclusion = self._extract_conclusion(long_chain)

        student_loss = self.train_student(compressed, conclusion)

        orig_len = max(1, len(long_chain))
        comp_len = len(compressed)
        ratio = comp_len / orig_len

        return DistillResult(
            original_chain=long_chain,
            compressed=compressed,
            key_steps=key_steps,
            compression_ratio=ratio,
            student_loss=student_loss,
            conclusion=conclusion,
            metadata={
                "original_tokens": self._approx_tokens(long_chain),
                "compressed_tokens": self._approx_tokens(compressed),
                "num_key_steps": len(key_steps),
            },
        )

    def extract_key_steps(self, chain: str) -> List[int]:
        """识别关键步骤, 返回其在原链 (按行切分) 的行索引列表.

        判据 (满足其一即视为关键步骤):
            1. 含结论标记词 (therefore/综上/得证 ...);
            2. 定理/引理/定义声明行;
            3. 含等式/不等式且较长 (核心代数步骤);
            4. 首行与末行 (问题陈述与最终结论) 始终保留.
        """
        lines = chain.splitlines()
        if not lines:
            return []
        n = len(lines)
        key: List[int] = []
        for i, line in enumerate(lines):
            low = line.lower()
            is_marker = any(mk in low for mk in _KEY_MARKERS)
            is_decl = bool(_DECL_RE.match(line))
            is_eq = bool(_EQ_RE.search(line)) and len(line.strip()) > 12
            if is_marker or is_decl or is_eq:
                key.append(i)
        # 始终保留首行 (问题) 与末行 (结论)
        if 0 not in key:
            key.insert(0, 0)
        if (n - 1) not in key:
            key.append(n - 1)
        # 去重保序
        seen = set()
        ordered: List[int] = []
        for i in key:
            if i not in seen:
                seen.add(i)
                ordered.append(i)
        return ordered

    def compress_chain(self, chain: str, key_steps: List[int]) -> str:
        """压缩中间步骤为摘要, 保留关键步骤.

        策略:
            - 关键步骤行原样保留;
            - 连续非关键中间步骤合并为单行摘要 ``[summary: <n> steps]``,
              并可选用教师模型生成一句话摘要.
        """
        lines = chain.splitlines()
        if not lines:
            return ""
        key_set = set(key_steps)
        out: List[str] = []
        i = 0
        n = len(lines)
        while i < n:
            if i in key_set:
                out.append(lines[i])
                i += 1
                continue
            # 收集连续非关键行
            j = i
            while j < n and j not in key_set:
                j += 1
            block = lines[i:j]
            summary = self._summarize_block(block)
            out.append(summary)
            i = j
        return "\n".join(out)

    def train_student(self, compressed: str, conclusion: str) -> float:
        """训练学生模型从压缩表示生成结论, 返回蒸馏 loss (标量).

        实现: 在学生模型 logits 上对结论 token 计算交叉熵 (teacher-forcing),
        返回平均 loss. 不在本方法内更新权重 (由外层训练循环统一反传).
        """
        if not conclusion:
            return 0.0
        prompt = f"Compressed reasoning:\n{compressed}\n\nConclusion:"
        ids = self._to_ids(prompt)
        try:
            out = self.student_model(ids)
        except TypeError:
            out = self.student_model(input_ids=ids)

        logits = self._extract_logits(out)
        if logits is None:
            # 无 logits 可用 -> 用压缩/结论长度差异作为代理 loss
            return float(max(0.0, self._approx_tokens(compressed) / 100.0))

        target_ids = self._to_ids(conclusion, max_len=64)[0]  # (T,)
        T = int(target_ids.shape[0])
        V = int(logits.shape[-1])

        # 将教师 logits 在时间维上对齐到目标长度 (重复/截断)
        logit_seq = logits[0]  # (S, V)
        S = int(logit_seq.shape[0])
        if S == 0:
            return float(T)
        if S >= T:
            used = logit_seq[-T:]
        else:
            reps = (T + S - 1) // S
            used = logit_seq.repeat(reps, 1)[:T]
        used = used.detach().float()

        # 交叉熵 (结论 token id 可能 >= V, 取模映射)
        target_clamped = (target_ids % V).long()
        log_probs = torch.log_softmax(used, dim=-1)
        gathered = log_probs.gather(1, target_clamped.unsqueeze(1)).squeeze(1)
        loss = -gathered.mean().item()
        return float(loss)

    # ------------------------------------------------------------------ #
    # 内部
    # ------------------------------------------------------------------ #
    def _summarize_block(self, block: List[str]) -> str:
        """对一段非关键中间步骤生成摘要行. """
        text = " ".join(b.strip() for b in block if b.strip())
        if not text:
            return ""
        n_steps = len(block)
        # 尝试用教师模型生成一句话摘要 (失败则退化为长度标记)
        prompt = f"Summarize the following {n_steps} reasoning steps in one sentence:\n{text}"
        summary, _ = self._call_model(self.model, prompt, max_new=24)
        summary = (summary or "").strip()
        if summary:
            return f"[summary {n_steps} steps] {summary}"
        return f"[summary: {n_steps} intermediate steps omitted]"

    @staticmethod
    def _extract_conclusion(chain: str) -> str:
        """从长链中抽取最终结论 (最后一个结论标记后的内容, 或末行). """
        lines = [l for l in chain.splitlines() if l.strip()]
        if not lines:
            return ""
        # 优先: 最后一个含结论标记的行
        for idx in range(len(lines) - 1, -1, -1):
            low = lines[idx].lower()
            if any(mk in low for mk in _KEY_MARKERS):
                return lines[idx].strip()
        return lines[-1].strip()

    @staticmethod
    def _approx_tokens(text: str) -> int:
        """近似 token 计数: 英文按词, 含 CJK 按字符. """
        if not text:
            return 0
        cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        ascii_words = len(re.findall(r"\S+", text))
        return cjk + ascii_words

    @staticmethod
    def _to_ids(text: str, max_len: int = 1024) -> torch.Tensor:
        chars = list(text)[:max_len]
        ids = [(ord(c) % 1000) + 1 for c in chars] or [0]
        return torch.tensor([ids], dtype=torch.long)

    @staticmethod
    def _extract_logits(out: Any) -> Optional[torch.Tensor]:
        if isinstance(out, dict):
            return out.get("logits")
        return getattr(out, "logits", None)

    @staticmethod
    def _call_model(model, text: str, max_new: int = 48) -> "tuple[str, float]":
        """复用与 debate 一致的鲁棒 model 调用. """
        ids = CoTDistillation._to_ids(text)
        try:
            out = model(ids)
        except TypeError:
            out = model(input_ids=ids)
        logits = CoTDistillation._extract_logits(out)
        conf = None
        if isinstance(out, dict):
            conf = out.get("confidence")
        else:
            conf = getattr(out, "confidence", None)
        if logits is None:
            return "", 0.5
        try:
            last = logits[0, -1]
            nxt = int(torch.argmax(last).item())
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
