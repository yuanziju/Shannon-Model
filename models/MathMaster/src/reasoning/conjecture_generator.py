"""数学猜想生成器 - 生成 + 数值验证 + 新颖性评估.

自动生成候选数学猜想, 对每个猜想执行数值验证 (在样本点测试谓词), 并评估相对
已知猜想库的新颖性. 通过的猜想可接入 ``self_play_debate.SelfPlayDebate`` 进
行正/反方辩论检验, 再交由 ``self_play_solver.SelfPlaySolver`` 尝试证明.

验证器内置常见猜想模式识别:
    - Goldbach 型: "偶数 > N 是两素数之和"
    - 素数生成多项式: "n^2 + n + 41 是素数"
    - 整除性命题: "若 p 素数则 2^p - 2 被 p 整除" (Fermat 小定理)
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

__all__ = ["ConjectureGenerator", "Conjecture", "VerificationResult"]

# 默认已知猜想库 (用于新颖性评估)
_DEFAULT_KNOWN = [
    "all even numbers greater than 2 are the sum of two primes",   # Goldbach
    "n^2 + n + 41 is prime",                                       # Euler
    "if p is prime then 2^p - 2 is divisible by p",                # Fermat
    "the number of primes less than n is approximately n / log n", # PNT
    "every even integer is the difference of two primes",          # Polignac
    "x^n + y^n = z^n has no positive integer solutions for n > 2",# Fermat's Last
]


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


@dataclass
class VerificationResult:
    """数值验证结果. """

    passed: bool = False
    pass_rate: float = 0.0           # 通过样本占比
    samples: int = 0
    failures: List[int] = field(default_factory=list)   # 反例
    pattern: str = "unknown"         # 识别的猜想模式
    notes: str = ""


@dataclass
class Conjecture:
    """一条候选猜想. """

    text: str = ""
    domain: str = ""
    verification: Optional[VerificationResult] = None
    novelty: float = 0.0             # [0,1], 越大越新颖
    score: float = 0.0               # 综合 = verification.pass_rate * 0.6 + novelty * 0.4
    accepted: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConjectureGenerator:
    """数学猜想生成器.

    Args:
        model: 15B 循环主体.
        num_candidates: 单次生成的候选数.
        num_verification_samples: 数值验证采样数.
        novelty_threshold: 新颖度低于此值的猜想丢弃.
        verification_pass_rate: 数值验证通过率阈值.
        known_conjectures: 已知猜想库 (新颖性对比基准).
    """

    def __init__(
        self,
        model,
        num_candidates: int = 5,
        num_verification_samples: int = 100,
        novelty_threshold: float = 0.5,
        verification_pass_rate: float = 0.95,
        known_conjectures: Optional[List[str]] = None,
    ) -> None:
        self.model = model
        self.num_candidates = max(1, int(num_candidates))
        self.num_verification_samples = max(1, int(num_verification_samples))
        self.novelty_threshold = float(novelty_threshold)
        self.verification_pass_rate = float(verification_pass_rate)
        self.known_conjectures = list(known_conjectures) if known_conjectures else list(_DEFAULT_KNOWN)

    # ------------------------------------------------------------------ #
    # 公共接口
    # ------------------------------------------------------------------ #
    def generate(
        self,
        domain: str = "number theory",
        seed_conjectures: Optional[List[str]] = None,
    ) -> List[Conjecture]:
        """生成并筛选候选猜想.

        流程:
            1. 模型生成 num_candidates 条候选;
            2. 对每条数值验证;
            3. 对每条评估新颖性;
            4. 综合打分, 返回按 score 降序的猜想列表.
        """
        t0 = time.time()
        candidates: List[Conjecture] = []
        for i in range(self.num_candidates):
            text = self._generate_candidate(domain, seed_conjectures or self.known_conjectures, i)
            conj = Conjecture(text=text, domain=domain)
            conj.verification = self.verify_numerically(text)
            conj.novelty = self.assess_novelty(text, self.known_conjectures)
            conj.score = (
                0.6 * conj.verification.pass_rate + 0.4 * conj.novelty
            )
            conj.accepted = (
                conj.verification.pass_rate >= self.verification_pass_rate
                and conj.novelty >= self.novelty_threshold
            )
            conj.metadata = {"candidate_index": i, "elapsed": time.time() - t0}
            candidates.append(conj)

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    def verify_numerically(
        self, conjecture: str, num_samples: Optional[int] = None
    ) -> VerificationResult:
        """数值验证: 识别模式后在样本点测试谓词. """
        n = int(num_samples) if num_samples is not None else self.num_verification_samples
        text = (conjecture or "").strip()
        if not text:
            return VerificationResult(pattern="empty")

        # 依次尝试各模式识别器
        for recognizer, verifier in (
            (self._recognize_goldbach, self._verify_goldbach),
            (self._recognize_prime_poly, self._verify_prime_poly),
            (self._recognize_fermat_div, self._verify_fermat_div),
        ):
            params = recognizer(text)
            if params is not None:
                return verifier(params, n)

        # 未知模式: 退化为结构合理性检查 (含等式/数学符号视为部分通过)
        return self._verify_generic(text, n)

    def assess_novelty(self, conjecture: str, known: List[str]) -> float:
        """新颖性评估: 1 - 与已知猜想库的最大 token-Jaccard 相似度. """
        if not conjecture:
            return 0.0
        if not known:
            return 1.0
        conj_tokens = self._token_set(conjecture)
        if not conj_tokens:
            return 0.0
        max_sim = 0.0
        for k in known:
            kt = self._token_set(k)
            if not kt:
                continue
            sim = len(conj_tokens & kt) / len(conj_tokens | kt)
            if sim > max_sim:
                max_sim = sim
        return max(0.0, min(1.0, 1.0 - max_sim))

    # ------------------------------------------------------------------ #
    # 模式识别器
    # ------------------------------------------------------------------ #
    @staticmethod
    def _recognize_goldbach(text: str) -> Optional[Dict[str, Any]]:
        """识别 Goldbach 型: 偶数 > N 是两素数之和. """
        low = text.lower()
        if "even" in low and ("sum of two primes" in low or "two primes" in low):
            # 提取下界 N
            m = re.search(r"(?:>|greater than|above|>=)\s*(\d+)", low)
            n_bound = int(m.group(1)) if m else 2
            return {"type": "goldbach", "bound": n_bound}
        return None

    @staticmethod
    def _recognize_prime_poly(text: str) -> Optional[Dict[str, Any]]:
        """识别素数生成多项式: <poly> is prime. """
        low = text.lower()
        if "prime" not in low:
            return None
        # 匹配 n^2 + n + 41 / n^2 - n + 41 / 2n^2 + 2 等
        m = re.search(r"([-+n^\d *]+n\^?\d?[-+n^\d *]*)\s*(?:is|are)\s*prime", low)
        if not m:
            return None
        expr = m.group(1).replace("^", "**")
        coeffs = ConjectureGenerator._parse_poly(expr)
        if coeffs is None:
            return None
        return {"type": "prime_poly", "coeffs": coeffs}

    @staticmethod
    def _recognize_fermat_div(text: str) -> Optional[Dict[str, Any]]:
        """识别整除性: 若 p 素数则 a^p - a 被 p 整除 (Fermat 小定理型). """
        low = text.lower()
        if "divisible" in low and "prime" in low and "^" in low:
            m = re.search(r"(\d+)\s*\^\s*p\s*-\s*(\d+)", low)
            if m:
                a = int(m.group(1))
                b = int(m.group(2))
                return {"type": "fermat_div", "a": a, "b": b}
        return None

    # ------------------------------------------------------------------ #
    # 验证器
    # ------------------------------------------------------------------ #
    @staticmethod
    def _verify_goldbach(params: Dict[str, Any], n: int) -> VerificationResult:
        bound = int(params.get("bound", 2))
        failures: List[int] = []
        tested = 0
        # 在 [bound+1, bound+2*n] 范围采样偶数
        start = bound + 1 if bound % 2 == 0 else bound + 2
        primes_under = [p for p in range(2, bound + 4 * n) if _is_prime(p)]
        primes_set = set(primes_under)
        for k in range(n):
            ev = start + 2 * k
            if ev <= 2:
                continue
            tested += 1
            ok = any((ev - p) in primes_set for p in primes_under if p <= ev)
            if not ok:
                failures.append(ev)
        pass_rate = (tested - len(failures)) / max(1, tested)
        return VerificationResult(
            passed=pass_rate >= 0.95,
            pass_rate=pass_rate,
            samples=tested,
            failures=failures[:10],
            pattern="goldbach",
            notes=f"bound={bound}",
        )

    @staticmethod
    def _verify_prime_poly(params: Dict[str, Any], n: int) -> VerificationResult:
        coeffs = params["coeffs"]  # dict {0:c0, 1:c1, 2:c2}
        failures: List[int] = []
        tested = 0
        for v in range(1, n + 1):
            val = sum(c * (v ** d) for d, c in coeffs.items())
            tested += 1
            if not _is_prime(val):
                failures.append(v)
        pass_rate = (tested - len(failures)) / max(1, tested)
        return VerificationResult(
            passed=pass_rate >= 0.95,
            pass_rate=pass_rate,
            samples=tested,
            failures=failures[:10],
            pattern="prime_poly",
            notes=f"coeffs={coeffs}",
        )

    @staticmethod
    def _verify_fermat_div(params: Dict[str, Any], n: int) -> VerificationResult:
        a = int(params["a"])
        b = int(params["b"])
        failures: List[int] = []
        tested = 0
        # 采样前 n 个素数 p
        p = 2
        count = 0
        while count < n and p < 10_000:
            if _is_prime(p):
                tested += 1
                if (a ** p - b) % p != 0:
                    failures.append(p)
                count += 1
            p += 1
        pass_rate = (tested - len(failures)) / max(1, tested)
        return VerificationResult(
            passed=pass_rate >= 0.95,
            pass_rate=pass_rate,
            samples=tested,
            failures=failures[:10],
            pattern="fermat_div",
            notes=f"a={a}, b={b}",
        )

    @staticmethod
    def _verify_generic(text: str, n: int) -> VerificationResult:
        """未知模式的退化验证: 检查数学结构合理性. """
        has_eq = bool(re.search(r"[=<>≤≥]", text))
        has_var = bool(re.search(r"\bn\b|[a-z]\d?", text))
        score = 0.5
        if has_eq:
            score += 0.2
        if has_var:
            score += 0.1
        score = max(0.0, min(1.0, score))
        return VerificationResult(
            passed=score >= 0.7,
            pass_rate=score,
            samples=0,
            failures=[],
            pattern="unknown",
            notes="no recognizer matched; structural check only",
        )

    # ------------------------------------------------------------------ #
    # 内部: 多项式解析 / token / 候选生成
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_poly(expr: str) -> Optional[Dict[int, int]]:
        """解析简单多项式 (仅支持 n, n^2 系数), 返回 {degree: coeff}. """
        expr = expr.replace(" ", "")
        coeffs: Dict[int, int] = {}
        # 拆成项
        terms = re.findall(r"[+-][^+-]+", "+" + expr)
        for term in terms:
            term = term.lstrip("+")
            sign = 1
            if term.startswith("-"):
                sign = -1
                term = term[1:]
            # n^k
            m = re.match(r"^(\d*)\*?n\^(\d+)$", term)
            if m:
                coeff = int(m.group(1)) if m.group(1) else 1
                deg = int(m.group(2))
                coeffs[deg] = coeffs.get(deg, 0) + sign * coeff
                continue
            # n
            m = re.match(r"^(\d*)\*?n$", term)
            if m:
                coeff = int(m.group(1)) if m.group(1) else 1
                coeffs[1] = coeffs.get(1, 0) + sign * coeff
                continue
            # 常数
            m = re.match(r"^(\d+)$", term)
            if m:
                coeffs[0] = coeffs.get(0, 0) + sign * int(m.group(1))
                continue
            return None  # 含无法解析项
        return coeffs if coeffs else None

    @staticmethod
    def _token_set(text: str) -> set:
        return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1}

    def _generate_candidate(
        self, domain: str, seeds: List[str], index: int
    ) -> str:
        """调用模型生成一条候选猜想. """
        seed_sample = "; ".join(seeds[:3])
        prompt = (
            f"[Conjecture Generation {index + 1}/{self.num_candidates}]\n"
            f"Domain: {domain}\n"
            f"Known conjectures (for reference, do NOT repeat): {seed_sample}\n"
            f"Propose a NEW, plausible and testable mathematical conjecture:"
        )
        text, _ = self._call_model(self.model, prompt, max_new=64, temperature=0.7)
        text = (text or "").strip()
        if not text:
            text = f"conjecture-{index}: every integer n > {index + 2} satisfies a testable property"
        return text

    # ------------------------------------------------------------------ #
    # 内部: model 调用
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_ids(text: str, max_len: int = 1024) -> torch.Tensor:
        chars = list(text)[:max_len]
        ids = [(ord(c) % 1000) + 1 for c in chars] or [0]
        return torch.tensor([ids], dtype=torch.long)

    @staticmethod
    def _call_model(model, text: str, max_new: int = 48, temperature: float = 0.7) -> "tuple[str, float]":
        ids = ConjectureGenerator._to_ids(text)
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
            last = logits[0, -1]
            if temperature > 0:
                probs = torch.softmax(last / max(temperature, 1e-6), dim=-1)
                nxt = int(torch.multinomial(probs, num_samples=1).item())
            else:
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
