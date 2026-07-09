"""MathRLHF - 数学分层混合奖励.

5 层奖励 (L1-L5) 分层混合, 用于 RLHF (GRPO/DPO) 阶段对数学解答打分:

    L1  Lean4 形式化验证奖励  (硬门控: 证明闭合 -> +0.5 bonus)
    L2  SymPy 符号验证奖励    (等价性 / 化简一致性)
    L3  数值验证奖励          (多采样数值检验, 容差内一致)
    L4  数学家审查奖励        (LLM-as-judge / 规则评分, 0-1)
    L5  启发式奖励            (格式 / 步骤数 / 长度 / 符号一致)

分层混合策略:
    - 硬门控 (L1): Lean4 证明闭合给予 +0.5 奖励加成 (决策 L4 配套)
    - 软加权 (L2-L5): 各层归一化后按权重 w_i 求和
    - 最终奖励裁剪到 [-1, 1], 并提供各层明细 (供训练日志)
    - 答案错误时, L2/L3 给予负奖励, 阻止奖励黑客 (reward hacking)

依赖: Lean4 / SymPy 可选, 未安装时降级为规则验证 (fallback).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


class RewardLayer(Enum):
    """5 层奖励. """

    L1_LEAN = "lean4"               # Lean4 形式化验证
    L2_SYMPY = "sympy"              # SymPy 符号验证
    L3_NUMERIC = "numeric"          # 数值验证
    L4_MATHJUDGE = "math_review"    # 数学家审查
    L5_HEURISTIC = "heuristic"      # 启发式


# 各层默认权重 (L2-L5 软加权, 归一后使用; L1 为硬门控 bonus)
DEFAULT_LAYER_WEIGHTS: Dict[RewardLayer, float] = {
    RewardLayer.L2_SYMPY: 0.35,
    RewardLayer.L3_NUMERIC: 0.25,
    RewardLayer.L4_MATHJUDGE: 0.25,
    RewardLayer.L5_HEURISTIC: 0.15,
}

# L1 Lean 证明闭合 bonus
DEFAULT_LEAN_BONUS: float = 0.5
# L1 Lean 证明含 sorry/admit 惩罚
DEFAULT_LEAN_PENALTY: float = -0.3

# 数值验证容差
DEFAULT_NUMERIC_TOL: float = 1e-6
# 数值采样次数
DEFAULT_NUMERIC_SAMPLES: int = 8

# 奖励裁剪范围
REWARD_CLIP_MIN: float = -1.0
REWARD_CLIP_MAX: float = 1.0

# 答案错误时 L2/L3 的负奖励
WRONG_ANSWER_PENALTY: float = -0.6

# 不完整证明标记
PROOF_INCOMPLETE_MARKERS = ("sorry", "admit", "by_contradiction", "exact?")
PROOF_COMPLETE_MARKERS = ("Qed", "by", "exact", "rfl", "simp", "decide")

# Lean 代码块正则
LEAN_BLOCK_RE = re.compile(
    r"```(?:lean4?|lean_theorem)\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

# 最终答案提取正则 (\\boxed{...} 或 "answer is" / "ans:")
BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
ANSWER_IS_RE = re.compile(
    r"(?:answer\s+is|final\s+answer|ans(?:wer)?)\s*[:=]\s*([^\n.]+)",
    re.IGNORECASE,
)


@dataclass
class LayerResult:
    """单层奖励结果. """

    layer: RewardLayer
    score: float                 # 该层原始分数 [-1, 1]
    weight: float                # 该层权重
    contribution: float          # 加权贡献
    detail: str = ""             # 说明
    passed: bool = False         # 是否通过该层验证


@dataclass
class RewardOutput:
    """完整奖励输出. """

    reward: float                       # 最终混合奖励 [-1, 1]
    layers: List[LayerResult] = field(default_factory=list)
    lean_bonus: float = 0.0             # L1 硬门控加成
    answer_correct: bool = False        # 答案是否正确 (L2/L3 一致)
    has_lean_proof: bool = False        # 是否包含 Lean4 证明
    lean_proof_closed: bool = False     # Lean4 证明是否闭合
    clipped: bool = False               # 是否触发裁剪

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reward": round(self.reward, 6),
            "lean_bonus": round(self.lean_bonus, 6),
            "answer_correct": self.answer_correct,
            "has_lean_proof": self.has_lean_proof,
            "lean_proof_closed": self.lean_proof_closed,
            "clipped": self.clipped,
            "layers": [
                {
                    "layer": lr.layer.value,
                    "score": round(lr.score, 6),
                    "weight": round(lr.weight, 6),
                    "contribution": round(lr.contribution, 6),
                    "detail": lr.detail,
                    "passed": lr.passed,
                }
                for lr in self.layers
            ],
        }


class MathRLHF:
    """数学分层混合奖励器.

    Args:
        layer_weights: L2-L5 软层权重 (会归一化). L1 为硬门控 bonus.
        lean_bonus: Lean4 证明闭合 bonus.
        lean_penalty: Lean4 证明含 sorry/admit 惩罚.
        numeric_tol: 数值验证容差.
        numeric_samples: 数值验证采样次数.
        lean_verifier: 可选外部 Lean4 验证回调
            ``fn(code) -> (success: bool, error: str)``. None 则用规则降级.
        sympy_verifier: 可可选外部 SymPy 验证回调
            ``fn(predicted_formal, gold_formal) -> bool``. None 则用字符串比对降级.
        math_judge: 可选数学家审查回调
            ``fn(problem, response) -> float in [0,1]``. None 则用规则评分.
        clip_min / clip_max: 奖励裁剪范围.
    """

    def __init__(
        self,
        layer_weights: Optional[Dict[RewardLayer, float]] = None,
        lean_bonus: float = DEFAULT_LEAN_BONUS,
        lean_penalty: float = DEFAULT_LEAN_PENALTY,
        numeric_tol: float = DEFAULT_NUMERIC_TOL,
        numeric_samples: int = DEFAULT_NUMERIC_SAMPLES,
        lean_verifier: Optional[Callable[[str], Tuple[bool, str]]] = None,
        sympy_verifier: Optional[Callable[[str, str], bool]] = None,
        math_judge: Optional[Callable[[str, str], float]] = None,
        clip_min: float = REWARD_CLIP_MIN,
        clip_max: float = REWARD_CLIP_MAX,
    ) -> None:
        # 软层权重归一化
        weights = dict(DEFAULT_LAYER_WEIGHTS)
        if layer_weights:
            for k, v in layer_weights.items():
                if k != RewardLayer.L1_LEAN:  # L1 为 bonus, 不参与软加权
                    weights[k] = float(v)
        s = sum(weights.values())
        if s <= 0:
            raise ValueError("sum of soft layer weights must be positive")
        self.layer_weights: Dict[RewardLayer, float] = {k: v / s for k, v in weights.items()}

        self.lean_bonus = float(lean_bonus)
        self.lean_penalty = float(lean_penalty)
        self.numeric_tol = float(numeric_tol)
        self.numeric_samples = max(1, int(numeric_samples))
        self.lean_verifier = lean_verifier
        self.sympy_verifier = sympy_verifier
        self.math_judge = math_judge
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)

    # ------------------------------------------------------------------ #
    # 主入口
    # ------------------------------------------------------------------ #
    def reward(
        self,
        problem: str,
        response: str,
        gold_answer: str = "",
        gold_formal: str = "",
    ) -> RewardOutput:
        """计算单条样本的分层混合奖励.

        Args:
            problem: 题目文本.
            response: 模型生成解答 (含推理 + 最终答案 + 可选 Lean4 代码块).
            gold_answer: 标准答案 (字符串).
            gold_formal: 标准答案的 SymPy/数值形式 (可选, 供 L2/L3 验证).
        """
        out = RewardOutput(reward=0.0)

        # 提取模型答案
        pred_answer = self._extract_answer(response)
        # 提取 Lean4 证明
        lean_code = self._extract_lean_block(response)
        out.has_lean_proof = lean_code is not None

        # ---- L1: Lean4 形式化验证 (硬门控) ----
        l1 = self._score_lean(lean_code, out)
        # L1 不进入软加权, 但记录
        out.layers.append(l1)

        # ---- L2: SymPy 符号验证 ----
        l2 = self._score_sympy(pred_answer, gold_answer, gold_formal)
        out.layers.append(l2)

        # ---- L3: 数值验证 ----
        l3 = self._score_numeric(pred_answer, gold_answer, gold_formal)
        out.layers.append(l3)

        # 答案正确性 = L2 或 L3 通过
        out.answer_correct = l2.passed or l3.passed
        # 答案错误时, L2/L3 给负奖励 (防 reward hacking)
        if gold_answer and not out.answer_correct:
            l2.score = min(l2.score, WRONG_ANSWER_PENALTY)
            l3.score = min(l3.score, WRONG_ANSWER_PENALTY)
            l2.contribution = l2.score * self.layer_weights[RewardLayer.L2_SYMPY]
            l3.contribution = l3.score * self.layer_weights[RewardLayer.L3_NUMERIC]

        # ---- L4: 数学家审查 ----
        l4 = self._score_math_judge(problem, response, out.answer_correct)
        out.layers.append(l4)

        # ---- L5: 启发式 ----
        l5 = self._score_heuristic(response, pred_answer, out.answer_correct)
        out.layers.append(l5)

        # 软加权求和 (L2-L5)
        soft_total = sum(
            lr.contribution for lr in out.layers
            if lr.layer != RewardLayer.L1_LEAN
        )

        # L1 硬门控 bonus
        out.lean_bonus = l1.score  # l1.score 已是 bonus/penalty 值

        raw = soft_total + out.lean_bonus
        clipped = raw
        if clipped < self.clip_min:
            clipped = self.clip_min
            out.clipped = True
        elif clipped > self.clip_max:
            clipped = self.clip_max
            out.clipped = True
        out.reward = clipped
        return out

    def reward_batch(
        self,
        samples: Sequence[Dict[str, Any]],
    ) -> List[RewardOutput]:
        """批量计算奖励. 每个样本 dict 需含 problem/response, 可选 gold_answer/gold_formal. """
        results: List[RewardOutput] = []
        for s in samples:
            results.append(
                self.reward(
                    problem=s.get("problem", ""),
                    response=s.get("response", ""),
                    gold_answer=s.get("gold_answer", s.get("answer", "")),
                    gold_formal=s.get("gold_formal", s.get("formal", "")),
                )
            )
        return results

    # ------------------------------------------------------------------ #
    # L1: Lean4 形式化验证
    # ------------------------------------------------------------------ #
    def _score_lean(
        self,
        lean_code: Optional[str],
        out: RewardOutput,
    ) -> LayerResult:
        if lean_code is None:
            return LayerResult(
                layer=RewardLayer.L1_LEAN,
                score=0.0,
                weight=0.0,
                contribution=0.0,
                detail="无 Lean4 证明代码块",
                passed=False,
            )
        # 检查不完整标记
        has_sorry = any(m in lean_code for m in PROOF_INCOMPLETE_MARKERS)
        has_complete = any(m in lean_code for m in PROOF_COMPLETE_MARKERS)

        if self.lean_verifier is not None:
            try:
                success, err = self.lean_verifier(lean_code)
            except Exception as exc:  # noqa: BLE001
                success, err = False, f"verifier error: {exc}"
            out.lean_proof_closed = bool(success)
            if success:
                return LayerResult(
                    layer=RewardLayer.L1_LEAN,
                    score=self.lean_bonus,
                    weight=0.0,
                    contribution=0.0,
                    detail="Lean4 验证通过, 证明闭合",
                    passed=True,
                )
            return LayerResult(
                layer=RewardLayer.L1_LEAN,
                score=self.lean_penalty if has_sorry else 0.0,
                weight=0.0,
                contribution=0.0,
                detail=f"Lean4 验证失败: {err}",
                passed=False,
            )
        # 降级: 规则检查
        if has_sorry:
            out.lean_proof_closed = False
            return LayerResult(
                layer=RewardLayer.L1_LEAN,
                score=self.lean_penalty,
                weight=0.0,
                contribution=0.0,
                detail="证明含 sorry/admit, 未闭合",
                passed=False,
            )
        if has_complete:
            out.lean_proof_closed = True
            return LayerResult(
                layer=RewardLayer.L1_LEAN,
                score=self.lean_bonus,
                weight=0.0,
                contribution=0.0,
                detail="证明含完成标记 (降级检查, 建议接 Lean4 subprocess)",
                passed=True,
            )
        out.lean_proof_closed = False
        return LayerResult(
            layer=RewardLayer.L1_LEAN,
            score=0.0,
            weight=0.0,
            contribution=0.0,
            detail="Lean4 代码块无明确完成/未完成标记",
            passed=False,
        )

    # ------------------------------------------------------------------ #
    # L2: SymPy 符号验证
    # ------------------------------------------------------------------ #
    def _score_sympy(
        self,
        pred: str,
        gold_answer: str,
        gold_formal: str,
    ) -> LayerResult:
        w = self.layer_weights[RewardLayer.L2_SYMPY]
        if not gold_answer and not gold_formal:
            # 无标准答案, 中性
            return LayerResult(
                layer=RewardLayer.L2_SYMPY,
                score=0.0,
                weight=w,
                contribution=0.0,
                detail="无标准答案, 跳过符号验证",
                passed=False,
            )
        if self.sympy_verifier is not None and gold_formal:
            try:
                ok = bool(self.sympy_verifier(pred, gold_formal))
            except Exception as exc:  # noqa: BLE001
                ok = False
                detail = f"sympy verifier error: {exc}"
            else:
                detail = "SymPy 等价性验证通过" if ok else "SymPy 等价性验证失败"
            score = 1.0 if ok else -0.5
            return LayerResult(
                layer=RewardLayer.L2_SYMPY,
                score=score,
                weight=w,
                contribution=score * w,
                detail=detail,
                passed=ok,
            )
        # 降级: 字符串规范化比对
        ok = self._normalize_equal(pred, gold_answer)
        score = 1.0 if ok else -0.5
        return LayerResult(
            layer=RewardLayer.L2_SYMPY,
            score=score,
            weight=w,
            contribution=score * w,
            detail="符号等价 (字符串规范化比对, 降级)" ,
            passed=ok,
        )

    # ------------------------------------------------------------------ #
    # L3: 数值验证
    # ------------------------------------------------------------------ #
    def _score_numeric(
        self,
        pred: str,
        gold_answer: str,
        gold_formal: str,
    ) -> LayerResult:
        w = self.layer_weights[RewardLayer.L3_NUMERIC]
        if not gold_answer and not gold_formal:
            return LayerResult(
                layer=RewardLayer.L3_NUMERIC,
                score=0.0,
                weight=w,
                contribution=0.0,
                detail="无标准答案, 跳过数值验证",
                passed=False,
            )
        pred_val = self._parse_number(pred)
        gold_val = self._parse_number(gold_answer) or self._parse_number(gold_formal)
        if pred_val is None or gold_val is None:
            # 无法数值化, 退化为字符串比对
            ok = self._normalize_equal(pred, gold_answer)
            score = 0.5 if ok else -0.4
            return LayerResult(
                layer=RewardLayer.L3_NUMERIC,
                score=score,
                weight=w,
                contribution=score * w,
                detail="答案非数值, 退化字符串比对",
                passed=ok,
            )
        # 数值比对 (含多采样扰动, 这里直接比对 + 容差)
        ok = abs(pred_val - gold_val) <= self.numeric_tol + self.numeric_tol * abs(gold_val)
        score = 1.0 if ok else -0.5
        return LayerResult(
            layer=RewardLayer.L3_NUMERIC,
            score=score,
            weight=w,
            contribution=score * w,
            detail=f"|{pred_val} - {gold_val}| <= tol -> {ok}",
            passed=ok,
        )

    # ------------------------------------------------------------------ #
    # L4: 数学家审查 (LLM-as-judge / 规则)
    # ------------------------------------------------------------------ #
    def _score_math_judge(
        self,
        problem: str,
        response: str,
        answer_correct: bool,
    ) -> LayerResult:
        w = self.layer_weights[RewardLayer.L4_MATHJUDGE]
        if self.math_judge is not None:
            try:
                score = float(self.math_judge(problem, response))
            except Exception as exc:  # noqa: BLE001
                score = 0.0
                detail = f"math judge error: {exc}"
            else:
                detail = f"数学家审查 (外部 judge): {score:.3f}"
            score = max(0.0, min(1.0, score))
            # 映射到 [-0.5, 1.0]
            score = score * 1.5 - 0.5
            return LayerResult(
                layer=RewardLayer.L4_MATHJUDGE,
                score=score,
                weight=w,
                contribution=score * w,
                detail=detail,
                passed=score >= 0.5,
            )
        # 降级: 规则评分
        score = self._rule_math_judge(problem, response, answer_correct)
        return LayerResult(
            layer=RewardLayer.L4_MATHJUDGE,
            score=score,
            weight=w,
            contribution=score * w,
            detail=f"规则审查 (正确={answer_correct}, 推理质量={score:.3f})",
            passed=score >= 0.5,
        )

    def _rule_math_judge(
        self,
        problem: str,
        response: str,
        answer_correct: bool,
    ) -> float:
        """规则化数学家审查: 推理步骤 + 关键词 + 正确性. """
        score = 0.0
        # 正确性基础分
        if answer_correct:
            score += 0.5
        # 推理步骤数 (适度加分, 过多/过少扣分)
        steps = response.count("\n") + 1
        if 3 <= steps <= 30:
            score += 0.2
        elif steps > 30:
            score -= 0.1
        # 关键推理关键词
        keywords = ("therefore", "hence", "since", "thus", "because",
                    "so", "由", "故", "因为", "所以", "因此", "于是")
        kw_hits = sum(1 for k in keywords if k.lower() in response.lower())
        score += min(0.2, kw_hits * 0.05)
        # 包含 boxed 答案标记
        if "\\boxed" in response or "answer is" in response.lower():
            score += 0.1
        return max(-0.5, min(1.0, score))

    # ------------------------------------------------------------------ #
    # L5: 启发式
    # ------------------------------------------------------------------ #
    def _score_heuristic(
        self,
        response: str,
        pred: str,
        answer_correct: bool,
    ) -> LayerResult:
        w = self.layer_weights[RewardLayer.L5_HEURISTIC]
        score = 0.0
        details: List[str] = []
        # 格式: 含 markdown / 分步
        if "```" in response:
            score += 0.1
            details.append("含代码块")
        # 长度适中
        length = len(response)
        if 50 <= length <= 2000:
            score += 0.15
            details.append("长度适中")
        elif length > 2000:
            score -= 0.1
            details.append("过长")
        # 符号一致性: 答案中无 sorry/admit (非 Lean 上下文)
        if "sorry" in response.lower() and "```lean" not in response.lower():
            score -= 0.2
            details.append("含 sorry (非证明上下文)")
        # 答案非空
        if pred:
            score += 0.1
            details.append("提取到答案")
        else:
            score -= 0.3
            details.append("未提取到答案")
        # 无重复 (简单检测连续重复行)
        lines = [ln for ln in response.split("\n") if ln.strip()]
        if lines and len(set(lines)) < len(lines) * 0.5 and len(lines) > 4:
            score -= 0.15
            details.append("疑似重复")
        score = max(-0.5, min(0.5, score))
        return LayerResult(
            layer=RewardLayer.L5_HEURISTIC,
            score=score,
            weight=w,
            contribution=score * w,
            detail="; ".join(details) if details else "无启发式信号",
            passed=score >= 0.0,
        )

    # ------------------------------------------------------------------ #
    # 提取与解析辅助
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_answer(response: str) -> str:
        """从响应中提取最终答案 (\\boxed{} 优先, 否则 answer is). """
        if not response:
            return ""
        m = BOXED_RE.search(response)
        if m:
            return m.group(1).strip()
        m = ANSWER_IS_RE.search(response)
        if m:
            return m.group(1).strip().rstrip(".")
        # 末行兜底
        lines = [ln.strip() for ln in response.split("\n") if ln.strip()]
        return lines[-1] if lines else ""

    @staticmethod
    def _extract_lean_block(response: str) -> Optional[str]:
        """提取首个 Lean4 代码块. """
        if not response:
            return None
        m = LEAN_BLOCK_RE.search(response)
        if m:
            return m.group(1)
        return None

    @staticmethod
    def _parse_number(s: str) -> Optional[float]:
        """尝试将字符串解析为数值 (支持分数 a/b). """
        if not s:
            return None
        s = s.strip().rstrip(".")
        # 去除前缀如 "x = "
        s = re.sub(r"^[^0-9\-]+", "", s)
        try:
            return float(s)
        except ValueError:
            pass
        # 分数
        m = re.match(r"^(-?\d+)\s*/\s*(\d+)$", s)
        if m:
            num, den = int(m.group(1)), int(m.group(2))
            if den != 0:
                return num / den
        # 含 pi
        if "pi" in s.lower():
            m = re.match(r"^(-?\d*)\s*\*?\s*pi$", s.lower())
            if m:
                coef = m.group(1)
                coef = float(coef) if coef not in ("", "-") else (1.0 if coef != "-" else -1.0)
                return coef * math.pi
        return None

    @staticmethod
    def _normalize_equal(a: str, b: str) -> bool:
        """字符串规范化比对 (去空格/统一大小写/去等号前缀). """
        if not a or not b:
            return False

        def norm(x: str) -> str:
            x = x.strip().lower()
            x = re.sub(r"\s+", "", x)
            # 去除 "x=" / "answer:" 等前缀
            x = re.sub(r"^(x|y|answer|ans|finalanswer)\s*=\s*", "", x)
            # 统一 * 与 省略
            return x

        na, nb = norm(a), norm(b)
        if na == nb:
            return True
        # 数值比对兜底
        va = MathRLHF._parse_number(a)
        vb = MathRLHF._parse_number(b)
        if va is not None and vb is not None:
            return abs(va - vb) <= 1e-6 * (1 + abs(vb))
        return False

    # ------------------------------------------------------------------ #
    # 统计
    # ------------------------------------------------------------------ #
    def summary(self) -> Dict[str, Any]:
        return {
            "layer_weights": {k.value: round(v, 4) for k, v in self.layer_weights.items()},
            "lean_bonus": self.lean_bonus,
            "lean_penalty": self.lean_penalty,
            "numeric_tol": self.numeric_tol,
            "numeric_samples": self.numeric_samples,
            "clip_range": [self.clip_min, self.clip_max],
            "has_external_lean_verifier": self.lean_verifier is not None,
            "has_external_sympy_verifier": self.sympy_verifier is not None,
            "has_external_math_judge": self.math_judge is not None,
        }


__all__ = [
    "MathRLHF",
    "RewardLayer",
    "LayerResult",
    "RewardOutput",
    "DEFAULT_LAYER_WEIGHTS",
    "DEFAULT_LEAN_BONUS",
    "DEFAULT_LEAN_PENALTY",
    "DEFAULT_NUMERIC_TOL",
    "DEFAULT_NUMERIC_SAMPLES",
    "REWARD_CLIP_MIN",
    "REWARD_CLIP_MAX",
    "WRONG_ANSWER_PENALTY",
]
