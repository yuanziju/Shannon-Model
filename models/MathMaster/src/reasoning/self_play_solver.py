"""Self-Play 迭代逼近求解器 - 开放问题多轮逼近.

针对 *开放型* 数学问题 (无已知闭式解/猜想待证), 通过 Proposer(提议者) ->
Solver(求解者) -> Judge(裁判) 三方多轮迭代, 逐步逼近一个可接受的解答.
与 ``common.agent.self_play.SelfPlay`` (任务求解) 和
``self_play_debate.SelfPlayDebate`` (命题检验) 的区别:

    - ``SelfPlay``         固定任务, 一次性 propose-solve-judge;
    - ``SelfPlayDebate``   正/反方辩论 *命题是否成立*;
    - ``SelfPlaySolver``   迭代逼近 *开放问题*, 每轮基于裁判反馈 refine 解答,
      逐步提升 judge 置信度直至收敛.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

__all__ = ["SelfPlaySolver", "SolverResult", "SolverJudgment"]


@dataclass
class SolverJudgment:
    """裁判对一次解答的评判. """

    correct: bool = False
    confidence: float = 0.0
    feedback: str = ""          # 给 proposer/solver 的改进建议
    score: float = 0.0          # 解答质量分 [0, 1]
    errors: List[str] = field(default_factory=list)


@dataclass
class SolverIteration:
    """单轮迭代记录. """

    iteration: int
    proposal: str
    solution: str
    judgment: SolverJudgment


@dataclass
class SolverResult:
    """Self-Play 求解结果. """

    problem: str = ""
    iterations: List[SolverIteration] = field(default_factory=list)
    best_solution: str = ""
    best_score: float = 0.0
    converged: bool = False
    total_iterations: int = 0
    final_confidence: float = 0.0
    elapsed: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class SelfPlaySolver:
    """Self-Play 迭代逼近求解器.

    Args:
        model: 15B 循环主体.
        max_iterations: 最大迭代轮数.
        convergence_threshold: judge 置信度达到该阈值即收敛.
        proposer_temp / solver_temp: 生成温度 (影响探索性).
    """

    def __init__(
        self,
        model,
        max_iterations: int = 10,
        convergence_threshold: float = 0.9,
        proposer_temp: float = 0.8,
        solver_temp: float = 0.4,
    ) -> None:
        self.model = model
        self.max_iterations = max(1, int(max_iterations))
        self.convergence_threshold = float(convergence_threshold)
        self.proposer_temp = float(proposer_temp)
        self.solver_temp = float(solver_temp)

    # ------------------------------------------------------------------ #
    # 公共接口
    # ------------------------------------------------------------------ #
    def solve(self, problem: str, seed_proposal: Optional[str] = None) -> SolverResult:
        """多轮迭代逼近开放问题.

        每轮:
            1. Proposer 给出问题切入/分解提议 (首轮可由 seed_proposal 提供);
            2. Solver 依据提议产出解答;
            3. Judge 评判解答正确性, 给出反馈与置信度;
            4. 若未收敛, 将反馈回灌给下一轮 Proposer.
        """
        t0 = time.time()
        iterations: List[SolverIteration] = []
        best_solution = ""
        best_score = -1.0
        prev_feedback = ""
        proposal = seed_proposal or ""

        for it in range(self.max_iterations):
            proposal = self.propose(problem, proposal if it == 0 else None, prev_feedback)
            solution = self.solve_step(problem, proposal, iterations)
            judgment = self.judge(problem, proposal, solution)

            iterations.append(
                SolverIteration(
                    iteration=it,
                    proposal=proposal,
                    solution=solution,
                    judgment=judgment,
                )
            )

            if judgment.score > best_score:
                best_score = judgment.score
                best_solution = solution

            prev_feedback = judgment.feedback

            if judgment.confidence >= self.convergence_threshold and judgment.correct:
                break

        final = iterations[-1].judgment if iterations else SolverJudgment()
        return SolverResult(
            problem=problem,
            iterations=iterations,
            best_solution=best_solution,
            best_score=max(0.0, best_score),
            converged=final.correct and final.confidence >= self.convergence_threshold,
            total_iterations=len(iterations),
            final_confidence=final.confidence,
            elapsed=time.time() - t0,
            metadata={
                "convergence_threshold": self.convergence_threshold,
                "max_iterations": self.max_iterations,
            },
        )

    # ------------------------------------------------------------------ #
    # 三角色
    # ------------------------------------------------------------------ #
    def propose(
        self,
        problem: str,
        previous_proposal: Optional[str],
        judge_feedback: str,
    ) -> str:
        """Proposer: 给出问题分解/切入提议. """
        prompt = (
            "[ROLE: Proposer] Decompose the open problem into an attack plan.\n"
            f"Problem: {problem}\n"
            f"Previous proposal: {previous_proposal or '(none)'}\n"
            f"Judge feedback: {judge_feedback or '(none)'}\n"
            f"Refined attack plan:"
        )
        text, conf = self._call_model(self.model, prompt, max_new=64, temperature=self.proposer_temp)
        return f"[plan conf={conf:.2f}] {text or 'direct-attack'}"

    def solve_step(
        self,
        problem: str,
        proposal: str,
        history: List[SolverIteration],
    ) -> str:
        """Solver: 依据提议产出解答 (可参考历史尝试避免重复错误). """
        recent = ""
        if history:
            last = history[-1]
            recent = f"Previous attempt (score={last.judgment.score:.2f}): {last.solution[:200]}"
        prompt = (
            "[ROLE: Solver] Produce a solution following the plan.\n"
            f"Problem: {problem}\n"
            f"Plan: {proposal}\n"
            f"{recent}\n"
            f"Solution:"
        )
        text, conf = self._call_model(self.model, prompt, max_new=96, temperature=self.solver_temp)
        return f"[sol conf={conf:.2f}] {text or 'partial-solution'}"

    def judge(self, problem: str, proposal: str, solution: str) -> SolverJudgment:
        """Judge: 评判解答正确性, 产出反馈. """
        prompt = (
            "[ROLE: Judge] Check correctness of the solution.\n"
            f"Problem: {problem}\n"
            f"Solution: {solution}\n"
            f"Is it correct? Give a score (0-1), confidence (0-1), and improvement feedback."
        )
        text, conf = self._call_model(self.model, prompt, max_new=48)

        score = self._score_solution(solution, conf)
        correct = score >= 0.6
        errors = self._detect_errors(solution, text)

        # 反馈: 若不正确, 指出错误方向
        if correct:
            feedback = "solution accepted; consider tighter proof" if score < 0.9 else "well done"
        else:
            feedback = text or "incorrect; revise the key step and retry"

        return SolverJudgment(
            correct=correct,
            confidence=conf,
            feedback=feedback,
            score=score,
            errors=errors,
        )

    # ------------------------------------------------------------------ #
    # 内部
    # ------------------------------------------------------------------ #
    @staticmethod
    def _score_solution(solution: str, conf: float) -> float:
        """解答质量启发式: confidence + 长度充实度. """
        if not solution:
            return 0.0
        len_term = min(1.0, len(solution) / 300.0) * 0.4
        return max(0.0, min(1.0, 0.6 * conf + len_term))

    @staticmethod
    def _detect_errors(solution: str, judge_text: str) -> List[str]:
        errs: List[str] = []
        low = (judge_text + " " + solution).lower()
        if "wrong" in low or "incorrect" in low or "错误" in low:
            errs.append("incorrect-step")
        if "gap" in low or "missing" in low or "跳跃" in low:
            errs.append("logical-gap")
        if "counterexample" in low or "反例" in low:
            errs.append("counterexample-found")
        return errs

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
        ids = SelfPlaySolver._to_ids(text)
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
