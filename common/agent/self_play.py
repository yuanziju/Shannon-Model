"""Self-Play 自我对弈 - proposer / solver / judge 三方对弈.

SelfPlay 通过生成-验证-修复闭环产生自监督训练数据, 用于 Phase6 自我进化
训练. 与代码生成 (CodeAgent) 和理科推理 Self-Play 联动.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


class GameRole(Enum):
    PROPOSER = "proposer"  # 出题者: 生成挑战任务
    SOLVER = "solver"      # 解题者: 求解任务
    JUDGE = "judge"        # 裁判: 验证解答正确性


class Outcome(Enum):
    SOLVED = "SOLVED"          # solver 正确解答
    FAILED = "FAILED"          # solver 解答错误
    INVALID_TASK = "INVALID"   # proposer 出题无效
    DISPUTED = "DISPUTED"      # judge 无法裁定


@dataclass
class PlayEpisode:
    """一次 Self-Play 对弈记录. """

    task: str
    proposer_output: str
    solver_output: str
    judge_verdict: str
    outcome: Outcome
    reward_proposer: float
    reward_solver: float
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class SelfPlay:
    """Self-Play 对弈引擎.

    三角色均由 (可能是同一) 15B 模型扮演, 通过角色 prompt 切换:
        - proposer: 生成难度自适应的挑战任务
        - solver:   在给定预算内求解
        - judge:    验证解答, 给出 verdict 与置信度

    Args:
        actor_fn: 角色扮演调用, ``fn(role: GameRole, prompt: str, ctx: dict) -> str``.
        verifier: 可选的硬验证器 (如代码执行/Lean 证明器),
            ``fn(task: str, solution: str) -> Optional[bool]``. 优先于 judge.
        difficulty: 初始难度 (0-1), proposer 据此调节任务难度.
    """

    def __init__(
        self,
        actor_fn: Optional[Callable[[GameRole, str, Dict[str, Any]], str]] = None,
        verifier: Optional[Callable[[str, str], Optional[bool]]] = None,
        difficulty: float = 0.3,
        rng_seed: Optional[int] = None,
    ) -> None:
        self.actor_fn = actor_fn or self._default_actor
        self.verifier = verifier
        self.difficulty = max(0.0, min(1.0, float(difficulty)))
        self._rng = random.Random(rng_seed)
        self._episodes: List[PlayEpisode] = []
        # 经验回放缓冲 (强化学习用)
        self._replay: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # 单轮对弈
    # ------------------------------------------------------------------ #
    def play(
        self,
        task_seed: Optional[str] = None,
        solver_budget: int = 3,
        context: Optional[Dict[str, Any]] = None,
    ) -> PlayEpisode:
        """执行一轮 proposer -> solver -> judge 对弈. """
        ctx = dict(context or {})
        # 1. proposer 出题
        task = task_seed
        if task is None:
            task = self.actor_fn(
                GameRole.PROPOSER,
                f"difficulty={self.difficulty:.2f} generate_one_task",
                ctx,
            )
        proposer_output = task

        # 2. solver 解题 (允许 budget 次尝试)
        solver_output = ""
        for attempt in range(max(1, int(solver_budget))):
            ctx_attempt = dict(ctx)
            ctx_attempt["attempt"] = attempt
            ctx_attempt["prev_solution"] = solver_output
            solver_output = self.actor_fn(
                GameRole.SOLVER, task, ctx_attempt
            )
            # 若有硬验证器, 提前短路
            if self.verifier is not None:
                verdict = self.verifier(task, solver_output)
                if verdict is True:
                    return self._finalize(
                        task, proposer_output, solver_output, "verified_correct",
                        Outcome.SOLVED, ctx
                    )
                if verdict is False and attempt == solver_budget - 1:
                    return self._finalize(
                        task, proposer_output, solver_output, "verified_wrong",
                        Outcome.FAILED, ctx
                    )

        # 3. judge 裁定
        judge_input = f"TASK:{task}\nSOLUTION:{solver_output}"
        judge_verdict = self.actor_fn(GameRole.JUDGE, judge_input, ctx)
        outcome, confidence = self._parse_verdict(judge_verdict)

        # 任务有效性检查
        if outcome == Outcome.INVALID_TASK:
            return self._finalize(
                task, proposer_output, solver_output, judge_verdict, Outcome.INVALID_TASK, ctx, confidence
            )

        ep = self._finalize(
            task, proposer_output, solver_output, judge_verdict, outcome, ctx, confidence
        )
        self._update_difficulty(outcome)
        return ep

    # ------------------------------------------------------------------ #
    # 批量对弈 + 经验回放
    # ------------------------------------------------------------------ #
    def tournament(self, n_episodes: int, task_seeds: Optional[Sequence[str]] = None) -> List[PlayEpisode]:
        """批量对弈, 返回所有 episode. """
        episodes: List[PlayEpisode] = []
        seeds = list(task_seeds) if task_seeds else [None] * n_episodes
        for i in range(int(n_episodes)):
            seed = seeds[i] if i < len(seeds) else None
            ep = self.play(task_seed=seed)
            episodes.append(ep)
            self._push_replay(ep)
        return episodes

    def sample_replay(self, batch_size: int = 8, prefer_solved: bool = True) -> List[Dict[str, Any]]:
        """从经验回放缓冲采样. ``prefer_solved=True`` 时优先采样成功 episode. """
        if not self._replay:
            return []
        pool = self._replay
        if prefer_solved:
            solved = [e for e in pool if e["outcome"] == Outcome.SOLVED.value]
            rest = [e for e in pool if e["outcome"] != Outcome.SOLVED.value]
            picked = self._rng.sample(solved, min(len(solved), batch_size))
            remain = max(0, batch_size - len(picked))
            if remain:
                picked += self._rng.sample(rest, min(len(rest), remain))
            return picked
        return self._rng.sample(pool, min(len(pool), batch_size))

    # ------------------------------------------------------------------ #
    # 内部
    # ------------------------------------------------------------------ #
    def _finalize(
        self,
        task: str,
        proposer_output: str,
        solver_output: str,
        judge_verdict: str,
        outcome: Outcome,
        ctx: Dict[str, Any],
        confidence: float = 0.0,
    ) -> PlayEpisode:
        # 奖励设计: proposer 出难题且被解出 -> 高奖励 (鼓励有挑战性但可解)
        # solver 解出 -> 正奖励, 失败 -> 负奖励
        if outcome == Outcome.SOLVED:
            r_proposer = 0.5 + 0.5 * self.difficulty
            r_solver = 1.0
        elif outcome == Outcome.FAILED:
            r_proposer = -0.2 if self.difficulty > 0.8 else 0.1
            r_solver = -0.5
        elif outcome == Outcome.INVALID_TASK:
            r_proposer = -1.0
            r_solver = 0.0
        else:  # DISPUTED
            r_proposer = 0.0
            r_solver = 0.0
        ep = PlayEpisode(
            task=task,
            proposer_output=proposer_output,
            solver_output=solver_output,
            judge_verdict=judge_verdict,
            outcome=outcome,
            reward_proposer=r_proposer,
            reward_solver=r_solver,
            confidence=confidence,
            metadata=dict(ctx),
        )
        self._episodes.append(ep)
        return ep

    def _update_difficulty(self, outcome: Outcome) -> None:
        # 自适应难度: solver 频繁成功 -> 提升难度; 频繁失败 -> 降低
        if outcome == Outcome.SOLVED:
            self.difficulty = min(1.0, self.difficulty + 0.05)
        elif outcome == Outcome.FAILED:
            self.difficulty = max(0.0, self.difficulty - 0.03)

    def _parse_verdict(self, verdict: str) -> Tuple[Outcome, float]:
        v = (verdict or "").lower()
        if "invalid" in v:
            return Outcome.INVALID_TASK, 0.6
        if "correct" in v or "solved" in v or "true" in v:
            return Outcome.SOLVED, 0.85
        if "wrong" in v or "incorrect" in v or "false" in v:
            return Outcome.FAILED, 0.85
        if "uncertain" in v or "disputed" in v:
            return Outcome.DISPUTED, 0.4
        return Outcome.DISPUTED, 0.3

    def _push_replay(self, ep: PlayEpisode) -> None:
        self._replay.append(
            {
                "task": ep.task,
                "solution": ep.solver_output,
                "outcome": ep.outcome.value,
                "reward_solver": ep.reward_solver,
                "confidence": ep.confidence,
            }
        )
        # 限制回放缓冲大小
        if len(self._replay) > 10_000:
            self._replay = self._replay[-10_000:]

    @staticmethod
    def _default_actor(role: GameRole, prompt: str, ctx: Dict[str, Any]) -> str:
        if role == GameRole.PROPOSER:
            return f"task_{len(prompt)}"
        if role == GameRole.SOLVER:
            return f"solution_to:{prompt[:30]}"
        return "correct"

    # ------------------------------------------------------------------ #
    # 统计
    # ------------------------------------------------------------------ #
    @property
    def episodes(self) -> List[PlayEpisode]:
        return list(self._episodes)

    @property
    def solve_rate(self) -> float:
        if not self._episodes:
            return 0.0
        solved = sum(1 for e in self._episodes if e.outcome == Outcome.SOLVED)
        return solved / len(self._episodes)

    @property
    def replay_size(self) -> int:
        return len(self._replay)


__all__ = ["SelfPlay", "PlayEpisode", "GameRole", "Outcome"]
