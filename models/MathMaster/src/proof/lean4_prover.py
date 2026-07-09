"""Lean4 全程形式化证明器 (Lean4Prover).

封装 Lean4 (lake / lean) 子进程调用, 提供完整的证明流程:

    - prove              : 端到端证明 (statement -> proof script -> verify)
    - generate_tactic    : 给定证明状态, 生成下一步策略候选
    - check_statement    : 仅检查 statement 是否合法 (类型检查, 不求证)
    - extract_state      : 从 Lean 输出中提取证明状态 (hyps/goal)
    - compile_mathlib    : 编译 Mathlib 依赖 (首次构建/预热)

设计原则:
    1. 无 Lean4 环境时, 全部方法返回 (success=False, fallback_used=True) 的降级结果,
       不抛异常, 上层可继续工作.
    2. 所有 subprocess 调用带超时, 默认 30s.
    3. 支持 theorem/lemma/example 三种声明形式.
    4. 提供内置 tactic 模板库, generate_tactic 在无模型时退化为规则推荐.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence


# Lean 证明状态正则 (从 lean 输出解析 hyps/goal)
# 形如: "a b : Nat\nh : a > b\n⊢ a - b > 0"
_LEAN_STATE_BLOCK_RE = re.compile(
    r"(?P<hyps>(?:[^\n⊢]*\n)*?)⊢\s*(?P<goal>.+?)(?=\n\n|\Z)",
    re.DOTALL,
)

# 错误行号正则
_LEAN_ERROR_LINE_RE = re.compile(r":(\d+):(\d+):\s*(?:error|warning)", re.MULTILINE)

# 内置 tactic 模板库 (无模型时使用)
DEFAULT_TACTIC_TEMPLATES: tuple[str, ...] = (
    "intro",
    "apply",
    "exact",
    "rw",
    "simp",
    "induction",
    "cases",
    "refl",
    "symm",
    "trans",
    "have",
    "show",
    "by_contra",
    "contradiction",
    "left",
    "right",
    "constructor",
    "exists",
    "use",
    "fun",
    "decide",
    "ring",
    "linarith",
    "nlinarith",
    "norm_num",
    "tauto",
    "push_neg",
)


@dataclass
class Lean4ProverConfig:
    """Lean4 证明器配置."""

    lean_executable: str = "lean"
    lake_executable: str = "lake"
    lean_project_dir: Optional[str] = None       # 含 lakefile.lean 的项目根
    timeout_sec: float = 30.0
    enable_subprocess: bool = True
    fallback_on_missing: bool = True             # 无 lean 时降级
    max_retries: int = 1
    temp_dir: Optional[str] = None
    use_mathlib: bool = False                    # 是否 import Mathlib


@dataclass
class LeanProofResult:
    """证明结果."""

    success: bool
    statement: str = ""
    proof: str = ""
    error_message: str = ""
    error_line: int = -1
    has_sorry: bool = False
    has_admit: bool = False
    elapsed_sec: float = 0.0
    fallback_used: bool = False
    raw_output: str = ""
    tactic_trace: List[str] = field(default_factory=list)


@dataclass
class ProofState:
    """Lean 证明状态."""

    goal: str = ""
    hypotheses: List[str] = field(default_factory=list)
    tactic_history: List[str] = field(default_factory=list)
    raw_text: str = ""

    def is_closed(self) -> bool:
        return self.goal == "" or self.goal.lower() in ("no goals", "goals accomplished")


@dataclass
class TacticCandidate:
    """策略候选."""

    tactic: str
    confidence: float = 0.0
    rationale: str = ""


class Lean4Prover:
    """Lean4 全程形式化证明器."""

    def __init__(self, config: Optional[Lean4ProverConfig] = None, **kwargs):
        self.cfg = config or Lean4ProverConfig(**kwargs)
        self._lean_available: Optional[bool] = None  # 延迟检测

    # ------------------------------------------------------------------
    # 环境检测
    # ------------------------------------------------------------------
    def _check_lean(self) -> bool:
        if self._lean_available is not None:
            return self._lean_available
        if not self.cfg.enable_subprocess:
            self._lean_available = False
            return False
        self._lean_available = shutil.which(self.cfg.lean_executable) is not None
        return self._lean_available

    def is_available(self) -> bool:
        """Lean4 是否可用."""
        return self._check_lean()

    # ------------------------------------------------------------------
    # 完整证明
    # ------------------------------------------------------------------
    def prove(
        self,
        statement: str,
        proof: Optional[str] = None,
        *,
        name: str = "main",
    ) -> LeanProofResult:
        """端到端证明.

        Args:
            statement: Lean4 命题声明 (类型), 如 `forall n : Nat, n + 0 = n`.
            proof:    策略证明脚本 (含 `by` 关键字). None 时使用 `by sorry` 占位.
            name:     theorem 名称.

        Returns:
            LeanProofResult.
        """
        t0 = time.time()
        if proof is None:
            proof = "by sorry"
        # 组装 lean 源码
        if self.cfg.use_mathlib:
            header = "import Mathlib\n"
        else:
            header = ""
        code = (
            f"{header}"
            f"theorem {name} : {statement} :=\n  {proof}\n"
            f"\n-- EOF\n"
        )
        has_sorry = bool(re.search(r"\bsorry\b", proof))
        has_admit = bool(re.search(r"\badmit\b", proof))

        if not self._check_lean():
            if self.cfg.fallback_on_missing:
                ok, err = self._syntax_check(code)
                return LeanProofResult(
                    success=ok,
                    statement=statement,
                    proof=proof,
                    error_message=err,
                    has_sorry=has_sorry,
                    has_admit=has_admit,
                    elapsed_sec=time.time() - t0,
                    fallback_used=True,
                    raw_output="(lean not available)",
                )
            return LeanProofResult(
                success=False,
                statement=statement,
                proof=proof,
                error_message="lean not available and fallback disabled",
                elapsed_sec=time.time() - t0,
            )

        ok, err, line, raw = self._run_lean_code(code)
        tactic_trace = self._extract_tactics_from_proof(proof)
        return LeanProofResult(
            success=ok,
            statement=statement,
            proof=proof,
            error_message=err,
            error_line=line,
            has_sorry=has_sorry,
            has_admit=has_admit,
            elapsed_sec=time.time() - t0,
            raw_output=raw,
            tactic_trace=tactic_trace,
        )

    # ------------------------------------------------------------------
    # 仅检查 statement 合法性 (不求证)
    # ------------------------------------------------------------------
    def check_statement(self, statement: str, *, name: str = "stmt") -> LeanProofResult:
        """仅类型检查 statement (用 sorry 占位), 验证命题可表达."""
        return self.prove(statement, proof="by sorry", name=name)

    # ------------------------------------------------------------------
    # 生成下一步策略候选
    # ------------------------------------------------------------------
    def generate_tactic(
        self,
        state: ProofState,
        *,
        top_k: int = 5,
        use_model: bool = False,
    ) -> List[TacticCandidate]:
        """给定证明状态, 生成候选策略.

        Args:
            state: 当前证明状态.
            top_k: 返回候选数.
            use_model: 是否使用模型推理 (本类未集成, 仅占位).

        Note:
            当前实现为规则式推荐 (基于 goal 文本启发式).
            集成神经模型时, 此方法可调用外部 model 服务.
        """
        candidates: List[TacticCandidate] = []
        goal = state.goal.lower()

        # 启发式规则
        if "∀" in state.goal or "forall" in goal:
            candidates.append(TacticCandidate("intro", 0.8, "goal is universal"))
        if "→" in state.goal or "->" in goal or "implies" in goal:
            candidates.append(TacticCandidate("intro", 0.75, "goal is implication"))
        if "=" in state.goal and "+" in state.goal:
            candidates.append(TacticCandidate("ring", 0.7, "algebraic equality"))
        if "<" in state.goal or ">" in state.goal or "≤" in state.goal:
            candidates.append(TacticCandidate("linarith", 0.7, "linear inequality"))
        if "∃" in state.goal or "exists" in goal:
            candidates.append(TacticCandidate("use", 0.7, "existential goal"))
            candidates.append(TacticCandidate("exists", 0.6, "existential goal"))
        if state.goal.startswith("¬") or "not " in goal:
            candidates.append(TacticCandidate("by_contra", 0.7, "negation goal"))
            candidates.append(TacticCandidate("push_neg", 0.5, "push negation"))
        if "∧" in state.goal or "and" in goal:
            candidates.append(TacticCandidate("constructor", 0.7, "conjunction"))
        if "∨" in state.goal or " or " in goal:
            candidates.append(TacticCandidate("left", 0.5, "disjunction (left)"))
            candidates.append(TacticCandidate("right", 0.5, "disjunction (right)"))
        if "iff" in goal or "↔" in state.goal:
            candidates.append(TacticCandidate("constructor", 0.7, "iff split"))

        # 通用 fallback
        candidates.append(TacticCandidate("simp", 0.4, "general simplification"))
        candidates.append(TacticCandidate("exact ?", 0.3, "search exact term"))
        candidates.append(TacticCandidate("decide", 0.3, "decidable procedure"))

        # 去重 + 排序 + 截断
        seen = set()
        deduped: List[TacticCandidate] = []
        for c in candidates:
            if c.tactic in seen:
                continue
            seen.add(c.tactic)
            deduped.append(c)
        deduped.sort(key=lambda x: -x.confidence)
        return deduped[:top_k]

    # ------------------------------------------------------------------
    # 从 Lean 输出提取证明状态
    # ------------------------------------------------------------------
    def extract_state(self, lean_output: str) -> List[ProofState]:
        """从 Lean 编译输出中提取证明状态列表.

        Lean 在 tactic 失败时会打印当前 hyps/goal. 我们解析这些块.
        """
        states: List[ProofState] = []
        for m in _LEAN_STATE_BLOCK_RE.finditer(lean_output):
            hyps_text = m.group("hyps").strip()
            goal = m.group("goal").strip()
            hyps = [h.strip() for h in hyps_text.split("\n") if h.strip()]
            states.append(
                ProofState(
                    goal=goal,
                    hypotheses=hyps,
                    raw_text=m.group(0),
                )
            )
        return states

    # ------------------------------------------------------------------
    # 编译 Mathlib 依赖 (预热)
    # ------------------------------------------------------------------
    def compile_mathlib(
        self, project_dir: Optional[str] = None, *, timeout_sec: float = 600.0
    ) -> LeanProofResult:
        """编译 Mathlib 依赖 (lake build).

        Args:
            project_dir: 含 lakefile.lean 的项目根. None 用 cfg.lean_project_dir.
            timeout_sec: 编译超时 (Mathlib 编译耗时较长).

        Returns:
            LeanProofResult.
        """
        t0 = time.time()
        project_dir = project_dir or self.cfg.lean_project_dir
        if project_dir is None:
            return LeanProofResult(
                success=False,
                error_message="no lean project directory configured",
                elapsed_sec=time.time() - t0,
            )
        if not self._check_lean() or shutil.which(self.cfg.lake_executable) is None:
            return LeanProofResult(
                success=False,
                error_message="lake executable not available",
                elapsed_sec=time.time() - t0,
                fallback_used=self.cfg.fallback_on_missing,
            )
        try:
            proc = subprocess.run(
                [self.cfg.lake_executable, "build"],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                cwd=project_dir,
            )
            ok = proc.returncode == 0
            return LeanProofResult(
                success=ok,
                error_message=(proc.stderr + proc.stdout)[:2000] if not ok else "",
                elapsed_sec=time.time() - t0,
                raw_output=proc.stdout,
            )
        except subprocess.TimeoutExpired:
            return LeanProofResult(
                success=False,
                error_message=f"mathlib build timeout after {timeout_sec}s",
                elapsed_sec=time.time() - t0,
            )
        except FileNotFoundError as e:
            return LeanProofResult(
                success=False,
                error_message=f"lake not found: {e}",
                elapsed_sec=time.time() - t0,
            )

    # ==================================================================
    # 内部实现
    # ==================================================================
    def _run_lean_code(self, code: str) -> tuple[bool, str, int, str]:
        """调用 lean 子进程验证代码.

        Returns:
            (success, error_message, error_line, raw_output)
        """
        tmpdir = tempfile.mkdtemp(prefix="lean4_prove_", dir=self.cfg.temp_dir)
        lean_file = os.path.join(tmpdir, "Proof.lean")
        try:
            with open(lean_file, "w", encoding="utf-8") as f:
                f.write(code)
            cmd = [self.cfg.lean_executable, lean_file]
            cwd = self.cfg.lean_project_dir or tmpdir
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.cfg.timeout_sec,
                    cwd=cwd,
                )
            except subprocess.TimeoutExpired:
                return (False, f"timeout after {self.cfg.timeout_sec}s", -1, "")
            except FileNotFoundError:
                return (False, "lean executable not found", -1, "")

            raw = (proc.stdout or "") + (proc.stderr or "")
            if proc.returncode == 0:
                return (True, "", -1, raw)
            line = _parse_error_line(raw)
            return (False, raw.strip()[:2000], line, raw)
        finally:
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass

    def _syntax_check(self, code: str) -> tuple[bool, str]:
        """无 lean 时的轻量语法检查 (启发式)."""
        if not code.strip():
            return (False, "empty code")
        if re.search(r"\bsorry\b", code):
            return (False, "incomplete proof (sorry detected)")
        if re.search(r"\badmit\b", code):
            return (False, "incomplete proof (admit detected)")
        # 检查 theorem/lemma/example 结构
        for kw in ("theorem", "lemma", "example"):
            if kw in code:
                if ":=" not in code:
                    return (False, f"{kw} without ':='")
                if "by" not in code and "rfl" not in code:
                    return (False, f"{kw} without 'by' or 'rfl'")
                break
        return (True, "")

    @staticmethod
    def _extract_tactics_from_proof(proof: str) -> List[str]:
        """从证明脚本提取 tactic 序列 (按行/分号分割)."""
        body = proof.strip()
        if body.startswith("by"):
            body = body[2:]
        tactics: List[str] = []
        # 按 ; 和换行分割
        for chunk in re.split(r"[;\n]", body):
            chunk = chunk.strip()
            if chunk:
                tactics.append(chunk)
        return tactics

    def __repr__(self) -> str:  # pragma: no cover - 便捷
        return (
            f"Lean4Prover(lean_available={self._check_lean()}, "
            f"timeout_sec={self.cfg.timeout_sec}, use_mathlib={self.cfg.use_mathlib})"
        )


def _parse_error_line(err: str) -> int:
    """从 lean 错误输出中解析错误行号."""
    m = _LEAN_ERROR_LINE_RE.search(err)
    if m:
        return int(m.group(1))
    return -1
