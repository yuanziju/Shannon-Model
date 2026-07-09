"""Coq 证明器 (CoqProver).

可选补充实现, 封装 coqc / coqtop 子进程调用, 提供与 Lean4Prover 对齐的接口:

    - prove           : 端到端证明 (statement -> proof -> verify)
    - check_statement : 仅类型检查 (admit 占位)
    - extract_state   : 从 Coq 输出提取证明状态
    - compile         : 编译 .v 文件 / 加载依赖

设计原则与 Lean4Prover 一致: 无 Coq 环境时降级返回, 不抛异常.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import List, Optional


# Coq 证明状态块: "1 subgoal\n  H : ...\n  ============================\n  goal"
_COQ_STATE_RE = re.compile(
    r"(?P<subgoals>\d+)\s+subgoal(?:s)?\s*\n(?P<hyps>(?:.+\n)*?)={4,}\s*\n(?P<goal>.+?)(?=\n\n|\Z)",
    re.DOTALL,
)

_COQ_ERROR_LINE_RE = re.compile(r"File\s+\"[^\"]+\",\s+line\s+(\d+)", re.MULTILINE)


@dataclass
class CoqProverConfig:
    """Coq 证明器配置."""

    coqc_executable: str = "coqc"
    coqtop_executable: str = "coqtop"
    timeout_sec: float = 30.0
    enable_subprocess: bool = True
    fallback_on_missing: bool = True
    temp_dir: Optional[str] = None
    require_imports: tuple[str, ...] = ()     # 默认 require 的库


@dataclass
class CoqProofResult:
    """Coq 证明结果."""

    success: bool
    statement: str = ""
    proof: str = ""
    error_message: str = ""
    error_line: int = -1
    has_admit: bool = False       # Coq 用 admit/Abort 表示未完成
    elapsed_sec: float = 0.0
    fallback_used: bool = False
    raw_output: str = ""


@dataclass
class CoqProofState:
    """Coq 证明状态."""

    goal: str = ""
    hypotheses: List[str] = field(default_factory=list)
    num_subgoals: int = 0
    raw_text: str = ""

    def is_closed(self) -> bool:
        return self.num_subgoals == 0


class CoqProver:
    """Coq 证明器."""

    def __init__(self, config: Optional[CoqProverConfig] = None, **kwargs):
        self.cfg = config or CoqProverConfig(**kwargs)
        self._coq_available: Optional[bool] = None

    # ------------------------------------------------------------------
    # 环境检测
    # ------------------------------------------------------------------
    def _check_coq(self) -> bool:
        if self._coq_available is not None:
            return self._coq_available
        if not self.cfg.enable_subprocess:
            self._coq_available = False
            return False
        self._coq_available = shutil.which(self.cfg.coqc_executable) is not None
        return self._coq_available

    def is_available(self) -> bool:
        return self._check_coq()

    # ------------------------------------------------------------------
    # 证明
    # ------------------------------------------------------------------
    def prove(
        self,
        statement: str,
        proof: Optional[str] = None,
        *,
        name: str = "main",
    ) -> CoqProofResult:
        """端到端证明.

        Args:
            statement: Coq 命题, 如 `forall n : nat, n + 0 = n`.
            proof:    tactic 脚本. None 时使用 `admit.` 占位.
            name:     Theorem 名称.
        """
        t0 = time.time()
        if proof is None:
            proof = "admit."
        has_admit = bool(re.search(r"\badmit\b", proof)) or "Abort" in proof

        header = ""
        for imp in self.cfg.require_imports:
            header += f"Require Import {imp}.\n"

        code = (
            f"{header}"
            f"Theorem {name} : {statement}.\n"
            f"Proof.\n"
            f"  {proof}\n"
            f"Qed.\n"
        )

        if not self._check_coq():
            if self.cfg.fallback_on_missing:
                ok, err = self._syntax_check(code)
                return CoqProofResult(
                    success=ok,
                    statement=statement,
                    proof=proof,
                    error_message=err,
                    has_admit=has_admit,
                    elapsed_sec=time.time() - t0,
                    fallback_used=True,
                    raw_output="(coqc not available)",
                )
            return CoqProofResult(
                success=False,
                statement=statement,
                proof=proof,
                error_message="coqc not available and fallback disabled",
                elapsed_sec=time.time() - t0,
            )

        ok, err, line, raw = self._run_coq_code(code)
        return CoqProofResult(
            success=ok,
            statement=statement,
            proof=proof,
            error_message=err,
            error_line=line,
            has_admit=has_admit,
            elapsed_sec=time.time() - t0,
            raw_output=raw,
        )

    # ------------------------------------------------------------------
    # 仅类型检查 (admit 占位)
    # ------------------------------------------------------------------
    def check_statement(self, statement: str, *, name: str = "stmt") -> CoqProofResult:
        return self.prove(statement, proof="admit.", name=name)

    # ------------------------------------------------------------------
    # 提取证明状态
    # ------------------------------------------------------------------
    def extract_state(self, coq_output: str) -> List[CoqProofState]:
        """从 coqtop 输出提取证明状态."""
        states: List[CoqProofState] = []
        for m in _COQ_STATE_RE.finditer(coq_output):
            hyps_text = m.group("hyps").strip()
            goal = m.group("goal").strip()
            hyps = [h.strip() for h in hyps_text.split("\n") if h.strip()]
            num_sub = int(m.group("subgoals"))
            states.append(
                CoqProofState(
                    goal=goal,
                    hypotheses=hyps,
                    num_subgoals=num_sub,
                    raw_text=m.group(0),
                )
            )
        return states

    # ------------------------------------------------------------------
    # 编译 .v 文件 / 加载依赖
    # ------------------------------------------------------------------
    def compile(
        self,
        code_or_path: str,
        *,
        is_file: bool = False,
        timeout_sec: Optional[float] = None,
    ) -> CoqProofResult:
        """编译 .v 源码或文件."""
        t0 = time.time()
        if not self._check_coq():
            return CoqProofResult(
                success=False,
                error_message="coqc not available",
                elapsed_sec=time.time() - t0,
                fallback_used=self.cfg.fallback_on_missing,
            )
        timeout = timeout_sec or self.cfg.timeout_sec

        if is_file:
            target = code_or_path
            cleanup = False
        else:
            tmpdir = tempfile.mkdtemp(prefix="coq_prove_", dir=self.cfg.temp_dir)
            target = os.path.join(tmpdir, "Proof.v")
            with open(target, "w", encoding="utf-8") as f:
                f.write(code_or_path)
            cleanup = True

        try:
            proc = subprocess.run(
                [self.cfg.coqc_executable, target],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.path.dirname(target) or None,
            )
            raw = (proc.stdout or "") + (proc.stderr or "")
            if proc.returncode == 0:
                return CoqProofResult(
                    success=True,
                    elapsed_sec=time.time() - t0,
                    raw_output=raw,
                )
            line = _parse_coq_error_line(raw)
            return CoqProofResult(
                success=False,
                error_message=raw.strip()[:2000],
                error_line=line,
                elapsed_sec=time.time() - t0,
                raw_output=raw,
            )
        except subprocess.TimeoutExpired:
            return CoqProofResult(
                success=False,
                error_message=f"timeout after {timeout}s",
                elapsed_sec=time.time() - t0,
            )
        finally:
            if cleanup:
                try:
                    shutil.rmtree(os.path.dirname(target), ignore_errors=True)
                except Exception:
                    pass

    # ==================================================================
    # 内部
    # ==================================================================
    def _run_coq_code(self, code: str) -> tuple[bool, str, int, str]:
        tmpdir = tempfile.mkdtemp(prefix="coq_prove_", dir=self.cfg.temp_dir)
        v_file = os.path.join(tmpdir, "Proof.v")
        try:
            with open(v_file, "w", encoding="utf-8") as f:
                f.write(code)
            try:
                proc = subprocess.run(
                    [self.cfg.coqc_executable, v_file],
                    capture_output=True,
                    text=True,
                    timeout=self.cfg.timeout_sec,
                    cwd=tmpdir,
                )
            except subprocess.TimeoutExpired:
                return (False, f"timeout after {self.cfg.timeout_sec}s", -1, "")
            except FileNotFoundError:
                return (False, "coqc not found", -1, "")

            raw = (proc.stdout or "") + (proc.stderr or "")
            if proc.returncode == 0:
                return (True, "", -1, raw)
            line = _parse_coq_error_line(raw)
            return (False, raw.strip()[:2000], line, raw)
        finally:
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass

    def _syntax_check(self, code: str) -> tuple[bool, str]:
        if not code.strip():
            return (False, "empty code")
        if "admit" in code:
            return (False, "incomplete proof (admit detected)")
        for kw in ("Theorem", "Lemma", "Proposition", "Goal"):
            if kw in code:
                if "Proof." not in code:
                    return (False, f"{kw} without 'Proof.'")
                if "Qed." not in code and "Defined." not in code:
                    return (False, f"{kw} without 'Qed.' or 'Defined.'")
                break
        return (True, "")

    def __repr__(self) -> str:  # pragma: no cover - 便捷
        return f"CoqProver(coq_available={self._check_coq()}, timeout_sec={self.cfg.timeout_sec})"


def _parse_coq_error_line(err: str) -> int:
    m = _COQ_ERROR_LINE_RE.search(err)
    if m:
        return int(m.group(1))
    return -1
