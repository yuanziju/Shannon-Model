"""Isabelle / Metamath 预留接口 (IsabelleProver / MetamathProver).

当前为预留接口骨架, 不实际调用 Isabelle / Metamath. 提供:
    - 统一的 ProverResult / ProverState 数据结构.
    - is_available() 返回 False (未实现).
    - prove / check_statement / extract_state 返回 not_implemented 结果.

未来集成时, 只需替换 prove / extract_state 的实现, 保持接口不变.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class IsabelleConfig:
    """Isabelle 配置."""

    isabelle_executable: str = "isabelle"
    timeout_sec: float = 60.0
    enable_subprocess: bool = True
    fallback_on_missing: bool = True
    session_name: str = "HOL"


@dataclass
class MetamathConfig:
    """Metamath 配置."""

    metamath_executable: str = "metamath"
    timeout_sec: float = 30.0
    enable_subprocess: bool = True
    fallback_on_missing: bool = True
    database: str = "set.mm"


@dataclass
class ProverResult:
    """通用证明结果 (Isabelle / Metamath 共用)."""

    success: bool
    prover: str = ""
    statement: str = ""
    proof: str = ""
    error_message: str = ""
    elapsed_sec: float = 0.0
    fallback_used: bool = False
    raw_output: str = ""
    not_implemented: bool = False


@dataclass
class ProverState:
    """通用证明状态."""

    goal: str = ""
    hypotheses: List[str] = field(default_factory=list)
    raw_text: str = ""

    def is_closed(self) -> bool:
        return self.goal == ""


class IsabelleProver:
    """Isabelle/HOL 证明器 (预留).

    未来集成路径:
        1. isabelle process -T <theory> 调用 Isabelle Scala API.
        2. 解析 PIDE 输出提取证明状态.
        3. 支持 Isar 与 apply 风格脚本.
    """

    NAME = "isabelle"

    def __init__(self, config: Optional[IsabelleConfig] = None, **kwargs):
        self.cfg = config or IsabelleConfig(**kwargs)

    def is_available(self) -> bool:
        if not self.cfg.enable_subprocess:
            return False
        return shutil.which(self.cfg.isabelle_executable) is not None

    def prove(
        self, statement: str, proof: Optional[str] = None, *, name: str = "main"
    ) -> ProverResult:
        t0 = time.time()
        return ProverResult(
            success=False,
            prover=self.NAME,
            statement=statement,
            proof=proof or "",
            error_message="Isabelle integration not yet implemented",
            elapsed_sec=time.time() - t0,
            fallback_used=self.cfg.fallback_on_missing,
            not_implemented=True,
        )

    def check_statement(self, statement: str, *, name: str = "stmt") -> ProverResult:
        return self.prove(statement, proof="sorry", name=name)

    def extract_state(self, isabelle_output: str) -> List[ProverState]:
        # 预留: Isabelle PIDE 输出解析
        return []

    def __repr__(self) -> str:  # pragma: no cover
        return f"IsabelleProver(available={self.is_available()}, session={self.cfg.session_name})"


class MetamathProver:
    """Metamath 证明器 (预留).

    未来集成路径:
        1. metamath verify <database> 验证 .mm 证明.
        2. 解析 step 输出提取证明状态.
    """

    NAME = "metamath"

    def __init__(self, config: Optional[MetamathConfig] = None, **kwargs):
        self.cfg = config or MetamathConfig(**kwargs)

    def is_available(self) -> bool:
        if not self.cfg.enable_subprocess:
            return False
        return shutil.which(self.cfg.metamath_executable) is not None

    def prove(
        self, statement: str, proof: Optional[str] = None, *, name: str = "main"
    ) -> ProverResult:
        t0 = time.time()
        return ProverResult(
            success=False,
            prover=self.NAME,
            statement=statement,
            proof=proof or "",
            error_message="Metamath integration not yet implemented",
            elapsed_sec=time.time() - t0,
            fallback_used=self.cfg.fallback_on_missing,
            not_implemented=True,
        )

    def check_statement(self, statement: str, *, name: str = "stmt") -> ProverResult:
        return self.prove(statement, name=name)

    def extract_state(self, metamath_output: str) -> List[ProverState]:
        return []

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"MetamathProver(available={self.is_available()}, "
            f"database={self.cfg.database})"
        )
