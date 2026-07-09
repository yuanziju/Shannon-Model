"""proof: MathMaster 形式化证明集成模块.

提供多种形式化/符号后端的统一接口:

    - Lean4Prover             : Lean4 全程形式化证明器
    - SymPySolver             : SymPy 符号计算求解器
    - NumberTheoryVerifier    : 自研数论验证器 (含反例发现强化)
    - CoqProver               : Coq 形式化证明器 (可选补充)
    - ProofRouter             : 证明路由器 (自动路由 + 批量路由)
    - IsabelleProver          : Isabelle/HOL 预留接口
    - MetamathProver          : Metamath 预留接口

参考: AGENTS.md (T2.3.2 Lean/Coq 通道, T2.5.3 形式化表示解析器),
       spec §5.6 / §7.1 SymPy/Lean 通道.
"""

from __future__ import annotations

from .sympy_solver import (
    SymPySolver,
    SymPyResult,
)
from .number_theory_verifier import (
    NumberTheoryVerifier,
    GoldbachResult,
    RiemannZeroResult,
    CounterexampleResult,
)
from .lean4_prover import (
    Lean4Prover,
    Lean4ProverConfig,
    LeanProofResult,
    ProofState,
    TacticCandidate,
    DEFAULT_TACTIC_TEMPLATES,
)
from .coq_prover import (
    CoqProver,
    CoqProverConfig,
    CoqProofResult,
    CoqProofState,
)
from .proof_router import (
    ProofRouter,
    ProofRequest,
    ProofResponse,
    BACKEND_LEAN4,
    BACKEND_Coq,
    BACKEND_SYMPY,
    BACKEND_NUMBER_THEORY,
    BACKEND_ISABELLE,
    BACKEND_METAMATH,
    ALL_BACKENDS,
)
from .isabelle_metamath import (
    IsabelleProver,
    IsabelleConfig,
    MetamathProver,
    MetamathConfig,
    ProverResult,
    ProverState,
)

__all__ = [
    # sympy_solver
    "SymPySolver",
    "SymPyResult",
    # number_theory_verifier
    "NumberTheoryVerifier",
    "GoldbachResult",
    "RiemannZeroResult",
    "CounterexampleResult",
    # lean4_prover
    "Lean4Prover",
    "Lean4ProverConfig",
    "LeanProofResult",
    "ProofState",
    "TacticCandidate",
    "DEFAULT_TACTIC_TEMPLATES",
    # coq_prover
    "CoqProver",
    "CoqProverConfig",
    "CoqProofResult",
    "CoqProofState",
    # proof_router
    "ProofRouter",
    "ProofRequest",
    "ProofResponse",
    "BACKEND_LEAN4",
    "BACKEND_Coq",
    "BACKEND_SYMPY",
    "BACKEND_NUMBER_THEORY",
    "BACKEND_ISABELLE",
    "BACKEND_METAMATH",
    "ALL_BACKENDS",
    # isabelle_metamath
    "IsabelleProver",
    "IsabelleConfig",
    "MetamathProver",
    "MetamathConfig",
    "ProverResult",
    "ProverState",
]

__version__ = "1.0.0"
