"""证明路由器 (ProofRouter).

根据命题类型 / 关键字 / 用户偏好, 自动将证明/计算请求路由到合适的后端:

    - lean4         : 形式化定理证明 (theorem / lemma / iff / equality 需形式化)
    - coq           : Coq 风格形式化 (备份)
    - sympy         : 符号计算 (代数化简 / 微积分 / 方程求解)
    - number_theory : 数论验证 (素数 / 哥德巴赫 / 黎曼零点 / 反例搜索)
    - isabelle      : 预留
    - metamath      : 预留

提供:
    - route          : 路由单个请求
    - route_batch    : 批量路由 (并行)
    - get_capabilities : 查询各后端能力描述
    - 自动路由策略   : 关键字 + 启发式 + 可用性优先
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


# 后端标识
BACKEND_LEAN4 = "lean4"
BACKEND_Coq = "coq"
BACKEND_SYMPY = "sympy"
BACKEND_NUMBER_THEORY = "number_theory"
BACKEND_ISABELLE = "isabelle"
BACKEND_METAMATH = "metamath"

ALL_BACKENDS: tuple[str, ...] = (
    BACKEND_LEAN4,
    BACKEND_Coq,
    BACKEND_SYMPY,
    BACKEND_NUMBER_THEORY,
    BACKEND_ISABELLE,
    BACKEND_METAMATH,
)


@dataclass
class ProofRequest:
    """证明 / 计算请求."""

    statement: str
    operation: str = "prove"        # prove / solve / simplify / verify / counterexample / ...
    backend: Optional[str] = None   # 用户指定后端; None 触发自动路由
    variables: List[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)


@dataclass
class ProofResponse:
    """路由响应."""

    backend: str
    operation: str
    success: bool
    result: Any = None
    error: str = ""
    elapsed_sec: float = 0.0
    fallback_chain: List[str] = field(default_factory=list)  # 已尝试的后端序列
    auto_routed: bool = False


class ProofRouter:
    """证明路由器."""

    def __init__(
        self,
        *,
        sympy_solver: Optional[Any] = None,
        lean4_prover: Optional[Any] = None,
        coq_prover: Optional[Any] = None,
        number_theory_verifier: Optional[Any] = None,
        isabelle_prover: Optional[Any] = None,
        metamath_prover: Optional[Any] = None,
        num_workers: int = 4,
        auto_init: bool = True,
    ):
        self.num_workers = max(1, int(num_workers))
        # 懒加载: 仅在需要时初始化后端
        self._extern = {
            BACKEND_SYMPY: sympy_solver,
            BACKEND_LEAN4: lean4_prover,
            BACKEND_Coq: coq_prover,
            BACKEND_NUMBER_THEORY: number_theory_verifier,
            BACKEND_ISABELLE: isabelle_prover,
            BACKEND_METAMATH: metamath_prover,
        }
        self._auto_init = auto_init
        self._initialized: dict[str, Any] = {}

    # ==================================================================
    # 后端懒加载
    # ==================================================================
    def _get_backend(self, name: str) -> Optional[Any]:
        if name in self._initialized:
            return self._initialized[name]
        # 优先用外部注入
        if self._extern.get(name) is not None:
            self._initialized[name] = self._extern[name]
            return self._initialized[name]
        if not self._auto_init:
            return None
        # 懒加载
        try:
            obj = self._lazy_init(name)
        except Exception:
            obj = None
        self._initialized[name] = obj
        return obj

    def _lazy_init(self, name: str) -> Optional[Any]:
        if name == BACKEND_SYMPY:
            from .sympy_solver import SymPySolver
            return SymPySolver()
        if name == BACKEND_LEAN4:
            from .lean4_prover import Lean4Prover
            return Lean4Prover()
        if name == BACKEND_Coq:
            from .coq_prover import CoqProver
            return CoqProver()
        if name == BACKEND_NUMBER_THEORY:
            from .number_theory_verifier import NumberTheoryVerifier
            return NumberTheoryVerifier()
        if name == BACKEND_ISABELLE:
            from .isabelle_metamath import IsabelleProver
            return IsabelleProver()
        if name == BACKEND_METAMATH:
            from .isabelle_metamath import MetamathProver
            return MetamathProver()
        return None

    # ==================================================================
    # 能力查询
    # ==================================================================
    CAPABILITIES: Dict[str, dict] = {
        BACKEND_LEAN4: {
            "description": "Lean4 全程形式化证明",
            "operations": ("prove", "check_statement", "generate_tactic", "compile_mathlib"),
            "strengths": ("theorem", "lemma", "equality", "iff", "induction"),
            "available_when": "lean executable installed",
        },
        BACKEND_Coq: {
            "description": "Coq 形式化证明 (备份)",
            "operations": ("prove", "check_statement", "compile"),
            "strengths": ("theorem", "inductive", "tactic"),
            "available_when": "coqc executable installed",
        },
        BACKEND_SYMPY: {
            "description": "SymPy 符号计算",
            "operations": (
                "solve", "simplify", "diff", "integrate", "limit",
                "series", "matrix_ops", "verify_identity", "factor", "expand",
            ),
            "strengths": ("algebra", "calculus", "equation", "identity"),
            "available_when": "sympy python package",
        },
        BACKEND_NUMBER_THEORY: {
            "description": "自研数论验证 + 反例发现",
            "operations": (
                "goldbach_check", "prime_check", "riemann_zeros_check",
                "prime_counting", "modular_arithmetic", "discrete_log",
                "crt", "gcd_extended", "find_counterexample",
                "batch_counterexample_search", "goldbach_counterexample_search",
                "riemann_zero_off_line_search",
            ),
            "strengths": ("prime", "goldbach", "riemann", "modular", "counterexample"),
            "available_when": "pure python (no external deps)",
        },
        BACKEND_ISABELLE: {
            "description": "Isabelle/HOL (预留)",
            "operations": ("prove", "check_statement"),
            "strengths": ("higher-order-logic", "isar"),
            "available_when": "not implemented yet",
        },
        BACKEND_METAMATH: {
            "description": "Metamath (预留)",
            "operations": ("prove", "check_statement"),
            "strengths": ("minimal-axiom", "database"),
            "available_when": "not implemented yet",
        },
    }

    def get_capabilities(self, backend: Optional[str] = None) -> Dict[str, dict]:
        """查询后端能力. backend=None 返回全部."""
        if backend is None:
            return dict(self.CAPABILITIES)
        if backend not in self.CAPABILITIES:
            return {}
        info = dict(self.CAPABILITIES[backend])
        # 附加实时可用性
        obj = self._get_backend(backend)
        info["instantiated"] = obj is not None
        if obj is not None and hasattr(obj, "is_available"):
            try:
                info["available"] = bool(obj.is_available())
            except Exception:
                info["available"] = False
        else:
            info["available"] = False
        return info

    # ==================================================================
    # 自动路由
    # ==================================================================
    def _auto_route(self, request: ProofRequest) -> str:
        """启发式自动路由: 关键字 + operation 匹配.

        优先级: number_theory > sympy > lean4 > coq > isabelle > metamath
        """
        stmt = (request.statement or "").lower()
        op = (request.operation or "prove").lower()

        # 1) 数论关键字
        nt_keywords = (
            "prime", "goldbach", "riemann", "zeta", "modular",
            "discrete log", "crt", "chinese remainder", "counterexample",
            "mersenne", "fermat", "collatz", "π(x)", "pi(x)",
        )
        for kw in nt_keywords:
            if kw in stmt:
                return BACKEND_NUMBER_THEORY

        # 2) operation 显式为数论/反例
        if op in {
            "prime_check", "goldbach_check", "riemann_zeros_check",
            "prime_counting", "find_counterexample", "batch_counterexample_search",
            "goldbach_counterexample_search", "riemann_zero_off_line_search",
            "discrete_log", "crt", "modular_arithmetic",
        }:
            return BACKEND_NUMBER_THEORY

        # 3) 符号计算关键字
        sp_keywords = (
            "integrate", "derivative", "differentiate", "limit",
            "series", "expand", "factor", "simplify", "matrix",
            "solve for", "equation",
        )
        for kw in sp_keywords:
            if kw in stmt:
                return BACKEND_SYMPY

        # 4) operation 显式为 sympy
        if op in {
            "solve", "simplify", "diff", "integrate", "limit",
            "series", "matrix_ops", "verify_identity", "factor", "expand",
        }:
            return BACKEND_SYMPY

        # 5) 形式化定理关键字
        formal_keywords = (
            "theorem", "lemma", "iff", "induction", "forall", "exists",
            "rational", "natural", "inductive", "predicate",
        )
        for kw in formal_keywords:
            if kw in stmt:
                # 优先 Lean4, 不可用时 Coq
                lean = self._get_backend(BACKEND_LEAN4)
                if lean is not None and getattr(lean, "is_available", lambda: False)():
                    return BACKEND_LEAN4
                coq = self._get_backend(BACKEND_Coq)
                if coq is not None and getattr(coq, "is_available", lambda: False)():
                    return BACKEND_Coq
                return BACKEND_LEAN4

        # 6) 默认: 数论 (反例搜索) > sympy > lean4
        for fallback in (BACKEND_NUMBER_THEORY, BACKEND_SYMPY, BACKEND_LEAN4):
            obj = self._get_backend(fallback)
            if obj is not None:
                return fallback
        return BACKEND_LEAN4

    # ==================================================================
    # 单请求路由 + 执行
    # ==================================================================
    def route(
        self,
        request: ProofRequest,
        *,
        fallback: bool = True,
    ) -> ProofResponse:
        """路由单个请求到合适后端并执行.

        Args:
            request: 证明/计算请求.
            fallback: 主后端失败时是否自动降级到备选后端.

        Returns:
            ProofResponse.
        """
        import time
        t0 = time.time()
        backend = request.backend or self._auto_route(request)
        auto_routed = request.backend is None
        fallback_chain: List[str] = []
        last_error = ""

        # 候选降级链
        candidates: List[str] = [backend]
        if fallback:
            for b in ALL_BACKENDS:
                if b not in candidates:
                    candidates.append(b)

        for cand in candidates:
            obj = self._get_backend(cand)
            if obj is None:
                continue
            fallback_chain.append(cand)
            try:
                result = self._dispatch(obj, cand, request)
                if result is not None:
                    return ProofResponse(
                        backend=cand,
                        operation=request.operation,
                        success=True,
                        result=result,
                        elapsed_sec=time.time() - t0,
                        fallback_chain=fallback_chain,
                        auto_routed=auto_routed,
                    )
            except Exception as e:
                last_error = f"{cand}: {e!r}"
                continue
        return ProofResponse(
            backend=backend,
            operation=request.operation,
            success=False,
            error=last_error or "no available backend",
            elapsed_sec=time.time() - t0,
            fallback_chain=fallback_chain,
            auto_routed=auto_routed,
        )

    # ==================================================================
    # 批量路由
    # ==================================================================
    def route_batch(
        self,
        requests: Sequence[ProofRequest],
        *,
        fallback: bool = True,
    ) -> List[ProofResponse]:
        """批量并行路由.

        Args:
            requests: 请求列表.
            fallback: 单请求内是否触发后端降级.

        Returns:
            响应列表 (顺序与 requests 一致).
        """
        if not requests:
            return []
        if len(requests) == 1 or self.num_workers <= 1:
            return [self.route(r, fallback=fallback) for r in requests]
        results: List[Optional[ProofResponse]] = [None] * len(requests)
        with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
            futures = {
                ex.submit(self.route, r, fallback=fallback): i for i, r in enumerate(requests)
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:
                    results[idx] = ProofResponse(
                        backend="unknown",
                        operation=requests[idx].operation,
                        success=False,
                        error=f"router error: {e!r}",
                    )
        return [r for r in results if r is not None]

    # ==================================================================
    # 分派到具体后端方法
    # ==================================================================
    def _dispatch(self, backend_obj: Any, backend_name: str, request: ProofRequest) -> Any:
        """根据 backend_name + operation 调用对应方法."""
        op = (request.operation or "prove").lower()
        stmt = request.statement
        params = dict(request.params)
        variables = request.variables

        if backend_name == BACKEND_SYMPY:
            return self._dispatch_sympy(backend_obj, op, stmt, variables, params)
        if backend_name == BACKEND_NUMBER_THEORY:
            return self._dispatch_nt(backend_obj, op, stmt, params)
        if backend_name == BACKEND_LEAN4:
            return self._dispatch_lean4(backend_obj, op, stmt, params)
        if backend_name == BACKEND_Coq:
            return self._dispatch_coq(backend_obj, op, stmt, params)
        if backend_name == BACKEND_ISABELLE:
            proof = params.get("proof")
            if op == "check_statement":
                return backend_obj.check_statement(stmt)
            return backend_obj.prove(stmt, proof=proof)
        if backend_name == BACKEND_METAMATH:
            proof = params.get("proof")
            if op == "check_statement":
                return backend_obj.check_statement(stmt)
            return backend_obj.prove(stmt, proof=proof)
        return None

    def _dispatch_sympy(self, obj, op, stmt, variables, params):
        if op in ("solve", "equation"):
            var = variables[0] if variables else params.get("var", "x")
            return obj.solve(stmt, var)
        if op == "simplify":
            return obj.simplify(stmt)
        if op == "diff":
            var = variables[0] if variables else params.get("var", "x")
            order = int(params.get("order", 1))
            return obj.diff(stmt, var, order=order)
        if op == "integrate":
            var = variables[0] if variables else params.get("var", "x")
            bounds = params.get("bounds")
            return obj.integrate(stmt, var, bounds=bounds)
        if op == "limit":
            var = variables[0] if variables else params.get("var", "x")
            to = params.get("to", 0)
            direction = params.get("direction", "+")
            return obj.limit(stmt, var, to, direction=direction)
        if op == "series":
            var = variables[0] if variables else params.get("var", "x")
            around = params.get("around", 0)
            n = int(params.get("n", 6))
            return obj.series(stmt, var, around=around, n=n)
        if op == "matrix_ops":
            m = params.get("matrix", stmt)
            mop = params.get("matrix_op", "det")
            return obj.matrix_ops(m, op=mop)
        if op == "verify_identity":
            rhs = params.get("rhs", "")
            var = variables[0] if variables else params.get("var", "x")
            return obj.verify_identity(stmt, rhs, var=var)
        if op == "factor":
            return obj.factor(stmt)
        if op == "expand":
            return obj.expand(stmt)
        # 默认 prove -> solve
        var = variables[0] if variables else params.get("var", "x")
        return obj.solve(stmt, var)

    def _dispatch_nt(self, obj, op, stmt, params):
        if op == "prime_check":
            n = int(params.get("n", stmt))
            return obj.prime_check(n)
        if op == "goldbach_check":
            n = int(params.get("n", stmt))
            return obj.goldbach_check(n)
        if op == "riemann_zeros_check":
            num = int(params.get("num_zeros", stmt if str(stmt).isdigit() else 10))
            return obj.riemann_zeros_check(num)
        if op == "prime_counting":
            x = int(params.get("x", stmt))
            return obj.prime_counting(x)
        if op == "modular_arithmetic":
            return obj.modular_arithmetic(
                params["a"], params["b"], params["m"],
                op=params.get("mod_op", "+"),
            )
        if op == "discrete_log":
            return obj.discrete_log(
                params["base"], params["target"], params["mod"],
            )
        if op == "crt":
            return obj.crt(params["remainders"], params["moduli"])
        if op == "gcd_extended":
            return obj.gcd_extended(params["a"], params["b"])
        if op == "find_counterexample":
            search_range = int(params.get("search_range", 1000))
            return obj.find_counterexample(stmt, search_range)
        if op == "batch_counterexample_search":
            ranges = params.get("ranges", [100, 1000, 10000])
            return obj.batch_counterexample_search(stmt, ranges)
        if op == "goldbach_counterexample_search":
            max_n = int(params.get("max_n", stmt))
            return obj.goldbach_counterexample_search(max_n)
        if op == "riemann_zero_off_line_search":
            max_imag = float(params.get("max_imag", stmt))
            return obj.riemann_zero_off_line_search(max_imag)
        # 默认: 当作 prime_check
        try:
            return obj.prime_check(int(stmt))
        except (TypeError, ValueError):
            return None

    def _dispatch_lean4(self, obj, op, stmt, params):
        proof = params.get("proof")
        name = params.get("name", "main")
        if op == "check_statement":
            return obj.check_statement(stmt, name=name)
        if op == "compile_mathlib":
            return obj.compile_mathlib(
                params.get("project_dir"), timeout_sec=float(params.get("timeout_sec", 600.0))
            )
        if op == "generate_tactic":
            from .lean4_prover import ProofState
            state = params.get("state")
            if isinstance(state, dict):
                state = ProofState(
                    goal=state.get("goal", ""),
                    hypotheses=state.get("hypotheses", []),
                )
            elif state is None:
                state = ProofState(goal=stmt)
            return obj.generate_tactic(state, top_k=int(params.get("top_k", 5)))
        return obj.prove(stmt, proof=proof, name=name)

    def _dispatch_coq(self, obj, op, stmt, params):
        proof = params.get("proof")
        name = params.get("name", "main")
        if op == "check_statement":
            return obj.check_statement(stmt, name=name)
        if op == "compile":
            return obj.compile(
                params.get("code", stmt),
                is_file=bool(params.get("is_file", False)),
                timeout_sec=params.get("timeout_sec"),
            )
        return obj.prove(stmt, proof=proof, name=name)

    def __repr__(self) -> str:  # pragma: no cover - 便捷
        return f"ProofRouter(num_workers={self.num_workers}, auto_init={self._auto_init})"
