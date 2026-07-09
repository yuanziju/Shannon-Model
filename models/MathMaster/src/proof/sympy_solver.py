"""SymPy 符号计算求解器 (SymPySolver).

提供统一的符号计算接口, 封装 SymPy 的常用功能:
    - solve     : 方程/方程组求解
    - simplify  : 表达式化简
    - diff      : 微分
    - integrate : 积分 (不定/定)
    - limit     : 极限
    - series    : 幂级数展开
    - matrix_ops: 矩阵运算
    - verify_identity : 恒等式验证
    - factor    : 因式分解
    - expand    : 展开表达式

设计原则:
    1. 延迟导入 sympy, 缺失时给出明确错误而非 ImportError.
    2. 接受字符串表达式, 内部 parse, 避免上层传入不安全对象.
    3. 全部方法返回 SymPyResult / 原生 sympy 对象, 不抛业务异常.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence


@dataclass
class SymPyResult:
    """统一的符号计算结果."""

    success: bool
    value: Any = None                # 原生 sympy 对象或其列表
    repr: str = ""                   # 字符串表示
    error: str = ""
    elapsed_sec: float = 0.0
    meta: dict = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - 便捷
        if not self.success:
            return f"[SymPySolver:FAILED] {self.error}"
        return self.repr if self.repr else repr(self.value)


def _require_sympy():
    try:
        import sympy as sp  # noqa: F401
        return sp
    except ImportError as e:  # pragma: no cover - 环境依赖
        raise RuntimeError(
            "SymPySolver requires `sympy`. Install with `pip install sympy`."
        ) from e


class SymPySolver:
    """SymPy 符号计算统一封装."""

    def __init__(self, *, timeout_sec: float = 30.0, **kwargs):
        self.timeout_sec = timeout_sec
        self._sp = _require_sympy()
        self._kwargs = kwargs
        # 缓存已声明的符号, 避免重复 Symbol 创建导致不同对象
        self._symbol_cache: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------
    def _get_symbol(self, name: str):
        if name not in self._symbol_cache:
            self._symbol_cache[name] = self._sp.Symbol(name)
        return self._symbol_cache[name]

    def _parse(self, expr_str: str):
        """安全解析表达式字符串, 注入已声明符号."""
        sp = self._sp
        # 收集表达式中可能出现的符号名 (简单标识符)
        import re
        names = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr_str or ""))
        # 排除 SymPy 内建函数/关键字
        reserved = {
            "sin", "cos", "tan", "cot", "sec", "csc",
            "asin", "acos", "atan", "acot", "asec", "acsc",
            "sinh", "cosh", "tanh", "coth",
            "log", "ln", "exp", "sqrt", "Abs", "abs",
            "pi", "E", "I", "oo", "zoo", "nan",
            "Sum", "Product", "Integral", "Derivative",
            "Rational", "Integer", "Float", "Matrix",
            "Piecewise", "Function", "Symbol",
            "And", "Or", "Not", "Xor", "Implies",
            "Eq", "Ne", "Lt", "Le", "Gt", "Ge",
            "Min", "Max", "floor", "ceiling",
            "gamma", "factorial", "binomial",
        }
        local_dict = {}
        for n in names:
            if n in reserved:
                continue
            if n in self._symbol_cache:
                local_dict[n] = self._symbol_cache[n]
            else:
                sym = sp.Symbol(n)
                self._symbol_cache[n] = sym
                local_dict[n] = sym
        return sp.sympify(expr_str, locals=local_dict, evaluate=True)

    def _ok(self, value: Any, *, meta: Optional[dict] = None) -> SymPyResult:
        return SymPyResult(
            success=True,
            value=value,
            repr=self._to_str(value),
            meta=meta or {},
        )

    def _fail(self, err: str, *, meta: Optional[dict] = None) -> SymPyResult:
        return SymPyResult(success=False, error=err, meta=meta or {})

    @staticmethod
    def _to_str(value: Any) -> str:
        try:
            return str(value)
        except Exception:  # pragma: no cover
            return repr(value)

    @staticmethod
    def _coerce_symbols(symbols):
        """接受 str / Symbol / list/tuple, 返回 list[Symbol]."""
        sp = _require_sympy()
        if symbols is None:
            return []
        if isinstance(symbols, str):
            return [sp.Symbol(s) for s in symbols.replace(",", " ").split()]
        if isinstance(symbols, (list, tuple)):
            out = []
            for s in symbols:
                if isinstance(s, str):
                    out.append(sp.Symbol(s))
                else:
                    out.append(s)
            return out
        # 单个 Symbol
        return [symbols]

    # ------------------------------------------------------------------
    # 求解
    # ------------------------------------------------------------------
    def solve(self, equation: str, var: Any = "x", **kwargs) -> SymPyResult:
        """求解方程 (组).

        Args:
            equation: 表达式字符串. 等式可用 `=` 或 `==` 或 `Eq(...)`,
                      如 `x**2 - 4` (默认=0) 或 `x**2 = 4`.
            var: 待求变量 (str / Symbol / list).
            **kwargs: 透传 sympy.solve (如 dict=True, domain=S.Reals).
        """
        try:
            sp = self._sp
            # 处理等号
            eq_str = equation.strip()
            if "=" in eq_str and "==" not in eq_str and "Eq(" not in eq_str:
                lhs, rhs = eq_str.split("=", 1)
                expr = self._parse(lhs) - self._parse(rhs)
            elif "==" in eq_str and "Eq(" not in eq_str:
                lhs, rhs = eq_str.split("==", 1)
                expr = self._parse(lhs) - self._parse(rhs)
            else:
                expr = self._parse(eq_str)
            symbols = self._coerce_symbols(var)
            sol = sp.solve(expr, *symbols, **kwargs)
            return self._ok(sol, meta={"equation": eq_str, "var": str(var)})
        except Exception as e:
            return self._fail(f"solve failed: {e!r}", meta={"equation": equation})

    # ------------------------------------------------------------------
    # 化简
    # ------------------------------------------------------------------
    def simplify(self, expr: str, **kwargs) -> SymPyResult:
        try:
            e = self._parse(expr)
            return self._ok(self._sp.simplify(e, **kwargs))
        except Exception as e:
            return self._fail(f"simplify failed: {e!r}")

    # ------------------------------------------------------------------
    # 微分
    # ------------------------------------------------------------------
    def diff(self, expr: str, var: Any = "x", order: int = 1, **kwargs) -> SymPyResult:
        try:
            e = self._parse(expr)
            symbols = self._coerce_symbols(var)
            if not symbols:
                return self._fail("diff requires a variable")
            res = e
            for _ in range(int(order)):
                res = self._sp.diff(res, symbols[0], **kwargs)
            return self._ok(res, meta={"var": str(var), "order": order})
        except Exception as e:
            return self._fail(f"diff failed: {e!r}")

    # ------------------------------------------------------------------
    # 积分
    # ------------------------------------------------------------------
    def integrate(
        self,
        expr: str,
        var: Any = "x",
        bounds: Optional[Sequence[float]] = None,
        **kwargs,
    ) -> SymPyResult:
        """积分. bounds=(a, b) 表示定积分, 否则不定积分."""
        try:
            e = self._parse(expr)
            symbols = self._coerce_symbols(var)
            if not symbols:
                return self._fail("integrate requires a variable")
            v = symbols[0]
            if bounds is not None and len(bounds) == 2:
                res = self._sp.integrate(e, (v, bounds[0], bounds[1]), **kwargs)
            else:
                res = self._sp.integrate(e, v, **kwargs)
            return self._ok(res, meta={"var": str(v), "bounds": bounds})
        except Exception as e:
            return self._fail(f"integrate failed: {e!r}")

    # ------------------------------------------------------------------
    # 极限
    # ------------------------------------------------------------------
    def limit(
        self, expr: str, var: Any = "x", to: Any = 0, direction: str = "+", **kwargs
    ) -> SymPyResult:
        try:
            e = self._parse(expr)
            symbols = self._coerce_symbols(var)
            if not symbols:
                return self._fail("limit requires a variable")
            v = symbols[0]
            res = self._sp.limit(e, v, to, direction, **kwargs)
            return self._ok(res, meta={"var": str(v), "to": str(to), "dir": direction})
        except Exception as e:
            return self._fail(f"limit failed: {e!r}")

    # ------------------------------------------------------------------
    # 幂级数
    # ------------------------------------------------------------------
    def series(
        self, expr: str, var: Any = "x", around: Any = 0, n: int = 6, **kwargs
    ) -> SymPyResult:
        try:
            e = self._parse(expr)
            symbols = self._coerce_symbols(var)
            if not symbols:
                return self._fail("series requires a variable")
            v = symbols[0]
            res = self._sp.series(e, v, around, n, **kwargs)
            return self._ok(res, meta={"var": str(v), "around": str(around), "n": n})
        except Exception as e:
            return self._fail(f"series failed: {e!r}")

    # ------------------------------------------------------------------
    # 矩阵运算
    # ------------------------------------------------------------------
    def matrix_ops(
        self, matrix: Any, op: str = "det", **kwargs
    ) -> SymPyResult:
        """矩阵运算.

        Args:
            matrix: 嵌套 list / Matrix 字符串.
            op: det / inv / transpose / rank / trace / eigenvals / eigenvectors /
                rref / charpoly.
        """
        try:
            sp = self._sp
            if isinstance(matrix, str):
                m = self._parse(matrix)
            else:
                m = sp.Matrix(matrix)
            op = op.lower()
            if op == "det":
                res = m.det()
            elif op == "inv":
                res = m.inv()
            elif op in ("transpose", "t"):
                res = m.T
            elif op == "rank":
                res = m.rank()
            elif op == "trace":
                res = m.trace()
            elif op == "eigenvals":
                res = m.eigenvals()
            elif op == "eigenvectors":
                res = m.eigenvects()
            elif op == "rref":
                res = m.rref()
            elif op == "charpoly":
                res = m.charpoly()
            else:
                return self._fail(f"unknown matrix op: {op}")
            return self._ok(res, meta={"op": op})
        except Exception as e:
            return self._fail(f"matrix_ops({op}) failed: {e!r}")

    # ------------------------------------------------------------------
    # 恒等式验证
    # ------------------------------------------------------------------
    def verify_identity(self, lhs: str, rhs: str, var: Any = "x") -> SymPyResult:
        """验证 lhs ≡ rhs (化简为零).

        实现: simplify(lhs - rhs) == 0 视为恒等.
        """
        try:
            sp = self._sp
            l = self._parse(lhs)
            r = self._parse(rhs)
            diff_expr = sp.simplify(l - r)
            is_id = diff_expr == 0
            return SymPyResult(
                success=True,
                value=is_id,
                repr=str(is_id),
                meta={
                    "lhs": lhs,
                    "rhs": rhs,
                    "diff_simplified": self._to_str(diff_expr),
                },
            )
        except Exception as e:
            return self._fail(f"verify_identity failed: {e!r}")

    # ------------------------------------------------------------------
    # 因式分解
    # ------------------------------------------------------------------
    def factor(self, expr: str, **kwargs) -> SymPyResult:
        try:
            e = self._parse(expr)
            return self._ok(self._sp.factor(e, **kwargs))
        except Exception as e:
            return self._fail(f"factor failed: {e!r}")

    # ------------------------------------------------------------------
    # 展开
    # ------------------------------------------------------------------
    def expand(self, expr: str, **kwargs) -> SymPyResult:
        try:
            e = self._parse(expr)
            return self._ok(self._sp.expand(e, **kwargs))
        except Exception as e:
            return self._fail(f"expand failed: {e!r}")

    # ------------------------------------------------------------------
    # 便捷: 批量求解
    # ------------------------------------------------------------------
    def solve_system(self, equations: Sequence[str], vars: Sequence[str], **kwargs) -> SymPyResult:
        """求解方程组."""
        try:
            sp = self._sp
            eqs = []
            for eq_str in equations:
                if "=" in eq_str and "==" not in eq_str:
                    lhs, rhs = eq_str.split("=", 1)
                    eqs.append(self._parse(lhs) - self._parse(rhs))
                else:
                    eqs.append(self._parse(eq_str))
            symbols = [self._get_symbol(v) if isinstance(v, str) else v for v in vars]
            sol = sp.solve(eqs, *symbols, **kwargs)
            return self._ok(sol, meta={"equations": list(equations), "vars": list(vars)})
        except Exception as e:
            return self._fail(f"solve_system failed: {e!r}")

    def __repr__(self) -> str:  # pragma: no cover - 便捷
        return f"SymPySolver(timeout_sec={self.timeout_sec})"
