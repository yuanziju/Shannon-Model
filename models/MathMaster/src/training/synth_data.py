"""合成数学数据生成 - 9 领域 + 竞赛 + 前沿.

MathDataGenerator 程序化生成可验证的数学训练数据, 覆盖:

    9 大基础领域:
        arithmetic        算术 (整数/分数/小数四则运算)
        algebra           代数 (方程/不等式/多项式)
        geometry          几何 (平面/立体/解析几何)
        analysis          分析 (微积分/级数/极限)
        number_theory     数论 (素数/同余/整除)
        abstract_algebra  抽象代数 (群/环/域)
        topology          拓扑 (开集/连通/紧致)
        discrete_math     离散 (图论/组合/逻辑)
        probability       概率 (分布/期望/贝叶斯)

    2 类高难度:
        competition       竞赛 (AMC/AIME/IMO 风格)
        frontier          前沿 (研究级开放问题)

每条样本为 dict:
    {id, domain, difficulty, problem, answer, solution, formal (可选), tags}
所有答案均可程序化验证 (数值/符号/布尔), 供 RLHF 奖励层复用.
"""

from __future__ import annotations

import math
import random
import string
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .curriculum import CurriculumLevel


# 9 大基础领域 + 竞赛 + 前沿
MATH_DOMAINS: Tuple[str, ...] = (
    "arithmetic",
    "algebra",
    "geometry",
    "analysis",
    "number_theory",
    "abstract_algebra",
    "topology",
    "discrete_math",
    "probability",
    "competition",
    "frontier",
)

# 领域 -> 默认难度
DOMAIN_DEFAULT_DIFFICULTY: Dict[str, str] = {
    "arithmetic": "basic",
    "algebra": "intermediate",
    "geometry": "intermediate",
    "analysis": "advanced",
    "number_theory": "intermediate",
    "abstract_algebra": "advanced",
    "topology": "advanced",
    "discrete_math": "intermediate",
    "probability": "intermediate",
    "competition": "advanced",
    "frontier": "frontier",
}


@dataclass
class ProblemSpec:
    """单条数学问题规格 (内部用). """

    domain: str
    difficulty: str
    problem: str
    answer: str
    solution: str = ""
    formal: str = ""           # Lean4 / SymPy 形式 (可选)
    tags: List[str] = field(default_factory=list)
    verify_kind: str = "exact"  # exact / numeric / boolean / symbolic


class MathDataGenerator:
    """程序化合成数学训练数据.

    所有生成方法均保证:
        - 可复现 (给定 seed)
        - 答案可验证 (供 MathRLHF 奖励层与 MathEvaluator 使用)
        - 形式化字段 (formal) 提供可执行的 Lean4/SymPy 表达

    Args:
        seed: 随机种子.
        max_num: 整数生成上界 (控制难度).
    """

    def __init__(self, seed: Optional[int] = None, max_num: int = 100) -> None:
        self.seed = seed
        self.max_num = max_num
        self._rng = random.Random(seed)
        self._counter = 0

    # ------------------------------------------------------------------ #
    # 公共入口
    # ------------------------------------------------------------------ #
    def generate(self, domain: str, n: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """按领域批量生成 ``n`` 条问题. """
        domain = domain.lower().strip()
        if domain not in MATH_DOMAINS:
            raise ValueError(
                f"unknown domain: {domain!r}; expected one of {list(MATH_DOMAINS)}"
            )
        method = getattr(self, f"generate_{domain}", None)
        if method is None:
            raise NotImplementedError(f"generator for domain {domain!r} not implemented")
        return method(n=n, **kwargs)

    def generate_mixed(
        self,
        n: int = 100,
        domain_weights: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        """按领域权重混合生成 ``n`` 条问题. """
        weights = domain_weights or {d: 1.0 / len(MATH_DOMAINS) for d in MATH_DOMAINS}
        # 归一
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("sum of domain_weights must be positive")
        weights = {d: w / total for d, w in weights.items()}
        # 分配配额
        quotas: Dict[str, int] = {d: 0 for d in MATH_DOMAINS}
        for d, w in weights.items():
            if d in MATH_DOMAINS:
                quotas[d] = int(round(n * w))
        # 修正舍入误差
        diff = n - sum(quotas.values())
        if diff != 0:
            # 把误差加到权重最大的领域
            top = max(weights, key=lambda d: weights.get(d, 0))
            quotas[top] += diff
        results: List[Dict[str, Any]] = []
        for d in MATH_DOMAINS:
            q = quotas[d]
            if q > 0:
                results.extend(self.generate(d, n=q))
        self._rng.shuffle(results)
        return results[:n]

    def generate_by_curriculum(
        self,
        level: CurriculumLevel,
        n: int = 100,
    ) -> List[Dict[str, Any]]:
        """按课程级别的领域分布生成数据. """
        from .curriculum import LEVEL_DOMAIN_DIST
        lv = level if isinstance(level, CurriculumLevel) else CurriculumLevel(level)
        dist = LEVEL_DOMAIN_DIST.get(lv, {})
        # 仅保留有生成器的领域
        weights = {d: w for d, w in dist.items() if d in MATH_DOMAINS}
        return self.generate_mixed(n=n, domain_weights=weights)

    # ------------------------------------------------------------------ #
    # 1. 算术 arithmetic
    # ------------------------------------------------------------------ #
    def generate_arithmetic(self, n: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """生成算术四则运算问题 (含整数/分数/混合). """
        results: List[Dict[str, Any]] = []
        ops_pool = ("+", "-", "*", "+", "-", "*", "+-")
        for _ in range(n):
            kind = self._rng.choice(["int", "int", "fraction", "mixed"])
            if kind == "int":
                a = self._rng.randint(1, self.max_num)
                b = self._rng.randint(1, self.max_num)
                op = self._rng.choice(ops_pool)
                if op == "+-":
                    # 两步: a + b - c
                    c = self._rng.randint(1, self.max_num)
                    problem = f"Compute {a} + {b} - {c}."
                    answer = str(a + b - c)
                    solution = f"{a} + {b} = {a + b}; {a + b} - {c} = {a + b - c}"
                else:
                    problem = f"Compute {a} {op} {b}."
                    if op == "+":
                        ans = a + b
                    elif op == "-":
                        a, b = max(a, b), min(a, b)  # 保证非负
                        ans = a - b
                    else:
                        ans = a * b
                    answer = str(ans)
                    solution = f"{a} {op} {b} = {ans}"
            elif kind == "fraction":
                d1 = self._rng.randint(2, 12)
                n1 = self._rng.randint(1, d1 - 1)
                d2 = self._rng.randint(2, 12)
                n2 = self._rng.randint(1, d2 - 1)
                op = self._rng.choice(["+", "-", "*"])
                problem = f"Compute {n1}/{d1} {op} {n2}/{d2}."
                if op == "+":
                    num = n1 * d2 + n2 * d1
                    den = d1 * d2
                elif op == "-":
                    num = abs(n1 * d2 - n2 * d1)
                    den = d1 * d2
                else:
                    num = n1 * n2
                    den = d1 * d2
                g = math.gcd(num, den)
                num, den = num // g, den // g
                answer = f"{num}/{den}" if den != 1 else str(num)
                solution = f"结果 = {num}/{den}" if den != 1 else f"结果 = {num}"
            else:  # mixed
                a = self._rng.randint(2, self.max_num)
                b = self._rng.randint(2, self.max_num)
                c = self._rng.randint(1, 20)
                problem = f"Compute ({a} * {b}) + {c}."
                ans = a * b + c
                answer = str(ans)
                solution = f"{a} * {b} = {a * b}; + {c} = {ans}"
            results.append(self._make(
                domain="arithmetic",
                difficulty="basic",
                problem=problem,
                answer=answer,
                solution=solution,
                formal=f"({answer})",
                tags=["arithmetic", kind],
                verify_kind="exact",
            ))
        return results

    # ------------------------------------------------------------------ #
    # 2. 代数 algebra
    # ------------------------------------------------------------------ #
    def generate_algebra(self, n: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """生成一元/二元线性方程与简单多项式求值问题. """
        results: List[Dict[str, Any]] = []
        for _ in range(n):
            kind = self._rng.choice(["linear1", "linear1", "linear2", "polyeval"])
            if kind == "linear1":
                # a*x + b = c  =>  x = (c-b)/a, 取整除
                a = self._rng.randint(2, 12)
                x = self._rng.randint(-10, 10)
                b = self._rng.randint(-20, 20)
                c = a * x + b
                sign_b = f"+ {b}" if b >= 0 else f"- {-b}"
                problem = f"Solve for x: {a}x {sign_b} = {c}."
                answer = f"x = {x}"
                solution = f"{a}x = {c} - ({b}) = {c - b}; x = {(c - b) // a}"
                formal = f"sympy.solve({a}*x + ({b}) - ({c}), x)"
            elif kind == "linear2":
                # 二元一次: x + y = s, x - y = d
                x = self._rng.randint(1, 20)
                y = self._rng.randint(1, 20)
                s, d = x + y, x - y
                problem = (
                    f"Solve the system: x + y = {s}, x - y = {d}."
                )
                answer = f"x = {x}, y = {y}"
                solution = (
                    f"两式相加: 2x = {s + d} => x = {(s + d) // 2}; "
                    f"y = {s} - x = {y}"
                )
                formal = f"sympy.solve([x+y-{s}, x-y-{d}], [x, y])"
            else:  # polyeval
                # 多项式 a*x^2 + b*x + c 在 x0 处求值
                a = self._rng.randint(1, 5)
                b = self._rng.randint(-5, 5)
                c = self._rng.randint(-10, 10)
                x0 = self._rng.randint(-3, 3)
                val = a * x0 * x0 + b * x0 + c
                problem = f"Evaluate {a}x^2 + {b}x + {c} at x = {x0}."
                answer = str(val)
                solution = (
                    f"{a}*{x0}^2 + {b}*{x0} + {c} = {a * x0 * x0} + {b * x0} + {c} = {val}"
                )
                formal = f"sympy.expand({a}*x**2 + {b}*x + {c}).subs(x, {x0})"
            results.append(self._make(
                domain="algebra",
                difficulty="intermediate",
                problem=problem,
                answer=answer,
                solution=solution,
                formal=formal,
                tags=["algebra", kind],
                verify_kind="symbolic",
            ))
        return results

    # ------------------------------------------------------------------ #
    # 3. 几何 geometry
    # ------------------------------------------------------------------ #
    def generate_geometry(self, n: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """生成平面/解析几何问题 (面积/周长/距离). """
        results: List[Dict[str, Any]] = []
        for _ in range(n):
            kind = self._rng.choice(["triangle_area", "circle_area", "rect_perim", "distance"])
            if kind == "triangle_area":
                base = self._rng.randint(2, 20)
                height = self._rng.randint(2, 20)
                area = base * height / 2
                problem = f"A triangle has base {base} and height {height}. Find its area."
                answer = self._fmt_num(area)
                solution = f"面积 = 1/2 * 底 * 高 = 1/2 * {base} * {height} = {area}"
                formal = f"sympy.Rational(1,2)*{base}*{height}"
            elif kind == "circle_area":
                r = self._rng.randint(1, 15)
                problem = f"A circle has radius {r}. Find its area (in terms of pi)."
                answer = f"{r*r}*pi"
                solution = f"面积 = pi * r^2 = pi * {r}^2 = {r*r}*pi"
                formal = f"sympy.pi*{r}**2"
            elif kind == "rect_perim":
                w = self._rng.randint(2, 20)
                h = self._rng.randint(2, 20)
                problem = f"A rectangle has width {w} and height {h}. Find its perimeter."
                answer = str(2 * (w + h))
                solution = f"周长 = 2*(w+h) = 2*({w}+{h}) = {2 * (w + h)}"
                formal = f"2*({w}+{h})"
            else:  # distance
                x1 = self._rng.randint(-10, 10)
                y1 = self._rng.randint(-10, 10)
                x2 = self._rng.randint(-10, 10)
                y2 = self._rng.randint(-10, 10)
                dx, dy = x2 - x1, y2 - y1
                d2 = dx * dx + dy * dy
                d = math.isqrt(d2) if math.isqrt(d2) ** 2 == d2 else math.sqrt(d2)
                problem = (
                    f"Find the distance between ({x1}, {y1}) and ({x2}, {y2})."
                )
                if isinstance(d, int):
                    answer = str(d)
                else:
                    answer = f"sqrt({d2})" if abs(d * d - d2) > 1e-9 else f"{d:.4f}"
                solution = f"d = sqrt(({dx})^2 + ({dy})^2) = sqrt({d2}) = {answer}"
                formal = f"sympy.sqrt(({dx})**2 + ({dy})**2)"
            results.append(self._make(
                domain="geometry",
                difficulty="intermediate",
                problem=problem,
                answer=answer,
                solution=solution,
                formal=formal,
                tags=["geometry", kind],
                verify_kind="symbolic",
            ))
        return results

    # ------------------------------------------------------------------ #
    # 4. 分析 analysis (微积分)
    # ------------------------------------------------------------------ #
    def generate_analysis(self, n: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """生成微积分问题 (求导/定积分/极限). """
        results: List[Dict[str, Any]] = []
        for _ in range(n):
            kind = self._rng.choice(["deriv_poly", "deriv_poly", "integral_poly", "limit"])
            if kind == "deriv_poly":
                # d/dx (a*x^k + b*x) = a*k*x^(k-1) + b
                a = self._rng.randint(1, 6)
                k = self._rng.randint(2, 5)
                b = self._rng.randint(-5, 5)
                problem = f"Find d/dx of {a}x^{k} + {b}x."
                ak = a * k
                answer = f"{ak}x^{k - 1} + {b}"
                solution = f"d/dx({a}x^{k}) = {ak}x^{k - 1}; d/dx({b}x) = {b}"
                formal = f"sympy.diff({a}*x**{k} + {b}*x, x)"
            elif kind == "integral_poly":
                # integral_0^t (a*x + b) dx = a*t^2/2 + b*t
                a = self._rng.randint(1, 6)
                b = self._rng.randint(-5, 5)
                t = self._rng.randint(1, 8)
                val = a * t * t / 2 + b * t
                problem = f"Compute the definite integral of {a}x + {b} from 0 to {t}."
                answer = self._fmt_num(val)
                solution = (
                    f"∫({a}x+{b})dx = {a}/2*x^2 + {b}x; "
                    f"代入 [{0},{t}] = {a * t * t / 2} + {b * t} = {val}"
                )
                formal = f"sympy.integrate({a}*x + {b}, (x, 0, {t}))"
            else:  # limit
                # lim_{x->a} (x^2 - a^2)/(x - a) = 2a
                a = self._rng.randint(1, 10)
                problem = f"Find lim_{{x -> {a}}} (x^2 - {a**2}) / (x - {a})."
                answer = str(2 * a)
                solution = f"(x^2 - {a**2}) = (x-{a})(x+{a}); 极限 = x+{a} -> {2 * a}"
                formal = f"sympy.limit((x**2 - {a**2})/(x - {a}), x, {a})"
            results.append(self._make(
                domain="analysis",
                difficulty="advanced",
                problem=problem,
                answer=answer,
                solution=solution,
                formal=formal,
                tags=["analysis", kind],
                verify_kind="symbolic",
            ))
        return results

    # ------------------------------------------------------------------ #
    # 5. 数论 number_theory
    # ------------------------------------------------------------------ #
    def generate_number_theory(self, n: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """生成数论问题 (整除/素数/同余/GCD). """
        results: List[Dict[str, Any]] = []
        for _ in range(n):
            kind = self._rng.choice(["gcd", "lcm", "prime_check", "modular", "divisor_count"])
            if kind == "gcd":
                a = self._rng.randint(2, 200)
                b = self._rng.randint(2, 200)
                problem = f"Find gcd({a}, {b})."
                answer = str(math.gcd(a, b))
                solution = f"gcd({a}, {b}) = {math.gcd(a, b)}"
                formal = f"sympy.gcd({a}, {b})"
            elif kind == "lcm":
                a = self._rng.randint(2, 50)
                b = self._rng.randint(2, 50)
                problem = f"Find lcm({a}, {b})."
                ans = a * b // math.gcd(a, b)
                answer = str(ans)
                solution = f"lcm = a*b/gcd = {a}*{b}/{math.gcd(a, b)} = {ans}"
                formal = f"sympy.lcm({a}, {b})"
            elif kind == "prime_check":
                a = self._rng.randint(2, 200)
                is_p = self._is_prime(a)
                problem = f"Is {a} a prime number? Answer True or False."
                answer = str(is_p)
                solution = f"{a} {'是' if is_p else '不是'}素数"
                formal = f"sympy.isprime({a})"
            elif kind == "modular":
                a = self._rng.randint(1, 100)
                m = self._rng.randint(2, 30)
                b = self._rng.randint(1, 100)
                problem = f"Find ({a} * {b}) mod {m}."
                ans = (a * b) % m
                answer = str(ans)
                solution = f"{a}*{b} = {a * b}; {a * b} mod {m} = {ans}"
                formal = f"({a}*{b}) % {m}"
            else:  # divisor_count
                a = self._rng.randint(2, 200)
                cnt = self._count_divisors(a)
                problem = f"How many positive divisors does {a} have?"
                answer = str(cnt)
                solution = f"{a} 的正因子个数 = {cnt}"
                formal = f"sympy.divisor_count({a})"
            results.append(self._make(
                domain="number_theory",
                difficulty="intermediate",
                problem=problem,
                answer=answer,
                solution=solution,
                formal=formal,
                tags=["number_theory", kind],
                verify_kind="exact",
            ))
        return results

    # ------------------------------------------------------------------ #
    # 6. 抽象代数 abstract_algebra (群/环/域, 用小阶群验证)
    # ------------------------------------------------------------------ #
    def generate_abstract_algebra(self, n: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """生成抽象代数问题 (群阶/元素阶/循环群). """
        results: List[Dict[str, Any]] = []
        for _ in range(n):
            kind = self._rng.choice(["group_order", "element_order", "cyclic", "subgroup"])
            if kind == "group_order":
                # Z_n 的阶
                nn = self._rng.randint(2, 24)
                problem = f"What is the order of the cyclic group Z_{nn}?"
                answer = str(nn)
                solution = f"Z_{nn} 有 {nn} 个元素, 阶 = {nn}"
                formal = f"sympy.ntheory.n_order(1, {nn})"
            elif kind == "element_order":
                # Z_n 中元素 k 的阶 = n / gcd(n, k)
                nn = self._rng.randint(4, 30)
                k = self._rng.randint(1, nn - 1)
                order = nn // math.gcd(nn, k)
                problem = f"In Z_{nn}, what is the order of element {k}?"
                answer = str(order)
                solution = f"ord({k}) = {nn} / gcd({nn},{k}) = {nn}/{math.gcd(nn, k)} = {order}"
                formal = f"sympy.n_order({k}, {nn})"
            elif kind == "cyclic":
                nn = self._rng.randint(2, 20)
                # Z_n 恒为循环群
                problem = f"Is Z_{nn} a cyclic group? Answer True or False."
                answer = "True"
                solution = f"Z_{nn} 由 1 生成, 是循环群"
                formal = "True"
            else:  # subgroup
                # Z_n 的子群个数 = n 的因子个数
                nn = self._rng.randint(2, 36)
                cnt = self._count_divisors(nn)
                problem = f"How many subgroups does the cyclic group Z_{nn} have?"
                answer = str(cnt)
                solution = f"Z_{nn} 的子群与其因子一一对应, 因子数 = {cnt}"
                formal = f"sympy.divisor_count({nn})"
            results.append(self._make(
                domain="abstract_algebra",
                difficulty="advanced",
                problem=problem,
                answer=answer,
                solution=solution,
                formal=formal,
                tags=["abstract_algebra", kind],
                verify_kind="exact",
            ))
        return results

    # ------------------------------------------------------------------ #
    # 7. 拓扑 topology (用有限拓扑/集合性质验证)
    # ------------------------------------------------------------------ #
    def generate_topology(self, n: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """生成拓扑问题 (开集/连通/紧致, 有限可验证). """
        results: List[Dict[str, Any]] = []
        for _ in range(n):
            kind = self._rng.choice(["open_set_count", "discrete_topology", "indiscrete", "hausdorff"])
            if kind == "open_set_count":
                # 离散拓扑在 n 元集上有 2^n 个开集
                nn = self._rng.randint(2, 6)
                problem = (
                    f"How many open sets does the discrete topology on a set "
                    f"with {nn} elements have?"
                )
                answer = str(2 ** nn)
                solution = f"离散拓扑: 每个子集皆开, 共 2^{nn} = {2 ** nn} 个开集"
                formal = f"2**{nn}"
            elif kind == "discrete_topology":
                nn = self._rng.randint(2, 6)
                problem = (
                    f"In the discrete topology on a {nn}-element set, is every subset open? "
                    f"Answer True or False."
                )
                answer = "True"
                solution = "离散拓扑定义: 每个子集都是开集"
                formal = "True"
            elif kind == "indiscrete":
                nn = self._rng.randint(2, 6)
                problem = (
                    f"How many open sets does the indiscrete topology on a "
                    f"{nn}-element set have?"
                )
                answer = "2"
                solution = "密着拓扑仅有空集与全集两个开集"
                formal = "2"
            else:  # hausdorff
                problem = "Is every discrete space Hausdorff (T2)? Answer True or False."
                answer = "True"
                solution = "离散空间中任意两点可用单点开集分离, 满足 T2"
                formal = "True"
            results.append(self._make(
                domain="topology",
                difficulty="advanced",
                problem=problem,
                answer=answer,
                solution=solution,
                formal=formal,
                tags=["topology", kind],
                verify_kind="boolean",
            ))
        return results

    # ------------------------------------------------------------------ #
    # 8. 离散数学 discrete_math (组合/图论/逻辑)
    # ------------------------------------------------------------------ #
    def generate_discrete_math(self, n: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """生成离散数学问题 (排列组合/图论/逻辑). """
        results: List[Dict[str, Any]] = []
        for _ in range(n):
            kind = self._rng.choice(["permutation", "combination", "factorial", "graph_degree", "logic"])
            if kind == "permutation":
                nn = self._rng.randint(3, 8)
                k = self._rng.randint(2, nn)
                ans = math.perm(nn, k)
                problem = f"Compute P({nn}, {k}) = {nn}! / ({nn}-{k})!."
                answer = str(ans)
                solution = f"P({nn},{k}) = {nn}!/({nn - k})! = {ans}"
                formal = f"math.perm({nn}, {k})"
            elif kind == "combination":
                nn = self._rng.randint(4, 12)
                k = self._rng.randint(2, nn)
                ans = math.comb(nn, k)
                problem = f"Compute C({nn}, {k}) = {nn}! / ({k}! * ({nn}-{k})!)."
                answer = str(ans)
                solution = f"C({nn},{k}) = {ans}"
                formal = f"math.comb({nn}, {k})"
            elif kind == "factorial":
                nn = self._rng.randint(2, 10)
                problem = f"Compute {nn}!."
                answer = str(math.factorial(nn))
                solution = f"{nn}! = {math.factorial(nn)}"
                formal = f"math.factorial({nn})"
            elif kind == "graph_degree":
                # 完全图 K_n 的边数 = n(n-1)/2
                nn = self._rng.randint(3, 10)
                edges = nn * (nn - 1) // 2
                problem = f"How many edges does the complete graph K_{nn} have?"
                answer = str(edges)
                solution = f"K_{nn} 边数 = {nn}*{nn - 1}/2 = {edges}"
                formal = f"{nn}*({nn}-1)//2"
            else:  # logic
                # 简单命题逻辑真值
                p = self._rng.choice([True, False])
                q = self._rng.choice([True, False])
                # 求 (p and q) or (not p)
                val = (p and q) or (not p)
                problem = (
                    f"Let p = {p}, q = {q}. Evaluate the truth value of (p and q) or (not p)."
                )
                answer = str(val)
                solution = f"(p∧q)={p and q}; (¬p)={not p}; or = {val}"
                formal = f"({p} and {q}) or (not {p})"
            results.append(self._make(
                domain="discrete_math",
                difficulty="intermediate",
                problem=problem,
                answer=answer,
                solution=solution,
                formal=formal,
                tags=["discrete_math", kind],
                verify_kind="exact",
            ))
        return results

    # ------------------------------------------------------------------ #
    # 9. 概率 probability
    # ------------------------------------------------------------------ #
    def generate_probability(self, n: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """生成概率问题 (古典概型/期望/贝叶斯). """
        results: List[Dict[str, Any]] = []
        for _ in range(n):
            kind = self._rng.choice(["dice", "urn", "coin", "expectation"])
            if kind == "dice":
                # 掷两枚骰子点数和为 s 的概率
                s = self._rng.randint(2, 12)
                # 计数
                cnt = sum(1 for a in range(1, 7) for b in range(1, 7) if a + b == s)
                problem = f"Two fair dice are rolled. What is the probability that the sum is {s}?"
                g = math.gcd(cnt, 36)
                answer = f"{cnt // g}/{36 // g}"
                solution = f"有利事件数 = {cnt}, 总事件 = 36, P = {cnt}/{36} = {answer}"
                formal = f"sympy.Rational({cnt}, 36)"
            elif kind == "urn":
                red = self._rng.randint(2, 8)
                blue = self._rng.randint(2, 8)
                total = red + blue
                # 抽一个红球概率
                g = math.gcd(red, total)
                problem = (
                    f"An urn contains {red} red and {blue} blue balls. "
                    f"What is the probability of drawing a red ball?"
                )
                answer = f"{red // g}/{total // g}"
                solution = f"P(红) = {red}/{total} = {answer}"
                formal = f"sympy.Rational({red}, {total})"
            elif kind == "coin":
                # n 次公平硬币全正面概率
                nn = self._rng.randint(2, 6)
                problem = f"A fair coin is flipped {nn} times. What is the probability of all heads?"
                answer = f"1/{2 ** nn}"
                solution = f"P = (1/2)^{nn} = 1/{2 ** nn}"
                formal = f"sympy.Rational(1, {2 ** nn})"
            else:  # expectation
                # 公平骰子期望
                problem = "What is the expected value of a single fair six-sided die roll?"
                answer = "3.5"
                solution = "E = (1+2+3+4+5+6)/6 = 21/6 = 3.5"
                formal = "sympy.Rational(21, 6)"
            results.append(self._make(
                domain="probability",
                difficulty="intermediate",
                problem=problem,
                answer=answer,
                solution=solution,
                formal=formal,
                tags=["probability", kind],
                verify_kind="symbolic",
            ))
        return results

    # ------------------------------------------------------------------ #
    # 10. 竞赛 competition (AMC/AIME/IMO 风格, 参数化模板)
    # ------------------------------------------------------------------ #
    def generate_competition(self, n: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """生成竞赛风格问题 (AMC/AIME 难度, 参数化). """
        results: List[Dict[str, Any]] = []
        for _ in range(n):
            kind = self._rng.choice(["amc_arith", "aime_mod", "imo_ineq", "amc_count"])
            if kind == "amc_arith":
                # AMC: 一组数的和/积
                nn = self._rng.randint(3, 6)
                nums = [self._rng.randint(1, 20) for _ in range(nn)]
                s = sum(nums)
                problem = (
                    f"Find the sum of {', '.join(map(str, nums))}."
                )
                answer = str(s)
                solution = " + ".join(map(str, nums)) + f" = {s}"
                formal = f"sum({nums})"
            elif kind == "aime_mod":
                # AIME: 求 x mod m (大数)
                a = self._rng.randint(100, 999)
                b = self._rng.randint(100, 999)
                m = self._rng.randint(10, 99)
                ans = (a ** 2 + b) % m
                problem = f"Find the remainder when {a}^2 + {b} is divided by {m}."
                answer = str(ans)
                solution = f"{a**2} + {b} = {a**2 + b}; mod {m} = {ans}"
                formal = f"({a}**2 + {b}) % {m}"
            elif kind == "imo_ineq":
                # AM-GM: 两正数 a, b, a+b=s, 求 ab 最大值 = s^2/4
                a = self._rng.randint(1, 20)
                b = self._rng.randint(1, 20)
                s = a + b
                problem = (
                    f"If x + y = {s} for positive reals x, y, what is the maximum value of xy?"
                )
                answer = self._fmt_num(s * s / 4)
                solution = f"由 AM-GM, xy <= ((x+y)/2)^2 = ({s}/2)^2 = {s * s / 4}"
                formal = f"({s}/2)**2"
            else:  # amc_count
                # 计数: 从 n 人中选 k 人的方式
                nn = self._rng.randint(5, 12)
                k = self._rng.randint(2, 4)
                ans = math.comb(nn, k)
                problem = f"In how many ways can {k} students be chosen from {nn} students?"
                answer = str(ans)
                solution = f"C({nn},{k}) = {ans}"
                formal = f"math.comb({nn}, {k})"
            results.append(self._make(
                domain="competition",
                difficulty="advanced",
                problem=problem,
                answer=answer,
                solution=solution,
                formal=formal,
                tags=["competition", kind],
                verify_kind="exact",
            ))
        return results

    # ------------------------------------------------------------------ #
    # 11. 前沿 frontier (研究级开放问题, 标注 open)
    # ------------------------------------------------------------------ #
    def generate_frontier(self, n: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """生成前沿研究问题 (开放问题, 标注 open=True, 无标准答案). """
        open_problems = [
            (
                "Is every even integer greater than 2 the sum of two primes? (Goldbach's conjecture)",
                "open", "数论", "Goldbach",
            ),
            (
                "Are there infinitely many twin primes? (Twin prime conjecture)",
                "open", "数论", "TwinPrime",
            ),
            (
                "Does P = NP? (P vs NP problem)",
                "open", "计算理论", "PvsNP",
            ),
            (
                "Is the Riemann Hypothesis true? (All non-trivial zeros of zeta have Re(s)=1/2)",
                "open", "分析", "Riemann",
            ),
            (
                "Is every simply connected closed 3-manifold homeomorphic to S^3? (Poincaré conjecture - proven)",
                "solved", "拓扑", "Poincare",
            ),
            (
                "Does there exist a polynomial-time algorithm for graph isomorphism?",
                "open", "图论", "GraphIso",
            ),
            (
                "Is the Collatz conjecture true for all positive integers?",
                "open", "数论", "Collatz",
            ),
            (
                "What is the exact value of the de Bruijn–Newman constant?",
                "open", "分析", "DeBruijnNewman",
            ),
        ]
        results: List[Dict[str, Any]] = []
        for _ in range(n):
            prob, status, area, name = self._rng.choice(open_problems)
            problem = prob
            answer = "open problem" if status == "open" else "solved (Perelman, 2003)"
            solution = (
                f"[{area}] {name}: {'开放问题, 尚未解决' if status == 'open' else '已解决'}"
            )
            results.append(self._make(
                domain="frontier",
                difficulty="frontier",
                problem=problem,
                answer=answer,
                solution=solution,
                formal=f"-- {name} ({status})",
                tags=["frontier", area, name, status],
                verify_kind="boolean",
            ))
        return results

    # ------------------------------------------------------------------ #
    # 辅助
    # ------------------------------------------------------------------ #
    def _make(
        self,
        domain: str,
        difficulty: str,
        problem: str,
        answer: str,
        solution: str,
        formal: str = "",
        tags: Optional[List[str]] = None,
        verify_kind: str = "exact",
    ) -> Dict[str, Any]:
        self._counter += 1
        return {
            "id": f"{domain}-{self._counter:06d}",
            "domain": domain,
            "difficulty": difficulty,
            "problem": problem,
            "answer": answer,
            "solution": solution,
            "formal": formal,
            "tags": list(tags or []),
            "verify_kind": verify_kind,
            "seed": self.seed,
        }

    @staticmethod
    def _is_prime(x: int) -> bool:
        if x < 2:
            return False
        if x < 4:
            return True
        if x % 2 == 0:
            return False
        i = 3
        while i * i <= x:
            if x % i == 0:
                return False
            i += 2
        return True

    @staticmethod
    def _count_divisors(x: int) -> int:
        if x < 1:
            return 0
        cnt = 0
        i = 1
        while i * i <= x:
            if x % i == 0:
                cnt += 1 if i * i == x else 2
            i += 1
        return cnt

    @staticmethod
    def _fmt_num(x: float) -> str:
        """格式化数值: 整数去 .0, 否则保留 4 位. """
        if isinstance(x, int):
            return str(x)
        if x == int(x):
            return str(int(x))
        return f"{x:.4f}"

    # ------------------------------------------------------------------ #
    # 统计
    # ------------------------------------------------------------------ #
    def supported_domains(self) -> Tuple[str, ...]:
        return MATH_DOMAINS

    def summary(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "max_num": self.max_num,
            "generated_total": self._counter,
            "domains": list(MATH_DOMAINS),
        }


__all__ = [
    "MathDataGenerator",
    "ProblemSpec",
    "MATH_DOMAINS",
    "DOMAIN_DEFAULT_DIFFICULTY",
]
