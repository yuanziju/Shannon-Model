"""自研数论验证器 (NumberTheoryVerifier).

提供数论命题验证 + 反例发现能力, 重点强化反例搜索:

基础验证:
    - goldbach_check          : 验证哥德巴赫猜想 (偶数 = 两素数之和)
    - prime_check             : Miller-Rabin 素性测试
    - riemann_zeros_check     : 计算黎曼 ζ 函数前 N 个非平凡零点, 验证实部=1/2
    - prime_counting          : 素数计数 π(x) (Meissel-Mertens / 简单实现)
    - modular_arithmetic      : 模运算
    - discrete_log            : 离散对数 (baby-step giant-step)
    - crt                     : 中国剩余定理
    - gcd_extended            : 扩展欧几里得

反例发现强化:
    - find_counterexample     : 自然语言猜想 -> 自动解析 -> 反例搜索
    - batch_counterexample_search : 多区间并行批量搜索
    - goldbach_counterexample_search : 哥德巴赫反例专门搜索
    - riemann_zero_off_line_search  : 搜索偏离临界线 Re(s)=1/2 的零点

实现说明:
    - 素性测试使用确定性 Miller-Rabin (n < 3.3e24 时, 12 个 witness 即可确定),
      工程实现采用确定性 witness 集合 {2,3,5,7,11,13,17,19,23,29,31,37}.
    - ζ 零点计算使用 mpmath 的 mpmath.zetazero (依赖 mpmath), 若不可用则降级.
    - 离散对数使用 baby-step giant-step, O(sqrt(n)).
    - 反例搜索对常见猜想 (奇素数/偶数/哥德巴赫/费马/3n+1 等) 内置解析器,
      未知格式返回 (found=False, reason="unparseable").
"""

from __future__ import annotations

import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple


# =============================================================================
# 确定性 Miller-Rabin witness (覆盖 n < 3.3 * 10^24)
# =============================================================================
_MR_WITNESSES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def _is_small_prime(n: int) -> Optional[bool]:
    """对极小 n 给出确定答案 (None 表示继续走 MR)."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    if n < 9:
        return True
    if n % 3 == 0:
        return False
    return None


def _miller_rabin_witness(a: int, d: int, n: int, r: int) -> bool:
    """单次 witness 测试. 返回 True 表示合数."""
    x = pow(a, d, n)
    if x == 1 or x == n - 1:
        return False
    for _ in range(r - 1):
        x = (x * x) % n
        if x == n - 1:
            return False
    return True


def _miller_rabin(n: int, witnesses: Tuple[int, ...] = _MR_WITNESSES) -> bool:
    """确定性 Miller-Rabin 素性测试."""
    fast = _is_small_prime(n)
    if fast is not None:
        return fast
    # n - 1 = d * 2^r
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in witnesses:
        if a % n == 0:
            continue
        if _miller_rabin_witness(a, d, n, r):
            return False
    return True


# =============================================================================
# 数据结构
# =============================================================================
@dataclass
class GoldbachResult:
    """哥德巴赫猜想验证结果."""

    n: int
    all_verified: bool
    first_failure: Optional[int] = None
    representations: dict = field(default_factory=dict)   # n -> (p, q)
    checked_count: int = 0
    elapsed_sec: float = 0.0

    def __bool__(self) -> bool:
        return self.all_verified


@dataclass
class RiemannZeroResult:
    """黎曼 ζ 零点检查结果."""

    num_zeros: int
    zeros: List[Tuple[float, float]] = field(default_factory=list)   # (imag, real_part)
    all_on_critical_line: bool = True
    max_deviation: float = 0.0          # 实部偏离 0.5 的最大值
    off_critical_zeros: List[Tuple[float, float]] = field(default_factory=list)
    elapsed_sec: float = 0.0


@dataclass
class CounterexampleResult:
    """反例搜索结果."""

    found: bool
    conjecture: str = ""
    counterexample: Any = None          # 反例值或证据
    witness: Any = None                 # 见证值 (如哥德巴赫的不可分解偶数)
    searched_range: int = 0
    elapsed_sec: float = 0.0
    reason: str = ""                    # found=False 时的原因
    extra: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.found

    def __repr__(self) -> str:  # pragma: no cover - 便捷
        if self.found:
            return (
                f"CounterexampleResult(found=True, conjecture={self.conjecture!r}, "
                f"counterexample={self.counterexample!r})"
            )
        return (
            f"CounterexampleResult(found=False, conjecture={self.conjecture!r}, "
            f"reason={self.reason!r})"
        )


# =============================================================================
# 主验证器
# =============================================================================
class NumberTheoryVerifier:
    """自研数论验证器 (含反例发现强化)."""

    def __init__(self, *, max_check: int = 1000, num_workers: int = 4):
        self.max_check = max_check
        self.num_workers = max(1, int(num_workers))
        # 素数筛缓存 (避免重复计算)
        self._sieve_cache: dict[int, List[int]] = {}

    # ------------------------------------------------------------------
    # 素性测试
    # ------------------------------------------------------------------
    def prime_check(self, n: int) -> bool:
        """Miller-Rabin 确定性素性测试."""
        try:
            n = int(n)
        except (TypeError, ValueError):
            return False
        return _miller_rabin(n)

    # 别名
    def is_prime(self, n: int) -> bool:
        return self.prime_check(n)

    # ------------------------------------------------------------------
    # 埃氏筛 (缓存)
    # ------------------------------------------------------------------
    def _sieve(self, limit: int) -> List[int]:
        limit = int(limit)
        if limit < 2:
            return []
        if limit in self._sieve_cache:
            return self._sieve_cache[limit]
        sieve = bytearray([1]) * (limit + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                sieve[i * i :: i] = bytearray(len(sieve[i * i :: i]))
        primes = [i for i, v in enumerate(sieve) if v]
        self._sieve_cache[limit] = primes
        return primes

    # ------------------------------------------------------------------
    # 哥德巴赫验证
    # ------------------------------------------------------------------
    def goldbach_check(self, n: int) -> GoldbachResult:
        """验证哥德巴赫猜想: 每个偶数 4..n 可表示为两素数之和."""
        import time
        t0 = time.time()
        n = int(n)
        if n < 4:
            return GoldbachResult(n=n, all_verified=True, checked_count=0, elapsed_sec=0.0)
        # 先筛出所有 ≤ n 的素数, 然后双指针/集合查找
        primes = self._sieve(n)
        prime_set = set(primes)
        reps: dict[int, Tuple[int, int]] = {}
        all_ok = True
        first_fail = None
        # 仅检查偶数 ≥ 4
        for even in range(4, n + 1, 2):
            found = None
            for p in primes:
                if p > even // 2:
                    break
                if (even - p) in prime_set:
                    found = (p, even - p)
                    break
            if found is None:
                all_ok = False
                first_fail = even
                break
            reps[even] = found
        return GoldbachResult(
            n=n,
            all_verified=all_ok,
            first_failure=first_fail,
            representations=reps,
            checked_count=len(reps) + (0 if all_ok else 1),
            elapsed_sec=time.time() - t0,
        )

    # ------------------------------------------------------------------
    # 黎曼 ζ 零点检查
    # ------------------------------------------------------------------
    def riemann_zeros_check(self, num_zeros: int = 10) -> RiemannZeroResult:
        """计算黎曼 ζ 函数前 N 个非平凡零点, 验证 Re(s)=0.5."""
        import time
        t0 = time.time()
        num_zeros = max(1, int(num_zeros))
        try:
            import mpmath
        except ImportError:
            return RiemannZeroResult(
                num_zeros=0,
                all_on_critical_line=False,
                elapsed_sec=time.time() - t0,
            )
        zeros: List[Tuple[float, float]] = []
        off_line: List[Tuple[float, float]] = []
        max_dev = 0.0
        for k in range(1, num_zeros + 1):
            try:
                z = mpmath.zetazero(k)
                imag = float(z.imag)
                real = float(z.real)
            except Exception:
                continue
            zeros.append((imag, real))
            dev = abs(real - 0.5)
            if dev > max_dev:
                max_dev = dev
            # mpmath 默认精度可能产生小波动, 容差 1e-6
            if dev > 1e-6:
                off_line.append((imag, real))
        # 容差范围内视为都在临界线上
        all_on_line = (len(off_line) == 0) and (len(zeros) > 0)
        return RiemannZeroResult(
            num_zeros=len(zeros),
            zeros=zeros,
            all_on_critical_line=all_on_line,
            max_deviation=max_dev,
            off_critical_zeros=off_line,
            elapsed_sec=time.time() - t0,
        )

    # ------------------------------------------------------------------
    # 素数计数 π(x)
    # ------------------------------------------------------------------
    def prime_counting(self, x: int) -> int:
        """素数计数 π(x). 简单实现: 筛 + count.

        Args:
            x: 上界 (含).
        """
        x = int(x)
        if x < 2:
            return 0
        return len(self._sieve(x))

    # 别名
    def pi(self, x: int) -> int:
        return self.prime_counting(x)

    # ------------------------------------------------------------------
    # 模运算
    # ------------------------------------------------------------------
    @staticmethod
    def modular_arithmetic(a: int, b: int, m: int, op: str = "+") -> int:
        """模运算: a op b mod m.

        op: + - * pow
        """
        m = int(m)
        if m <= 0:
            raise ValueError("modulus must be positive")
        a = int(a) % m
        b = int(b) % m
        op = op.lower()
        if op == "+":
            return (a + b) % m
        if op == "-":
            return (a - b) % m
        if op in ("*", "mul"):
            return (a * b) % m
        if op in ("pow", "power"):
            return pow(a, b, m)
        if op == "/":
            # 模逆: a * inv(b) mod m
            g, inv, _ = NumberTheoryVerifier.gcd_extended(b, m)
            if g != 1:
                raise ValueError(f"{b} has no inverse mod {m}")
            return (a * inv) % m
        raise ValueError(f"unknown op: {op}")

    # ------------------------------------------------------------------
    # 扩展欧几里得
    # ------------------------------------------------------------------
    @staticmethod
    def gcd_extended(a: int, b: int) -> Tuple[int, int, int]:
        """扩展欧几里得: 返回 (g, x, y) 使得 a*x + b*y = g = gcd(a, b)."""
        a, b = int(a), int(b)
        if b == 0:
            return (a, 1, 0)
        g, x1, y1 = NumberTheoryVerifier.gcd_extended(b, a % b)
        return (g, y1, x1 - (a // b) * y1)

    # ------------------------------------------------------------------
    # 中国剩余定理
    # ------------------------------------------------------------------
    @staticmethod
    def crt(remainders: List[int], moduli: List[int]) -> Optional[Tuple[int, int]]:
        """中国剩余定理. 返回 (x, M) 满足 x ≡ r_i (mod m_i), M = prod(m_i).

        无解时返回 None.
        """
        if len(remainders) != len(moduli) or not remainders:
            return None
        r0, m0 = int(remainders[0]) % int(moduli[0]), int(moduli[0])
        for i in range(1, len(remainders)):
            r1, m1 = int(remainders[i]) % int(moduli[i]), int(moduli[i])
            g, p, q = NumberTheoryVerifier.gcd_extended(m0, m1)
            if (r1 - r0) % g != 0:
                return None
            lcm = m0 // g * m1
            x = (r0 + (r1 - r0) // g * p % (m1 // g) * m0) % lcm
            r0, m0 = x, lcm
        return (r0, m0)

    # ------------------------------------------------------------------
    # 离散对数 (baby-step giant-step)
    # ------------------------------------------------------------------
    @staticmethod
    def discrete_log(base: int, target: int, mod: int) -> Optional[int]:
        """求解 x 使 base^x ≡ target (mod mod), 无解返回 None.

        baby-step giant-step, 复杂度 O(sqrt(mod)).
        """
        base, target, mod = int(base), int(target), int(mod)
        if mod <= 0:
            return None
        base %= mod
        target %= mod
        if target == 1:
            return 0
        n = int(math.isqrt(mod)) + 1
        # baby step: base^j -> j
        table: dict[int, int] = {}
        cur = 1
        for j in range(n):
            if cur not in table:
                table[cur] = j
            cur = (cur * base) % mod
        # giant step: base^(-n) 的逆
        # 计算 base^n 的逆
        g, inv, _ = NumberTheoryVerifier.gcd_extended(pow(base, n, mod), mod)
        if g != 1:
            # 退化: 模不互素, 简单返回 None (可扩展 Pohlig-Hellman)
            return None
        gamma = inv % mod
        cur = target
        for i in range(n + 1):
            if cur in table:
                return i * n + table[cur]
            cur = (cur * gamma) % mod
        return None

    # ==================================================================
    # 反例发现强化 (Counterexample Discovery)
    # ==================================================================
    def find_counterexample(
        self, conjecture: str, search_range: int
    ) -> CounterexampleResult:
        """自动搜索反例.

        支持的猜想模式 (自然语言, 自动解析):
            - "all odd numbers > N are prime"
            - "all even numbers > N are sum of two primes" (哥德巴赫反例)
            - "n**2 + n + 41 is prime for all n"            (欧拉多项式)
            - "fermat: 2**(2**k) + 1 is prime"               (费马数)
            - "3n+1: collatz reaches 1"                      (Collatz)
            - "2**n - 1 is prime"                            (梅森数)
            - 含 `prime`/`odd`/`even`/`collatz`/`fermat`/`mersenne` 关键字

        Args:
            conjecture: 猜想自然语言描述.
            search_range: 搜索上限 (n in [1, search_range]).

        Returns:
            CounterexampleResult.
        """
        import time
        t0 = time.time()
        conjecture_l = (conjecture or "").lower()
        search_range = max(1, int(search_range))

        # 1) "all odd numbers > K are prime"
        m = re.search(r"all odd numbers?\s*>\s*(\d+).*prime", conjecture_l)
        if m:
            k = int(m.group(1))
            for n in range(k + 1, search_range + 1):
                if n % 2 == 1 and not self.prime_check(n):
                    return CounterexampleResult(
                        found=True,
                        conjecture=conjecture,
                        counterexample=n,
                        searched_range=search_range,
                        elapsed_sec=time.time() - t0,
                        reason="odd non-prime found",
                        extra={"threshold": k},
                    )
            return CounterexampleResult(
                found=False,
                conjecture=conjecture,
                searched_range=search_range,
                elapsed_sec=time.time() - t0,
                reason=f"no odd non-prime in ({k}, {search_range}]",
            )

        # 2) "all even numbers > K are sum of two primes" / goldbach
        m = re.search(r"(goldbach|even.*sum.*two\s*primes|sum of two primes)", conjecture_l)
        if m:
            return self._goldbach_counterexample_impl(search_range, conjecture, t0)

        # 3) 欧拉素数生成多项式 n**2 + n + 41
        m = re.search(r"n\*\*2\s*\+\s*n\s*\+\s*(\d+)", conjecture_l) or \
            re.search(r"n\^2\s*\+\s*n\s*\+\s*(\d+)", conjecture_l)
        if m and "prime" in conjecture_l:
            c = int(m.group(1))
            for n in range(0, search_range + 1):
                val = n * n + n + c
                if not self.prime_check(val):
                    return CounterexampleResult(
                        found=True,
                        conjecture=conjecture,
                        counterexample=n,
                        witness=val,
                        searched_range=search_range,
                        elapsed_sec=time.time() - t0,
                        reason=f"n={n} -> {val} is composite",
                        extra={"polynomial_const": c},
                    )
            return CounterexampleResult(
                found=False,
                conjecture=conjecture,
                searched_range=search_range,
                elapsed_sec=time.time() - t0,
                reason=f"polynomial prime in [0, {search_range}]",
            )

        # 4) 费马数 2**(2**k) + 1
        if "fermat" in conjecture_l:
            for k in range(0, min(search_range, 20)):  # k 过大计算不可行
                val = (1 << (1 << k)) + 1
                if not self.prime_check(val):
                    return CounterexampleResult(
                        found=True,
                        conjecture=conjecture,
                        counterexample=k,
                        witness=val,
                        searched_range=search_range,
                        elapsed_sec=time.time() - t0,
                        reason=f"F_{k} = {val} is composite",
                    )
            return CounterexampleResult(
                found=False,
                conjecture=conjecture,
                searched_range=search_range,
                elapsed_sec=time.time() - t0,
                reason="all Fermat numbers in range are prime (unexpected)",
            )

        # 5) 梅森数 2**n - 1
        if "mersenne" in conjecture_l or re.search(r"2\*\*n\s*-\s*1", conjecture_l):
            for n in range(2, min(search_range, 200)):
                val = (1 << n) - 1
                if not self.prime_check(val):
                    return CounterexampleResult(
                        found=True,
                        conjecture=conjecture,
                        counterexample=n,
                        witness=val,
                        searched_range=search_range,
                        elapsed_sec=time.time() - t0,
                        reason=f"2^{n}-1 = {val} is composite",
                    )
            return CounterexampleResult(
                found=False,
                conjecture=conjecture,
                searched_range=search_range,
                elapsed_sec=time.time() - t0,
                reason="all Mersenne numbers in range are prime (unexpected)",
            )

        # 6) Collatz 猜想 (3n+1 reaches 1)
        if "collatz" in conjecture_l or "3n+1" in conjecture_l:
            for n in range(2, search_range + 1):
                steps = 0
                x = n
                visited = {x}
                while x != 1 and steps < 10_000:
                    if x % 2 == 0:
                        x //= 2
                    else:
                        x = 3 * x + 1
                    if x in visited:
                        # 进入循环 (非 1)
                        return CounterexampleResult(
                            found=True,
                            conjecture=conjecture,
                            counterexample=n,
                            witness=x,
                            searched_range=search_range,
                            elapsed_sec=time.time() - t0,
                            reason=f"collatz sequence loops at {x}",
                        )
                    visited.add(x)
                    steps += 1
                if x != 1:
                    return CounterexampleResult(
                        found=True,
                        conjecture=conjecture,
                        counterexample=n,
                        searched_range=search_range,
                        elapsed_sec=time.time() - t0,
                        reason="collatz sequence did not reach 1 within step budget",
                    )
            return CounterexampleResult(
                found=False,
                conjecture=conjecture,
                searched_range=search_range,
                elapsed_sec=time.time() - t0,
                reason=f"collatz holds in [2, {search_range}]",
            )

        # 7) 通用 "all primes satisfy P" — 当前无法解析, 返回 not found
        return CounterexampleResult(
            found=False,
            conjecture=conjecture,
            searched_range=search_range,
            elapsed_sec=time.time() - t0,
            reason="unparseable conjecture (no matching pattern)",
        )

    def batch_counterexample_search(
        self, conjecture: str, ranges: List[int]
    ) -> List[CounterexampleResult]:
        """批量并行反例搜索.

        将不同搜索区间 [1, r_i] 分配到多个 worker 并行执行.
        任一 worker 发现反例即提前返回该区间结果, 其余仍在后台执行.
        """
        if not ranges:
            return []
        # 去重并排序
        uniq = sorted(set(max(1, int(r)) for r in ranges))
        results: List[CounterexampleResult] = []

        # 若只一个区间, 直接同步调用 (避免线程开销)
        if len(uniq) == 1 or self.num_workers <= 1:
            return [self.find_counterexample(conjecture, r) for r in uniq]

        with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
            future_to_range = {
                ex.submit(self.find_counterexample, conjecture, r): r for r in uniq
            }
            for fut in as_completed(future_to_range):
                r = future_to_range[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    res = CounterexampleResult(
                        found=False,
                        conjecture=conjecture,
                        searched_range=r,
                        reason=f"worker error: {e!r}",
                    )
                results.append(res)
                # 找到反例后可提前取消 (此处保留全部结果以便对比)
                # if res.found:
                #     break
        # 按区间排序返回
        results.sort(key=lambda x: x.searched_range)
        return results

    # ------------------------------------------------------------------
    # 哥德巴赫反例专门搜索
    # ------------------------------------------------------------------
    def goldbach_counterexample_search(self, max_n: int) -> Optional[int]:
        """专门搜索哥德巴赫反例.

        遍历偶数 4..max_n, 检查是否可表示为两素数之和.
        返回第一个不可分解的偶数, 无则 None.
        """
        max_n = int(max_n)
        if max_n < 4:
            return None
        primes = self._sieve(max_n)
        prime_set = set(primes)
        for even in range(4, max_n + 1, 2):
            ok = False
            for p in primes:
                if p > even // 2:
                    break
                if (even - p) in prime_set:
                    ok = True
                    break
            if not ok:
                return even
        return None

    def _goldbach_counterexample_impl(
        self, max_n: int, conjecture: str, t0: float
    ) -> CounterexampleResult:
        """内部: 哥德巴赫反例搜索 + 包装为 CounterexampleResult."""
        import time
        ce = self.goldbach_counterexample_search(max_n)
        if ce is not None:
            return CounterexampleResult(
                found=True,
                conjecture=conjecture,
                counterexample=ce,
                witness="no two-prime decomposition",
                searched_range=max_n,
                elapsed_sec=time.time() - t0,
                reason=f"even {ce} cannot be written as p+q",
            )
        return CounterexampleResult(
            found=False,
            conjecture=conjecture,
            searched_range=max_n,
            elapsed_sec=time.time() - t0,
            reason=f"goldbach holds in [4, {max_n}]",
        )

    # ------------------------------------------------------------------
    # 黎曼 ζ 零点偏离临界线搜索
    # ------------------------------------------------------------------
    def riemann_zero_off_line_search(
        self, max_imag: float
    ) -> Optional[Tuple[float, float, float]]:
        """搜索偏离临界线 Re(s)=1/2 的零点.

        扫描虚部 [1, max_imag] 内的 ζ 零点, 返回首个 Re(s) ≠ 0.5 的 (real, imag, deviation),
        无偏离则返回 None.

        Args:
            max_imag: 最大虚部扫描上界.

        Note:
            - 使用 mpmath.zetazero 枚举已知零点 (已知 RH 验证到极大虚部均满足).
            - 工程上为演示接口; 真实零点偏离搜索需要 Riemann-Siegel 公式自实现.
        """
        try:
            import mpmath
        except ImportError:
            return None
        max_imag = float(max_imag)
        if max_imag <= 1:
            return None
        k = 1
        # 限制最大枚举次数, 防止无限循环
        max_iter = 100_000
        while k <= max_iter:
            try:
                z = mpmath.zetazero(k)
            except Exception:
                return None
            imag = float(z.imag)
            if imag > max_imag:
                return None
            real = float(z.real)
            dev = abs(real - 0.5)
            # 容差 1e-6 (mpmath 默认 15 位精度)
            if dev > 1e-6:
                return (real, imag, dev)
            k += 1
        return None

    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - 便捷
        return f"NumberTheoryVerifier(max_check={self.max_check}, num_workers={self.num_workers})"
