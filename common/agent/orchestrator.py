"""工具编排 - CHAIN / PARALLEL / MIXED 调度.

ToolOrchestrator 负责将 <ACTION> 阶段产生的工具调用按依赖关系编排执行:
    - CHAIN:    串行链式, 上一步输出作为下一步输入.
    - PARALLEL: 并行扇出, 互不依赖的工具同时执行.
    - MIXED:    DAG 混合调度, 拓扑排序 + 就绪队列并行.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


class OrchestratorMode(Enum):
    CHAIN = "CHAIN"
    PARALLEL = "PARALLEL"
    MIXED = "MIXED"


class ToolCall:
    """单次工具调用描述. """

    __slots__ = ("name", "args", "depends_on", "id")

    def __init__(
        self,
        name: str,
        args: Optional[Dict[str, Any]] = None,
        depends_on: Optional[Sequence[str]] = None,
        call_id: Optional[str] = None,
    ) -> None:
        self.name = name
        self.args = dict(args or {})
        self.depends_on = list(depends_on or [])
        self.id = call_id or f"{name}_{id(self)}"

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return f"ToolCall(id={self.id}, name={self.name}, deps={self.depends_on})"


class ToolOrchestrator:
    """工具编排器.

    Args:
        registry: 工具名 -> 可调用对象 的映射. 每个工具签名
            ``fn(args: dict, context: dict) -> Any``.
        max_workers: PARALLEL/MIXED 模式的并发线程数.
        default_timeout: 单工具执行超时 (秒), 超时返回错误观测.
    """

    def __init__(
        self,
        registry: Optional[Dict[str, Callable[..., Any]]] = None,
        max_workers: int = 8,
        default_timeout: float = 30.0,
    ) -> None:
        self.registry: Dict[str, Callable[..., Any]] = dict(registry or {})
        self.max_workers = max(1, int(max_workers))
        self.default_timeout = float(default_timeout)
        self._last_plan: List[ToolCall] = []

    # ------------------------------------------------------------------ #
    # 注册 / 查询
    # ------------------------------------------------------------------ #
    def register(self, name: str, fn: Callable[..., Any]) -> None:
        self.registry[name] = fn

    def has(self, name: str) -> bool:
        return name in self.registry

    # ------------------------------------------------------------------ #
    # 编排入口
    # ------------------------------------------------------------------ #
    def execute(
        self,
        calls: Sequence[ToolCall],
        mode: OrchestratorMode = OrchestratorMode.MIXED,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """按 ``mode`` 编排执行 ``calls``, 返回 ``{call_id: result}`` 字典. """
        ctx = dict(context or {})
        self._last_plan = list(calls)
        if not calls:
            return {}
        if mode == OrchestratorMode.CHAIN:
            return self._run_chain(calls, ctx)
        if mode == OrchestratorMode.PARALLEL:
            return self._run_parallel(calls, ctx)
        return self._run_mixed(calls, ctx)

    # ------------------------------------------------------------------ #
    # CHAIN
    # ------------------------------------------------------------------ #
    def _run_chain(self, calls: Sequence[ToolCall], ctx: Dict[str, Any]) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        prev_output: Any = None
        for call in calls:
            args = dict(call.args)
            # 链式: 自动注入上游输出
            if prev_output is not None:
                args.setdefault("prev", prev_output)
            res = self._invoke(call.name, args, ctx)
            results[call.id] = res
            if isinstance(res, dict) and "error" in res:
                # 错误不中断链, 但下游 prev 保留 error 描述
                prev_output = res["error"]
            else:
                prev_output = res
        return results

    # ------------------------------------------------------------------ #
    # PARALLEL
    # ------------------------------------------------------------------ #
    def _run_parallel(self, calls: Sequence[ToolCall], ctx: Dict[str, Any]) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            future_map = {
                pool.submit(self._invoke, c.name, dict(c.args), dict(ctx)): c.id for c in calls
            }
            for fut in as_completed(future_map):
                cid = future_map[fut]
                results[cid] = fut.result()
        return results

    # ------------------------------------------------------------------ #
    # MIXED (DAG 拓扑 + 就绪并行)
    # ------------------------------------------------------------------ #
    def _run_mixed(self, calls: Sequence[ToolCall], ctx: Dict[str, Any]) -> Dict[str, Any]:
        by_id: Dict[str, ToolCall] = {c.id: c for c in calls}
        # 校验依赖存在性
        for c in calls:
            for dep in c.depends_on:
                if dep not in by_id and dep not in ctx.get("results", {}):
                    by_id.setdefault(dep, ToolCall("__external__", call_id=dep))

        indeg: Dict[str, int] = {cid: 0 for cid in by_id}
        adj: Dict[str, List[str]] = {cid: [] for cid in by_id}
        for c in calls:
            for dep in c.depends_on:
                adj.setdefault(dep, []).append(c.id)
                indeg[c.id] += 1

        results: Dict[str, Any] = dict(ctx.get("results", {}))
        ready = [cid for cid, d in indeg.items() if d == 0 and by_id[cid].name != "__external__"]
        processed = 0
        total = len(calls)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            while processed < total:
                if not ready:
                    # 死锁/环: 剩余调用直接报错
                    for cid, c in by_id.items():
                        if cid not in results and c.name != "__external__":
                            results[cid] = {"error": "dependency_unresolvable", "call_id": cid}
                    break
                batch, ready = ready, []
                fut_map = {}
                for cid in batch:
                    c = by_id[cid]
                    args = dict(c.args)
                    # 注入依赖结果
                    for dep in c.depends_on:
                        if dep in results:
                            args.setdefault(dep, results[dep])
                    fut_map[pool.submit(self._invoke, c.name, args, dict(ctx))] = cid

                for fut in as_completed(fut_map):
                    cid = fut_map[fut]
                    results[cid] = fut.result()
                    processed += 1
                    # 解锁下游
                    for nxt in adj.get(cid, []):
                        indeg[nxt] -= 1
                        if indeg[nxt] == 0 and by_id[nxt].name != "__external__":
                            ready.append(nxt)
        return {c.id: results[c.id] for c in calls if c.id in results}

    # ------------------------------------------------------------------ #
    # 单工具执行 (带超时与错误捕获)
    # ------------------------------------------------------------------ #
    def _invoke(self, name: str, args: Dict[str, Any], ctx: Dict[str, Any]) -> Any:
        fn = self.registry.get(name)
        if fn is None:
            return {"error": f"unknown_tool:{name}", "name": name}
        start = time.time()
        try:
            out = fn(args, ctx)
            return {"ok": True, "value": out, "elapsed": time.time() - start}
        except Exception as exc:  # noqa: BLE001 - 编排层需捕获所有工具异常
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "elapsed": time.time() - start}

    @property
    def last_plan(self) -> List[ToolCall]:
        return list(self._last_plan)


__all__ = ["ToolOrchestrator", "ToolCall", "OrchestratorMode"]
