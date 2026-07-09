"""请求调度器 - P0-P4 优先级 + 抢占 + 连续批处理.

RequestScheduler 实现推理服务的请求调度:
    - 5 级优先级 (P0 最高 / P4 最低), 同级 FIFO.
    - 抢占式调度: 高优先级请求可抢占低优先级正在处理的请求.
    - 连续批处理 (continuous batching): 动态加入/移除请求, 最大化吞吐.
"""

from __future__ import annotations

import heapq
import itertools
import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple


class Priority(IntEnum):
    """5 级优先级 (P0 实时 / P1 高 / P2 中 / P3 低 / P4 后台). """

    P0 = 0  # 实时对话 (流式, 抢占一切)
    P1 = 1  # 高优先交互
    P2 = 2  # 中优先 (默认)
    P3 = 3  # 低优先批处理
    P4 = 4  # 后台离线

    @classmethod
    def from_label(cls, label: str) -> "Priority":
        m = {"P0": cls.P0, "P1": cls.P1, "P2": cls.P2, "P3": cls.P3, "P4": cls.P4}
        return m.get(label.upper(), cls.P2)


@dataclass(order=True)
class _HeapItem:
    sort_key: Tuple[int, int]          # (priority, seq) -> 保证同级 FIFO
    request: "InferRequest" = field(compare=False)


@dataclass
class InferRequest:
    req_id: str
    prompt: str
    priority: Priority = Priority.P2
    max_tokens: int = 512
    arrived_at: float = field(default_factory=time.time)
    state: str = "queued"              # queued / running / preempted / done / cancelled
    generated: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)
    # 抢占恢复用
    kv_handle: Optional[str] = None

    @property
    def is_preemptable(self) -> bool:
        return self.priority >= Priority.P3


class RequestScheduler:
    """推理请求调度器.

    Args:
        max_batch_size: 单批次最大并发请求数 (连续批处理上限).
        max_running: 同时处于 running 状态的请求上限 (显存约束).
        preemption_enabled: 是否允许抢占 (P0/P1 抢占 P3/P4).
        time_slice: 调度时间片 (秒), 仅用于统计.
    """

    def __init__(
        self,
        max_batch_size: int = 32,
        max_running: int = 256,
        preemption_enabled: bool = True,
        time_slice: float = 0.05,
    ) -> None:
        self.max_batch_size = max(1, int(max_batch_size))
        self.max_running = max(1, int(max_running))
        self.preemption_enabled = preemption_enabled
        self.time_slice = float(time_slice)

        self._heap: List[_HeapItem] = []
        self._counter = itertools.count()
        self._running: Dict[str, InferRequest] = {}
        self._preempted: Deque[InferRequest] = deque()
        self._done: List[InferRequest] = []
        self._cancelled: set = set()

    # ------------------------------------------------------------------ #
    # 提交
    # ------------------------------------------------------------------ #
    def submit(self, request: InferRequest) -> str:
        request.state = "queued"
        seq = next(self._counter)
        heapq.heappush(self._heap, _HeapItem((int(request.priority), seq), request))
        return request.req_id

    def cancel(self, req_id: str) -> bool:
        if req_id in self._running:
            self._running[req_id].state = "cancelled"
            self._running.pop(req_id, None)
            self._cancelled.add(req_id)
            return True
        self._cancelled.add(req_id)
        return False

    # ------------------------------------------------------------------ #
    # 调度
    # ------------------------------------------------------------------ #
    def schedule(self) -> List[InferRequest]:
        """返回本时间片应 running 的请求批次 (连续批处理).

        策略:
            1. 若 running 不足, 从堆中弹出最高优先级请求补充.
            2. 若高优先级请求到达且 running 已满, 抢占最低优先级可抢占请求.
        """
        # 清理已完成/取消
        for rid in list(self._running):
            r = self._running[rid]
            if r.state in ("done", "cancelled"):
                self._running.pop(rid, None)
                if r.state == "done":
                    self._done.append(r)

        # 恢复被抢占请求
        while self._preempted and len(self._running) < self.max_running:
            r = self._preempted.popleft()
            r.state = "running"
            self._running[r.req_id] = r

        # 从队列补充
        while self._heap and len(self._running) < self.max_running:
            item = heapq.heappop(self._heap)
            r = item.request
            if r.req_id in self._cancelled:
                continue
            r.state = "running"
            self._running[r.req_id] = r

        # 抢占逻辑: 队列中有更高优先级请求, 但 running 已满
        if self.preemption_enabled and self._heap:
            self._try_preempt()

        # 返回当前批次 (受 max_batch_size 限制)
        running_list = sorted(self._running.values(), key=lambda r: (int(r.priority), r.arrived_at))
        return running_list[: self.max_batch_size]

    def _try_preempt(self) -> None:
        # 找到 running 中最低优先级且可抢占的请求
        if not self._running:
            return
        victim_id = None
        victim_pri = -1
        for rid, r in self._running.items():
            if r.is_preemptable and int(r.priority) > victim_pri:
                victim_pri = int(r.priority)
                victim_id = rid
        if victim_id is None:
            return
        # 队首请求优先级是否严格更高
        top = self._heap[0]
        if top.sort_key[0] < victim_pri:
            heapq.heappop(self._heap)
            new_req = top.request
            victim = self._running.pop(victim_id)
            victim.state = "preempted"
            self._preempted.append(victim)
            new_req.state = "running"
            self._running[new_req.req_id] = new_req

    # ------------------------------------------------------------------ #
    # 完成回调
    # ------------------------------------------------------------------ #
    def complete(self, req_id: str, generated_tokens: Optional[int] = None) -> None:
        r = self._running.pop(req_id, None)
        if r is None:
            return
        if generated_tokens is not None:
            r.generated = int(generated_tokens)
        r.state = "done"
        self._done.append(r)

    # ------------------------------------------------------------------ #
    # 状态查询
    # ------------------------------------------------------------------ #
    @property
    def queue_size(self) -> int:
        return len(self._heap)

    @property
    def running_count(self) -> int:
        return len(self._running)

    @property
    def preempted_count(self) -> int:
        return len(self._preempted)

    @property
    def done_count(self) -> int:
        return len(self._done)

    def stats(self) -> Dict[str, int]:
        return {
            "queued": self.queue_size,
            "running": self.running_count,
            "preempted": self.preempted_count,
            "done": self.done_count,
            "cancelled": len(self._cancelled),
        }


__all__ = ["RequestScheduler", "InferRequest", "Priority"]
