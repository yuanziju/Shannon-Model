"""长程记忆 - cosine top-k 检索.

LongTermMemory 维护 Agent 跨会话的向量记忆库, 提供基于 cosine 相似度的
top-k 检索. 向量以纯 Python list[float] 存储, 不依赖 numpy.
复用长程记忆模块 (T2.7.3), 避免与对话状态管理器重复实现.
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    na, nb = _norm(a), _norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return _dot(a, b) / (na * nb)


class MemoryEntry:
    """单条记忆条目. """

    __slots__ = ("id", "vector", "content", "metadata", "timestamp", "importance", "access_count")

    def __init__(
        self,
        entry_id: str,
        vector: List[float],
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 1.0,
        timestamp: Optional[float] = None,
    ) -> None:
        self.id = entry_id
        self.vector = list(vector)
        self.content = content
        self.metadata = dict(metadata or {})
        self.importance = float(importance)
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.access_count = 0

    def score(self, decay: float = 0.0) -> float:
        """综合评分: 相似度 * 重要性 * 时效衰减 * 访问增益. """
        recency = math.exp(-decay * (time.time() - self.timestamp) / 86400.0)
        return self.importance * recency * (1.0 + 0.1 * math.log1p(self.access_count))


class LongTermMemory:
    """长程记忆库, 支持 cosine top-k 检索与重要性遗忘.

    Args:
        capacity: 最大条目数, 超出时按综合评分淘汰最低者 (LRU+重要性混合).
        dim: 向量维度 (仅用于校验, 不强制).
        decay: 时效衰减系数 (每天), 0 表示不衰减.
    """

    def __init__(
        self,
        capacity: int = 100_000,
        dim: Optional[int] = None,
        decay: float = 0.05,
    ) -> None:
        self.capacity = max(1, int(capacity))
        self.dim = dim
        self.decay = float(decay)
        self._store: Dict[str, MemoryEntry] = {}
        self._counter = 0
        # 工作记忆 (短期, FIFO 滑动窗口), 与长程记忆解耦
        self._working: Deque[Dict[str, Any]] = deque(maxlen=64)

    # ------------------------------------------------------------------ #
    # 写入
    # ------------------------------------------------------------------ #
    def add(
        self,
        vector: Sequence[float],
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 1.0,
        entry_id: Optional[str] = None,
    ) -> str:
        self._counter += 1
        eid = entry_id or f"mem_{self._counter}"
        entry = MemoryEntry(eid, list(vector), content, metadata, importance)
        if self.dim is None and entry.vector:
            self.dim = len(entry.vector)
        self._store[eid] = entry
        self._evict_if_needed()
        return eid

    def push_working(self, item: Dict[str, Any]) -> None:
        """压入工作记忆 (短期上下文, 不参与检索). """
        self._working.append(item)

    # ------------------------------------------------------------------ #
    # 检索
    # ------------------------------------------------------------------ #
    def retrieve(
        self,
        query: Sequence[float],
        top_k: int = 5,
        min_score: float = 0.0,
        filter_fn: Optional[callable] = None,
    ) -> List[Tuple[MemoryEntry, float]]:
        """cosine top-k 检索.

        Returns:
            ``[(entry, combined_score), ...]`` 按分数降序. combined_score =
            cosine_similarity * entry.score(decay).
        """
        q = list(query)
        if not q or not self._store:
            return []
        top_k = max(1, int(top_k))
        scored: List[Tuple[float, MemoryEntry]] = []
        for entry in self._store.values():
            if filter_fn is not None and not filter_fn(entry):
                continue
            sim = cosine_similarity(q, entry.vector)
            combined = sim * entry.score(self.decay)
            if combined >= min_score:
                scored.append((combined, entry))
        scored.sort(key=lambda t: t[0], reverse=True)
        results: List[Tuple[MemoryEntry, float]] = []
        for combined, entry in scored[:top_k]:
            entry.access_count += 1
            results.append((entry, combined))
        return results

    def recall_text(self, query: Sequence[float], top_k: int = 5) -> List[str]:
        """便捷方法: 仅返回检索到的 content 列表. """
        return [e.content for e, _ in self.retrieve(query, top_k=top_k)]

    # ------------------------------------------------------------------ #
    # 维护
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self._store)

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        return self._store.get(entry_id)

    def remove(self, entry_id: str) -> bool:
        return self._store.pop(entry_id, None) is not None

    def consolidate(self, threshold: float = 0.98) -> int:
        """合并高度相似的冗余记忆, 返回合并掉的条目数. """
        entries = list(self._store.values())
        removed = 0
        for i in range(len(entries)):
            if entries[i].id not in self._store:
                continue
            for j in range(i + 1, len(entries)):
                if entries[j].id not in self._store:
                    continue
                if cosine_similarity(entries[i].vector, entries[j].vector) >= threshold:
                    # 保留重要性更高者, 合并 metadata
                    keep, drop = (entries[i], entries[j]) if entries[i].importance >= entries[j].importance else (entries[j], entries[i])
                    keep.importance = max(keep.importance, drop.importance)
                    keep.access_count += drop.access_count
                    self._store.pop(drop.id, None)
                    removed += 1
        return removed

    def _evict_if_needed(self) -> None:
        while len(self._store) > self.capacity:
            # 淘汰综合评分最低的条目
            worst_id = min(self._store, key=lambda eid: self._store[eid].score(self.decay))
            self._store.pop(worst_id, None)

    @property
    def working_memory(self) -> List[Dict[str, Any]]:
        return list(self._working)


__all__ = ["LongTermMemory", "MemoryEntry", "cosine_similarity"]
