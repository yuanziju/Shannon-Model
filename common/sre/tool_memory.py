"""工具记忆 (ToolMemory).

spec §7.3: ToolMemory 实现跨工具变量共享命名空间, 让不同工具的输出
可作为下游工具的输入. 提供统一的 get/set 接口, 支持类型标注、命名空间
隔离、变量来源追踪与生命周期管理.

核心特性:
    - 跨工具命名空间: 不同工具 (sympy/lean/python) 共享同一命名空间,
      通过 ${var} 引用.
    - 类型感知: 每个变量记录类型 (symbolic / numeric / proof / dataframe /
      image / text), 供下游工具做类型检查.
    - 来源追踪: 记录变量由哪个工具产生, 何时写入.
    - 生命周期: 支持 TTL (自动过期) 与显式清理.
    - 命名空间隔离: 工具可声明私有命名空间, 避免冲突.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import torch
import torch.nn as nn


class VariableType(str, Enum):
    """工具变量类型."""

    SYMBOLIC = "symbolic"       # 符号 (sympy 表达式)
    NUMERIC = "numeric"         # 数值 (int/float/array)
    PROOF = "proof"             # 证明状态 (Lean)
    DATAFRAME = "dataframe"     # 表格数据
    IMAGE = "image"             # 图像
    TEXT = "text"               # 文本
    EXCEPTION = "exception"     # 异常
    UNKNOWN = "unknown"


@dataclass
class MemoryEntry:
    """单个变量条目."""

    name: str
    value: Any
    var_type: VariableType = VariableType.UNKNOWN
    source_tool: str = ""          # 产生该变量的工具
    timestamp: float = field(default_factory=time.time)
    ttl: Optional[float] = None    # 秒, None=永不过期
    namespace: str = "global"
    metadata: dict = field(default_factory=dict)
    version: int = 1               # 写入版本 (覆盖时递增)


@dataclass
class ToolMemoryConfig:
    """工具记忆配置."""

    max_entries: int = 1024         # 最大条目数
    default_ttl: Optional[float] = None  # 默认 TTL
    enable_versioning: bool = True  # 是否版本化
    max_versions: int = 4           # 每变量保留版本数
    enable_expiry: bool = True      # 是否启用过期清理


class ToolMemory(nn.Module):
    """跨工具变量共享命名空间.

    继承 nn.Module 仅为接口统一 (无参模块, 持有可选的嵌入投影头用于
    将变量值编码为向量注入主干).
    """

    def __init__(
        self,
        config: ToolMemoryConfig | None = None,
        value_encoder: Optional[callable] = None,
        **kwargs,
    ):
        super().__init__()
        cfg = config or ToolMemoryConfig(**kwargs)
        self.cfg = cfg
        # 主存储: namespace -> name -> entry
        self._store: dict[str, dict[str, MemoryEntry]] = {"global": {}}
        # 版本历史: namespace -> name -> [old entries]
        self._history: dict[str, dict[str, list[MemoryEntry]]] = {"global": {}}
        # 访问统计
        self._access_count: dict[str, int] = {}
        self._write_count: dict[str, int] = {}
        # 可选值编码器 (将变量值编码为向量)
        self.value_encoder = value_encoder

    # ------------------------------------------------------------------
    # 命名空间管理
    # ------------------------------------------------------------------
    def create_namespace(self, ns: str) -> None:
        """创建命名空间."""
        if ns not in self._store:
            self._store[ns] = {}
            self._history[ns] = {}

    def list_namespaces(self) -> list[str]:
        return list(self._store.keys())

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def set(
        self,
        name: str,
        value: Any,
        var_type: VariableType | str = VariableType.UNKNOWN,
        source_tool: str = "",
        namespace: str = "global",
        ttl: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> MemoryEntry:
        """写入变量.

        Args:
            name: 变量名.
            value: 变量值.
            var_type: 变量类型 (枚举或字符串).
            source_tool: 产生该变量的工具名.
            namespace: 命名空间.
            ttl: 生存时间 (秒), None 用默认.
            metadata: 附加元数据.
        """
        self.create_namespace(namespace)
        if isinstance(var_type, str):
            var_type = VariableType(var_type)
        effective_ttl = ttl if ttl is not None else self.cfg.default_ttl

        # 版本化: 保存旧版本
        if self.cfg.enable_versioning and name in self._store[namespace]:
            old = self._store[namespace][name]
            if namespace not in self._history:
                self._history[namespace] = {}
            self._history[namespace].setdefault(name, []).append(old)
            # 限制版本数
            if len(self._history[namespace][name]) > self.cfg.max_versions:
                self._history[namespace][name] = self._history[namespace][name][-self.cfg.max_versions:]
            version = old.version + 1
        else:
            version = 1

        entry = MemoryEntry(
            name=name,
            value=value,
            var_type=var_type,
            source_tool=source_tool,
            ttl=effective_ttl,
            namespace=namespace,
            metadata=metadata or {},
            version=version,
        )
        self._store[namespace][name] = entry
        self._write_count[name] = self._write_count.get(name, 0) + 1

        # 容量限制 (LRU 式淘汰: 删除最旧条目)
        if len(self._store[namespace]) > self.cfg.max_entries:
            oldest = min(
                self._store[namespace].values(),
                key=lambda e: e.timestamp,
            )
            del self._store[namespace][oldest.name]
        return entry

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------
    def get(
        self,
        name: str,
        namespace: str = "global",
        default: Any = None,
        check_ttl: bool = True,
    ) -> Any:
        """读取变量值."""
        self._access_count[name] = self._access_count.get(name, 0) + 1
        if namespace not in self._store or name not in self._store[namespace]:
            return default
        entry = self._store[namespace][name]
        if check_ttl and self.cfg.enable_expiry and entry.ttl is not None:
            if time.time() - entry.timestamp > entry.ttl:
                # 过期, 删除并返回 default
                del self._store[namespace][name]
                return default
        return entry.value

    def get_entry(
        self, name: str, namespace: str = "global"
    ) -> Optional[MemoryEntry]:
        """读取完整条目 (含元数据)."""
        if namespace not in self._store or name not in self._store[namespace]:
            return None
        return self._store[namespace][name]

    def get_history(
        self, name: str, namespace: str = "global"
    ) -> list[MemoryEntry]:
        """获取变量版本历史."""
        return self._history.get(namespace, {}).get(name, [])

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def list_names(
        self, namespace: str = "global"
    ) -> list[str]:
        """列出命名空间内所有变量名."""
        return list(self._store.get(namespace, {}).keys())

    def list_by_type(
        self, var_type: VariableType | str, namespace: str = "global"
    ) -> list[str]:
        """按类型筛选变量."""
        if isinstance(var_type, str):
            var_type = VariableType(var_type)
        return [
            name for name, entry in self._store.get(namespace, {}).items()
            if entry.var_type == var_type
        ]

    def list_by_source(
        self, source_tool: str, namespace: str = "global"
    ) -> list[str]:
        """按来源工具筛选变量."""
        return [
            name for name, entry in self._store.get(namespace, {}).items()
            if entry.source_tool == source_tool
        ]

    # ------------------------------------------------------------------
    # 删除 / 清理
    # ------------------------------------------------------------------
    def delete(self, name: str, namespace: str = "global") -> bool:
        """删除变量."""
        if namespace in self._store and name in self._store[namespace]:
            del self._store[namespace][name]
            return True
        return False

    def clear_namespace(self, namespace: str) -> int:
        """清空命名空间, 返回删除条目数."""
        if namespace not in self._store:
            return 0
        n = len(self._store[namespace])
        self._store[namespace].clear()
        return n

    def clear_expired(self) -> int:
        """清理所有过期变量, 返回清理数."""
        cleared = 0
        for ns, entries in self._store.items():
            expired = [
                name for name, entry in entries.items()
                if entry.ttl is not None
                and time.time() - entry.timestamp > entry.ttl
            ]
            for name in expired:
                del entries[name]
                cleared += 1
        return cleared

    # ------------------------------------------------------------------
    # 跨命名空间: 变量导出 / 导入 (工具间传递)
    # ------------------------------------------------------------------
    def export(
        self, names: list[str], from_ns: str = "global", to_ns: str = "global"
    ) -> int:
        """将变量从一个命名空间导出到另一个."""
        if from_ns not in self._store:
            return 0
        count = 0
        for name in names:
            if name in self._store[from_ns]:
                entry = self._store[from_ns][name]
                self.set(
                    name, entry.value, entry.var_type, entry.source_tool,
                    to_ns, entry.ttl, entry.metadata,
                )
                count += 1
        return count

    def resolve_reference(self, ref: str, namespace: str = "global") -> Any:
        """解析 ${var} 引用 (供 ToolCoordinator 调用)."""
        if not isinstance(ref, str):
            return ref
        if ref.startswith("${") and ref.endswith("}"):
            var_name = ref[2:-1]
            # 支持 ns::var 格式
            if "::" in var_name:
                ns, var_name = var_name.split("::", 1)
            else:
                ns = namespace
            return self.get(var_name, ns)
        return ref

    # ------------------------------------------------------------------
    # 统计 / 快照
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        """返回统计信息."""
        total = sum(len(entries) for entries in self._store.values())
        return {
            "total_entries": total,
            "namespaces": len(self._store),
            "per_namespace": {
                ns: len(entries) for ns, entries in self._store.items()
            },
            "top_accessed": sorted(
                self._access_count.items(), key=lambda x: -x[1]
            )[:10],
            "top_written": sorted(
                self._write_count.items(), key=lambda x: -x[1]
            )[:10],
        }

    def snapshot(self, namespace: str = "global") -> dict:
        """返回命名空间快照 (深拷贝)."""
        import copy
        return {
            name: {
                "value": copy.deepcopy(entry.value),
                "type": entry.var_type.value,
                "source": entry.source_tool,
                "version": entry.version,
                "timestamp": entry.timestamp,
            }
            for name, entry in self._store.get(namespace, {}).items()
        }

    # ------------------------------------------------------------------
    # 编码: 将命名空间内变量编码为向量 (供注入主干)
    # ------------------------------------------------------------------
    def encode_namespace(
        self, namespace: str = "global", device: torch.device | None = None
    ) -> torch.Tensor | None:
        """将命名空间内所有变量编码为向量矩阵 [M, D].

        需要提供 value_encoder. 无变量或无编码器返回 None.
        """
        if self.value_encoder is None:
            return None
        entries = self._store.get(namespace, {})
        if not entries:
            return None
        vectors = []
        for name, entry in entries.items():
            try:
                vec = self.value_encoder(entry.value, entry.var_type)
                if vec is not None:
                    vectors.append(vec)
            except Exception:
                continue
        if not vectors:
            return None
        return torch.stack(vectors).to(device or torch.device("cpu"))

    def extra_repr(self) -> str:
        return (
            f"max_entries={self.cfg.max_entries}, "
            f"namespaces={len(self._store)}, "
            f"versioning={self.cfg.enable_versioning}"
        )
