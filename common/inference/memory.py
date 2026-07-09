"""显存管理 - Paged KV + SSM Swap + 24-48GB 目标.

MemoryManager 统一管理推理显存:
    - Paged KV:         分页 KV Cache (复用 CacheManager 抽象)
    - SSM Swap:         SSM 循环状态在显存 <-> 内存(CPU)间换入换出
    - 动态 1-32 循环状态: RDT 循环块动态迭代状态
    - 目标显存:         24-48GB (双卡 4090/3090 或单卡 A100)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class Device(Enum):
    GPU = "GPU"
    CPU = "CPU"   # SSM swap 目标


class MemoryPool:
    """显存池 (简化: 字节级账本). """

    def __init__(self, total_bytes: float) -> None:
        self.total = float(total_bytes)
        self.used = 0.0

    @property
    def free(self) -> float:
        return max(0.0, self.total - self.used)

    def allocate(self, nbytes: float) -> bool:
        if nbytes > self.free:
            return False
        self.used += float(nbytes)
        return True

    def release(self, nbytes: float) -> None:
        self.used = max(0.0, self.used - float(nbytes))


@dataclass
class SSMState:
    """单个 SSM (状态空间模型) 循环状态. """

    state_id: str
    layer: int
    loop_depth: int               # 1-32 动态循环深度
    bytes: float
    device: Device = Device.GPU
    last_access: float = 0.0
    pinned: bool = False          # 常驻不换出


# 默认显存目标 (spec: 双卡 4090/3090 24GB 或单卡 A100 40GB)
DEFAULT_GPU_BUDGET = 40.0 * 1024 ** 3   # 40 GB
DEFAULT_CPU_BUDGET = 128.0 * 1024 ** 3  # 128 GB
MIN_LOOP_DEPTH = 1
MAX_LOOP_DEPTH = 32


class MemoryManager:
    """推理显存管理器.

    Args:
        gpu_budget: GPU 显存预算 (字节).
        cpu_budget: CPU 内存预算 (字节, 用于 SSM swap).
        reserve_ratio: 预留给激活值/权重的显存比例.
    """

    def __init__(
        self,
        gpu_budget: float = DEFAULT_GPU_BUDGET,
        cpu_budget: float = DEFAULT_CPU_BUDGET,
        reserve_ratio: float = 0.4,
    ) -> None:
        self.gpu = MemoryPool(gpu_budget)
        self.cpu = MemoryPool(cpu_budget)
        self.reserve_ratio = max(0.0, min(0.9, float(reserve_ratio)))
        # KV Cache 与 SSM 状态共享 (1 - reserve_ratio) 的显存
        self._kv_bytes = 0.0
        self._ssm_states: Dict[str, SSMState] = {}
        self._swap_count = 0
        self._oom_count = 0

    # ------------------------------------------------------------------ #
    # KV Cache 配额
    # ------------------------------------------------------------------ #
    @property
    def kv_budget(self) -> float:
        """KV Cache 可用显存 = GPU总 * (1 - reserve). """
        return self.gpu.total * (1.0 - self.reserve_ratio)

    @property
    def kv_used(self) -> float:
        return self._kv_bytes

    def allocate_kv(self, nbytes: float) -> bool:
        if self._kv_bytes + nbytes > self.kv_budget:
            return False
        self._kv_bytes += float(nbytes)
        return True

    def release_kv(self, nbytes: float) -> None:
        self._kv_bytes = max(0.0, self._kv_bytes - float(nbytes))

    # ------------------------------------------------------------------ #
    # SSM 循环状态管理 (1-32 动态循环深度)
    # ------------------------------------------------------------------ #
    def register_ssm(self, state_id: str, layer: int, loop_depth: int, nbytes: float) -> bool:
        depth = max(MIN_LOOP_DEPTH, min(MAX_LOOP_DEPTH, int(loop_depth)))
        # 实际占用 = 单步状态 * 循环深度
        actual = float(nbytes) * depth
        state = SSMState(state_id=state_id, layer=layer, loop_depth=depth, bytes=actual)
        # 先尝试 GPU
        if self.gpu.allocate(actual):
            state.device = Device.GPU
            self._ssm_states[state_id] = state
            return True
        # 换出部分冷状态腾出空间
        self._swap_out_cold(needed=actual)
        if self.gpu.allocate(actual):
            state.device = Device.GPU
            self._ssm_states[state_id] = state
            return True
        # 直接放 CPU
        if self.cpu.allocate(actual):
            state.device = Device.CPU
            self._ssm_states[state_id] = state
            return True
        self._oom_count += 1
        return False

    def access_ssm(self, state_id: str) -> bool:
        """访问某状态: 若在 CPU 则换入 GPU. """
        state = self._ssm_states.get(state_id)
        if state is None:
            return False
        if state.device == Device.CPU:
            return self._swap_in(state_id)
        return True

    def release_ssm(self, state_id: str) -> None:
        state = self._ssm_states.pop(state_id, None)
        if state is None:
            return
        if state.device == Device.GPU:
            self.gpu.release(state.bytes)
        else:
            self.cpu.release(state.bytes)

    def set_loop_depth(self, state_id: str, loop_depth: int) -> bool:
        """动态调整循环深度 (1-32), 重新分配显存. """
        state = self._ssm_states.get(state_id)
        if state is None:
            return False
        new_depth = max(MIN_LOOP_DEPTH, min(MAX_LOOP_DEPTH, int(loop_depth)))
        if new_depth == state.loop_depth:
            return True
        # 释放旧占用
        pool = self.gpu if state.device == Device.GPU else self.cpu
        pool.release(state.bytes)
        old_depth = state.loop_depth
        state.loop_depth = new_depth
        state.bytes = state.bytes / old_depth * new_depth
        if not pool.allocate(state.bytes):
            # 重新分配失败, 尝试换出
            self._swap_out_cold(needed=state.bytes)
            if not pool.allocate(state.bytes):
                # 放 CPU
                if self.cpu.allocate(state.bytes):
                    state.device = Device.CPU
                else:
                    self._oom_count += 1
                    return False
        return True

    # ------------------------------------------------------------------ #
    # Swap 机制
    # ------------------------------------------------------------------ #
    def _swap_out_cold(self, needed: float) -> float:
        """按 LRU 换出非 pinned 的 GPU 冷状态到 CPU, 返回释放字节数. """
        freed = 0.0
        # 按 last_access 升序 (最久未访问优先换出)
        candidates = sorted(
            [s for s in self._ssm_states.values() if s.device == Device.GPU and not s.pinned],
            key=lambda s: s.last_access,
        )
        for state in candidates:
            if freed >= needed:
                break
            if self.cpu.allocate(state.bytes):
                self.gpu.release(state.bytes)
                state.device = Device.CPU
                self._swap_count += 1
                freed += state.bytes
        return freed

    def _swap_in(self, state_id: str) -> bool:
        state = self._ssm_states.get(state_id)
        if state is None or state.device != Device.CPU:
            return False
        if not self.gpu.allocate(state.bytes):
            self._swap_out_cold(needed=state.bytes)
            if not self.gpu.allocate(state.bytes):
                return False
        self.cpu.release(state.bytes)
        state.device = Device.GPU
        self._swap_count += 1
        return True

    # ------------------------------------------------------------------ #
    # 统计
    # ------------------------------------------------------------------ #
    def stats(self) -> Dict[str, Any]:
        gpu_states = sum(1 for s in self._ssm_states.values() if s.device == Device.GPU)
        cpu_states = sum(1 for s in self._ssm_states.values() if s.device == Device.CPU)
        return {
            "gpu_total_gb": round(self.gpu.total / 1024 ** 3, 2),
            "gpu_used_gb": round(self.gpu.used / 1024 ** 3, 2),
            "gpu_free_gb": round(self.gpu.free / 1024 ** 3, 2),
            "kv_budget_gb": round(self.kv_budget / 1024 ** 3, 2),
            "kv_used_gb": round(self.kv_used / 1024 ** 3, 2),
            "ssm_states_gpu": gpu_states,
            "ssm_states_cpu": cpu_states,
            "swap_count": self._swap_count,
            "oom_count": self._oom_count,
            "target_range_gb": [24, 48],
        }


__all__ = [
    "MemoryManager",
    "MemoryPool",
    "SSMState",
    "Device",
    "DEFAULT_GPU_BUDGET",
    "DEFAULT_CPU_BUDGET",
    "MIN_LOOP_DEPTH",
    "MAX_LOOP_DEPTH",
]
