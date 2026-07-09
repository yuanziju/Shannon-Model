"""设备管理与后端检测 (Shannon / MathMaster 共享基础设施).

按优先级惰性检测计算后端: torch_npu (CANN/昇腾) > CUDA > MLX (Apple Silicon) > CPU.
提供单例 :func:`get_default_manager` 以便全局复用检测结果。
"""

from __future__ import annotations

import enum
import logging
import os
import threading
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class Backend(enum.Enum):
    """支持的计算后端."""

    CANN = "cann"   # 华为昇腾, 通过 torch_npu
    CUDA = "cuda"   # NVIDIA GPU
    MLX = "mlx"     # Apple Silicon
    CPU = "cpu"


class DeviceManager:
    """惰性检测后端并缓存结果的设备管理器.

    检测顺序遵循 ``torch_npu > torch.cuda > mlx > CPU``.
    所有检测只执行一次, 后续调用直接返回缓存值.
    """

    def __init__(self) -> None:
        self._backend: Optional[Backend] = None
        self._device: Optional[torch.device] = None
        self._device_count: Optional[int] = None
        self._lock = threading.Lock()

    # -- backend --------------------------------------------------------
    def detect_backend(self) -> Backend:
        """惰性检测当前可用的最佳后端."""
        if self._backend is not None:
            return self._backend
        with self._lock:
            if self._backend is not None:
                return self._backend
            # 1. CANN / 昇腾 (torch_npu)
            try:
                import torch_npu  # type: ignore  noqa: F401
                if torch.npu.is_available():  # type: ignore[attr-defined]
                    self._backend = Backend.CANN
                    logger.info("Detected backend: CANN (torch_npu)")
                    return self._backend
            except Exception:
                pass
            # 2. CUDA
            try:
                if torch.cuda.is_available():
                    self._backend = Backend.CUDA
                    logger.info("Detected backend: CUDA")
                    return self._backend
            except Exception:
                pass
            # 3. MLX (Apple Silicon)
            try:
                import mlx  # type: ignore  noqa: F401
                import mlx.core as mx  # type: ignore  noqa: F401
                self._backend = Backend.MLX
                logger.info("Detected backend: MLX")
                return self._backend
            except Exception:
                pass
            # 4. CPU fallback
            self._backend = Backend.CPU
            logger.info("Detected backend: CPU")
            return self._backend

    # -- device ---------------------------------------------------------
    def get_device(self) -> torch.device:
        """返回当前后端对应的 torch.device.

        对于 MLX, 因为 MLX 自行管理设备, 这里返回 ``cpu`` 以便与 torch 互操作.
        分布式场景下根据 ``LOCAL_RANK`` 选择对应设备.
        """
        if self._device is not None:
            return self._device
        backend = self.detect_backend()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if backend == Backend.CANN:
            idx = min(local_rank, max(self.device_count() - 1, 0))
            self._device = torch.device("npu", idx)  # type: ignore[attr-defined]
        elif backend == Backend.CUDA:
            idx = min(local_rank, max(self.device_count() - 1, 0))
            self._device = torch.device("cuda", idx)
        elif backend == Backend.MLX:
            # MLX 自行管理显存, torch 侧默认 cpu
            self._device = torch.device("cpu")
        else:
            self._device = torch.device("cpu")
        logger.debug("Using device: %s", self._device)
        return self._device

    # -- distributed ----------------------------------------------------
    def is_distributed(self) -> bool:
        """判断是否处于分布式训练环境."""
        world_size = os.environ.get("WORLD_SIZE")
        if world_size is None:
            return False
        try:
            return int(world_size) > 1
        except ValueError:
            return False

    # -- device count ---------------------------------------------------
    def device_count(self) -> int:
        """返回当前后端可用设备数."""
        if self._device_count is not None:
            return self._device_count
        backend = self.detect_backend()
        if backend == Backend.CANN:
            try:
                import torch_npu  # type: ignore
                self._device_count = torch.npu.device_count()  # type: ignore[attr-defined]
            except Exception:
                self._device_count = 0
        elif backend == Backend.CUDA:
            self._device_count = torch.cuda.device_count()
        elif backend == Backend.MLX:
            # MLX 通常是单设备 (统一内存)
            self._device_count = 1
        else:
            self._device_count = 1
        return self._device_count

    # -- reset (主要用于测试) -------------------------------------------
    def reset(self) -> None:
        """清除缓存, 强制下次重新检测."""
        with self._lock:
            self._backend = None
            self._device = None
            self._device_count = None


# -- 单例 -----------------------------------------------------------------
_default_manager: Optional[DeviceManager] = None
_singleton_lock = threading.Lock()


def get_default_manager() -> DeviceManager:
    """返回全局共享的 :class:`DeviceManager` 单例."""
    global _default_manager
    if _default_manager is None:
        with _singleton_lock:
            if _default_manager is None:
                _default_manager = DeviceManager()
    return _default_manager
