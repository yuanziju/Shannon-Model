"""分布式训练支持 (Shannon / MathMaster 共享基础设施).

提供 5D 并行配置 (:class:`ParallelConfig`)、分布式进程组管理
(:class:`DistributedManager`, 兼容 HCCL/NCCL/Gloo) 以及专家并行所需的
all-to-all 通信原语 :func:`expert_all_to_all`.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


@dataclass
class ParallelConfig:
    """5D 并行配置.

    维度: TP (张量) + PP (流水线) + DP (数据) + SP (序列) + EP (专家).
    EP 通常与 DP 或 TP 重叠, 不单独占用额外 rank 维度, 因此
    ``world_size = tp_size * pp_size * dp_size * sp_size``.
    """

    tp_size: int = 1
    pp_size: int = 1
    dp_size: int = 1
    sp_size: int = 1
    ep_size: int = 1
    world_size: int = 1
    rank: int = 0

    def __post_init__(self) -> None:
        computed = self.tp_size * self.pp_size * self.dp_size * self.sp_size
        if self.world_size < computed:
            self.world_size = computed
        if any(d <= 0 for d in (self.tp_size, self.pp_size, self.dp_size, self.sp_size, self.ep_size)):
            raise ValueError("All parallel dimensions must be positive integers")

    # -- rank 解算 ------------------------------------------------------
    def compute_5d_rank(
        self,
        tp_idx: int,
        pp_idx: int,
        dp_idx: int,
        sp_idx: int,
    ) -> int:
        """根据各并行维度的索引计算全局 rank.

        采用层次化分解 (DP 最外层, TP 最内层)::

            rank = ((dp_idx * pp_size + pp_idx) * sp_size + sp_idx) * tp_size + tp_idx

        索引从 0 开始, 越界会抛出 :class:`IndexError`.
        """
        if not (0 <= tp_idx < self.tp_size):
            raise IndexError(f"tp_idx {tp_idx} out of [0, {self.tp_size})")
        if not (0 <= pp_idx < self.pp_size):
            raise IndexError(f"pp_idx {pp_idx} out of [0, {self.pp_size})")
        if not (0 <= dp_idx < self.dp_size):
            raise IndexError(f"dp_idx {dp_idx} out of [0, {self.dp_size})")
        if not (0 <= sp_idx < self.sp_size):
            raise IndexError(f"sp_idx {sp_idx} out of [0, {self.sp_size})")
        return (
            ((dp_idx * self.pp_size + pp_idx) * self.sp_size + sp_idx) * self.tp_size
            + tp_idx
        )

    # -- 子组划分 -------------------------------------------------------
    def get_group_ranks(
        self, kind: str, anchor: Optional[Tuple[int, int, int, int]] = None
    ) -> list:
        """返回某个并行维度子组包含的全部 rank.

        kind ∈ {"tp","pp","dp","sp"}. anchor 为 (tp_idx, pp_idx, dp_idx, sp_idx)
        锚点, 默认全 0. 该方法仅用于参考, 实际通信组应在 :class:`DistributedManager`
        中通过 ``dist.new_group`` 创建.
        """
        if anchor is None:
            anchor = (0, 0, 0, 0)
        t, p, d, s = anchor
        ranks = []
        if kind == "tp":
            for ti in range(self.tp_size):
                ranks.append(self.compute_5d_rank(ti, p, d, s))
        elif kind == "pp":
            for pi in range(self.pp_size):
                ranks.append(self.compute_5d_rank(t, pi, d, s))
        elif kind == "dp":
            for di in range(self.dp_size):
                ranks.append(self.compute_5d_rank(t, p, di, s))
        elif kind == "sp":
            for si in range(self.sp_size):
                ranks.append(self.compute_5d_rank(t, p, d, si))
        else:
            raise ValueError(f"Unknown group kind: {kind}")
        return ranks


class DistributedManager:
    """初始化并管理分布式进程组 (HCCL/NCCL/Gloo).

    单进程 (``world_size == 1``) 时所有集合通信原语降级为 no-op,
    方便在无分布式环境下复用同一份代码.
    """

    def __init__(self, config: Optional[ParallelConfig] = None) -> None:
        self.config = config or ParallelConfig()
        self._initialized = False
        self._backend: Optional[str] = None

    # -- 初始化 ---------------------------------------------------------
    def init_process_group(self, backend: Optional[str] = None) -> None:
        """初始化进程组.

        backend 为 None 时自动选择: 昇腾环境使用 ``hccl``; CUDA 环境使用 ``nccl``;
        其余使用 ``gloo``. 仅当环境变量 ``WORLD_SIZE`` 已设置且 > 1 时才真正初始化
        (即由 torchrun / mpirun 等启动器拉起的场景); 否则降级为单进程模式, 所有
        集合通信原语变为 no-op, 方便本地复用同一份代码.
        """
        if self._initialized:
            return
        if not dist.is_available():
            logger.warning("torch.distributed is not available; running in single-process mode")
            return

        env_ws = os.environ.get("WORLD_SIZE")
        if env_ws is None:
            logger.info("WORLD_SIZE env not set; single-process mode")
            self._initialized = False
            self._backend = None
            return
        try:
            world_size = int(env_ws)
        except ValueError:
            logger.warning("Invalid WORLD_SIZE env %r; single-process mode", env_ws)
            return

        if world_size <= 1:
            logger.info("world_size=%d, skip init_process_group", world_size)
            self._initialized = False
            self._backend = None
            return

        rank = int(os.environ.get("RANK", self.config.rank))
        if backend is None:
            backend = self._auto_backend()

        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        self._initialized = True
        self._backend = backend
        logger.info(
            "Initialized process group: backend=%s rank=%d world_size=%d",
            backend, rank, world_size,
        )

    @staticmethod
    def _auto_backend() -> str:
        try:
            import torch_npu  # type: ignore  noqa: F401
            return "hccl"
        except Exception:
            pass
        if torch.cuda.is_available():
            return "nccl"
        return "gloo"

    # -- 查询 -----------------------------------------------------------
    def is_initialized(self) -> bool:
        return self._initialized and dist.is_available() and dist.is_initialized()

    def get_global_rank(self) -> int:
        if self.is_initialized():
            return dist.get_rank()
        return int(os.environ.get("RANK", self.config.rank))

    def get_world_size(self) -> int:
        if self.is_initialized():
            return dist.get_world_size()
        return int(os.environ.get("WORLD_SIZE", self.config.world_size))

    # -- 集合通信 -------------------------------------------------------
    def barrier(self) -> None:
        if self.is_initialized():
            dist.barrier()

    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: dist.ReduceOp = dist.ReduceOp.SUM,
        async_op: bool = False,
    ):
        """全局规约. 未初始化时直接返回原 tensor."""
        if self.is_initialized():
            work = dist.all_reduce(tensor, op=op, async_op=async_op)
            if async_op:
                return work
        return tensor

    def all_gather(self, tensor_list, tensor: torch.Tensor) -> None:
        if self.is_initialized():
            dist.all_gather(tensor_list, tensor)
        else:
            if tensor_list and len(tensor_list) == 1:
                tensor_list[0].copy_(tensor)

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        if self.is_initialized():
            dist.broadcast(tensor, src=src)
        return tensor


def expert_all_to_all(
    input_tensor: torch.Tensor,
    group: Optional["dist.ProcessGroup"] = None,
    async_op: bool = False,
) -> torch.Tensor:
    """专家并行的 all-to-all 通信.

    输入形如 ``[num_local_experts, tokens_per_expert, hidden]``, 在专家并行组内
    交换: 每个 rank 把本地各专家收到的 token 发给持有该专家的 rank, 同时接收
    自己负责的专家对应的 token. 使用 :func:`torch.distributed.all_to_all_single`
    实现, 假设各 rank 的 split 大小相等 (均匀划分).

    单进程或组大小为 1 时直接返回原 tensor.
    """
    world_size = dist.get_world_size(group) if dist.is_available() and dist.is_initialized() else 1
    if world_size <= 1:
        return input_tensor

    input_contig = input_tensor.contiguous()
    output = torch.empty_like(input_contig)
    work = dist.all_to_all_single(output, input_contig, group=group, async_op=async_op)
    if async_op:
        return output, work
    return output
