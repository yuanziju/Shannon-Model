"""梯度检查点工具模块.

提供 use_reentrant=False 的梯度检查点包装器与序列检查点工具,
用于在循环主体中节省显存 (spec: 省内存 32-47%).
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class GradientCheckpoint(nn.Module):
    """梯度检查点包装器 (use_reentrant=False).

    包装任意子模块, 在前向时不保存中间激活, 反向时重新计算,
    从而以约 33% 额外计算换取显存节省. use_reentrant=False 为
    PyTorch 推荐的安全模式, 支持如 dropout 等随机操作.
    """

    def __init__(self, module: nn.Module, use_reentrant: bool = False):
        super().__init__()
        self.module = module
        self.use_reentrant = use_reentrant

    def forward(self, *args, **kwargs):
        # 需要将所有可微输入作为位置参数传入 checkpoint.
        # kwargs 不参与重计算图的保护, 这里直接透传给 module.
        def _run(*inputs):
            return self.module(*inputs, **kwargs)

        if not args:
            # 无可微位置参数时直接调用, 避免 checkpoint 报错
            return self.module(*args, **kwargs)
        return checkpoint(_run, *args, use_reentrant=self.use_reentrant)

    def extra_repr(self) -> str:
        return f"use_reentrant={self.use_reentrant}"


def checkpoint_sequential(
    modules: Sequence[nn.Module],
    inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    use_reentrant: bool = False,
) -> torch.Tensor:
    """对模块序列依次应用梯度检查点.

    modules: 模块序列 (如 nn.ModuleList), 每个模块接受单个张量并返回单个张量.
    inputs: 初始输入张量或张量元组 (元组仅用于首模块; 后续模块串联单张量).
    返回最后一个模块的输出.
    """
    if isinstance(inputs, torch.Tensor):
        current = inputs
    else:
        # 首模块接收元组输入
        assert len(modules) > 0
        current = checkpoint(modules[0], *inputs, use_reentrant=use_reentrant)
        modules = modules[1:]

    for module in modules:
        current = checkpoint(module, current, use_reentrant=use_reentrant)
    return current
