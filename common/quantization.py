"""量化 (Shannon / MathMaster 共享基础设施).

采用 *伪量化* (pseudo-quantization): 把权重 / 激活量化后再反量化回 float,
量化噪声被注入到浮点张量中, 而 scale 因子单独保存为 buffer. 不使用真正的
低比特整数存储, 兼容所有后端 (CANN / CUDA / MLX / CPU).

逐组件量化策略 (与 spec.md 推理引擎一致):
    - 大专家 FFN 权重 -> W8A16
    - KV Cache        -> FP8
    - 其余            -> FP16
    - NVFP4 仅 Blackwell 可用, 当前硬件回退到 W4A16 伪量化
"""

from __future__ import annotations

import enum
import logging
from typing import Any, Callable, Dict, Iterable, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class Precision(enum.Enum):
    """支持的精度档位."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    FP8 = "fp8"
    W8A16 = "w8a16"
    W4A16 = "w4a16"
    NVFP4 = "nvfp4"

    @property
    def bits(self) -> int:
        if self in (Precision.INT8, Precision.FP8, Precision.W8A16):
            return 8
        if self in (Precision.W4A16, Precision.NVFP4):
            return 4
        return 0  # 纯浮点


_DTYPE_MAP: Dict[Precision, torch.dtype] = {
    Precision.FP32: torch.float32,
    Precision.FP16: torch.float16,
    Precision.BF16: torch.bfloat16,
}

#: 精度档位 <-> 整数编码双向表, 用于把精度存入 buffer (torch 无法存字符串).
_PRECISION_CODE: Dict[Precision, int] = {p: i for i, p in enumerate(Precision)}
_CODE_TO_PRECISION: Dict[int, Precision] = {i: p for p, i in _PRECISION_CODE.items()}


class Quantizer:
    """逐组件伪量化器.

    用法:
        q = Quantizer()
        q.quantize(model)                # 按默认规则量化整个模型
        q.quantize(tensor, Precision.W8A16, name="expert.ffn.w1")  # 量化单个张量
        q.dynamic_switch(model, Precision.BF16)  # 运行时整体精度切换
        q.sensitivity_analysis(model, eval_fn)   # 逐层敏感度分析
    """

    def __init__(self, default_precision: Precision = Precision.FP16) -> None:
        self.default_precision = default_precision
        # 组件名 -> 精度 (覆盖默认规则)
        self.component_map: Dict[str, Precision] = {}
        # 参数名 -> scale 张量 (伪量化时计算并缓存)
        self.scales: Dict[str, torch.Tensor] = {}
        # 敏感度分析结果: 参数名 -> {precision: deviation}
        self.sensitivity_scores: Dict[str, Dict[str, float]] = {}

    # ==================================================================
    # 公共 API
    # ==================================================================
    def register_component(self, name: str, precision: Precision) -> None:
        """显式注册某组件的量化精度, 优先于默认规则."""
        self.component_map[name] = precision

    def set_default(self, precision: Precision) -> None:
        self.default_precision = precision

    def quantize(
        self,
        target: Any,
        precision: Optional[Precision] = None,
        name: Optional[str] = None,
    ) -> Any:
        """量化目标.

        - ``target`` 为 :class:`torch.Tensor`: 返回伪量化后的 float 张量,
          若提供 ``name`` 则 scale 会被缓存到 :attr:`scales`.
        - ``target`` 为 :class:`torch.nn.Module`: 按逐组件规则就地量化全部参数.
        """
        if isinstance(target, torch.Tensor):
            p = precision or self.component_map.get(name, self.default_precision)
            return self._pseudo_quantize(target, p, name)
        if isinstance(target, nn.Module):
            return self._quantize_module(target, precision)
        raise TypeError(f"unsupported quantize target: {type(target)}")

    def dynamic_switch(self, module: nn.Module, precision: Precision) -> nn.Module:
        """运行时动态精度切换 (整体 dtype 转换).

        用于不同阶段 (训练 BF16 / 推理 FP16 / 低资源 FP32) 间的快速切换.
        """
        dtype = _DTYPE_MAP.get(precision)
        if dtype is None:
            # 低比特档位: 用伪量化近似
            logger.info("dynamic_switch to %s via pseudo-quantization", precision.value)
            return self._quantize_module(module, precision)
        module.to(dtype=dtype)
        logger.info("dynamic_switch module to %s", precision.value)
        return module

    def sensitivity_analysis(
        self,
        module: nn.Module,
        eval_fn: Callable[[nn.Module], Any],
        precisions: Optional[Iterable[Precision]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """逐层敏感度分析.

        对每个参数, 分别用各精度伪量化后, 比较 :func:`eval_fn` 输出与基线
        (不量化) 的偏差. 偏差越大表示该层对量化越敏感. 结果同时存入
        :attr:`sensitivity_scores`.
        """
        precisions = list(precisions) if precisions else [Precision.W8A16, Precision.W4A16]
        with torch.no_grad():
            baseline = eval_fn(module)
        results: Dict[str, Dict[str, float]] = {}
        for pname, param in module.named_parameters():
            orig = param.data.detach().clone()
            scores: Dict[str, float] = {}
            for prec in precisions:
                with torch.no_grad():
                    param.data.copy_(self._pseudo_quantize(orig, prec, pname))
                    out = eval_fn(module)
                    scores[prec.value] = self._deviation(baseline, out)
                with torch.no_grad():
                    param.data.copy_(orig)
            results[pname] = scores
            self.sensitivity_scores[pname] = scores
        logger.info("sensitivity_analysis done for %d parameters", len(results))
        return results

    # ==================================================================
    # 内部: 伪量化核心
    # ==================================================================
    def _pseudo_quantize(
        self,
        tensor: torch.Tensor,
        precision: Precision,
        name: Optional[str] = None,
    ) -> torch.Tensor:
        """伪量化: quantize -> dequantize 回 float, scale 缓存.

        - 纯浮点档位 (FP32/FP16/BF16): 仅做 dtype 转换;
        - FP8 / INT8 / W8A16: 8-bit 对称逐通道 (末维) 量化;
        - W4A16 / NVFP4: 4-bit 对称逐通道量化, 输出保持原 dtype.
        """
        # 纯浮点档位
        dtype = _DTYPE_MAP.get(precision)
        if dtype is not None:
            return tensor.to(dtype)

        # NVFP4 在非 Blackwell 上不可用, 回退到 W4A16 伪量化
        if precision == Precision.NVFP4 and not _has_nvfp4():
            logger.warning("NVFP4 unavailable on current hardware; falling back to W4A16 pseudo-quant")
            precision = Precision.W4A16

        bits = precision.bits
        if bits <= 0:
            return tensor

        return self._symmetric_pseudo_quant(tensor, bits, name)

    def _symmetric_pseudo_quant(
        self, tensor: torch.Tensor, bits: int, name: Optional[str]
    ) -> torch.Tensor:
        """对称逐通道 (沿最后一维) 伪量化.

        步骤: 计算 scale -> 量化到整数 -> 反量化回 float. scale 存入
        :attr:`scales` (当 ``name`` 给出时), 而返回值是与原张量同 dtype 的
        带量化噪声的 float 张量.
        """
        orig_dtype = tensor.dtype
        work = tensor.detach()
        if work.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            work = work.to(torch.float32)
        else:
            work = work.to(torch.float32)

        qmax = (1 << (bits - 1)) - 1  # 对称: [-qmax, qmax]

        # 逐通道 (最后一维); 标量/1D 则逐张量
        if work.dim() >= 2:
            reduce_dims = tuple(range(work.dim() - 1))
            absmax = work.abs().amax(dim=reduce_dims, keepdim=True).clamp(min=1e-8)
        else:
            # 0-d (标量) 或 1-d: 逐张量, 保持 0-d 形状以正确广播
            absmax = work.abs().amax().clamp(min=1e-8)

        scale = absmax / float(qmax)
        q = torch.round(work / scale).clamp(-qmax, qmax)
        deq = q * scale

        if name is not None:
            # scale 存为 buffer 友好的 1D/张量, 转回原精度
            self.scales[name] = scale.squeeze().to(orig_dtype).clone()

        return deq.to(orig_dtype)

    # ==================================================================
    # 内部: 模块级逐组件量化
    # ==================================================================
    def _quantize_module(
        self, module: nn.Module, override: Optional[Precision] = None
    ) -> nn.Module:
        """按逐组件规则就地量化模块参数.

        默认规则 (可用 :meth:`register_component` 覆盖或 ``override`` 强制):
            - 专家 FFN 权重 -> W8A16
            - KV cache 相关 -> FP8
            - 其余          -> FP16
        """
        with torch.no_grad():
            for pname, param in module.named_parameters():
                prec = override or self.component_map.get(pname) or self._infer_precision(pname)
                quantized = self._pseudo_quantize(param.data, prec, pname)
                param.data.copy_(quantized.to(param.data.dtype))
                # 仅对真正做了低比特伪量化 (产生 scale) 的参数注册 buffer,
                # 纯浮点档位 (FP16/BF16/FP32) 不注册, 避免无意义的 buffer 堆积.
                scale = self.scales.get(pname)
                if scale is not None:
                    self._register_scale_buffer(module, pname, prec, scale)
        logger.info("quantize_module applied per-component rules to %s", type(module).__name__)
        return module

    @staticmethod
    def _infer_precision(pname: str) -> Precision:
        lower = pname.lower()
        # KV cache
        if any(tag in lower for tag in ("kv_cache", "k_cache", "v_cache", "kv_proj")):
            return Precision.FP8
        # 大专家 FFN 权重 (排除 router/gate 这些小参数)
        is_expert = any(tag in lower for tag in ("expert", "experts", "moe"))
        is_ffn = any(tag in lower for tag in ("ffn", "mlp", "w1", "w2", "w3", "fc", "linear"))
        is_router = any(tag in lower for tag in ("gate", "router"))
        if is_expert and is_ffn and not is_router:
            return Precision.W8A16
        # 其余
        return Precision.FP16

    @staticmethod
    def _register_scale_buffer(
        module: nn.Module,
        pname: str,
        prec: Precision,
        scale: Optional[torch.Tensor],
    ) -> None:
        """在模块上注册 scale 与精度元信息 buffer.

        - ``<pname>__quant_scale``: 伪量化 scale 张量 (供推理/恢复使用);
        - ``<pname>__quant_meta``:  精度档位整数编码 (标量 int32 张量, 见
          :data:`_PRECISION_CODE`, 可反向查表恢复精度名).
        两者均为非持久化 buffer, 不写入 state_dict, 避免与权重键冲突.
        """
        base = pname.replace(".", "_")
        scale_name = base + "__quant_scale"
        meta_name = base + "__quant_meta"
        existing = dict(module.named_buffers())
        try:
            if scale is not None and scale_name not in existing:
                module.register_buffer(scale_name, scale.detach().clone(), persistent=False)
            if meta_name not in existing:
                code = _PRECISION_CODE.get(prec, -1)
                module.register_buffer(meta_name, torch.tensor(code, dtype=torch.int32), persistent=False)
        except (AttributeError, RuntimeError, KeyError, ValueError):
            # 命名冲突或模块不允许注册额外 buffer 时静默跳过
            pass

    # ==================================================================
    # 内部: 偏差度量
    # ==================================================================
    @staticmethod
    def _deviation(baseline: Any, other: Any) -> float:
        """度量两次 eval_fn 输出之间的偏差."""
        if isinstance(baseline, torch.Tensor) and isinstance(other, torch.Tensor):
            if baseline.shape != other.shape:
                return float("inf")
            denom = baseline.abs().mean().clamp(min=1e-8)
            return float((other - baseline).abs().mean() / denom)
        if isinstance(baseline, dict) and isinstance(other, dict):
            devs = [
                Quantizer._deviation(baseline[k], other[k])
                for k in baseline if k in other
            ]
            return sum(devs) / len(devs) if devs else 0.0
        # 标量 / 其它
        try:
            return abs(float(other) - float(baseline))
        except (TypeError, ValueError):
            return 0.0 if other == baseline else 1.0


# =========================================================================
# 辅助
# =========================================================================
def _has_nvfp4() -> bool:
    """检测当前硬件是否真正支持 NVFP4 (仅 Blackwell 及以上)."""
    try:
        if not torch.cuda.is_available():
            return False
        major, _minor = torch.cuda.get_device_capability()
        # Blackwell sm_100+
        return major >= 10
    except Exception:
        return False
