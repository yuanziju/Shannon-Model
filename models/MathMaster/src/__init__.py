"""MathMaster 源码包.

公共 API:
    MathConfig - 模型配置数据类 (src.config)
    MathModel  - 完整模型 (src.model) — 可选, 模型实现损坏时降级为 None

子模块 (proof / reasoning / training 等) 可独立导入, 不依赖 MathModel 是否可用.
"""

from .config import MathConfig

# MathModel 依赖完整的模型实现; 若实现源码尚未就绪 (语法/依赖缺失), 降级为 None
# 而不阻断 proof / reasoning / training 等子模块的导入.
try:
    from .model import MathModel  # noqa: F401
except Exception:  # pragma: no cover - 模型实现不可用时降级
    MathModel = None  # type: ignore[assignment]

__all__ = ["MathConfig", "MathModel"]
