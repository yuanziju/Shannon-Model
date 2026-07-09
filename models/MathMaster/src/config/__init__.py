"""MathMaster 配置模块.

导出 :class:`MathConfig` 数据类, 定义 MathMaster 30-70B MoE 数学专精模型的全部
超参数 (模型主体 / 循环深度 / AB堆叠 / 专家池 / 残差池 / 直觉层 / 注意力 / NSL /
CTM / 位置编码 / 形式化 / 训练 / 评估 / 领域权重).
"""

from .config import MathConfig

__all__ = ["MathConfig"]
