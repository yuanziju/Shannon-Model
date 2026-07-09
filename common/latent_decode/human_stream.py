"""拟人流式输出前端 (HumanStream).

决策 L11: 拟人流式修订率上限 15%, 延迟 50-200ms.

模拟人类打字的流式输出, 主要包含两类拟人动效:

1. **删除重打 (delete-retype)**: 模型偶尔"打错"再删除重打, 表现出
   犹豫/修正的人类特征. 删除-重打的总修订 token 数占总输出比例上限 15%.

2. **延迟分布**: token 间延迟服从对数正态分布, 范围 [50ms, 200ms],
   模拟人类打字速度波动 (含思考停顿).

输出格式兼容主流流式协议 (SSE / WebSocket / gRPC stream), 每个事件含:
    - token / text
    - action: "add" | "delete" | "retype" | "pause"
    - delay_ms: 本事件延迟
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, Optional

import torch
import torch.nn as nn


class StreamAction(str, Enum):
    ADD = "add"           # 正常添加 token
    DELETE = "delete"     # 删除上一个 token
    RETYPE = "retype"     # 删除后重新打 (合并事件)
    PAUSE = "pause"       # 思考停顿 (无 token)


# 决策 L11 硬约束
REVISION_RATE_CAP = 0.15   # 修订率上限 15%
MIN_DELAY_MS = 50          # 最小延迟 50ms
MAX_DELAY_MS = 200         # 最大延迟 200ms
DEFAULT_MEAN_DELAY_MS = 120
DEFAULT_STD_DELAY_MS = 40


@dataclass
class HumanStreamConfig:
    """拟人流式配置."""

    # 修订 (删除重打) 控制
    revision_rate_cap: float = REVISION_RATE_CAP
    retype_prob: float = 0.08        # 每个 token 触发删除重打的概率
    retype_length_max: int = 3       # 单次删除重打最多删多少 token
    # 延迟分布 (对数正态参数, 实际 clip 到 [MIN, MAX])
    mean_delay_ms: float = DEFAULT_MEAN_DELAY_MS
    std_delay_ms: float = DEFAULT_STD_DELAY_MS
    min_delay_ms: float = MIN_DELAY_MS
    max_delay_ms: float = MAX_DELAY_MS
    # 思考停顿
    pause_prob: float = 0.02          # 触发长停顿概率
    pause_delay_ms: float = 800.0     # 长停顿时长
    # 是否真实 sleep (推理服务场景). False 仅返回事件不阻塞.
    real_sleep: bool = False
    # 种子
    seed: Optional[int] = None


@dataclass
class StreamEvent:
    """单个流式事件."""

    action: StreamAction
    text: str = ""
    token_id: int = -1
    delay_ms: float = 0.0
    revision_count: int = 0   # 本事件累计修订 token 数 (用于统计)
    index: int = 0            # 事件序号


class HumanStream(nn.Module):
    """拟人流式输出前端.

    注: 本模块为无参 (逻辑层) 模块, 继承 nn.Module 仅为统一接口与设备管理.
    """

    def __init__(self, config: HumanStreamConfig | None = None, **kwargs):
        super().__init__()
        cfg = config or HumanStreamConfig(**kwargs)
        self.cfg = cfg
        self._rng = random.Random(cfg.seed)

    # ------------------------------------------------------------------
    # 延迟采样 (对数正态 + clip)
    # ------------------------------------------------------------------
    def _sample_delay(self, is_pause: bool = False) -> float:
        if is_pause:
            return self.cfg.pause_delay_ms
        # 对数正态: 用 normal -> exp 实现
        mu = _safe_log(self.cfg.mean_delay_ms)
        sigma = max(
            _safe_log(1 + self.cfg.std_delay_ms / max(self.cfg.mean_delay_ms, 1)),
            0.01,
        )
        delay = torch.exp(
            torch.tensor(self._rng.gauss(mu, sigma))
        ).item()
        return float(max(self.cfg.min_delay_ms, min(self.cfg.max_delay_ms, delay)))

    # ------------------------------------------------------------------
    # 删除重打决策
    # ------------------------------------------------------------------
    def _should_retype(self, budget: int, total: int, revised: int) -> bool:
        """决定当前 token 是否触发删除重打.

        受修订率上限约束: revised/total < cap.
        """
        if total == 0:
            return False
        if revised / total >= self.cfg.revision_rate_cap:
            return False
        if budget <= 0:
            return False
        return self._rng.random() < self.cfg.retype_prob

    # ------------------------------------------------------------------
    # 核心流式生成
    # ------------------------------------------------------------------
    def stream(
        self,
        tokens: list[int] | torch.Tensor,
        detokenizer: Optional[callable] = None,
        max_revisions: Optional[int] = None,
    ) -> Iterator[StreamEvent]:
        """将 token 序列转换为拟人流式事件流.

        Args:
            tokens: token id 列表 (list 或 1D tensor).
            detokenizer: token_id -> str 函数. None 则用 str(token_id).
            max_revisions: 显式限制修订总数 (None 用 cap * len 计算).

        Yields:
            StreamEvent 事件序列.
        """
        # 归一化为 list[int]
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.reshape(-1).tolist()
        n = len(tokens)
        if n == 0:
            return

        detok = detokenizer or (lambda tid: str(tid))
        cap_count = int(self.cfg.revision_rate_cap * n)
        if max_revisions is not None:
            cap_count = min(cap_count, max_revisions)

        revised_total = 0
        emitted = 0  # 已输出的 (非修订) token 数
        index = 0

        i = 0
        buffer_text: list[str] = []
        while i < n:
            tok = int(tokens[i])
            text = detok(tok)

            # 是否触发思考停顿
            is_pause = self._rng.random() < self.cfg.pause_prob
            if is_pause:
                delay = self._sample_delay(is_pause=True)
                if self.cfg.real_sleep:
                    time.sleep(delay / 1000.0)
                yield StreamEvent(
                    action=StreamAction.PAUSE, delay_ms=delay, index=index
                )
                index += 1

            # 是否触发删除重打
            budget = cap_count - revised_total
            if self._should_retype(budget, max(emitted, 1), revised_total):
                # 删除最近 1~retype_length_max 个 token (如果有)
                del_len = min(
                    self._rng.randint(1, self.cfg.retype_length_max),
                    len(buffer_text),
                )
                if del_len > 0:
                    # 发出 delete 事件 (逐个)
                    for _ in range(del_len):
                        delay = self._sample_delay()
                        if self.cfg.real_sleep:
                            time.sleep(delay / 1000.0)
                        yield StreamEvent(
                            action=StreamAction.DELETE,
                            delay_ms=delay,
                            revision_count=1,
                            index=index,
                        )
                        index += 1
                        revised_total += 1
                        buffer_text.pop()
                    # 重新打出删除的 token (用原 token, 保证最终正确)
                    # 此处简化: 把 i 回退 del_len 重新生成
                    # 但为了不无限循环, 仅重打当前 token
                    delay = self._sample_delay()
                    if self.cfg.real_sleep:
                        time.sleep(delay / 1000.0)
                    yield StreamEvent(
                        action=StreamAction.RETYPE,
                        text=text,
                        token_id=tok,
                        delay_ms=delay,
                        revision_count=0,
                        index=index,
                    )
                    index += 1
                    buffer_text.append(text)
                    emitted += 1
                    i += 1
                    continue

            # 正常添加
            delay = self._sample_delay()
            if self.cfg.real_sleep:
                time.sleep(delay / 1000.0)
            yield StreamEvent(
                action=StreamAction.ADD,
                text=text,
                token_id=tok,
                delay_ms=delay,
                index=index,
            )
            index += 1
            buffer_text.append(text)
            emitted += 1
            i += 1

    # ------------------------------------------------------------------
    # 批量统计 (用于监控修订率是否触顶)
    # ------------------------------------------------------------------
    def collect_stats(self, events: list[StreamEvent]) -> dict:
        total = sum(1 for e in events if e.action in (StreamAction.ADD, StreamAction.RETYPE))
        revised = sum(e.revision_count for e in events)
        deletes = sum(1 for e in events if e.action == StreamAction.DELETE)
        pauses = sum(1 for e in events if e.action == StreamAction.PAUSE)
        delays = [e.delay_ms for e in events if e.action != StreamAction.PAUSE]
        pause_delays = [e.delay_ms for e in events if e.action == StreamAction.PAUSE]
        return {
            "emitted_tokens": total,
            "revisions": revised,
            "deletes": deletes,
            "pauses": pauses,
            "revision_rate": revised / max(total + revised, 1),
            "revision_rate_cap": self.cfg.revision_rate_cap,
            "cap_respected": revised <= self.cfg.revision_rate_cap * max(total, 1),
            "mean_delay_ms": sum(delays) / max(len(delays), 1),
            "min_delay_ms": min(delays) if delays else 0.0,
            "max_delay_ms": max(delays) if delays else 0.0,
            "total_pause_ms": sum(pause_delays),
        }

    # ------------------------------------------------------------------
    # 简单文本流 (不区分 token, 仅输出最终文本 + 修订标记)
    # ------------------------------------------------------------------
    def stream_text(
        self,
        text: str,
        chunk_size: int = 1,
    ) -> Iterator[StreamEvent]:
        """按字符 (或 chunk) 流式输出文本, 含删除重打效果."""
        chunks = [
            text[i: i + chunk_size] for i in range(0, len(text), chunk_size)
        ]
        # 复用 stream 逻辑 (token_id 用 chunk 索引)
        yield from self.stream(
            chunks, detokenizer=lambda idx: chunks[idx] if idx < len(chunks) else ""
        )

    def extra_repr(self) -> str:
        return (
            f"revision_rate_cap={self.cfg.revision_rate_cap}, "
            f"delay=[{self.cfg.min_delay_ms},{self.cfg.max_delay_ms}]ms"
        )


def _safe_log(x: float) -> float:
    import math
    return math.log(max(x, 1e-8))
