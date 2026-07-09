"""流式解码 - SSE + 拟人修订.

StreamDecoder 实现流式输出前端:
    - SSE (Server-Sent Events) 协议推送 token-by-token.
    - 拟人流式输出: 删除重打 + 修订 (spec 决策 L11, 修订率上限 15%).
    - 延迟模拟 (50-200ms) 提升自然感.
    - 三级置信度门控 (token/块/全局).
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple


class StreamEvent(Enum):
    """SSE 事件类型. """

    TOKEN = "token"          # 正常 token
    DELETE = "delete"        # 删除重打 (拟人修订)
    REPLACE = "replace"      # 替换
    DONE = "done"            # 流结束
    ERROR = "error"


# spec 决策 L11: 拟人修订率上限 15%, 延迟 50-200ms
MAX_REVISION_RATE = 0.15
MIN_DELAY_MS = 50
MAX_DELAY_MS = 200
DEFAULT_CHUNK_SIZE = 4  # 每次拟人 "犹豫" 的 token 粒度


@dataclass
class TokenChunk:
    """单个流式输出块. """

    event: StreamEvent
    content: str = ""
    token_ids: List[int] = field(default_factory=list)
    confidence: float = 1.0
    delay_ms: int = 0
    index: int = 0

    def to_sse(self) -> str:
        """序列化为 SSE 报文. """
        payload = {
            "event": self.event.value,
            "content": self.content,
            "confidence": round(self.confidence, 3),
            "index": self.index,
        }
        if self.token_ids:
            payload["token_ids"] = self.token_ids
        lines = [f"event: {self.event.value}", f"data: {self._encode(payload)}"]
        return "\n".join(lines) + "\n\n"

    @staticmethod
    def _encode(d: Dict[str, Any]) -> str:
        # 简易 JSON (避免依赖), 仅处理 str/int/float/list
        parts = []
        for k, v in d.items():
            if isinstance(v, str):
                parts.append(f'"{k}":"{v}"')
            elif isinstance(v, bool):
                parts.append(f'"{k}":{str(v).lower()}')
            elif isinstance(v, (int, float)):
                parts.append(f'"{k}":{v}')
            elif isinstance(v, list):
                parts.append(f'"{k}":[{",".join(str(x) for x in v)}]')
        return "{" + ",".join(parts) + "}"


class StreamDecoder:
    """流式解码前端.

    Args:
        confidence_threshold: 三级置信度门控阈值 (低于则触发修订).
            (token, block, global) 三级.
        revision_rate: 拟人修订率上限 (默认 15%).
        delay_range_ms: 拟人延迟区间 (ms).
        rng_seed: 随机种子.
        chunk_size: 拟人 "打字" 粒度 (token 数).
    """

    def __init__(
        self,
        confidence_threshold: Tuple[float, float, float] = (0.6, 0.7, 0.8),
        revision_rate: float = MAX_REVISION_RATE,
        delay_range_ms: Tuple[int, int] = (MIN_DELAY_MS, MAX_DELAY_MS),
        rng_seed: Optional[int] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        self.tok_thr, self.blk_thr, self.glob_thr = confidence_threshold
        self.revision_rate = max(0.0, min(1.0, float(revision_rate)))
        self.delay_range = delay_range_ms
        self._rng = random.Random(rng_seed)
        self.chunk_size = max(1, int(chunk_size))
        self._revision_count = 0
        self._total_emitted = 0
        self._buffer: List[TokenChunk] = []

    # ------------------------------------------------------------------ #
    # 流式生成
    # ------------------------------------------------------------------ #
    def stream(
        self,
        tokens: Sequence[Tuple[str, float]],
        final_text: Optional[str] = None,
    ) -> Iterator[TokenChunk]:
        """将 ``[(token, confidence), ...]`` 转为流式 SSE 事件迭代器.

        拟人修订逻辑:
            1. 按 chunk_size 分块, 计算块平均置信度.
            2. 若块置信度 < blk_thr 且修订率未超上限 -> 触发删除重打.
            3. 全局置信度低于 glob_thr -> 流末尾追加整体替换建议.
        """
        self._buffer.clear()
        idx = 0
        # 全局置信度
        global_conf = sum(c for _, c in tokens) / max(1, len(tokens))
        i = 0
        emitted_tokens: List[str] = []
        while i < len(tokens):
            chunk = tokens[i: i + self.chunk_size]
            i += self.chunk_size
            blk_conf = sum(c for _, c in chunk) / max(1, len(chunk))
            content = "".join(t for t, _ in chunk)
            ids = list(range(i - len(chunk), i))

            # 三级门控: token 级
            low_tok = any(c < self.tok_thr for _, c in chunk)
            # 块级门控 + 修订率检查
            should_revise = (
                (blk_conf < self.blk_thr or low_tok)
                and self._revision_ratio() < self.revision_rate
            )
            if should_revise:
                # 先发出删除事件 (拟人 "反悔")
                del_chunk = TokenChunk(
                    event=StreamEvent.DELETE,
                    content=content,
                    token_ids=ids,
                    confidence=blk_conf,
                    delay_ms=self._delay(),
                    index=idx,
                )
                idx += 1
                yield del_chunk
                self._buffer.append(del_chunk)
                self._revision_count += len(chunk)
                # 替换为修正版本 (此处用 final_text 片段或原样重打)
                revised = self._revise(chunk, final_text)
                rep_chunk = TokenChunk(
                    event=StreamEvent.REPLACE,
                    content=revised,
                    token_ids=ids,
                    confidence=min(1.0, blk_conf + 0.2),
                    delay_ms=self._delay(),
                    index=idx,
                )
                idx += 1
                yield rep_chunk
                self._buffer.append(rep_chunk)
                emitted_tokens.append(revised)
                self._total_emitted += len(chunk)
                continue

            # 正常 token 输出
            tok_chunk = TokenChunk(
                event=StreamEvent.TOKEN,
                content=content,
                token_ids=ids,
                confidence=blk_conf,
                delay_ms=self._delay(),
                index=idx,
            )
            idx += 1
            yield tok_chunk
            self._buffer.append(tok_chunk)
            emitted_tokens.append(content)
            self._total_emitted += len(chunk)

        # 全局门控: 整体置信度不足 -> 追加全局替换建议
        if global_conf < self.glob_thr and final_text:
            idx += 1
            glob_chunk = TokenChunk(
                event=StreamEvent.REPLACE,
                content=final_text,
                confidence=global_conf,
                delay_ms=self._delay(),
                index=idx,
            )
            yield glob_chunk
            self._buffer.append(glob_chunk)

        # 结束事件
        done = TokenChunk(event=StreamEvent.DONE, content="", confidence=1.0, index=idx + 1)
        yield done
        self._buffer.append(done)

    def stream_sse(self, tokens: Sequence[Tuple[str, float]], final_text: Optional[str] = None) -> Iterator[str]:
        """便捷: 直接产出 SSE 报文字符串. """
        for chunk in self.stream(tokens, final_text=final_text):
            yield chunk.to_sse()

    # ------------------------------------------------------------------ #
    # 内部
    # ------------------------------------------------------------------ #
    def _delay(self) -> int:
        lo, hi = self.delay_range
        return self._rng.randint(lo, hi)

    def _revision_ratio(self) -> float:
        if self._total_emitted == 0:
            return 0.0
        return self._revision_count / self._total_emitted

    def _revise(self, chunk: Sequence[Tuple[str, float]], final_text: Optional[str]) -> str:
        """生成修订版本. 优先用 final_text 对齐, 否则做简单修正. """
        if final_text:
            # 取 final_text 中与 chunk 等长的片段
            start = self._total_emitted
            return final_text[start: start + len(chunk)]
        # 简单修正: 去除重复 token / 修剪空白
        out = []
        seen = set()
        for tok, _ in chunk:
            if tok.strip() and tok not in seen:
                out.append(tok)
                seen.add(tok)
        return "".join(out) or "".join(t for t, _ in chunk)

    # ------------------------------------------------------------------ #
    # 统计
    # ------------------------------------------------------------------ #
    @property
    def revision_rate_actual(self) -> float:
        return self._revision_ratio()

    @property
    def buffer(self) -> List[TokenChunk]:
        return list(self._buffer)


__all__ = [
    "StreamDecoder",
    "TokenChunk",
    "StreamEvent",
    "MAX_REVISION_RATE",
    "MIN_DELAY_MS",
    "MAX_DELAY_MS",
]
