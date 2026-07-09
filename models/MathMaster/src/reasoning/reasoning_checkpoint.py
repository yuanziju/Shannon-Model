"""推理检查点管理 - 断点续推支持.

与 ``common.checkpoint.CheckpointManager`` (模型权重检查点) 互补:
    - ``CheckpointManager``   保存 *模型权重* (state_dict 分片);
    - ``ReasoningCheckpoint`` 保存 *推理过程状态* (问题/工作内存/步数), JSON 落盘.

每个命名空间 (name) 维护一个独立的检查点序列, 按步数排序, 保留最近
``max_to_keep`` 个. 支持 save / load / latest / list / remove.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 文件名: <name>_step_<step>.json
_FILE_RE = re.compile(r"^(?P<name>.+?)_step_(?P<step>\d+)\.json$")


class ReasoningCheckpoint:
    """推理过程检查点管理器.

    Args:
        save_dir: 检查点目录.
        max_to_keep: 每个命名空间最多保留的检查点数.
    """

    def __init__(self, save_dir: str, max_to_keep: int = 5) -> None:
        self.save_dir = save_dir
        self.max_to_keep = max(1, int(max_to_keep))
        self._lock = threading.Lock()
        os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 保存
    # ------------------------------------------------------------------ #
    def save(self, name: str, step: int, state: Dict[str, Any]) -> str:
        """保存一个推理检查点, 返回文件路径. """
        name = self._sanitize_name(name)
        step = int(step)
        payload = {
            "name": name,
            "step": step,
            "state": state,
            "timestamp": time.time(),
            "format": "reasoning-ckpt-v1",
        }
        path = os.path.join(self.save_dir, f"{name}_step_{step}.json")
        with self._lock:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            self._prune(name)
        logger.info("Reasoning checkpoint saved: %s (step=%d)", path, step)
        return path

    # ------------------------------------------------------------------ #
    # 加载
    # ------------------------------------------------------------------ #
    def load(self, name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """加载检查点状态. ``name=None`` 时加载全局最新. """
        path = self.latest(name)
        if path is None:
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return payload.get("state")
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load checkpoint %s: %s", path, exc)
            return None

    def load_step(self, name: str, step: int) -> Optional[Dict[str, Any]]:
        """加载指定命名空间下指定 step 的检查点. """
        name = self._sanitize_name(name)
        path = os.path.join(self.save_dir, f"{name}_step_{int(step)}.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f).get("state")
        except (OSError, json.JSONDecodeError):
            return None

    # ------------------------------------------------------------------ #
    # 查询
    # ------------------------------------------------------------------ #
    def latest(self, name: Optional[str] = None) -> Optional[str]:
        """返回最新检查点路径. ``name=None`` 时全局最新. """
        files = self._list_files(name)
        if not files:
            return None
        files.sort(key=self._step_of, reverse=True)
        return files[0]

    def list(self, name: Optional[str] = None) -> List[str]:
        """列出检查点路径 (按 step 升序). """
        files = self._list_files(name)
        files.sort(key=self._step_of)
        return files

    def remove(self, name: Optional[str] = None, step: Optional[int] = None) -> int:
        """删除检查点. ``name=None`` 删全部; 指定 ``step`` 仅删该步. 返回删除数. """
        removed = 0
        with self._lock:
            targets = self._list_files(name)
            for path in targets:
                if step is not None and self._step_of(path) != int(step):
                    continue
                try:
                    os.remove(path)
                    removed += 1
                except OSError as exc:
                    logger.warning("Failed to remove %s: %s", path, exc)
        return removed

    # ------------------------------------------------------------------ #
    # 内部
    # ------------------------------------------------------------------ #
    def _list_files(self, name: Optional[str]) -> List[str]:
        if not os.path.isdir(self.save_dir):
            return []
        results: List[str] = []
        for fn in os.listdir(self.save_dir):
            m = _FILE_RE.match(fn)
            if not m:
                continue
            if name is not None and m.group("name") != self._sanitize_name(name):
                continue
            results.append(os.path.join(self.save_dir, fn))
        return results

    def _prune(self, name: str) -> None:
        """保留每个命名空间最近 max_to_keep 个检查点. """
        if self.max_to_keep <= 0:
            return
        files = self._list_files(name)
        files.sort(key=self._step_of, reverse=True)
        for stale in files[self.max_to_keep:]:
            try:
                os.remove(stale)
            except OSError as exc:
                logger.warning("Failed to prune %s: %s", stale, exc)

    @staticmethod
    def _sanitize_name(name: str) -> str:
        # 仅保留字母数字下划线连字符, 其余替换为下划线
        return re.sub(r"[^A-Za-z0-9_\-]", "_", str(name)) or "ckpt"

    @staticmethod
    def _step_of(path: str) -> int:
        m = _FILE_RE.search(os.path.basename(path))
        return int(m.group("step")) if m else -1


__all__ = ["ReasoningCheckpoint"]
