"""检查点管理 (Shannon / MathMaster 共享基础设施).

提供分片保存 / 加载、异步保存 (基于 threading) 以及 "啊哈时刻"
(loss 突变) 检测, 用于在模型能力跃迁点自动多保存检查点.
"""

from __future__ import annotations

import logging
import os
import re
import statistics
import threading
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

_STEP_RE = re.compile(r"ckpt_step_(\d+)\.meta\.pt$")


class CheckpointManager:
    """分片检查点管理器.

    每个检查点由三片文件组成::

        <base>.model.pt  -- 模型 state_dict
        <base>.optim.pt  -- 优化器 state_dict
        <base>.meta.pt   -- step / metrics / 元信息

    ``base`` 形如 ``ckpt_step_<step>``. 保存时可选择同步或异步 (线程) 方式,
    异步保存会在派发线程前先把张量拷贝到 CPU, 避免训练线程继续修改.
    """

    def __init__(self, save_dir: str, max_to_keep: int = 5) -> None:
        self.save_dir = save_dir
        self.max_to_keep = max_to_keep
        self._lock = threading.Lock()
        self._async_thread: Optional[threading.Thread] = None
        os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 保存
    # ------------------------------------------------------------------
    def save(
        self,
        model,
        optimizer=None,
        step: int = 0,
        metrics: Optional[Dict[str, Any]] = None,
        path: Optional[str] = None,
        async_save: bool = False,
    ) -> str:
        """保存检查点.

        ``async_save=True`` 时调用 :meth:`save_async`, 立即返回目标路径
        而不阻塞训练; 否则同步落盘.
        """
        if async_save:
            return self.save_async(model, optimizer, step, metrics, path)
        return self._save_sync(model, optimizer, step, metrics, path)

    def save_async(
        self,
        model,
        optimizer=None,
        step: int = 0,
        metrics: Optional[Dict[str, Any]] = None,
        path: Optional[str] = None,
    ) -> str:
        """异步 (线程) 保存. 先快照 CPU 副本再后台落盘."""
        model_snapshot = self._snapshot_model(model)
        optim_snapshot = self._snapshot_optimizer(optimizer)
        target_path = path or os.path.join(self.save_dir, f"ckpt_step_{step}.pt")

        def _worker() -> None:
            try:
                self._save_from_snapshot(
                    model_snapshot, optim_snapshot, step, metrics, target_path
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Async checkpoint save failed at step %d: %s", step, exc)

        self._async_thread = threading.Thread(target=_worker, daemon=True, name="ckpt-save")
        self._async_thread.start()
        logger.info("Async checkpoint scheduled at step %d -> %s", step, target_path)
        return target_path

    def wait_async(self) -> None:
        """等待最近一次异步保存完成."""
        if self._async_thread is not None and self._async_thread.is_alive():
            self._async_thread.join()

    # -- 内部: 同步保存 -------------------------------------------------
    def _save_sync(self, model, optimizer, step, metrics, path) -> str:
        if path is None:
            path = os.path.join(self.save_dir, f"ckpt_step_{step}.pt")
        model_state = self._extract_state_dict(model)
        optim_state = self._extract_state_dict(optimizer)
        self._write_shards(path, model_state, optim_state, step, metrics)
        self._prune()
        logger.info("Checkpoint saved (sync) at step %d -> %s", step, path)
        return path

    def _save_from_snapshot(self, model_state, optim_state, step, metrics, path) -> None:
        with self._lock:
            self._write_shards(path, model_state, optim_state, step, metrics)
            self._prune()
            logger.info("Checkpoint saved (async) at step %d -> %s", step, path)

    def _write_shards(self, path, model_state, optim_state, step, metrics) -> None:
        torch.save({"model": model_state}, path.replace(".pt", ".model.pt"))
        torch.save({"optimizer": optim_state}, path.replace(".pt", ".optim.pt"))
        torch.save(
            {"step": step, "metrics": metrics, "format": "sharded-v1"},
            path.replace(".pt", ".meta.pt"),
        )

    # -- 内部: 快照 -----------------------------------------------------
    @staticmethod
    def _extract_state_dict(obj) -> Any:
        if obj is None:
            return None
        if hasattr(obj, "state_dict"):
            return obj.state_dict()
        return obj

    @staticmethod
    def _snapshot_model(model) -> Any:
        if model is None:
            return None
        if hasattr(model, "state_dict"):
            return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if isinstance(model, dict):
            return {k: (v.detach().cpu().clone() if torch.is_tensor(v) else v) for k, v in model.items()}
        return model

    @staticmethod
    def _snapshot_optimizer(optimizer) -> Any:
        if optimizer is None:
            return None
        if hasattr(optimizer, "state_dict"):
            return optimizer.state_dict()
        return optimizer

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------
    def load(
        self,
        model,
        optimizer=None,
        path: Optional[str] = None,
        map_location: Any = "cpu",
        strict: bool = False,
    ) -> Tuple[int, Optional[Dict[str, Any]]]:
        """加载检查点, 返回 ``(step, metrics)``.

        ``path`` 为 ``<base>.pt`` 或 ``<base>`` 形式均可. 若 ``path`` 为
        None 则自动选取最新的检查点.
        """
        if path is None:
            path = self.latest_checkpoint()
        if path is None:
            logger.warning("No checkpoint found in %s", self.save_dir)
            return 0, None

        base = path[:-3] if path.endswith(".pt") else path
        model_path = base + ".model.pt"
        optim_path = base + ".optim.pt"
        meta_path = base + ".meta.pt"

        step, metrics = 0, None
        if os.path.exists(model_path):
            msd = torch.load(model_path, map_location=map_location, weights_only=False)
            state = msd.get("model", msd)
            if model is not None and hasattr(model, "load_state_dict"):
                model.load_state_dict(state, strict=strict)
            elif isinstance(model, dict):
                model.update(state)
        if optimizer is not None and os.path.exists(optim_path):
            osd = torch.load(optim_path, map_location=map_location, weights_only=False)
            state = osd.get("optimizer", None)
            if state is not None and hasattr(optimizer, "load_state_dict"):
                optimizer.load_state_dict(state)
        if os.path.exists(meta_path):
            meta = torch.load(meta_path, map_location=map_location, weights_only=False)
            step = int(meta.get("step", 0))
            metrics = meta.get("metrics")
        logger.info("Checkpoint loaded from %s (step=%d)", path, step)
        return step, metrics

    # ------------------------------------------------------------------
    # 维护
    # ------------------------------------------------------------------
    def latest_checkpoint(self) -> Optional[str]:
        if not os.path.isdir(self.save_dir):
            return None
        metas = [f for f in os.listdir(self.save_dir) if f.endswith(".meta.pt")]
        if not metas:
            return None
        metas.sort(key=self._step_from_name, reverse=True)
        return os.path.join(self.save_dir, metas[0].replace(".meta.pt", ".pt"))

    def list_checkpoints(self) -> List[str]:
        if not os.path.isdir(self.save_dir):
            return []
        metas = [f for f in os.listdir(self.save_dir) if f.endswith(".meta.pt")]
        metas.sort(key=self._step_from_name)
        return [os.path.join(self.save_dir, m.replace(".meta.pt", ".pt")) for m in metas]

    def _prune(self) -> None:
        if self.max_to_keep <= 0:
            return
        metas = [f for f in os.listdir(self.save_dir) if f.endswith(".meta.pt")]
        metas.sort(key=self._step_from_name, reverse=True)
        for stale in metas[self.max_to_keep:]:
            base = stale.replace(".meta.pt", "")
            for ext in (".model.pt", ".optim.pt", ".meta.pt"):
                p = os.path.join(self.save_dir, base + ext)
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError as exc:
                        logger.warning("Failed to remove stale checkpoint %s: %s", p, exc)

    @staticmethod
    def _step_from_name(name: str) -> int:
        m = _STEP_RE.search(name)
        return int(m.group(1)) if m else -1

    # ------------------------------------------------------------------
    # 啊哈时刻检测
    # ------------------------------------------------------------------
    def detect_aha_moment(
        self,
        loss_history: List[float],
        window: int = 50,
        drop_threshold: float = 0.15,
        curvature_threshold: float = 0.0,
    ) -> bool:
        """检测 loss 突变 (能力跃迁) 点.

        判据 (满足其一即触发):
            1. 近 ``window`` 步平均 loss 相对前 ``window`` 步下降超过
               ``drop_threshold`` (默认 15%);
            2. loss 二阶差分符号反转 (曲率反转, ``curvature_threshold`` 用于
               过滤微小波动).

        历史不足 ``2 * window`` 时返回 False.
        """
        if len(loss_history) < 2 * window:
            return False
        recent = loss_history[-window:]
        prev = loss_history[-2 * window:-window]
        p_mean = statistics.mean(prev)
        r_mean = statistics.mean(recent)

        triggered = False
        if p_mean > 0 and (p_mean - r_mean) / p_mean > drop_threshold:
            triggered = True

        # 二阶差分符号反转
        if len(loss_history) >= 5:
            vals = loss_history[-5:]
            d1 = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
            d2 = [d1[i + 1] - d1[i] for i in range(len(d1) - 1)]
            for i in range(1, len(d2)):
                if abs(d2[i]) > curvature_threshold and d2[i] * d2[i - 1] < 0:
                    triggered = True
                    break

        if triggered:
            logger.info(
                "Aha moment detected: prev_loss=%.5f recent_loss=%.5f (drop %.1f%%)",
                p_mean, r_mean, 100 * (p_mean - r_mean) / max(p_mean, 1e-12),
            )
        return triggered
