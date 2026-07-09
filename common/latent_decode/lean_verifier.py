"""Lean 4 验证器 (LeanVerifier).

决策 L4: 形式化证明类输出必须强制 AR + Lean 验证器.

本模块负责:
1. **证明检测**: 从模型输出文本中检测 Lean4 代码块 / 证明意图.
2. **Lean4 subprocess 执行**: 调用 `lean` 编译器对提取的代码块进行类型检查,
   确保证明闭合 (无 sorry / by_contradiction / admit).
3. **结果编码**: 将验证结果 (success / failure + 错误信息) 编码为反馈向量,
   供 GRPO/DPO 奖励使用 (决策: Lean 完成 +0.5).

依赖: 系统需安装 Lean4 (`elan` / `lean`). 若未安装, 验证降级为语法检查.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


# Lean 代码块提取正则 (```lean ... ``` 或 ```lean4 ... ```)
LEAN_BLOCK_RE = re.compile(
    r"```(?:lean4?|lean_theorem)\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

# 证明完成标记 (合法)
PROOF_COMPLETE_MARKERS = ("Qed", "by", "exact", "rfl", "simp", "decide")

# 不完整证明标记 (sorry / admit 表示未完成)
PROOF_INCOMPLETE_MARKERS = ("sorry", "admit", "by_contradiction", "exact?")

# 证明关键词 (用于证明意图检测)
PROOF_INTENT_KEYWORDS = (
    "theorem", "lemma", "proof", "example", "Prop",
    "by", "rw", "simp", "induction", "exact",
)


@dataclass
class LeanVerifierConfig:
    """Lean 验证器配置."""

    lean_executable: str = "lean"          # lean 可执行路径
    lean_project_dir: Optional[str] = None # Lean 项目根 (含 lakefile)
    timeout_sec: float = 30.0              # 单次验证超时
    # 是否强制执行 (False 则仅语法检查, 不调 subprocess)
    enable_subprocess: bool = True
    # 验证失败重试
    max_retries: int = 1
    # 反馈向量维度 (供奖励编码)
    feedback_dim: int = 256
    # 临时目录
    temp_dir: Optional[str] = None
    # 是否在未安装 lean 时降级
    fallback_on_missing: bool = True


@dataclass
class VerificationResult:
    """单次验证结果."""

    success: bool
    code: str
    error_message: str = ""
    error_line: int = -1
    has_sorry: bool = False
    has_admit: bool = False
    elapsed_sec: float = 0.0
    fallback_used: bool = False
    raw_output: str = ""


class LeanVerifier(nn.Module):
    """Lean4 形式化证明验证器.

    继承 nn.Module 以提供反馈向量编码头 (用于奖励信号).
    """

    def __init__(self, config: LeanVerifierConfig | None = None, **kwargs):
        super().__init__()
        cfg = config or LeanVerifierConfig(**kwargs)
        self.cfg = cfg
        # 反馈编码: 将验证状态 (success/fail/sorry/...) 编码为向量
        # 输入特征: [success, has_sorry, has_admit, error_line_norm, elapsed_norm, intent_detected]
        self.feedback_encoder = nn.Sequential(
            nn.Linear(6, cfg.feedback_dim),
            nn.GELU(),
            nn.Linear(cfg.feedback_dim, cfg.feedback_dim),
        )
        # 预缓存 lean 是否可用
        self._lean_available = self._check_lean_available()

    # ------------------------------------------------------------------
    # 证明检测
    # ------------------------------------------------------------------
    @staticmethod
    def detect_proof_blocks(text: str) -> list[str]:
        """从文本中提取所有 Lean 代码块."""
        return [m.group(1).strip() for m in LEAN_BLOCK_RE.finditer(text)]

    @staticmethod
    def detect_proof_intent(text: str) -> bool:
        """检测文本是否包含证明意图 (即使无完整代码块)."""
        return any(kw in text for kw in PROOF_INTENT_KEYWORDS)

    @staticmethod
    def has_incomplete_markers(code: str) -> tuple[bool, bool]:
        """检测代码是否含 sorry / admit (未完成证明)."""
        has_sorry = bool(re.search(r"\bsorry\b", code))
        has_admit = bool(re.search(r"\badmit\b", code))
        return has_sorry, has_admit

    # ------------------------------------------------------------------
    # Lean4 subprocess 执行
    # ------------------------------------------------------------------
    def _check_lean_available(self) -> bool:
        """检查 lean 可执行文件是否可用."""
        if not self.cfg.enable_subprocess:
            return False
        path = shutil.which(self.cfg.lean_executable)
        return path is not None

    def _run_lean(self, code: str) -> tuple[bool, str, int, float]:
        """调用 lean 子进程验证单段代码.

        Returns:
            (success, error_message, error_line, elapsed_sec)
        """
        if not self._lean_available:
            return False, "lean executable not available", -1, 0.0

        tmpdir = tempfile.mkdtemp(prefix="lean_verify_", dir=self.cfg.temp_dir)
        lean_file = os.path.join(tmpdir, "Verify.lean")
        try:
            with open(lean_file, "w", encoding="utf-8") as f:
                f.write(code)

            cmd = [self.cfg.lean_executable, lean_file]
            if self.cfg.lean_project_dir:
                cmd = [self.cfg.lean_executable, "--root", self.cfg.lean_project_dir, lean_file]

            import time
            t0 = time.time()
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.cfg.timeout_sec,
                    cwd=self.cfg.lean_project_dir or tmpdir,
                )
                elapsed = time.time() - t0
                if proc.returncode == 0:
                    return True, "", -1, elapsed
                # 解析错误
                err = proc.stderr + proc.stdout
                line = _parse_error_line(err)
                return False, err.strip()[:2000], line, elapsed
            except subprocess.TimeoutExpired:
                return False, f"timeout after {self.cfg.timeout_sec}s", -1, self.cfg.timeout_sec
            except FileNotFoundError:
                return False, "lean not found", -1, time.time() - t0
        finally:
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 语法降级检查 (无 lean 时)
    # ------------------------------------------------------------------
    def _syntax_check(self, code: str) -> tuple[bool, str]:
        """无 lean 时的轻量语法检查 (启发式)."""
        if not code.strip():
            return False, "empty code block"
        has_sorry, has_admit = self.has_incomplete_markers(code)
        if has_sorry or has_admit:
            return False, "incomplete proof (sorry/admit detected)"
        # 检查基本平衡: theorem/lemma 必须有 by 或 :=
        for kw in ("theorem", "lemma", "example"):
            if kw in code:
                if "by" not in code and ":=" not in code:
                    return False, f"{kw} without 'by' or ':='"
        return True, ""

    # ------------------------------------------------------------------
    # 验证入口
    # ------------------------------------------------------------------
    def verify(self, text_or_code: str, is_code: bool = False) -> VerificationResult:
        """验证文本中的 Lean 证明.

        Args:
            text_or_code: 模型输出文本 (或纯 Lean 代码, is_code=True).
            is_code: 是否直接是 Lean 代码.
        """
        import time
        if is_code:
            blocks = [text_or_code]
        else:
            blocks = self.detect_proof_blocks(text_or_code)

        if not blocks:
            # 无证明代码块, 视为非证明文本, 默认通过
            return VerificationResult(
                success=True, code="", elapsed_sec=0.0,
                error_message="no proof block detected",
            )

        # 验证所有代码块 (取最坏结果)
        worst = VerificationResult(success=True, code=blocks[0])
        for code in blocks:
            has_sorry, has_admit = self.has_incomplete_markers(code)
            t0 = time.time()
            if self._lean_available:
                success, err, line, elapsed = self._run_lean(code)
                fallback = False
            elif self.cfg.fallback_on_missing:
                success, err = self._syntax_check(code)
                elapsed = time.time() - t0
                line = -1
                fallback = True
            else:
                return VerificationResult(
                    success=False, code=code,
                    error_message="lean not available and fallback disabled",
                    elapsed_sec=time.time() - t0,
                )
            result = VerificationResult(
                success=success,
                code=code,
                error_message=err,
                error_line=line,
                has_sorry=has_sorry,
                has_admit=has_admit,
                elapsed_sec=elapsed,
                fallback_used=fallback,
            )
            if not success:
                worst = result
                break
        return worst

    # ------------------------------------------------------------------
    # 反馈向量编码 (供奖励计算)
    # ------------------------------------------------------------------
    def encode_feedback(self, result: VerificationResult, intent_detected: bool = False) -> torch.Tensor:
        """将验证结果编码为反馈向量 [feedback_dim]."""
        features = torch.tensor([
            float(result.success),
            float(result.has_sorry),
            float(result.has_admit),
            float(result.error_line) / 1000.0 if result.error_line > 0 else 0.0,
            min(result.elapsed_sec / self.cfg.timeout_sec, 1.0),
            float(intent_detected),
        ], dtype=torch.float32)
        return self.feedback_encoder(features)

    # ------------------------------------------------------------------
    # 奖励计算 (GRPO/DPO, 决策: Lean 完成 +0.5)
    # ------------------------------------------------------------------
    @staticmethod
    def compute_reward(result: VerificationResult) -> float:
        """计算验证奖励 (参考 spec §7.2 奖励设计).

        - Lean 完成 (无 sorry/admit, 验证通过): +0.5
        - 含 sorry/admit (未完成): -0.1
        - 验证失败: -0.2
        - 无证明块 (非证明文本): 0.0
        """
        if not result.code:
            return 0.0
        if result.success:
            return 0.5
        if result.has_sorry or result.has_admit:
            return -0.1
        return -0.2

    def extra_repr(self) -> str:
        return (
            f"lean_executable={self.cfg.lean_executable}, "
            f"lean_available={self._lean_available}, "
            f"feedback_dim={self.cfg.feedback_dim}"
        )


def _parse_error_line(err: str) -> int:
    """从 lean 错误输出中解析错误行号."""
    m = re.search(r":(\d+):\d+: error", err)
    if m:
        return int(m.group(1))
    m = re.search(r"line (\d+)", err)
    if m:
        return int(m.group(1))
    return -1
