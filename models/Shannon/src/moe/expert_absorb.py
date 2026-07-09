"""专家能力吸收器 (ExpertAbsorber).

实现无需重训的新能力注入:
  1. 从外部数据/教师模型蒸馏知识到空专家
  2. 验证吸收质量 (防能力污染)
  3. 增量能力注入 (持续学习 Phase5)

参考: AGENTS.md Agent 9, spec 持续学习 + 空专家持续填充.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertAbsorber(nn.Module):
    """专家能力吸收器.

    通过知识蒸馏将新能力注入空专家, 支持:
      - 教师-学生蒸馏 (teacher -> empty expert)
      - 数据驱动吸收 (从新数据学习)
      - 能力验证 (防污染)
    """

    def __init__(
        self,
        hidden_dim: int,
        absorb_lr: float = 1e-3,
        absorb_steps: int = 20,
        validation_threshold: float = 0.1,
        max_absorb_per_round: int = 2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.absorb_lr = absorb_lr
        self.absorb_steps = absorb_steps
        self.validation_threshold = validation_threshold
        self.max_absorb_per_round = max_absorb_per_round

        # 能力评估头 (判断吸收质量)
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def absorb_from_teacher(
        self,
        empty_expert: nn.Module,
        teacher_expert: nn.Module,
        num_samples: int = 128,
        device: Optional[torch.device] = None,
    ) -> Dict[str, float]:
        """从教师专家蒸馏知识到空专家.

        Args:
            empty_expert: 目标空专家 (EmptyExpert).
            teacher_expert: 教师专家 (已训练好的 BigExpert 等).
            num_samples: 蒸馏样本数.
            device: 计算设备.

        Returns:
            吸收统计字典.
        """
        device = device or next(empty_expert.parameters()).device
        hd = self.hidden_dim

        # 生成蒸馏样本
        x = torch.randn(num_samples, hd, device=device)

        # 教师输出
        with torch.no_grad():
            if hasattr(teacher_expert, "ffn"):
                teacher_out = teacher_expert.ffn(x)
            else:
                teacher_out = teacher_expert(x)

        # 学生 (空专家) 参数
        if hasattr(empty_expert, "ffn"):
            student_params = list(empty_expert.ffn.parameters())
        else:
            student_params = list(empty_expert.parameters())
        opt = torch.optim.Adam(student_params, lr=self.absorb_lr)

        total_loss = 0.0
        for step in range(self.absorb_steps):
            opt.zero_grad()
            if hasattr(empty_expert, "ffn"):
                pred = empty_expert.ffn(x)
            else:
                pred = empty_expert(x)
            loss = F.mse_loss(pred, teacher_out)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())

        avg_loss = total_loss / self.absorb_steps

        # 验证吸收质量
        quality = self._validate(empty_expert, teacher_expert, x, device)

        # 质量达标则标记吸收
        absorbed = False
        if quality["similarity"] > (1.0 - self.validation_threshold):
            if hasattr(empty_expert, "mark_absorbed"):
                empty_expert.mark_absorbed()
            if hasattr(empty_expert, "gate"):
                with torch.no_grad():
                    empty_expert.gate.fill_(0.5)
            absorbed = True

        return {
            "avg_distill_loss": avg_loss,
            "similarity": quality["similarity"],
            "quality_score": quality["quality_score"],
            "absorbed": absorbed,
            "validation_passed": quality["similarity"] > (1.0 - self.validation_threshold),
        }

    def _validate(
        self,
        student: nn.Module,
        teacher: nn.Module,
        x: torch.Tensor,
        device: torch.device,
    ) -> Dict[str, float]:
        """验证学生专家与教师专家的输出一致性."""
        with torch.no_grad():
            if hasattr(student, "ffn"):
                s_out = student.ffn(x)
            else:
                s_out = student(x)
            if hasattr(teacher, "ffn"):
                t_out = teacher.ffn(x)
            else:
                t_out = teacher(x)
            # 余弦相似度
            sim = F.cosine_similarity(
                s_out.flatten(), t_out.flatten(), dim=0
            ).item()
            # 质量评估头
            quality = self.quality_head(s_out.mean(dim=0, keepdim=True)).item()
        return {"similarity": sim, "quality_score": quality}

    def absorb_batch(
        self,
        empty_experts: List[nn.Module],
        teacher_experts: List[nn.Module],
        num_samples: int = 128,
        device: Optional[torch.device] = None,
    ) -> List[Dict[str, float]]:
        """批量吸收: 每轮最多吸收 max_absorb_per_round 个空专家.

        Args:
            empty_experts: 空专家列表.
            teacher_experts: 教师专家列表.

        Returns:
            每个空专家的吸收统计.
        """
        results = []
        absorbed_count = 0
        for empty, teacher in zip(empty_experts, teacher_experts):
            if absorbed_count >= self.max_absorb_per_round:
                results.append({"skipped": True, "absorbed": False})
                continue
            result = self.absorb_from_teacher(empty, teacher, num_samples, device)
            results.append(result)
            if result["absorbed"]:
                absorbed_count += 1
        return results

    def extra_repr(self) -> str:
        return (
            f"lr={self.absorb_lr}, steps={self.absorb_steps}, "
            f"threshold={self.validation_threshold}, "
            f"max_per_round={self.max_absorb_per_round}"
        )
