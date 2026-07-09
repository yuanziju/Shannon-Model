"""Self-Play 多 Agent 辩论引擎.

通过正方 (Proponent) / 反方 (Opponent) / 裁判 (Judge) 三角色多轮辩论,
对数学猜想进行证伪式检验, 产出带有置信度的辩论结论.

与 ``common.agent.self_play`` 的 proposer/solver/judge 自我对弈互补:
    - ``SelfPlay``     面向 *任务求解* (出题-解题-验证)
    - ``SelfPlayDebate`` 面向 *命题检验* (证明-反驳-裁定), 用于猜想筛选与
      形式化证明前的启发式论证, 与 ``ConjectureGenerator`` 联动.

三方均由同一 (或多个) 15B 模型扮演, 通过角色 prompt 切换.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch

__all__ = [
    "SelfPlayDebate",
    "DebateResult",
    "DebateRound",
    "Judgment",
    "DebateSide",
]


class DebateSide:
    """辩论胜负方标识. """

    PROPOSITION = "proposition"  # 正方胜 (猜想成立)
    OPPOSITION = "opposition"    # 反方胜 (猜想被反驳)
    TIE = "tie"                  # 平局 / 无法裁定


@dataclass
class Judgment:
    """裁判对单轮辩论的裁定. """

    winner: str = DebateSide.TIE          # 'proposition' | 'opposition' | 'tie'
    confidence: float = 0.0               # 裁判置信度 [0, 1]
    reasoning: str = ""                   # 裁判理由
    prop_score: float = 0.0               # 正方论点强度
    opp_score: float = 0.0                # 反方论点强度


@dataclass
class DebateRound:
    """单轮辩论记录. """

    round: int
    prop_args: str
    opp_args: str
    judgment: Judgment


@dataclass
class DebateResult:
    """完整辩论结论. """

    winner: str = DebateSide.TIE
    rounds: List[DebateRound] = field(default_factory=list)
    prop_args: List[str] = field(default_factory=list)         # 各轮正方论点
    opp_args: List[str] = field(default_factory=list)          # 各轮反方论点
    judge_decisions: List[Judgment] = field(default_factory=list)
    confidence: float = 0.0
    conjecture: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


# --------------------------------------------------------------------- #
# 内部: 极简 tokenizer / model 调用 (兼容 MockModel)
# --------------------------------------------------------------------- #
def _to_ids(text: str, max_len: int = 1024) -> torch.Tensor:
    chars = list(text)[:max_len]
    ids = [(ord(c) % 1000) + 1 for c in chars] or [0]
    return torch.tensor([ids], dtype=torch.long)


def _call_model(model, text: str, max_new: int = 48) -> "tuple[str, float]":
    """调用模型并返回 (回复文本, 置信度). 对 MockModel 等返回 dict 的模型鲁棒.

    置信度从模型 ``confidence`` 输出读取, 缺失时回退 0.5.
    """
    ids = _to_ids(text)
    try:
        out = model(ids)
    except TypeError:
        out = model(input_ids=ids)

    logits = None
    conf = None
    if isinstance(out, dict):
        logits = out.get("logits")
        conf = out.get("confidence")
    else:
        logits = getattr(out, "logits", None)
        conf = getattr(out, "confidence", None)

    if logits is None:
        return "", 0.5

    # 从最后一步 logits 贪心解码一段简短续写 (mock 下近似 autoregressive)
    try:
        last = logits[0, -1]
        nxt = int(torch.argmax(last).item())
    except Exception:  # noqa: BLE001
        return "", 0.5

    generated = [nxt]
    for _ in range(max(0, max_new - 1)):
        try:
            row = logits[0, -1]
        except Exception:  # noqa: BLE001
            break
        v = int(torch.argmax(row).item())
        if v == 0:
            break
        generated.append(v)

    text_out = "".join(
        chr(v) if (32 <= v < 0x110000) else " " for v in generated
    ).strip()

    if conf is not None:
        try:
            c = float(conf.mean().item())
        except Exception:  # noqa: BLE001
            c = 0.5
    else:
        c = 0.5
    c = max(0.0, min(1.0, c))
    return text_out, c


class SelfPlayDebate:
    """Self-Play 多 Agent 辩论引擎.

    Args:
        model: 15B 循环主体 (或 MockModel). 调用签名 ``model(input_ids) -> dict``
            含 ``logits`` 与可选 ``confidence``.
        max_rounds: 最大辩论轮数.
        num_proponents: 正方 agent 数量 (多 agent 论点聚合).
        num_opponents: 反方 agent 数量.
        early_stop_confidence: 裁判置信度达到该阈值时提前结束.
    """

    def __init__(
        self,
        model,
        max_rounds: int = 10,
        num_proponents: int = 2,
        num_opponents: int = 2,
        early_stop_confidence: float = 0.9,
    ) -> None:
        self.model = model
        self.max_rounds = max(1, int(max_rounds))
        self.num_proponents = max(1, int(num_proponents))
        self.num_opponents = max(1, int(num_opponents))
        self.early_stop_confidence = float(early_stop_confidence)

    # ------------------------------------------------------------------ #
    # 公共接口
    # ------------------------------------------------------------------ #
    def debate(self, conjecture: str) -> DebateResult:
        """多 Agent 辩论主循环.

        流程:
            1. 正方 (多个 proponent) 联合产出证明论点;
            2. 反方 (多个 opponent) 针对正方论点产出反驳/反例;
            3. 裁判评估双方, 给出本轮裁定;
            4. 多轮迭代: 正方再反驳反方, 反方再反驳正方;
            5. 聚合各轮裁定决定最终胜方与置信度.
        """
        rounds: List[DebateRound] = []
        all_prop: List[str] = []
        all_opp: List[str] = []
        decisions: List[Judgment] = []

        prev_opp = ""
        prev_prop = ""

        for r in range(self.max_rounds):
            prop_args = self.generate_proponent_args(conjecture, prev_opp)
            opp_args = self.generate_opponent_args(conjecture, prev_prop)
            judgment = self.judge_round(prop_args, opp_args)

            rnd = DebateRound(
                round=r,
                prop_args=prop_args,
                opp_args=opp_args,
                judgment=judgment,
            )
            rounds.append(rnd)
            all_prop.append(prop_args)
            all_opp.append(opp_args)
            decisions.append(judgment)

            prev_opp = opp_args
            prev_prop = prop_args

            if judgment.confidence >= self.early_stop_confidence:
                break

        winner, confidence = self._aggregate(decisions)
        return DebateResult(
            winner=winner,
            rounds=rounds,
            prop_args=all_prop,
            opp_args=all_opp,
            judge_decisions=decisions,
            confidence=confidence,
            conjecture=conjecture,
            metadata={
                "num_rounds": len(rounds),
                "num_proponents": self.num_proponents,
                "num_opponents": self.num_opponents,
            },
        )

    def generate_proponent_args(self, conjecture: str, prev_opp_args: str) -> str:
        """正方: 尝试证明猜想, 并反驳上一轮反方论点. """
        args_list: List[str] = []
        for i in range(self.num_proponents):
            prompt = (
                f"[ROLE: Proponent {i + 1}/{self.num_proponents}]\n"
                f"Conjecture: {conjecture}\n"
                f"Opponent's previous counter-argument: {prev_opp_args or '(none)'}\n"
                f"Provide a proof or supporting argument, and refute the counter if any:"
            )
            text, conf = _call_model(self.model, prompt)
            args_list.append(
                f"[P{i + 1} conf={conf:.2f}] {text or 'argue-by-construction'}"
            )
        return self._merge_args(args_list)

    def generate_opponent_args(self, conjecture: str, prev_prop_args: str) -> str:
        """反方: 尝试找反例或反驳, 并针对上一轮正方论点反驳. """
        args_list: List[str] = []
        for i in range(self.num_opponents):
            prompt = (
                f"[ROLE: Opponent {i + 1}/{self.num_opponents}]\n"
                f"Conjecture: {conjecture}\n"
                f"Proponent's previous argument: {prev_prop_args or '(none)'}\n"
                f"Find a counterexample or refute the argument:"
            )
            text, conf = _call_model(self.model, prompt)
            args_list.append(
                f"[O{i + 1} conf={conf:.2f}] {text or 'counterexample-search'}"
            )
        return self._merge_args(args_list)

    def judge_round(self, prop_args: str, opp_args: str) -> Judgment:
        """裁判: 评估双方论点强度, 决定本轮胜方. """
        prompt = (
            "[ROLE: Judge]\n"
            f"Proponent arguments:\n{prop_args}\n\n"
            f"Opponent arguments:\n{opp_args}\n\n"
            "Evaluate both sides. State which side is stronger and your confidence."
        )
        text, conf = _call_model(self.model, prompt, max_new=32)

        # 论点强度: 用模型 confidence 作为弱信号, 同时按文本长度做轻微加权
        prop_score = self._side_score(prop_args)
        opp_score = self._side_score(opp_args)

        if prop_score > opp_score * 1.05:
            winner = DebateSide.PROPOSITION
        elif opp_score > prop_score * 1.05:
            winner = DebateSide.OPPOSITION
        else:
            winner = DebateSide.TIE

        # 关键词微调 (反方提到反例/counterexample 时倾向反方)
        lowered = (text + " " + opp_args).lower()
        if "counterexample" in lowered or "refut" in lowered:
            if winner == DebateSide.PROPOSITION:
                winner = DebateSide.TIE

        return Judgment(
            winner=winner,
            confidence=conf,
            reasoning=text or "auto-judged by score differential",
            prop_score=prop_score,
            opp_score=opp_score,
        )

    # ------------------------------------------------------------------ #
    # 内部
    # ------------------------------------------------------------------ #
    @staticmethod
    def _merge_args(args_list: List[str]) -> str:
        return " || ".join(a for a in args_list if a)

    @staticmethod
    def _side_score(args: str) -> float:
        """论点强度启发式: 长度 + 置信度标记. """
        if not args:
            return 0.0
        base = min(1.0, len(args) / 200.0) * 0.5
        # 解析 [Pn conf=0.xx] / [On conf=0.xx] 标记
        confs = []
        for tok in args.replace("||", " ").split():
            if "conf=" in tok:
                try:
                    confs.append(float(tok.split("conf=")[-1].rstrip("]")))
                except ValueError:
                    pass
        conf_term = (sum(confs) / len(confs)) if confs else 0.5
        return base + 0.5 * conf_term

    @staticmethod
    def _aggregate(decisions: List[Judgment]) -> "tuple[str, float]":
        """聚合各轮裁定决定最终胜方与置信度. """
        if not decisions:
            return DebateSide.TIE, 0.0
        weighted = {"prop": 0.0, "opp": 0.0, "tie": 0.0}
        for j in decisions:
            w = max(1e-3, j.confidence)
            if j.winner == DebateSide.PROPOSITION:
                weighted["prop"] += w
            elif j.winner == DebateSide.OPPOSITION:
                weighted["opp"] += w
            else:
                weighted["tie"] += w * 0.5
        total = weighted["prop"] + weighted["opp"] + weighted["tie"]
        if total <= 0:
            return DebateSide.TIE, 0.0
        if weighted["prop"] > weighted["opp"] and weighted["prop"] > weighted["tie"]:
            winner = DebateSide.PROPOSITION
        elif weighted["opp"] > weighted["prop"] and weighted["opp"] > weighted["tie"]:
            winner = DebateSide.OPPOSITION
        else:
            winner = DebateSide.TIE
        confidence = max(weighted["prop"], weighted["opp"]) / total
        return winner, float(confidence)
