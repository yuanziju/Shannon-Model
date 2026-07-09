"""社交部署 - 多 Agent 协作 / 角色扮演 / 图灵测试.

SocialDeploy 将 ReAct+CRA Agent 实例部署到社交场景, 支持多 Agent
角色扮演对话与图灵测试评估. 遵守平台合规要求 (T2.7.6/T3.8.3).
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


class TuringVerdict(Enum):
    """图灵测试裁定结果. """

    HUMAN_LIKE = "HUMAN_LIKE"      # 判官误判为人
    MACHINE_LIKE = "MACHINE_LIKE"  # 判官识破为机器
    UNCERTAIN = "UNCERTAIN"        # 无法判定


@dataclass
class Persona:
    """角色人设. """

    name: str
    description: str
    traits: List[str] = field(default_factory=list)
    style: str = "neutral"
    system_prompt: str = ""


@dataclass
class SocialMessage:
    role: str            # persona name 或 "judge" 或 "user"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SocialDeploy:
    """社交部署引擎.

    Args:
        agent_factory: 根据人设生成一个 Agent (具备 ``run(user_input, context)``)
            的可调用对象: ``fn(persona: Persona) -> Any``.
        judge_fn: 图灵测试判官, 签名 ``fn(dialogue: list) -> str``
            返回 "human" / "machine" / "uncertain".
        rng_seed: 随机种子 (可复现).
    """

    def __init__(
        self,
        agent_factory: Optional[Callable[[Persona], Any]] = None,
        judge_fn: Optional[Callable[[List[SocialMessage]], str]] = None,
        rng_seed: Optional[int] = None,
    ) -> None:
        self.agent_factory = agent_factory or self._default_agent_factory
        self.judge_fn = judge_fn or self._default_judge
        self._rng = random.Random(rng_seed)
        self._personas: Dict[str, Persona] = {}
        self._agents: Dict[str, Any] = {}
        self._deploy_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # 角色管理
    # ------------------------------------------------------------------ #
    def register_persona(self, persona: Persona) -> None:
        self._personas[persona.name] = persona
        self._agents[persona.name] = self.agent_factory(persona)

    def get_agent(self, name: str) -> Any:
        return self._agents.get(name)

    # ------------------------------------------------------------------ #
    # 多 Agent 角色扮演
    # ------------------------------------------------------------------ #
    def roleplay(
        self,
        participants: Sequence[str],
        seed_topic: str,
        rounds: int = 4,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[SocialMessage]:
        """多 Agent 轮流围绕 ``seed_topic`` 进行角色扮演对话. """
        dialogue: List[SocialMessage] = []
        ctx = dict(context or {})
        ctx["topic"] = seed_topic
        # 谁先开口
        speakers = [p for p in participants if p in self._agents]
        if not speakers:
            return dialogue
        # 随机化首个发言者以模拟自然对话
        self._rng.shuffle(speakers)
        current_topic = seed_topic
        for r in range(int(rounds)):
            for name in speakers:
                agent = self._agents[name]
                persona = self._personas[name]
                local_ctx = dict(ctx)
                local_ctx["persona"] = persona.description
                local_ctx["history"] = [m.content for m in dialogue[-6:]]
                prompt = self._build_prompt(persona, current_topic, dialogue)
                reply = self._call_agent(agent, prompt, local_ctx)
                dialogue.append(SocialMessage(role=name, content=reply, metadata={"round": r}))
                current_topic = reply  # 话题随对话演进
        self._deploy_log.append(
            {"type": "roleplay", "participants": list(speakers), "rounds": rounds, "ts": time.time()}
        )
        return dialogue

    # ------------------------------------------------------------------ #
    # 图灵测试
    # ------------------------------------------------------------------ #
    def turing_test(
        self,
        agent_persona: Persona,
        human_messages: Sequence[str],
        agent_messages: Optional[Sequence[str]] = None,
        rounds: int = 5,
    ) -> Dict[str, Any]:
        """模拟图灵测试: 判官与 (Agent + 真人) 对话后裁定.

        Returns:
            ``{"verdict": TuringVerdict, "pass_rate": float, "dialogue": [...]}``.
        """
        if agent_persona.name not in self._agents:
            self.register_persona(agent_persona)
        agent = self._agents[agent_persona.name]
        dialogue: List[SocialMessage] = []
        agent_replies: List[str] = list(agent_messages or [])
        human_pool = list(human_messages)
        self._rng.shuffle(human_pool)

        agent_turn = True
        for r in range(int(rounds)):
            if agent_turn and agent_replies:
                content = agent_replies.pop(0)
            elif not agent_turn and human_pool:
                content = human_pool.pop(0)
            else:
                # 由 Agent 生成
                prompt = self._build_prompt(agent_persona, "chat", dialogue)
                content = self._call_agent(agent, prompt, {"history": [m.content for m in dialogue]})
            speaker = agent_persona.name if agent_turn else "human_control"
            dialogue.append(SocialMessage(role=speaker, content=content, metadata={"round": r}))
            agent_turn = not agent_turn

        verdict_str = self.judge_fn(dialogue)
        verdict = TuringVerdict(
            {"human": TuringVerdict.HUMAN_LIKE, "machine": TuringVerdict.MACHINE_LIKE}.get(
                verdict_str, TuringVerdict.UNCERTAIN
            )
        )
        pass_rate = 1.0 if verdict == TuringVerdict.HUMAN_LIKE else (0.5 if verdict == TuringVerdict.UNCERTAIN else 0.0)
        result = {
            "verdict": verdict,
            "pass_rate": pass_rate,
            "dialogue": [{"role": m.role, "content": m.content} for m in dialogue],
        }
        self._deploy_log.append({"type": "turing", "verdict": verdict.value, "ts": time.time()})
        return result

    # ------------------------------------------------------------------ #
    # 默认实现
    # ------------------------------------------------------------------ #
    @staticmethod
    def _default_agent_factory(persona: Persona) -> Any:
        class _Echo:
            def __init__(self, p: Persona) -> None:
                self.persona = p

            def run(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> str:
                return f"[{self.persona.name}] {user_input}"

        return _Echo(persona)

    def _call_agent(self, agent: Any, prompt: str, ctx: Dict[str, Any]) -> str:
        try:
            out = agent.run(prompt, ctx)
        except Exception:  # noqa: BLE001
            out = "..."
        return str(out) if out is not None else "..."

    @staticmethod
    def _build_prompt(persona: Persona, topic: str, dialogue: Sequence[SocialMessage]) -> str:
        recent = " | ".join(m.content[:40] for m in list(dialogue)[-3:])
        return f"[{persona.name}/{persona.style}] topic={topic} recent={recent}"

    def _default_judge(self, dialogue: List[SocialMessage]) -> str:
        # 启发式: 含 persona 风格标记的回复更像机器
        machine_markers = sum(
            1 for m in dialogue if m.role != "human_control" and ("[" in m.content and "]" in m.content)
        )
        total = max(1, len([m for m in dialogue if m.role != "human_control"]))
        ratio = machine_markers / total
        if ratio > 0.6:
            return "machine"
        if ratio < 0.3:
            return "human"
        return "uncertain"

    @property
    def deploy_log(self) -> List[Dict[str, Any]]:
        return list(self._deploy_log)


__all__ = ["SocialDeploy", "Persona", "SocialMessage", "TuringVerdict"]
