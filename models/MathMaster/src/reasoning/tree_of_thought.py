"""Tree-of-Thought 数学搜索 - BFS + 价值函数 + 剪枝 + 回溯.

将数学推理建模为思想树搜索: 每个节点是一段 "思考状态", 通过模型展开子节点
(branching), 用价值函数评估节点前景, BFS + beam-width 剪枝保留最优分支,
并在死路时回溯到祖先节点重新展开.

与 ``latent_decode`` 的层次化 NAR / 掩码精化解码互补: ToT 在 *符号推理* 层
搜索, latent decode 在 *隐空间* 层搜索.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

__all__ = ["MathToT", "ThoughtNode", "ToTResult"]


@dataclass
class ThoughtNode:
    """思想树节点. """

    state: str
    value: float = 0.0
    depth: int = 0
    parent: Optional["ThoughtNode"] = None
    children: List["ThoughtNode"] = field(default_factory=list)
    expanded: bool = False
    pruned: bool = False
    visits: int = 0

    def path_to_root(self) -> List["ThoughtNode"]:
        """从根到本节点的路径. """
        chain: List[ThoughtNode] = []
        node: Optional[ThoughtNode] = self
        while node is not None:
            chain.append(node)
            node = node.parent
        chain.reverse()
        return chain

    def describe(self) -> str:
        return " -> ".join(n.state[:40] for n in self.path_to_root())


@dataclass
class ToTResult:
    """ToT 搜索结果. """

    problem: str = ""
    solution_path: List[str] = field(default_factory=list)
    solution_value: float = 0.0
    total_nodes: int = 0
    expanded_nodes: int = 0
    pruned_nodes: int = 0
    backtracks: int = 0
    reached_depth: int = 0
    converged: bool = False
    elapsed: float = 0.0
    best_node: Optional[ThoughtNode] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MathToT:
    """Tree-of-Thought 数学搜索.

    Args:
        model: 15B 循环主体.
        max_depth: 最大搜索深度.
        branching_factor: 每节点展开子节点数.
        beam_width: BFS beam 宽度 (每层保留的最优节点数).
        value_threshold: 价值低于此值的子节点剪枝.
        max_nodes: 节点总数上限 (防爆).
        solution_threshold: 价值达到此阈值视为找到解答.
    """

    def __init__(
        self,
        model,
        max_depth: int = 5,
        branching_factor: int = 3,
        beam_width: int = 3,
        value_threshold: float = 0.2,
        max_nodes: int = 200,
        solution_threshold: float = 0.85,
    ) -> None:
        self.model = model
        self.max_depth = max(1, int(max_depth))
        self.branching_factor = max(1, int(branching_factor))
        self.beam_width = max(1, int(beam_width))
        self.value_threshold = float(value_threshold)
        self.max_nodes = max(1, int(max_nodes))
        self.solution_threshold = float(solution_threshold)

    # ------------------------------------------------------------------ #
    # 公共接口
    # ------------------------------------------------------------------ #
    def search(self, problem: str) -> ToTResult:
        """BFS + 价值剪枝 + 回溯的树搜索. """
        t0 = time.time()
        root = ThoughtNode(state=f"[Problem] {problem}", value=0.5, depth=0)
        total = 1
        expanded = 0
        pruned = 0
        backtracks = 0
        best_node = root
        converged = False

        # BFS frontier (按 value 排序的 beam)
        frontier: List[ThoughtNode] = [root]

        for depth in range(1, self.max_depth + 1):
            if not frontier or total >= self.max_nodes:
                break

            candidates: List[ThoughtNode] = []
            for node in frontier:
                if total >= self.max_nodes:
                    break
                if node.expanded:
                    continue
                children = self.expand(node, problem)
                node.expanded = True
                expanded += 1

                kept: List[ThoughtNode] = []
                for child in children:
                    total += 1
                    child.value = self.evaluate(child, problem)
                    if child.value < self.value_threshold:
                        child.pruned = True
                        pruned += 1
                        continue
                    kept.append(child)
                    node.children.append(child)
                    if child.value > best_node.value:
                        best_node = child
                    if child.value >= self.solution_threshold:
                        converged = True
                        best_node = child
                candidates.extend(kept)

                if converged:
                    break

            if not candidates:
                # 死路 -> 回溯到 frontier 的祖先尝试新展开
                backtracks += 1
                frontier = self._backtrack_frontier(frontier)
                continue

            # beam 剪枝: 保留 beam_width 个最优
            candidates.sort(key=lambda n: n.value, reverse=True)
            frontier = candidates[: self.beam_width]

            if converged:
                break

        path = [n.state for n in best_node.path_to_root()]
        return ToTResult(
            problem=problem,
            solution_path=path,
            solution_value=best_node.value,
            total_nodes=total,
            expanded_nodes=expanded,
            pruned_nodes=pruned,
            backtracks=backtracks,
            reached_depth=best_node.depth,
            converged=converged,
            elapsed=time.time() - t0,
            best_node=best_node,
            metadata={
                "max_depth": self.max_depth,
                "branching_factor": self.branching_factor,
                "beam_width": self.beam_width,
            },
        )

    # ------------------------------------------------------------------ #
    # 核心: 展开 / 评估 / 回溯
    # ------------------------------------------------------------------ #
    def expand(self, node: ThoughtNode, problem: str) -> List[ThoughtNode]:
        """生成子思考节点 (branching_factor 个). """
        context = node.describe()
        children: List[ThoughtNode] = []
        seen_states = set()
        for k in range(self.branching_factor):
            prompt = (
                f"[ToT expand k={k}] Problem: {problem}\n"
                f"Current path: {context}\n"
                f"Propose the next distinct reasoning step:"
            )
            text, _ = self._call_model(self.model, prompt, max_new=48, temperature=0.6 + 0.1 * k)
            text = (text or f"step-{k}").strip()
            if text in seen_states:
                text = f"{text}#{k}"
            seen_states.add(text)
            children.append(
                ThoughtNode(
                    state=text,
                    depth=node.depth + 1,
                    parent=node,
                )
            )
        return children

    def evaluate(self, node: ThoughtNode, problem: str) -> float:
        """价值函数: 评估节点作为解答前景的分数 [0, 1]. """
        prompt = (
            f"[ToT value] Problem: {problem}\n"
            f"Reasoning path so far: {node.describe()}\n"
            f"Rate the promise of this path toward a correct solution (0-1)."
        )
        _, conf = self._call_model(self.model, prompt, max_new=16)
        # value = confidence + 深度推进奖励 (越深越接近解)
        depth_bonus = min(0.2, 0.04 * node.depth)
        value = 0.8 * conf + depth_bonus
        # 含结论标记的节点加分
        if any(mk in node.state.lower() for mk in ("therefore", "qed", "综上", "得证")):
            value = min(1.0, value + 0.15)
        return max(0.0, min(1.0, value))

    def backtrack(self, node: ThoughtNode) -> Optional[ThoughtNode]:
        """从死路节点回溯到第一个仍可展开的祖先. """
        cur: Optional[ThoughtNode] = node.parent
        while cur is not None:
            if not cur.expanded or any(not c.pruned for c in cur.children):
                return cur
            cur = cur.parent
        return None

    # ------------------------------------------------------------------ #
    # 内部
    # ------------------------------------------------------------------ #
    def _backtrack_frontier(self, frontier: List[ThoughtNode]) -> List[ThoughtNode]:
        """整层死路时, 回溯收集可重展开的祖先作为新 frontier. """
        new_frontier: List[ThoughtNode] = []
        seen: set = set()
        for node in frontier:
            anc = self.backtrack(node)
            if anc is not None and id(anc) not in seen:
                seen.add(id(anc))
                anc.expanded = False  # 允许重新展开
                new_frontier.append(anc)
        return new_frontier[: self.beam_width]

    # ------------------------------------------------------------------ #
    # 内部: model 调用
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_ids(text: str, max_len: int = 1024) -> torch.Tensor:
        chars = list(text)[:max_len]
        ids = [(ord(c) % 1000) + 1 for c in chars] or [0]
        return torch.tensor([ids], dtype=torch.long)

    @staticmethod
    def _call_model(model, text: str, max_new: int = 48, temperature: float = 0.7) -> "tuple[str, float]":
        ids = MathToT._to_ids(text)
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
        try:
            last = logits[0, -1]
            if temperature > 0:
                probs = torch.softmax(last / max(temperature, 1e-6), dim=-1)
                nxt = int(torch.multinomial(probs, num_samples=1).item())
            else:
                nxt = int(torch.argmax(last).item())
        except Exception:  # noqa: BLE001
            return "", 0.5
        generated = [nxt]
        for _ in range(max(0, max_new - 1)):
            try:
                v = int(torch.argmax(logits[0, -1]).item())
            except Exception:  # noqa: BLE001
                break
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
        return text_out, max(0.0, min(1.0, c))
