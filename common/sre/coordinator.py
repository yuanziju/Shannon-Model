"""工具协调器 (ToolCoordinator).

spec §7.3: Tool Coordinator 负责多工具链式调度, 支持:
    - Kahn 拓扑排序: 工具间存在数据依赖, 按拓扑序并行调度独立工具.
    - [IF:tool_failed] 条件分支: 工具失败时切换到备选路径.
    - 并行调用: 独立子问题同时调用多工具.
    - 流式返回: 工具输出实时流式返回模型.
    - 变量共享: 通过 ToolMemory 跨工具传递变量.

任务图 (DAG) 表示:
    node = (tool_name, args_template, output_var, on_fail_branch)
    edge = data依赖 (output_var -> input_arg)

调度算法:
    1. Kahn 拓扑排序得到可并行层.
    2. 同层工具并行执行.
    3. 失败节点触发 [IF:tool_failed] 分支.
    4. 收集所有输出到 ToolMemory.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

import torch
import torch.nn as nn


class ToolStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"   # 因依赖失败而跳过


@dataclass
class ToolNode:
    """工具调用节点 (DAG 中的一个节点)."""

    node_id: str
    tool_name: str
    args_template: dict          # 参数模板, 含 ${var} 引用
    output_var: str              # 输出变量名 (存入 ToolMemory)
    dependencies: list[str] = field(default_factory=list)  # 依赖的 node_id
    on_fail: str | None = None   # 失败时跳转的分支 node_id ([IF:tool_failed])
    on_fail_action: str = "branch"  # "branch" | "abort" | "continue"
    timeout_sec: float = 30.0
    retries: int = 0
    # 运行时状态
    status: ToolStatus = ToolStatus.PENDING
    result: object = None
    error: str = ""
    elapsed_sec: float = 0.0


@dataclass
class CoordinatorConfig:
    """协调器配置."""

    max_parallel: int = 8             # 最大并行工具数
    default_timeout: float = 30.0
    enable_streaming: bool = True
    # 失败传播策略: "skip_descendants" | "branch" | "abort_all"
    fail_propagation: str = "branch"
    # 重试
    default_retries: int = 0


class ToolCoordinator(nn.Module):
    """多工具链式调度协调器 (Kahn 拓扑排序 + 条件分支).

    继承 nn.Module 仅为接口统一 (无参模块).
    """

    def __init__(
        self,
        config: CoordinatorConfig | None = None,
        tool_registry: Optional[dict[str, Callable]] = None,
        tool_memory: Optional[object] = None,
        **kwargs,
    ):
        super().__init__()
        cfg = config or CoordinatorConfig(**kwargs)
        self.cfg = cfg
        self.tool_registry: dict[str, Callable] = dict(tool_registry or {})
        self.tool_memory = tool_memory  # ToolMemory 实例
        # DAG 节点表
        self.nodes: dict[str, ToolNode] = {}
        # 邻接表 (依赖 -> 被依赖)
        self.dependents: dict[str, list[str]] = defaultdict(list)
        # 入度表
        self.in_degree: dict[str, int] = {}

    # ------------------------------------------------------------------
    # DAG 构建
    # ------------------------------------------------------------------
    def register_tool(self, name: str, fn: Callable) -> None:
        """注册工具函数."""
        self.tool_registry[name] = fn

    def add_node(self, node: ToolNode) -> None:
        """添加节点到 DAG."""
        self.nodes[node.node_id] = node
        self.in_degree[node.node_id] = len(node.dependencies)
        for dep in node.dependencies:
            if dep not in self.nodes:
                # 前置节点尚未添加, 占位
                self.nodes[dep] = ToolNode(
                    node_id=dep, tool_name="", args_template={}, output_var=dep
                )
                self.in_degree[dep] = 0
            self.dependents[dep].append(node.node_id)

    def build_graph(self, nodes: list[ToolNode]) -> None:
        """批量构建 DAG."""
        self.nodes.clear()
        self.dependents.clear()
        self.in_degree.clear()
        # 先全部添加
        for n in nodes:
            self.nodes[n.node_id] = n
            self.in_degree[n.node_id] = 0
            self.dependents[n.node_id] = []
        # 再建立依赖
        for n in nodes:
            self.in_degree[n.node_id] = len(n.dependencies)
            for dep in n.dependencies:
                if dep in self.dependents:
                    self.dependents[dep].append(n.node_id)

    # ------------------------------------------------------------------
    # Kahn 拓扑排序
    # ------------------------------------------------------------------
    def kahn_topo_sort(self) -> list[list[str]]:
        """Kahn 算法拓扑排序, 返回可并行层列表.

        Returns:
            layers: list of list[node_id], 同层可并行.
        """
        in_deg = dict(self.in_degree)
        queue = deque(
            nid for nid, d in in_deg.items() if d == 0
        )
        layers = []
        visited = 0
        while queue:
            # 当前层全部出队 (并行)
            layer = list(queue)
            queue.clear()
            layers.append(layer)
            for nid in layer:
                visited += 1
                for child in self.dependents.get(nid, []):
                    in_deg[child] -= 1
                    if in_deg[child] == 0:
                        queue.append(child)
        if visited != len(self.nodes):
            raise ValueError(
                "DAG contains a cycle; cannot topologically sort"
            )
        return layers

    # ------------------------------------------------------------------
    # 参数解析: 替换 ${var} 引用
    # ------------------------------------------------------------------
    def _resolve_args(self, template: dict) -> dict:
        """解析参数模板中的 ${var} 引用, 从 ToolMemory 取值."""
        if self.tool_memory is None:
            return dict(template)
        resolved = {}
        for k, v in template.items():
            if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
                var_name = v[2:-1]
                resolved[k] = self.tool_memory.get(var_name)
            elif isinstance(v, dict):
                resolved[k] = self._resolve_args(v)
            else:
                resolved[k] = v
        return resolved

    # ------------------------------------------------------------------
    # 单工具执行
    # ------------------------------------------------------------------
    def _execute_node(self, node: ToolNode) -> ToolNode:
        """执行单个工具节点."""
        if node.tool_name not in self.tool_registry:
            node.status = ToolStatus.FAILED
            node.error = f"tool '{node.tool_name}' not registered"
            return node
        # 解析参数
        try:
            args = self._resolve_args(node.args_template)
        except Exception as e:
            node.status = ToolStatus.FAILED
            node.error = f"arg resolution failed: {e}"
            return node
        # 执行 (含重试)
        import time
        fn = self.tool_registry[node.tool_name]
        last_err = ""
        for attempt in range(node.retries + 1):
            t0 = time.time()
            try:
                result = fn(**args)
                node.result = result
                node.status = ToolStatus.SUCCESS
                node.elapsed_sec = time.time() - t0
                # 存入 ToolMemory
                if self.tool_memory is not None:
                    self.tool_memory.set(
                        node.output_var, result, source_tool=node.tool_name
                    )
                return node
            except Exception as e:
                last_err = str(e)
                node.elapsed_sec = time.time() - t0
        node.status = ToolStatus.FAILED
        node.error = last_err
        return node

    # ------------------------------------------------------------------
    # 失败处理: [IF:tool_failed] 分支
    # ------------------------------------------------------------------
    def _handle_failure(self, node: ToolNode) -> list[str]:
        """处理节点失败, 返回需要触发的新节点 id 列表.

        根据 on_fail_action:
            - "branch": 激活 on_fail 指向的备选节点.
            - "abort": 终止整个 DAG.
            - "continue": 跳过, 后代节点将被标记 SKIPPED.
        """
        triggered = []
        action = node.on_fail_action
        if action == "branch" and node.on_fail:
            triggered.append(node.on_fail)
            # 将备选节点入度置 0 使其可被调度
            if node.on_fail in self.in_degree:
                # 移除其对失败节点的依赖
                self.in_degree[node.on_fail] = 0
        elif action == "abort":
            # 标记所有未完成节点为 SKIPPED
            for nid, n in self.nodes.items():
                if n.status == ToolStatus.PENDING:
                    n.status = ToolStatus.SKIPPED
                    n.error = "aborted due to upstream failure"
        # "continue": 不做特殊处理, 后代在调度时检测
        return triggered

    def _propagate_skip(self, failed_id: str) -> None:
        """将依赖失败节点的后代标记为 SKIPPED (除非有备选分支)."""
        for child in self.dependents.get(failed_id, []):
            child_node = self.nodes.get(child)
            if child_node and child_node.status == ToolStatus.PENDING:
                # 若子节点有 on_fail 且 on_fail_action == branch, 不跳过
                if child_node.on_fail_action == "branch" and child_node.on_fail:
                    continue
                child_node.status = ToolStatus.SKIPPED
                child_node.error = f"upstream '{failed_id}' failed"
                self._propagate_skip(child)

    # ------------------------------------------------------------------
    # 完整调度 (Kahn 分层并行)
    # ------------------------------------------------------------------
    def execute(self) -> dict:
        """执行整个 DAG.

        Returns:
            dict 含 results / status / failed_nodes / execution_log.
        """
        layers = self.kahn_topo_sort()
        execution_log = []
        failed_nodes = []

        for layer in layers:
            # 过滤掉 SKIPPED 节点
            runnable = [
                nid for nid in layer
                if self.nodes[nid].status == ToolStatus.PENDING
            ]
            if not runnable:
                continue

            # 并行执行当前层 (简化: 顺序执行, 实际可用线程池)
            # 注: 真实生产环境使用 concurrent.futures 或 asyncio
            for nid in runnable:
                node = self.nodes[nid]
                node.status = ToolStatus.RUNNING
                execution_log.append({
                    "node_id": nid, "tool": node.tool_name,
                    "status": "start",
                })
                node = self._execute_node(node)
                execution_log.append({
                    "node_id": nid, "tool": node.tool_name,
                    "status": node.status.value,
                    "elapsed": node.elapsed_sec,
                    "error": node.error,
                })
                if node.status == ToolStatus.FAILED:
                    failed_nodes.append(nid)
                    # 触发 [IF:tool_failed] 分支
                    triggered = self._handle_failure(node)
                    # 传播跳过 (无备选的后代)
                    self._propagate_skip(nid)
                    # 将触发的备选节点加入下一轮
                    for t in triggered:
                        if t in self.nodes:
                            self.nodes[t].status = ToolStatus.PENDING

        # 收集结果
        results = {}
        for nid, node in self.nodes.items():
            if node.status == ToolStatus.SUCCESS:
                results[node.output_var] = node.result

        return {
            "results": results,
            "status": {
                nid: node.status.value for nid, node in self.nodes.items()
            },
            "failed_nodes": failed_nodes,
            "execution_log": execution_log,
            "all_succeeded": len(failed_nodes) == 0,
        }

    # ------------------------------------------------------------------
    # 流式执行 (生成器, 逐节点 yield)
    # ------------------------------------------------------------------
    def execute_streaming(self):
        """流式执行 DAG, 每完成一个节点 yield 一次结果.

        用于 spec §7.3 "流式返回: 工具输出实时流式返回模型".
        """
        layers = self.kahn_topo_sort()
        for layer in layers:
            runnable = [
                nid for nid in layer
                if self.nodes[nid].status == ToolStatus.PENDING
            ]
            for nid in runnable:
                node = self.nodes[nid]
                node.status = ToolStatus.RUNNING
                yield {"event": "start", "node_id": nid, "tool": node.tool_name}
                node = self._execute_node(node)
                yield {
                    "event": "finish",
                    "node_id": nid,
                    "tool": node.tool_name,
                    "status": node.status.value,
                    "result": node.result,
                    "error": node.error,
                }
                if node.status == ToolStatus.FAILED:
                    triggered = self._handle_failure(node)
                    self._propagate_skip(nid)
                    for t in triggered:
                        if t in self.nodes:
                            self.nodes[t].status = ToolStatus.PENDING
                    yield {
                        "event": "fail_branch",
                        "node_id": nid,
                        "triggered": triggered,
                    }

    def extra_repr(self) -> str:
        return (
            f"max_parallel={self.cfg.max_parallel}, "
            f"fail_propagation={self.cfg.fail_propagation}, "
            f"num_nodes={len(self.nodes)}, "
            f"num_tools={len(self.tool_registry)}"
        )
