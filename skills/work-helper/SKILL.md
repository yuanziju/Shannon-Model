---
name: work-helper
version: 1.0.0
description: "AI 代码工作流 Skill。当用户需要写代码、修 bug、重构、补测试时使用。强制 spec 驱动 + 完整 TDD + 子代理审查，绝对不允许偷懒（空实现/简化实现），写不了打 TODO/FIXME。"
---

# Work Helper — 代码工作流

## 触发条件

当用户请求涉及以下任意一种时触发本 Skill：
- 写新功能（new feature）
- 修 bug（bug fix）
- 重构代码（refactor）
- 补测试（add test）

如果用户请求是写文档 / 写文案 / 写创意内容，**不触发本 Skill**（本 Skill 只处理代码）。

---

## 核心铁律（不可违反）

### 铁律 1：绝对不能偷懒
- 不能空实现（`pass` / `return None` / `// TODO` 不算实现）
- 不能简化实现（用户要 A，不能只给 A 的 30%）
- 实在写不了的地方必须打 `TODO` 或 `FIXME` 注释，说明原因和后续计划
- 除非用户明确说"先简版""先 MVP""不用写完整"，否则一律完整实现

详见 [references/anti-lazy.md](references/anti-lazy.md)

### 铁律 2：全部写 spec
所有任务（不分轻重）都必须先写 spec 再动手。
- Spec 位置：项目根 `specs/<task-name>/`（**不放 dotfile 如 `.trae/`**）
- 重任务：`proposal.md` + `tasks.md` + delta-spec（OpenSpec 风格，带 `ADDED/MODIFIED/REMOVED/RENAMED` 标记）
- 轻任务：`proposal.md`（<20 行，含背景 + 目标 + 任务清单）

### 铁律 3：TDD 优先
完整 TDD（RED-GREEN-REFACTOR），例外清单见 [references/tdd-rules.md](references/tdd-rules.md)。
不过度 TDD（不给测试写测试、不给测试框架本身写测试）。

### 铁律 4：强制子代理审查
代码写完后，**必须**开一个新的子代理过 checklist 审查。
- 子代理提示词必须严格约束，不能放飞自我
- **绝对不允许子代理再开子代理**（防递归）
- 若主代理并行开了多个子代理执行任务，验收时主代理**另开一个独立子代理**验收，不能用执行任务的子代理自己验收

详见 [references/review-checklist.md](references/review-checklist.md)

### 铁律 5：Conventional Commits
commit message 必须用 Conventional Commits 格式：
```
<type>(<scope>): <description>

[optional body]

[optional footer]
```
type ∈ `feat / fix / refactor / test / docs / chore / perf / style / build / ci`

---

## 任务分流（Agent 自判）

详见 [references/task-tiers.md](references/task-tiers.md)

| 维度 | 重任务 | 轻任务 |
|------|--------|--------|
| 触发条件 | 新功能 / 大重构（影响 >3 文件 或 新增 >100 行） | bug fix / 小重构 / 补测试（≤3 文件 且 ≤100 行） |
| Spec | 完整 OpenSpec（proposal + tasks + delta-spec） | 简版 spec（proposal <20 行） |
| Brainstorm | 强制（列 3 种方案对比 + 风险点） | Agent 自决 |
| TDD | 完整 RED-GREEN-REFACTOR | 简化（bug fix 必须先写复现测试） |

边界模糊时按**重任务**处理（宁可多走流程，不可偷懒）。

---

## 必须停下来问用户的时机

以下情况**必须停下来用 AUQ 问用户**，不能自己决定：

1. **不可逆操作前**：删文件 / 覆盖大段代码 / 改 CI / 改配置文件
2. **多方案分歧时**：用户没指定，存在 2+ 等价实现方案
3. **调试循环超限时**：测试连挂 2 次或调试循环超过 2 轮
4. **环境改动前**：装新依赖 / 跑外部命令 / 访问网络

其他情况 Agent 自己决定，不要瞎问浪费用户轮次。

---

## 执行流程（9 步）

详见 [workflows/code.md](workflows/code.md)

```
1. 识别任务类型 + 轻重分级
2. 写 spec
3. brainstorm（重任务强制）
4. TDD 写测试（RED）
5. 实现（GREEN）
6. 重构（REFACTOR）
7. 验证（跑测试 / py_compile）
8. 子代理审查
9. Conventional Commit
```

---

## 测试指标（不能只看覆盖率）

详见 [references/test-metrics.md](references/test-metrics.md)

- 多维覆盖率：行 / 分支 / 函数（不只看一个数字）
- 边界覆盖：空值 / 负数 / 超长 / 并发
- 断言质量：避免伪测试（`assert True` / 只测 happy path）
- 关键路径覆盖：核心逻辑必须有测试，不看总数看重点

---

## 文件结构

```
skills/work-helper/
├── SKILL.md                       # 本文件（主入口）
├── workflows/
│   └── code.md                    # 代码工作流主体（9 步详解）
└── references/
    ├── task-tiers.md              # 轻重任务分级标准
    ├── tdd-rules.md               # TDD 规则 + 例外清单
    ├── test-metrics.md            # 测试指标清单
    ├── anti-lazy.md               # 防偷懒铁律
    └── review-checklist.md        # 子代理审查 checklist
```
