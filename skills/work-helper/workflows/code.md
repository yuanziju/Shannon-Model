# 代码工作流主体（9 步）

> 本文件是 [SKILL.md](../SKILL.md) 的执行流程详解。所有代码任务必须按这 9 步走，不可跳步。

---

## 步骤 1：识别任务类型 + 轻重分级

### 1.1 识别任务类型
从用户原话判断，分四类：
- **新功能**（new feature）：从无到有加东西
- **修 bug**（bug fix）：用户给了报错或现象
- **重构**（refactor）：改结构不改行为
- **补测试**（add test）：给已有代码加测试

判断不出来时**不问用户**，按**重任务**处理（宁可多走流程，不可偷懒），但在响应里说一句：
> "我按重任务流程走，如果是小改动告诉我。"

### 1.2 轻重分级
详见 [references/task-tiers.md](../references/task-tiers.md)。简表：

| 维度 | 重任务 | 轻任务 |
|------|--------|--------|
| 影响范围 | >3 文件 或 新增 >100 行 | ≤3 文件 且 ≤100 行 |
| 类型 | 新功能 / 大重构 | bug fix / 小重构 / 补测试 |

边界模糊按**重任务**处理。

---

## 步骤 2：写 spec

### 2.1 Spec 位置
项目根 `specs/<task-name>/`，**不放 dotfile**（如 `.trae/`、`.claude/`）。

```
specs/
└── add-login-feature/           # task-name 用 kebab-case
    ├── proposal.md              # 重任务 + 轻任务都写
    ├── tasks.md                 # 仅重任务写
    └── delta-spec.md            # 仅重任务写（OpenSpec 风格）
```

### 2.2 重任务 spec（完整 OpenSpec 风格）

**proposal.md**：
```markdown
# Proposal: <task-name>

## 背景
<为什么要做这个>

## 目标
<做成什么样子>

## 非目标
<明确不做什么，防范围蔓延>

## 方案对比
（brainstorm 阶段填充，3 种方案 + 优缺点）

## 风险点
（brainstorm 阶段填充）
```

**tasks.md**（任务清单，T-XX 编号）：
```markdown
# Tasks: <task-name>

- [ ] T-01: <第一步>
- [ ] T-02: <第二步>
- [ ] T-03: <测试编写>
- [ ] T-04: <实现>
- [ ] T-05: <重构>
- [ ] T-06: <验证>
```

**delta-spec.md**（OpenSpec 增量规范）：
```markdown
# Delta Spec: <task-name>

## ADDED
- <新增的需求/接口/行为>

## MODIFIED
- <修改的现有需求>

## REMOVED
- <删除的需求>

## RENAMED
- <重命名的需求>
```

### 2.3 轻任务 spec（简版）

**proposal.md**（<20 行）：
```markdown
# Proposal: <task-name>

背景: <一句话>
目标: <一句话>
任务:
1. <步骤1>
2. <步骤2>
3. <步骤3>
```

---

## 步骤 3：Brainstorm（重任务强制）

### 3.1 重任务必须 brainstorm
- 列出**至少 3 种**实现方案
- 每种方案写明：实现思路 + 优点 + 缺点 + 风险
- 识别边界条件和潜在风险点
- 把对比结果写回 `proposal.md` 的"方案对比"和"风险点"章节

### 3.2 轻任务自决
Agent 自己判断要不要 brainstorm。简单 bug fix 直接修，复杂的还是 brainstorm。

### 3.3 多方案分歧时
如果存在 2+ 等价方案且用户没指定，**必须停下来用 AUQ 问用户**（见 SKILL.md 停问时机）。

---

## 步骤 4：TDD 写测试（RED）

详见 [references/tdd-rules.md](../references/tdd-rules.md)。

### 4.1 完整 TDD 流程（重任务 + bug fix）
1. **RED**：先写失败的测试
   - 新功能：写覆盖目标行为的测试
   - bug fix：先写**复现 bug 的测试**（必须失败，证明 bug 存在）
2. **GREEN**：写最小实现让测试通过
3. **REFACTOR**：重构，测试必须仍然通过

### 4.2 例外清单（不走 TDD）
- 文档改动（只改 .md / 注释）
- 配置改动（只改 .yml / .toml / .json 配置）
- 原型探索（用户明确说"先试一下""探索性"）
- 纯样式改动（格式化 / import 排序）

### 4.3 不过度 TDD
- **不给测试写测试**（不测 pytest / unittest 框架本身）
- **不给测试工具写测试**（不测 mock / fixture）
- 测试的测试 = 测试本身的质量靠 review，不靠递归测试

---

## 步骤 5：实现（GREEN）

### 5.1 防偷懒铁律（详见 references/anti-lazy.md）
- **不能空实现**：`pass` / `return None` / `// TODO` 不算实现
- **不能简化实现**：用户要 A，不能只给 A 的 30%
- **写不了打 TODO/FIXME**：说明原因和后续计划
  ```python
  # TODO: 需要 SMTP 凭据，等运维提供后补全
  # 计划: 在 T-05 接入真实 SMTP
  # 临时: 写入 outbox 表，由后台 worker 异步发送
  def send_email(to, subject, body):
      Outbox.create(to=to, subject=subject, body=body)
      logger.info(f"Email queued for {to}")
  ```
- **除非用户明确说要简化**："先 MVP""先简版""不用写完整"——否则一律完整实现

### 5.2 多方案处理
- brainstorm 选定的方案直接实现
- 实现中发现新方案，停下来用 AUQ 问用户

### 5.3 工具使用
- 改老文件用 `Edit`
- 建新文件用 `Write`
- 改之前先把"我打算这么改"用一两句话讲清楚
- Edit 找不到 old_string → 重新 Read 文件再试，不靠猜

---

## 步骤 6：重构（REFACTOR）

### 6.1 重构原则
- 测试必须仍然通过（每改一点就跑一次）
- 不改行为，只改结构
- 命名统一（snake_case / camelCase 别混）
- 消除重复（DRY）

### 6.2 重构边界
- 单个函数 >50 行 → 拆分
- 嵌套 >3 层 → 拍平
- 重复代码 >3 处 → 抽函数

---

## 步骤 7：验证

### 7.1 默认验证方式
1. **TDD 写了测试** → 跑相关测试 `pytest tests/test_xxx.py -v`
2. **没测试**（例外清单内的任务）→ `python -m py_compile xxx.py` 验语法
3. **用户要求才跑**集成测试 / e2e

### 7.2 验证失败处理
- 测试挂了 → 报告失败原因 + 怀疑点
- **调试循环超 2 次** → 停下来用 AUQ 问用户（见停问时机）
- py_compile 报语法错 → 直接修，不再问

### 7.3 测试指标自检
详见 [references/test-metrics.md](../references/test-metrics.md)。
- 多维覆盖率：行 / 分支 / 函数（不只看一个数字）
- 边界覆盖：空值 / 负数 / 超长 / 并发 是否有测试
- 断言质量：避免 `assert True` / 只测 happy path
- 关键路径覆盖：核心逻辑必须有测试

---

## 步骤 8：子代理审查（强制）

详见 [references/review-checklist.md](../references/review-checklist.md)。

### 8.1 开子代理
- **必须**开一个新的子代理过 checklist
- 子代理提示词必须严格约束（见 review-checklist.md 的提示词模板）
- **绝对不允许子代理再开子代理**（防递归）

### 8.2 并行任务验收
- 若主代理并行开了多个子代理执行任务
- 验收时主代理**另开一个独立子代理**验收
- 不能用执行任务的子代理自己验收自己的产出

### 8.3 审查结果处理
- 子代理发现问题 → 主代理修复 → 再开新子代理复审
- 子代理通过 → 进入步骤 9

---

## 步骤 9：Conventional Commit

### 9.1 Commit 格式
```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

type ∈ `feat / fix / refactor / test / docs / chore / perf / style / build / ci`

### 9.2 例子
```
feat(auth): add JWT login endpoint

- Add /api/auth/login POST handler
- Issue JWT token with 7-day expiry
- Add tests for valid/invalid credentials

Closes #123
```

### 9.3 注意
- **不主动 commit**，除非用户明确要求
- 用户要求 commit 时，按 Conventional Commits 格式写 message
- 用户没要求 commit 时，只把改动留在工作区，告诉用户改了什么

---

## 流程图

```
用户需求
   │
   ▼
[1] 识别任务类型 + 轻重分级
   │
   ▼
[2] 写 spec（重任务完整 / 轻任务简版）
   │
   ▼
[3] Brainstorm（重任务强制，轻任务自决）
   │   └─ 多方案分歧 → AUQ 问用户
   ▼
[4] TDD 写测试（RED）
   │   └─ 例外清单内任务跳过
   ▼
[5] 实现（GREEN）
   │   └─ 防偷懒铁律：不能空实现/简化实现
   ▼
[6] 重构（REFACTOR）
   │
   ▼
[7] 验证（跑测试 / py_compile）
   │   └─ 调试超 2 次 → AUQ 问用户
   ▼
[8] 子代理审查（强制）
   │   └─ 不通过 → 修复 → 复审
   ▼
[9] Conventional Commit（用户要求才 commit）
```
