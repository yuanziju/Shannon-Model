# MathMaster v2 架构设计稿

> **版本**: v2.0-draft  
> **目标**: 数学专精多模态模型，文本 / 图像 / PDF 输入，自然语言 / 形式化证明 / 符号计算 / 猜想与反例输出  
> **架构底色**: MoE + 循环深度 + 神经符号混合（NSL）  
> **设计原则**: 复用 Shannon 公共底子，旧 MathMaster v1 代码保留，v2 另起设计稿  

---

## 1. 设计目标与范围

### 1.1 目标

MathMaster v2 是一个**面向数学推理的专用模型**，在 Shannon 通用多模态模型底子上做数学特化：

- **输入模态**（仅 3 种）：纯文本、图像、PDF。
- **输出形式**（4 类）：
  1. 自然语言解答（含步骤、解释）。
  2. 形式化证明（Lean4 / Coq）。
  3. 符号计算（SymPy / 可执行代码）。
  4. 猜想与反例（由外部工具推导/搜索，模型负责调用与整理）。

### 1.2 与 Shannon、旧 MathMaster 的关系

```
Shannon 15B MoE（通用底子）
    ├── common/          ← 公共组件库，v2 直接复用并补充
    │     ├── attention/      Hybrid-M3 注意力族
    │     ├── layers/         RMSNorm/SwiGLU/RoPE/AttnRes/mHC...
    │     ├── moe/            双层 MoE / 空专家 / 负载均衡
    │     ├── ctm/            NLM / MLA 同步 / CTM 动态损失
    │     ├── nsl/            神经语系统
    │     ├── latent_decode/  B+C 融合隐空间解码 / 拟人流式
    │     ├── sre/            特化推理引擎 / ToolCoordinator
    │     └── agent/          ReAct+CRA / AgentRuntime
    │
    └── models/MathMaster/    ← 旧 v1 保留
              ├── docs/design.md         v1 设计稿（保留）
              └── docs/design_v2.md      ← 本文档
```

- **v1 保留**：`models/MathMaster/src/` 下旧代码不动，作为遗产和可复用组件来源。
- **v2 复用 Shannon 底子**：Encoder–RecurrentBody–Decoder 结构、双层 MoE、Hybrid-M3、NSL、B+C 融合解码。
- **v2 新增数学特化层**：ABStack（3 层）、ResidualPool、数学工具调用头、双轨验证接口。

---

## 2. 架构总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           输入层（3 模态）                                      │
│  纯文本 ──┐                                                                   │
│  图像   ──┼──→ 复用 Shannon Encoder（ViT+Q-Former+VAE / PDF 双通道解析）        │
│  PDF    ──┘                                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MathEncoder（数学编码器）                              │
│  ├── 文本嵌入 + NSL 符号嵌入（门控融合）                                         │
│  ├── 位置编码（LongRoPE2 / YaRN，支持长上下文）                                  │
│  ├── 符号↔神经对齐（SymbolNeuralBridge）                                       │
│  └── 1 层 Transformer 做初始投影                                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MathRecurrentBody（循环主体）                           │
│                                                                             │
│   每轮迭代（1–32 次，ACT 自适应停止）：                                          │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ 1. ResidualPool（残差池）                                            │   │
│   │    ├── AttnRes + mHC                                                │   │
│   │    ├── attention 检索“有用笔记”                                       │   │
│   │    ├── 删除非 AB 残差 + 压缩                                          │   │
│   │    └── 每 N 轮 top-k 筛选                                             │   │
│   ├─────────────────────────────────────────────────────────────────────┤   │
│   │ 2. ABStack（3 层，负责广度）                                          │   │
│   │    ├── FivePathAttention：5 路 Hybrid-M3 注意力（复用清理后的实现）      │   │
│   │    ├── MetaRouter：1 对 1 置换（Sinkhorn / 匈牙利）                    │   │
│   │    ├── 5 个子 agent：路由策略让模型自己学                               │   │
│   │    └── ExpertPool：Shannon 双层 MoE + MathMaster 专家池融合             │   │
│   ├─────────────────────────────────────────────────────────────────────┤   │
│   │ 3. RDT Core（Shannon 循环块，负责深度）                                 │   │
│   │    ├── Hybrid-M3 注意力                                              │   │
│   │    ├── 双层 MoE FFN                                                  │   │
│   │    ├── CTM 集成（MLA 潜变量同步、NLM 增强）                             │   │
│   │    └── LTI / ACT / 深度嵌入                                           │   │
│   ├─────────────────────────────────────────────────────────────────────┤   │
│   │ 4. LoopControl（循环控制）                                             │   │
│   │    ├── 深度嵌入（迭代索引）                                            │   │
│   │    ├── AB 输出 + 残差 + hidden 融合                                   │   │
│   │    └── ACT 自适应停止                                                 │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MathDecoder（数学解码器）                             │
│  ├── B+C 融合隐空间解码（层次化 NAR + 掩码精化 + 流匹配可选 + AR 保底）          │
│  ├── 拟人流式输出前端（删除重打 + 延迟，修订率上限 15%）                         │
│  ├── text_head：自然语言主通道                                                 │
│  └── tool_call_head：输出 <ACTION>{...}</ACTION> JSON，服务解析后调用执行器       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         输出后处理 / 双轨验证                                   │
│  ├── 自然语言：直接输出                                                        │
│  ├── 形式化证明 / 符号计算：本地执行器草稿验证 → 外部严格验证器二次确认            │
│  └── 猜想 / 反例：调用外部生成器/搜索器，结果回灌模型整理                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 输入处理

仅支持 3 种输入模态，**直接复用 Shannon 的编码器**，不在 v2 中重新实现视觉/文档编码。

| 模态 | 处理方式 | 复用来源 |
|------|----------|----------|
| 纯文本 | BPE token 嵌入 + NSL 符号嵌入门控融合 | `common.layers` / `models/Shannon/src/encoder/text_embed.py` |
| 图像 | ViT patch + Q-Former 查询压缩 + VAE latent 双通道 | `models/Shannon/src/encoder/image_encoder.py` |
| PDF | 双通道：视觉渲染 + 结构提取，输出统一 token 序列 | `models/Shannon/src/encoder/doc_parser.py` |

所有模态最终转换为同一隐藏空间的 token 序列，送入 `MathEncoder`。

---

## 4. MathEncoder

在 Shannon 编码器基础上增加数学特化：

1. **文本嵌入** + **NSL 符号嵌入**：
   - `token_embed`：普通词表 embedding。
   - `symbol_embed`：NSL 符号词表 embedding。
   - `symbol_gate`：可学习门控，初始为 0，逐步引入符号信息。

2. **位置编码**：
   - 默认 `LongRoPE2`，支持 1M–10M 上下文。
   - 备选 `YaRN` / 标准 `RoPE`。

3. **一层 Transformer**：
   - 做模态对齐与初始投影。
   - 使用 `DynamicAttentionController` 在 `MLA` / `Gated` 之间路由。

4. **NSL 桥接**：
   - `SymbolNeuralBridge` 对可选 AST 输入做符号↔神经对齐。
   - 输出 `infonce_loss` 用于符号空间对齐训练。

---

## 5. MathRecurrentBody：深度 + 广度 + 记忆

循环主体是 v2 的核心创新区，**深度由 Shannon RDT 提供，广度由 ABStack 提供，记忆由 ResidualPool 管理**。

### 5.1 每轮迭代流程

```
hidden_t
   │
   ▼
┌───────────────────┐
│  ResidualPool     │  ← 检索/压缩/管理跨轮次记忆
└─────────┬─────────┘
          ▼
┌───────────────────┐
│  ABStack (3 层)   │  ← 5 路注意力 + 元路由 + 子 agent + 专家池
└─────────┬─────────┘
          ▼
┌───────────────────┐
│  RDT Core         │  ← Shannon 循环块（深度推理）
└─────────┬─────────┘
          ▼
┌───────────────────┐
│  LoopControl      │  ← 深度嵌入 + ACT 停止
└─────────┬─────────┘
          ▼
     hidden_{t+1}
```

### 5.2 ResidualPool（残差池）

跨轮次保存“有用笔记”：

- 只保留 **AB 输出残差**。
- 用 `AttnRes` + `mHC` 聚合历史残差。
- 用当前 `hidden` 做 query，检索相关性高的残差。
- 每 `pool_compress_every` 轮做一次 top-k 压缩。

### 5.3 ABStack（3 层，负责广度）

旧 MathMaster v1 的 10 层 ABStack 缩减为 **3 层**，内部结构保留：

- **FivePathAttention**：5 路注意力并行。
  - 注意力类型**直接复用清理后的 `common.attention` Hybrid-M3**。
  - 当前暂定：`MLA` + `KDA` + `MMA` + 2 路待清理后从 Hybrid-M3 中选取（设计稿迭代时填入）。
- **MetaRouter**：1 对 1 置换，将 5 路输出置换到 5 个子 agent。
  - 训练：Sinkhorn 双随机软置换。
  - 推理：可选匈牙利硬置换 + straight-through。
- **SubAgent × 5**：
  - 路由策略**让模型自己学**，不再手工固定 top-k。
  - 每个子 agent 有独立的路由参数，共享同一个 `ExpertPool`。
- **ExpertPool**：
  - 融合 Shannon 双层 MoE（16 大 × 16 小，Top-2~4）与 MathMaster v1 的常驻专家池。
  - 常驻专家负责通用/数学基础能力。
  - 大/小专家负责粗/细粒度稀疏路由。
  - 自学习空专家用于持续吸收新数学能力。

### 5.4 RDT Core（Shannon 循环块，负责深度）

复用 `models/Shannon/src/recurrent/body.py` 的 RDT 循环块：

- `Hybrid-M3` 注意力动态调度。
- 双层 MoE FFN。
- CTM 集成：`MLASync` 潜变量同步、`NLMLayer` 增强、`CTMDynamicLoss`。
- LTI 稳定性约束、ACT 自适应停止、深度嵌入。

### 5.5 LoopControl

- 根据当前迭代索引注入深度嵌入。
- 融合 AB 输出、残差池输出、上一 hidden。
- 计算 ACT 停止概率，动态决定 1–32 次迭代。

---

## 6. MoE 设计：两个都要

v2 同时使用 Shannon 双层 MoE 和 MathMaster 专家池，**不冲突，互补**：

```
ExpertPool
├── Shannon 双层 MoE
│     ├── 16 大专家（粗粒度，inter = 4d，Top-k）
│     ├── 16 小专家（细粒度，inter = d，Top-k）
│     ├── 常驻共享专家（DeepSeek 模式）
│     └── 自学习空专家（零初始化，持续填充）
│
└── MathMaster v1 常驻专家池
      ├── 4 固定常驻专家（密集，始终开启）
      └── 2 可学习常驻专家（EmptyExpert + NLM 增强）
```

- 子 agent 路由输出先经 Shannon 双层 MoE，再与常驻专家池输出门控融合。
- 负载均衡损失同时作用于大/小专家路由。

---

## 7. NSL 全链路集成

神经语系统（Neuro-Symbolic Language）贯穿编码器–循环主体–解码器：

| 位置 | 作用 |
|------|------|
| **编码器** | `SymbolNeuralBridge` 将数学符号、公式 AST、Lean/SymPy AST 映射到神经隐空间。 |
| **循环主体** | 每轮迭代维护一个 `symbol_state`，与 `hidden_state` 并行更新，支持符号推理的中间状态传递。 |
| **解码器** | `NSLDecoder` 将隐状态解码为符号树，再映射为 Lean / Coq / SymPy / Python 代码。 |

NSL 不替代自然语言通道，而是作为**符号形式的专业通道**，与工具调用协同工作。

---

## 8. MathDecoder：B+C 融合 + 拟人流式

### 8.1 保留 Shannon 的 B+C 融合隐空间解码

```
循环主体输出 hidden
   │
   ▼
┌─────────────────────────────────────┐
│  B+C 融合解码器                       │
│  ├── 层次化 NAR（段落→句子→token）      │
│  ├── 掩码精化（复用 RDT 权重）           │
│  ├── 流匹配全局规划（可选）              │
│  └── AR 保底通道（三级置信度门控）        │
└─────────────────────────────────────┘
```

- 自然语言解答走 B+C 融合主通道。
- 形式化证明、符号计算等**强制走 AR 保底 + Lean 验证器**（决策 L4）。

### 8.2 拟人流式输出

- 删除重打 + 思考停顿。
- 修订率上限 15%。
- token 间延迟 50–200 ms。

### 8.3 输出头

v2 仅保留两个输出头：

| 输出头 | 作用 |
|--------|------|
| `text_head` | 自然语言主通道，输出解答文本。 |
| `tool_call_head` | 输出 `<ACTION>{"tool": ..., "args": ...}</ACTION>`，由推理服务解析为 JSON 并调用执行器。 |

猜想、反例、形式化证明、符号计算全部通过 `tool_call_head` 表达。

---

## 9. 输出形式与工具调用

### 9.1 四种输出形式的技术路径

| 输出形式 | 生成方式 | 验证方式 |
|----------|----------|----------|
| 自然语言解答 | `text_head` 直接生成 | 人工/自动评估可读性、正确性 |
| 形式化证明（Lean4 / Coq） | `tool_call_head` 调用 prover | 本地执行器草稿验证 → 外部严格验证器 |
| 符号计算（SymPy / 可执行代码） | `tool_call_head` 调用 solver | 本地执行器草稿验证 → 外部严格验证器 |
| 猜想与反例 | `tool_call_head` 调用 conjecture_generator / counterexample_search | 外部生成器/搜索器推导，模型整理 |

### 9.2 工具调用格式

采用 **Function Calling JSON**，但在自然语言通道中以 `<ACTION>` token 嵌入：

```text
首先，我们对方程进行因式分解。
<ACTION>{"tool": "sympy.solve", "args": {"expr": "x**2 - 5*x + 6"}}</ACTION>
得到解 [2, 3]。
```

推理服务层解析 `<ACTION>...</ACTION>`，执行后把结果以 observation token 回灌。

---

## 10. 双轨验证：内部草稿 + 外部验证

### 10.1 内部草稿验证

- 循环主体生成形式化/符号草稿时，**同步调用本地执行器**（Lean/SymPy/Python）。
- 执行结果（成功/失败/错误信息）以 observation token 回灌模型。
- 失败时模型自动重试 1–3 次，修正后重新生成。

### 10.2 外部严格验证

- 最终输出提交给外部严格验证器：
  - Lean4 / Coq 形式化编译器。
  - SymPy 符号验证。
  - Python 沙箱执行。
- 验证通过才返回给用户；不通过时模型进入反思/修正循环。

---

## 11. 训练策略

### 11.1 训练起点

**从零独立训练**。不依赖 Shannon checkpoint 热启动，但复用 Shannon 的训练基础设施：

- 5D 并行（TP + PP + DP + SP + EP）。
- BF16 混合精度。
- 多优化器组合（SAGE / Muon / AdEMAMix / SCALE）。

### 11.2 六阶段训练流程

MathMaster v2 沿用 Shannon 的 6 阶段流程，但数据和奖励函数数学特化：

1. **预训练**：数学/理科/代码/文档数据，动态配比。
2. **中间训练**：Silent Thinking + 动态循环深度 + 自学习空专家激活。
3. **SFT**：数学指令、多模态问题、CRA 格式工具调用数据。
4. **对齐**：DPO / GRPO，工具使用正确性奖励 + 证明/计算验证奖励。
5. **持续学习**：在线数学数据接入，空专家持续填充。
6. **自我进化**：Self-Play 解题对弈 + 经验回放强化。

### 11.3 数据配比（数学特化）

| 数据类型 | 比例 | 说明 |
|----------|------|------|
| 数学文本/书籍 | 30% | 教材、论文、竞赛题解 |
| 代码/执行反馈 | 25% | SymPy/Python/Lean 代码及执行结果 |
| 理科（物理/化学/CS 理论） | 15% | 保持数理逻辑迁移能力 |
| 多模态（图文公式/PDF） | 15% | 几何图、公式截图、论文 PDF |
| 通用/语言 | 15% | 防止通用能力退化 |

---

## 12. 文件结构

```
models/MathMaster/
├── __init__.py
├── docs/
│   ├── design.md        ← v1 旧设计稿（保留）
│   └── design_v2.md     ← 本文档
└── src_v2/              ← v2 新代码目录（后续实现）
    ├── __init__.py
    ├── config/
    │   └── config.py         MathMasterV2Config
    ├── encoder/
    │   └── encoder.py        MathEncoder（复用 Shannon + 数学特化）
    ├── recurrent/
    │   ├── residual_pool.py  ResidualPool
    │   ├── abstack.py        ABStack / ABBlock / MetaRouter / SubAgent
    │   ├── expert_pool.py    融合专家池
    │   └── body.py           MathRecurrentBody（RDT + ABStack + ResidualPool）
    ├── decoder/
    │   └── decoder.py        MathDecoder（B+C 融合 + 拟人流式 + tool_call_head）
    ├── nsl/
    │   └── math_nsl.py       数学 NSL 适配层
    ├── tools/
    │   ├── registry.py       工具注册表
    │   ├── verifier.py       双轨验证接口
    │   └── executor.py       本地执行器封装
    └── training/
        └── math_trainer_v2.py  v2 训练器
```

---

## 13. 依赖项与前置任务

v2 实现前需要完成以下公共组件清理/补充：

| # | 前置任务 | 说明 | 位置 |
|---|----------|------|------|
| 1 | **重构 Hybrid-M3 注意力族** | 清理命名、对齐设计初衷，ABStack 直接复用清理后的实现 | `common/attention/` |
| 2 | 统一 `ExpertPool` 接口 | 让 Shannon 双层 MoE 与 MathMaster 常驻专家池能无缝融合 | `common/moe/` / `models/MathMaster/src_v2/recurrent/expert_pool.py` |
| 3 | 工具调用数据格式 | 定义 `<ACTION>{...}</ACTION>` 的训练数据转换和推理解析流程 | `common/agent/react_cra.py` / `common/sre/coordinator.py` |
| 4 | 数学验证器封装 | 本地 Lean/SymPy/Python 执行器 + 外部严格验证服务接口 | `models/MathMaster/src_v2/tools/` |

---

## 14. 设计决策记录

| 决策 | 选择 | 理由 |
|------|------|------|
| 骨架 | 复用 Shannon Encoder–RDT–Decoder | 基础设施成熟，训练代码可复用 |
| 深度 | RDT 1–32 次动态迭代 | 循环深度负责深度推理 |
| 广度 | ABStack 3 层 | 旧 v1 10 层缩减为 3 层，专注知识面广度 |
| 注意力 | 复用清理后的 Hybrid-M3 | 用户确认当前 Hybrid-M3 需重构，重构后 ABStack 直接复用 |
| MoE | Shannon 双层 + MathMaster 专家池融合 | 两者互补，都要 |
| 输出 | text_head + tool_call_head | 符号形式走工具调用，自然语言主通道 |
| 解码 | 保留 B+C 融合 + 拟人流式 | 自然语言质量与拟人输出体验 |
| 训练 | 从零独立训练 | 用户明确 |
| 验证 | 内部草稿 + 外部严格验证 | 双轨保证可靠性 |
| NSL | 全链路集成 | 编码器/循环主体/解码器都参与符号推理 |

---

## 15. 下一步工作

1. 重构 `common/attention/` Hybrid-M3，明确 8 种注意力的命名、职责和接口。
2. 确定 ABStack 5 路注意力的最终清单（待 Hybrid-M3 清理后填入）。
3. 设计 `tool_call_head` 的输出格式与训练数据构造流程。
4. 实现 `MathEncoder` / `MathRecurrentBody` / `MathDecoder` 的 v2 骨架代码。
5. 构建数学双轨验证工具链（本地执行器 + 外部验证服务）。
