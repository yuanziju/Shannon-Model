# MathMaster 设计文档 (新底子架构)

> MathMaster 是 30-70B MoE 数学专精模型, 基于"新底子"架构构建.
> 本文档描述架构设计、模块职责与组件复用关系.

---

## 1. 架构总览

```
输入 (input_ids)
  │
  ▼
[MathEncoder] ── 文本嵌入 + NSL符号嵌入 + 位置编码(LongRoPE2) + 1层Transformer
  │                  └─ DynamicAttentionController 路由 [MLA, Gated]
  │                  └─ SymbolNeuralBridge (符号-神经对齐, 可选 AST 输入)
  ▼
[MathRecurrentBody] ── Looped 循环主体 (1-32 次动态迭代)
  │  │
  │  │  每轮迭代包含 4 部分:
  │  │
  │  ├─ 1. ResidualPool (残差池)
  │  │      ├─ AttnRes (深度方向注意力残差, common.layers)
  │  │      ├─ mHC (流形约束超链接, Sinkhorn双随机, common.layers)
  │  │      ├─ attention 检索 "有用笔记" (query-key 相关性)
  │  │      ├─ 删除非AB残差 + 压缩 (线性投影)
  │  │      └─ 每 pool_compress_every 轮: 注意力索引 + top-k 筛选
  │  │
  │  ├─ 2. IntuitionLayer (直觉层, 基础版)
  │  │      ├─ 快通道: GatedRMSNorm + 多层 ExpertFFN (SwiGLU + down-proj)
  │  │      └─ 隐变量采样: VAE 风格 (mean + logvar → 重参数化)
  │  │
  │  ├─ 3. ABStack (10个AB固定堆叠)
  │  │      │
  │  │      └─ 每个 ABBlock 内部:
  │  │           ├─ FivePathAttention: 5路 Hybrid-M3 注意力 (A1-A5)
  │  │           │    └─ 从 8 种中选 5: MLA/KDA/Lightning/Sliding/MMA/MoH/Gated
  │  │           ├─ MetaRouter: 1对1置换 ("电线盒", Sinkhorn双随机矩阵)
  │  │           │    └─ 路径特征 → 代价矩阵 → Sinkhorn归一化 → 置换矩阵
  │  │           │    └─ 推理时可选匈牙利硬置换 (straight-through)
  │  │           ├─ SubAgent × 5 (G1-G5, 不同路由策略, 共享ExpertPool):
  │  │           │    ├─ G1: top-1 big + top-1 small (激进)
  │  │           │    ├─ G2: top-2 big + top-2 small
  │  │           │    ├─ G3: top-3 big + top-3 small
  │  │           │    ├─ G4: top-4 big + top-4 small (默认)
  │  │           │    └─ G5: top-4 big + top-4 small + NLM增强 (CTM)
  │  │           └─ ExpertPool (共享专家池):
  │  │                ├─ 6 常驻专家 (4固定 + 2可学习 EmptyExpert, 密集)
  │  │                │    └─ 2 可学习专家: 零初始化门控 + NLM增强 (NLMLayer)
  │  │                ├─ 16 大专家 (粗粒度, top-k路由, 稀疏)
  │  │                ├─ 16 小专家 (细粒度, top-k路由, 稀疏)
  │  │                └─ CTMRouter (复杂度驱动 NLM 增强开关)
  │  │
  │  │      AB之间: 固定1对1 (编号对编号, 不做路由)
  │  │
  │  ├─ 4. LoopControl (循环控制)
  │  │      ├─ 深度嵌入 (迭代索引 → embedding)
  │  │      ├─ AB输出 + 深度嵌入 + 残差 → 新hidden
  │  │      └─ ACT 自适应停止 (halt_prob, 累积阈值停止)
  │  │
  │  └─ CTM 集成:
  │       ├─ MLASync (c_kv·c_kv^T 同步矩阵, 迭代间潜变量同步)
  │       └─ CTMDynamicLoss (多 tick min-loss + max-certainty)
  │
  ▼
[MathDecoder] ── 多任务输出头
  │                  ├─ text_head      → 主文本 logits (vocab_size)
  │                  ├─ lean4_head     → Lean4 形式化 (nsl_vocab_size)
  │                  ├─ sympy_head     → SymPy 符号 (nsl_vocab_size)
  │                  ├─ conjecture_head → 猜想生成 (vocab_size)
  │                  ├─ proof_step_head → 证明步骤 (vocab_size)
  │                  ├─ confidence_head → 置信度标量 (per token)
  │                  └─ NSLDecoder (树结构符号解码, common.nsl)
  ▼
输出 dict: {logits, text_logits, lean4_logits, sympy_logits,
            conjecture_logits, proof_step_logits, confidence,
            aux_loss, kl_loss, ponder_loss, num_iterations, ...}
```

---

## 2. 模块详解

### 2.1 MathEncoder (编码器)

| 组件 | 说明 | 复用来源 |
|------|------|----------|
| token_embed | 文本嵌入 (vocab_size → hidden_dim) | nn.Embedding |
| symbol_embed | NSL 符号嵌入 (nsl_vocab_size → hidden_dim, 门控融合) | nn.Embedding |
| pos_encoding | LongRoPE2 位置编码 (支持 1M-10M 上下文) | common.layers |
| symbol_bridge | 符号-神经对齐 (AST编码 + InfoNCE) | common.nsl |
| grammar / parser | NSL 文法 + 形式化解析器 | common.nsl |
| attn_controller | DynamicAttentionController 路由 [MLA, Gated] | common.attention |
| ffn | ExpertFFN (SwiGLU + down-proj) | common.layers (SwiGLU) |

编码流程: `input_ids → token_embed + gate·symbol_embed → LongRoPE2 → 1层Transformer → hidden`

### 2.2 MathRecurrentBody (循环主体)

#### 2.2.1 ResidualPool (残差池)

每轮迭代管理残差池, 仅保留 AB 残差:

1. **AttnRes 聚合**: 将池中残差堆叠, 用注意力权重聚合 (common.layers.AttnRes)
2. **mHC 约束**: 流形约束超链接, Sinkhorn 双随机矩阵保证信号不爆炸 (common.layers.mHC)
3. **attention 检索**: 用 hidden 查询池中残差, 取相关性加权的"有用笔记"
4. **删除压缩**: 线性投影压缩残差表示
5. **每 N 轮 top-k**: 按 gate 分数筛选最有用的 `pool_topk` 个残差

#### 2.2.2 IntuitionLayer (直觉层, 基础版)

> 完整版待后续完善. 当前基础版:

- **快通道**: GatedRMSNorm → 多层 ExpertFFN (SwiGLU + down-proj), 提供快速直觉响应
- **隐变量采样**: VAE 风格重参数化 (mean + eps·std), 引入随机直觉
- **融合**: `hidden + dropout(fast) + dropout(latent)`
- **KL 损失**: 标准正态先验下的 KL 散度

#### 2.2.3 ABStack (AB堆叠)

**ABBlock 内部流程**:

```
输入 x + paths (来自上一AB)
  │
  ├─ FivePathAttention(x) → 5路注意力输出 [A1..A5]
  │    └─ 每路: RMSNorm → 不同Hybrid-M3注意力类型
  │
  ├─ 残差融合: attended[i] = norm(paths[i] + attn_out[i])
  │
  ├─ MetaRouter(attended) → perm [b, 5, 5] (Sinkhorn双随机)
  │
  ├─ 应用置换: permuted = perm @ attended (路径→子agent)
  │
  ├─ SubAgent[i](permuted[i], expert_pool) → processed[i]  (i=0..4)
  │    └─ 各子agent用不同top-k策略, 共享同一ExpertPool
  │
  ├─ 逆置换: output_paths = perm^T @ processed (子agent→路径)
  │
  └─ 返回 new_paths (传给下一AB, 固定1对1)
```

**MetaRouter (1对1置换)**:
- 路径池化 → 投影 → 与子agent键点积 → 代价矩阵 [b, n, n]
- Sinkhorn 归一化 (log空间, 10次迭代) → 双随机矩阵
- 推理时可选匈牙利硬置换 (straight-through 估计器保持可微)

**ExpertPool (共享专家池)**:

| 专家类型 | 数量 | 路由 | 说明 |
|----------|------|------|------|
| 固定常驻 | 4 | 密集 (始终开启) | ExpertFFN (SwiGLU + down) |
| 可学习常驻 | 2 | 密集 (始终开启) | EmptyExpert (零门控, NLM增强) |
| 大专家 | 16 | top-4 稀疏 | ExpertFFN (粗粒度, inter=4d) |
| 小专家 | 16 | top-4 稀疏 | ExpertFFN (细粒度, inter=d) |

- **EmptyExpert**: 参考 Shannon 设计, 零初始化 down 投影 + 零门控, 持续学习阶段逐步填充
- **NLMLayer**: 神经元级模型, 增强 2 可学习专家的激活函数 (common.ctm)
- **CTMRouter**: 复杂度驱动, 控制是否启用 NLM 增强 (common.ctm)
- **负载均衡损失**: 标准 MoE aux loss (frac_tokens × mean_prob)

**SubAgent 路由策略**:

| 子agent | top_k_big | top_k_small | NLM | 策略 |
|---------|-----------|-------------|-----|------|
| G1 | 1 | 1 | ✗ | 激进/最小路由 |
| G2 | 2 | 2 | ✗ | 中等路由 |
| G3 | 3 | 3 | ✗ | 较全路由 |
| G4 | 4 | 4 | ✗ | 默认/全路由 |
| G5 | 4 | 4 | ✓ | 全路由 + CTM增强 |

#### 2.2.4 LoopControl (循环控制)

- **深度嵌入**: 迭代索引 → embedding, 注入位置信号
- **融合**: `new_hidden = norm(ab_out + depth_emb + hidden)`
- **ACT 停止**: `halt_prob = sigmoid(halt_proj(new_hidden).mean())`
  - 累积 halt_prob, 超过阈值且达到最小迭代数时停止
  - ponder_loss 鼓励早停

#### 2.2.5 CTM 集成

| 组件 | 用途 | 复用来源 |
|------|------|----------|
| MLASync | c_kv·c_kv^T 同步矩阵, 迭代间潜变量同步 | common.ctm |
| CTMDynamicLoss | 多 tick min-loss + max-certainty (有标签时) | common.ctm |
| NLMLayer | 神经元级模型, 增强可学习专家激活 | common.ctm |
| CTMRouter | 复杂度驱动路由 | common.ctm |

### 2.3 MathDecoder (解码器)

多任务输出头, 支持数学推理的多种输出形式:

| 输出头 | 维度 | 用途 |
|--------|------|------|
| text_logits | vocab_size | 主文本输出 |
| lean4_logits | nsl_vocab_size | Lean4 形式化证明 |
| sympy_logits | nsl_vocab_size | SymPy 符号计算 |
| conjecture_logits | vocab_size | 数学猜想生成 |
| proof_step_logits | vocab_size | 证明步骤 |
| confidence | 1 (per token) | 置信度标量 |

NSLDecoder (common.nsl) 提供树结构符号解码能力.

---

## 3. 组件复用关系

所有公共组件从 `common/` 导入复用, 不重新实现:

### common.attention (Hybrid-M3 8种注意力)
- `MLAAttention` — DeepSeek 多头潜注意力 (编码器 + AB五路)
- `KDAAttention` — Kimi Delta 注意力 (AB五路)
- `LightningAttention` — Lightning 注意力 (AB五路)
- `SlidingWindowAttention` — 滑窗注意力 (AB五路)
- `MMAAttention` — 模态互注意力 (AB五路)
- `MoHAttention` — 混合头注意力 (AB五路)
- `GatedAttention` — 秩64门控注意力 (编码器 + AB五路)
- `DynamicAttentionController` — 动态注意力路由 (编码器)

### common.layers (基础层)
- `RMSNorm` / `GatedRMSNorm` — 归一化 (全局)
- `SwiGLU` — 门控线性单元 (ExpertFFN 内部)
- `RoPE` / `YaRN` — 旋转位置编码 (备选)
- `LongRoPE2` — 超长上下文位置编码 (编码器, 1M-10M)
- `AttnRes` — 深度方向注意力残差 (残差池)
- `mHC` — 流形约束超链接 (残差池)

### common.ctm (CTM 集成)
- `NLMLayer` — 神经元级模型 (ExpertPool 可学习专家增强)
- `MLASync` — c_kv 同步矩阵 (循环主体潜变量同步)
- `CTMDynamicLoss` — 动态损失 (多 tick 训练)
- `CTMRouter` — 复杂度驱动路由 (ExpertPool)

### common.nsl (神经语系统)
- `SymbolNeuralBridge` — 符号-神经双向翻译 (编码器)
- `NSLGrammar` — 符号文法 (编码器)
- `FormalParser` — 形式化解析器 (编码器)
- `NSLDecoder` — 树结构符号解码 (解码器)

---

## 4. 配置 (MathConfig)

### 4.1 模型主体 (70B 参考)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| model_name | "MathMaster-70B" | 模型名 |
| vocab_size | 128000 | 词表大小 |
| hidden_dim | 8192 | 隐藏维度 |
| num_layers | 40 | 层数 |
| num_heads | 64 | 注意力头数 |
| head_dim | 128 | 每头维度 |
| num_kv_heads | 16 | KV头数 (GQA) |
| max_seq_len | 1,000,000 | 最大序列长度 |

### 4.2 循环深度

| 参数 | 默认值 | 说明 |
|------|--------|------|
| dynamic_iterations | (1, 32) | 动态迭代范围 |
| silent_thinking | True | 静默思考 |
| act_halting_threshold | 0.95 | ACT 停止阈值 |

### 4.3 AB堆叠

| 参数 | 默认值 | 说明 |
|------|--------|------|
| num_ab_blocks | 10 | AB块数 |
| num_attention_paths | 5 | 注意力路数 (A1-A5) |
| num_sub_agents | 5 | 子agent数 (G1-G5) |

### 4.4 专家池

| 参数 | 默认值 | 说明 |
|------|--------|------|
| num_resident_experts | 6 | 常驻专家总数 |
| num_fixed_resident_experts | 4 | 固定常驻 |
| num_learnable_resident_experts | 2 | 可学习常驻 (EmptyExpert) |
| num_big_experts | 16 | 大专家数 |
| num_small_experts | 16 | 小专家数 |
| top_k_big | 4 | 大专家 top-k |
| top_k_small | 4 | 小专家 top-k |

### 4.5 其他配置组

- **残差池**: pool_topk(32), pool_compress_every(3), use_attn_res, use_mhc
- **直觉层**: intuition_hidden_dim, intuition_num_layers(2), intuition_latent_dim
- **注意力**: attention_types (Hybrid-M3 8种), rope_theta
- **NSL**: nsl_vocab_size(8192), nsl_num_layers(2), nsl_max_nodes(256)
- **CTM**: ctm_num_neurons(8), ctm_d_state(16), ctm_warmup_freeze
- **位置编码**: pos_encoding("longrope2"), pos_original_max(8192)
- **形式化**: formal_backend("lean4")
- **训练**: dropout, rms_eps, aux_loss_weight, kl_loss_weight, ponder_loss_weight
- **评估**: eval_confidence_threshold, eval_benchmarks
- **输出任务**: text/lean4/sympy/conjecture/proof_step/confidence
- **领域权重**: algebra/analysis/geometry/number_theory/topology/logic/probability/combinatorics
- **元路由器**: meta_router_sinkhorn_iters(10), meta_router_hard_perm

### 4.6 序列化

- `to_dict()`: 转为可序列化字典
- `from_dict(d)`: 从字典构造 (忽略未知字段, list→tuple 自动转换)
- `__post_init__`: 派生默认值 (moe_inter_dim, intuition_hidden_dim) + 一致性校验

---

## 5. 文件结构

```
models/MathMaster/
├── __init__.py
├── src/
│   ├── __init__.py              ← 导出 MathConfig, MathModel
│   ├── config/
│   │   ├── __init__.py          ← 导出 MathConfig
│   │   └── config.py            ← MathConfig 数据类
│   └── model/
│       ├── __init__.py          ← 导出所有模型组件
│       └── model.py             ← 完整模型实现
└── docs/
    └── design.md                ← 本文档
```

### model.py 导出的类

| 类 | 说明 |
|----|------|
| `MathModel` | 完整模型 (编码器→循环主体→解码器) |
| `MathEncoder` | 编码器 |
| `MathRecurrentBody` | 循环主体 |
| `MathDecoder` | 多任务解码器 |
| `ResidualPool` | 残差池 |
| `IntuitionLayer` | 直觉层 (基础版) |
| `ABStack` | AB堆叠 (10个AB) |
| `ABBlock` | 单个AB块 |
| `FivePathAttention` | 五路注意力 |
| `MetaRouter` | 元路由器 (1对1置换) |
| `SubAgent` | 子agent (路由策略) |
| `ExpertPool` | 共享专家池 |
| `LoopControl` | 循环控制 |
| `ExpertFFN` | 专家FFN (SwiGLU + down-proj) |
| `EmptyExpert` | 空专家 (零初始化, Shannon风格) |

---

## 6. 设计决策

### 6.1 MetaRouter 1对1置换

采用 **Sinkhorn 归一化** 实现可微的软置换:
- 路径池化特征 → 投影 → 代价矩阵 [b, n, n]
- log 空间 Sinkhorn 迭代 (行+列归一化) → 双随机矩阵
- 双随机矩阵的谱范数 ≤ 1, 保证数值稳定
- 逆置换 = 转置 (双随机矩阵性质)
- 推理时可选 **匈牙利硬置换** (straight-through 估计器保持反向传播)

### 6.2 EmptyExpert (可学习常驻专家)

参考 Shannon 的 EmptyExpert 设计:
- SwiGLU + **零初始化 down 投影** + **零初始化标量门控**
- 初始贡献为 0, 不影响模型输出
- 持续学习阶段逐步填充 (gate 和 down 权重逐渐学习)
- 配合 NLMLayer 实现 CTM 神经元级增强

### 6.3 双层 MoE (16大×16小)

- **大专家** (粗粒度): inter_dim = 4 × hidden_dim, 处理粗粒度模式
- **小专家** (细粒度): inter_dim = hidden_dim, 处理细粒度模式
- 各自独立 top-k 路由 (top_k_big=4, top_k_small=4)
- 负载均衡损失防止专家过载/饥饿

### 6.4 ACT 自适应停止

- 每轮迭代计算 halt_prob = sigmoid(halt_proj(hidden).mean())
- 累积 halt_prob, 超过阈值 (0.95) 且达到最小迭代数时停止
- ponder_loss 鼓励早停 (加权迭代索引)
- silent_thinking: 中间迭代不产生输出, 仅最终迭代解码

### 6.5 CTM 集成

- **NLMLayer**: 仅增强 2 可学习专家的激活 (决策 C7: 不主导状态转移)
- **MLASync**: 复用 MLA 潜变量 c_kv (决策 C5: 不引入独立模块)
- **CTMRouter**: 复杂度驱动 NLM 增强开关 (决策 C10: 仅实体专家用 NLM)
- **CTMDynamicLoss**: 多 tick min-loss + max-certainty (有标签时计算)
