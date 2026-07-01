# λ-Oracle 技术规格书 (Spec v2.1)

> **版本**: v2.1-final  
> **状态**: 基于17轮深度问答(68+问题) + 子agent调研完成 + 引擎架构深化  
> **最后更新**: 2026-06-27

---

## 目录

1. [架构总览与哲学](#1-架构总览与哲学)
2. [OpenMythos循环深度骨干](#2-openmythos循环深度骨干)
3. [嵌套MoE设计](#3-嵌套moe设计)
4. [混合注意力融合系统](#4-混合注意力融合系统)
5. [模态输入处理](#5-模态输入处理)
6. [输出模态与解码](#6-输出模态与解码)
7. [特化推理引擎](#7-特化推理引擎)
8. [推理引擎架构](#8-推理引擎架构)
9. [训练引擎架构](#9-训练引擎架构)
10. [CANN适配与优化](#10-cann适配与优化)
11. [评估体系](#11-评估体系)
12. [部署架构](#12-部署架构)
13. [附录: 设计决策汇总](#13-附录设计决策汇总)

---

## 1. 架构总览与哲学

### 1.1 设计哲学

λ-Oracle 采用**第一性原理**设计，融合三大前沿架构：

1. **OpenMythos 循环深度** —— Prelude→Recurrent Block→Coda，LTI稳定性保证
2. **嵌套MoE** —— 16×16专家双层路由，256小专家激活16个
3. **Transfusion/Neo-Unify 原生统一** —— 文本+图像+视频在同一token空间

并引入六大创新增强：
- **动态注意力权重生成** —— 小型交叉注意力网络学习最优注意力模式
- **特化推理引擎** —— 工具输出作为原生神经元输入
- **Kimi AttnRes** —— 深度方向注意力残差
- **DeepSeek mHC** —— 流形约束超链接
- **原生任意分辨率图像** —— 不resize，Native-RoPE编码
- **压缩视频记忆** —— SSM+可学习记忆token+关键事件摘要

### 1.2 架构层次

```
Layer 5: 应用接口层
  ├── Function Calling API (通用+深度推理双模式)
  ├── 流式对话接口 (token-by-token)
  └── 批量推理接口

Layer 4: 推理引擎层
  ├── 统一注意力内核层 (MLA/KDA/Lightning/Sliding/SSM动态调度)
  ├── 请求调度器 (连续批处理 + 优先级队列 + 多模态预处理)
  ├── 嵌套MoE推理优化 (双层路由 + 专家预加载)
  ├── 工具与图像执行层 (SymPy/Lean/Python + OpenCV/SAM2/ComfyUI/SVG)
  ├── 量化与压缩引擎 (FP16/BF16/INT8/W4A16/W8A8)
  └── 显存管理 (Paged KV Cache + SSM State Swap + 选择性重计算)

Layer 3: 模型核心层
  ├── Prelude (2层标准Transformer, 运行一次)
  ├── Recurrent Block × 6循环 (8 Unique Cells)
  │   ├── DeepSeek MLA注意力
  │   ├── 动态注意力控制器 (交叉注意力网络)
  │   ├── 嵌套MoE FFN (16×16专家, Top-4×Top-4)
  │   ├── LTI稳定性约束 (谱半径<1)
  │   ├── 循环索引嵌入 (正弦深度位置编码)
  │   ├── 深度LoRA适配器 (逐循环轻量适配)
  │   └── ACT自适应停止 (简单token提前退出)
  ├── Coda (2层标准Transformer, 运行一次)
  ├── 三维Native-RoPE + ALiBi/xPos外推
  ├── RMSNorm + 可学习门控 (DeepNorm风格)
  ├── Kimi AttnRes (深度方向注意力残差)
  └── DeepSeek mHC (流形约束超链接)

Layer 2: 模态接口层
  ├── 文本Tokenizer (BPE, 150K词表)
  ├── 视觉编码器 (VAE Latent, 原生任意分辨率)
  ├── 视频编码器 (4fps密集采样)
  ├── PDF处理器 (双路径)
  ├── SVG Tokenizer (分层几何tokenization)
  └── Tool Encoder (SymPy/Lean/Python, 动态维度)

Layer 1: 基础设施层
  ├── 华为昇腾910B集群 (16卡+)
  ├── CANN 8.0 + PyTorch NPU
  └── 分布式存储与数据Pipeline
```

### 1.3 核心参数规格

| 参数 | 目标值 | 来源/备注 |
|------|--------|----------|
| 总参数量 | ~14.0B | 专家12.9B + Attention 0.54B + 嵌入/投影 ~1B |
| 激活参数量 (推理) | ~872M | 16个专家×50.3M + Attention 67M |
| 模型结构 | 48层 / 4096维 / 32头 / 11008 FFN | 14B深配置，更深更窄 |
| 循环深度 | 8 Unique Cells × 6循环 = 48层 | 推理可扩展至8-12步 |
| 嵌套MoE | 16 Meta-Experts × 16 Sub-Experts | Top-4×Top-4，256小专家激活16个 |
| 文本词表 | 150K | 中英+代码+数学符号 |
| 上下文长度 | 理论无限 | 线性注意力+SSM |
| 训练数据 | 3T-5T tokens | 动态比例调整 |
| Batch Size | 全局8M tokens | 超大batch |
| 学习率 | 分层 (1e-3/3e-4/1e-4) | 嵌入/深层/输出层递减 |
| 优化器 | AdamW + Lion + WSD + 分层LR | 含AttnRes + mHC增强 |
| 图像分辨率 | 原生任意分辨率 | Native-RoPE编码位置 |
| 视频采样 | 4fps密集采样 | 时序注意力聚合 |
| 训练集群 | 16× Ascend 910B | TP + ZeRO-3/FSDP |
| 推理显存 | ~20-40GB (FP16, 8卡TP) | 目标单卡可运行 |

---

## 2. OpenMythos循环深度骨干

### 2.1 三阶段架构 (Prelude → Recurrent Block → Coda)

参考OpenMythos设计：

```
Input Tokens
    │
    ▼
┌─────────────┐
│   Prelude   │  ← 2层标准Transformer，运行一次
│  (2 layers) │     生成冻结锚点 e
└──────┬──────┘
       │ e (frozen)
       ▼
┌─────────────────────────────────────────┐
│         Recurrent Block                 │
│  ┌─────────────────────────────────┐    │
│  │  Loop 1: Cell 1→2→3→4→5→6→7→8 │    │
│  │  Loop 2: Cell 1→2→3→4→5→6→7→8 │    │
│  │  Loop 3: Cell 1→2→3→4→5→6→7→8 │    │
│  │  ... (最多6次循环，推理可扩展)   │    │
│  └─────────────────────────────────┘    │
│                                         │
│  每Cell包含:                            │
│  ├── DeepSeek MLA注意力                 │
│  ├── 动态注意力控制器                   │
│  ├── 嵌套MoE FFN                        │
│  └── RMSNorm + 可学习门控               │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────┐
│    Coda     │  ← 2层标准Transformer，运行一次
│  (2 layers) │     映射为输出logits
└─────────────┘
```

### 2.2 循环状态更新公式

```
h_{t+1} = A · h_t + B · e + TransformerBlock(h_t, e)
```

- `h_t`: 第t次循环后的隐藏状态
- `e`: Prelude产生的冻结输入锚点（防止漂移）
- `A`: 学习的状态转移矩阵（LTI稳定性约束）
- `B`: 学习的输入注入矩阵
- `TransformerBlock`: 注意力 + 嵌套MoE FFN

### 2.3 LTI稳定性约束

```python
# 双指数形式保证谱半径<1
A = exp(-exp(log_dt + log_A))
```

- 每个元素严格在 (0, 1) 范围内
- 谱半径 ρ(A) < 1，数学保证循环不发散
- 等价于连续时间线性系统的ZOH离散化

### 2.4 循环索引嵌入

每次循环前，基于当前循环索引 `t` 添加正弦位置编码：

```python
loop_emb = sin_pos_encoding(t, d_model)
h_t = h_t + loop_emb
```

使模型感知当前迭代深度，早期做粗粒度匹配，后期做细粒度推理。

### 2.5 深度LoRA适配器

为避免所有循环完全共享权重，每个迭代注入低秩适配：

```python
delta(x, t) = down(x) * scale[t] @ B
```

- `scale[t]`: 可学习的逐迭代缩放向量
- 使每次循环行为略有不同，代价远低于独立参数

### 2.6 ACT自适应停止

每个token位置维护停止概率，累积超过阈值(0.99)时提前退出：

```python
halt_prob += r_t  # 当前步停止概率
if halt_prob > 0.99:
    return h_t  # 提前退出
```

简单token可能2-3次循环收敛，复杂推理token可使用全部6次或更多。

---

## 3. 嵌套MoE设计

### 3.1 架构概述

传统MoE的每个专家（一个FFN）被替换为一个小型MoE结构——"多了一个子层级"。

```
标准MoE:
  Router → Top-K Experts → 输出

嵌套MoE:
  Meta-Router → Top-4 Meta-Experts
                       ├── Meta-Expert 1 → Sub-Router → Top-4 Sub-Experts → 输出
                       ├── Meta-Expert 2 → Sub-Router → Top-4 Sub-Experts → 输出
                       ├── ...
                       └── Meta-Expert 16 → Sub-Router → Top-4 Sub-Experts → 输出
```

### 3.2 数学公式

**Meta级路由**:
```
g_meta = softmax(W_meta · x)  # 16维logits
TopMeta = topk(g_meta, k=4)
g_meta_norm = g_meta[TopMeta] / sum(g_meta[TopMeta])
```

**Sub级路由** (对每个选中的Meta-Expert):
```
g_sub_i = softmax(W_sub_i · x)  # 16维logits
TopSub_i = topk(g_sub_i, k=4)
g_sub_norm_i = g_sub_i[TopSub_i] / sum(g_sub_i[TopSub_i])
```

**最终输出**:
```
output = sum_i(sum_j( g_meta_norm[i] · g_sub_norm_i[j] · Expert_{i,j}(x) ))
```

### 3.3 参数规模

| 组件 | 数量 | 每个参数量 | 总参数量 |
|------|------|-----------|---------|
| Meta-Router | 8个Cell | ~65K | ~520K |
| Sub-Router | 8×16 = 128个 | ~65K | ~8.3M |
| Sub-Experts | 256个 | ~50.3M | ~12.9B |
| Attention (MLA) | 48层 | ~11.3M | ~542M |
| 嵌入/投影/Norm | - | - | ~1B |
| **总计** | | | **~14.0B** |

激活参数量：16个Sub-Experts × 50.3M + Attention 67M ≈ **872M**

### 3.4 负载均衡

**双重辅助损失**:
```
L_total = L_task + λ_meta · L_meta_balance + λ_sub · L_sub_balance

L_meta_balance = N · sum_i(f_i · P_i)  # N=16 Meta-Experts
L_sub_balance = N · sum_j(f_j · P_j)   # N=16 Sub-Experts per Meta
```

- `f_i`: 专家i的负载比例
- `P_i`: 路由器分配给专家i的概率比例
- 目标: 负载均匀分布，防止路由崩溃

**补充策略**:
- 容量因子: 训练1.25 / 推理1.0
- 专家Dropout: 5%随机丢弃专家，增强鲁棒性

### 3.5 与循环深度的集成

**路由不固定**：每轮循环动态重选专家，不同Cell、不同Cycle均可选择不同Meta/Sub组合。

**参数共享策略**:
- Attention和LayerNorm: 按Cell共享
- 256个Sub-Experts: **全局跨层共享**（核心设计，控制总参数量在14B）
- 路由参数: Cell级独立
- LayerNorm和路由参数: 不跨Cell共享

**内存优化**:
- 专家分布在8卡上 (Expert Parallelism)
- 每卡2个Meta-Expert (32个Sub-Expert)
- All-to-All通信优化: 主NPU聚合策略

---

## 4. 混合注意力融合系统 (Hybrid-M3)

### 4.1 设计哲学：统一注意力调度

λ-Oracle不采用单一注意力机制，而是构建**分层异构注意力架构**，将6种前沿注意力算法融合为统一的"注意力内核层"，由调度器根据序列长度、模态类型、任务复杂度动态选择最优注意力路径。

```
输入token
    │
    ▼
┌─────────────────────────────────────────────┐
│         统一注意力调度器                      │
│  ┌───────────────────────────────────────┐  │
│  │ 序列长度 < 4K? → MLA Full (Softmax)   │  │
│  │ 序列长度 4K-128K? → KDA (线性)         │  │
│  │ 序列长度 > 128K? → KDA + SSM           │  │
│  │ 图像patch? → Sliding Window + Bi-Attn  │  │
│  │ 多模态对齐? → MMA Mask解锁             │  │
│  │ 冗余头? → MoH Top-K动态选择            │  │
│  └───────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

### 4.2 DeepSeek MLA (全局感知层, 25%)

**作用**: 跨模态全局对齐、长程依赖、精确检索。

```
标准MHA: 缓存 K, V ∈ R^{seq_len × d_model}
MLA: 缓存低秩潜变量 c_kv ∈ R^{seq_len × d_c}  (d_c = d_model/4)

K = W_{DK} · c_kv
V = W_{DV} · c_kv
Q = W_{DQ} · c_q

Attention = Softmax( Q·K^T / sqrt(d_k) ) · V
```

- KV Cache减少 **75-90%**
- 配合 **QK-Norm** (千问门控)：在点积前对Q、K施加RMSNorm，防止深度网络注意力分数失控
- 去除QKV Bias，使注意力更依赖方向而非绝对幅值
- 适用于：全局上下文理解、跨模态对齐、精确事实检索

### 4.3 Kimi KDA (线性记忆层, 50%)

**Kimi Delta Attention** — 目前最好的线性注意力算法之一。

**核心公式**:
```
S_t = (I - β_t · k_t · k_t^T) · Diag(α_t) · S_{t-1} + β_t · k_t · v_t^T
o_t = S_t^T · q_t
```

- `S_t ∈ R^{d_k × d_v}`: 矩阵值RNN状态（关联记忆）
- `β_t`: delta rule学习率（数据依赖）
- `Diag(α_t)`: **逐通道遗忘门控**（细粒度，超越标量头级门控）

**Chunkwise并行算法**: 采用Diagonal-Plus-Low-Rank (DPLR)转移矩阵，通过WY表示法压缩rank-1更新。

**效果** (Kimi Linear, 3B激活/48B总参):
- 1M上下文解码吞吐量提升 **6.3倍**
- KV缓存减少 **75%**
- RULER (128K): 84.3，Pareto最优

**适用于**: 长序列解码、状态压缩、流式生成。

### 4.4 Lightning Attention (恒定速度层, 辅助)

**核心突破**: 消除因果线性注意力的cumsum瓶颈。

**分块分解**:
```
O_t = [(Q_t · K_t^T) ⊙ Mask] · V_t   ← Intra-block: 标准Softmax (块大小B)
      + Q_t · (KV_accumulated)          ← Inter-block: 线性kernel trick

KV_accumulated ← KV_accumulated + K_t^T · V_t
```

- 块内使用Left-Product，块间使用Right-Product累积
- 序列长度1K→128K，训练速度(TGS)基本恒定
- 配合LRPE位置编码 + 指数衰减

**适用于**: 训练阶段的长序列恒定速度处理。

### 4.5 MMA — 模态互注意力 (多模态专用)

**Modality-Mutual Attention**: 解锁image token到text token的注意力路径。

```
标准因果掩码: image token只能被后续token attend，不能attend前面的text
MMA掩码:
  - 同模态内: 保持因果
  - image → text: 允许双向 (解锁！)
  - text → image: 保持因果
```

**效果**: 在12个多模态理解基准上平均提升 **+5.5%**，零额外参数。

**应用阶段**: SFT阶段启用，预训练保持标准因果掩码。

### 4.6 Mixture-of-Head Attention (MoH)

将多头注意力中的头视为MoE中的专家，每个token动态选择Top-K头。

```
O = sum_{i ∈ TopK(router(x))} g_i(x) · Head_i(x)

router(x) = Softmax( W_r · RMSNorm(x) )  # 头级别路由
```

- **共享头**: 部分头始终激活，捕获通用知识
- **动态头**: 根据输入选择Top-4/8头
- LLaMA3-8B仅用75%的头，14基准平均提升 **+2.4%**
- 不增加参数量，推理时可跳过未选中头

### 4.7 滑动窗口注意力 (局部感知)

用于图像patch和视频帧的局部空间建模：

```
窗口大小 W = 512 (文本) / 64×64 (图像局部)

SlidingAttn(Q, K, V):
    for each position i:
        attend to positions [i-W, i+W]
```

- 与全局MLA层互补：局部层处理细粒度空间关系，全局层处理语义关联
- 图像patch内部使用双向滑动窗口

### 4.8 注意力融合路由机制

**分层异构注意力模式** (每4层一个周期):

```
Layer 4k+1: KDA (线性注意力)              ← 长序列记忆
Layer 4k+2: KDA + MoH Top-4 (稀疏头)      ← 自适应计算
Layer 4k+3: KDA (线性注意力)              ← 长序列记忆
Layer 4k+4: MLA Full + QK-Norm + MMA      ← 全局对齐
```

**统一KV表示对齐**:
```
KDA层输出: 隐状态 S_t (矩阵值，无需KV cache)
MLA层输出: 标准KV cache (低秩压缩)

跨层对齐: h_{l+1} = W_proj · concat([h_l, h_{KDA/MLA}])
```

**模态感知路由偏置**:
```python
# 文本token偏好全局检索头
if modality_id == TEXT:
    router_bias += [0.5, 0.5, 0.1, 0.1, ...]  # 全局头加分

# 图像token偏好局部滑动窗口头
if modality_id == IMAGE:
    router_bias += [0.1, 0.1, 0.5, 0.5, ...]  # 局部头加分
```

### 4.9 动态注意力权重生成 (用户创新)

在统一注意力调度之上，增加一层动态控制器：

```python
class DynamicAttentionController:
    def __init__(self, hidden_size, num_heads):
        # 小型交叉注意力网络 (< 1%主网络参数)
        self.query_proj = nn.Linear(hidden_size, 256)
        self.key_proj = nn.Linear(hidden_size, 256)
        self.value_proj = nn.Linear(hidden_size, 256)
        self.out_proj = nn.Linear(256, num_heads * max_seq_len)
        self.modality_gate = nn.Embedding(num_modalities, 256)
        self.depth_gate = nn.Embedding(max_loop_depth, 256)
    
    def forward(self, hidden_state, modality_id, loop_step):
        q = self.query_proj(hidden_state)
        k = self.key_proj(hidden_state)
        v = self.value_proj(hidden_state)
        m = self.modality_gate(modality_id)
        d = self.depth_gate(loop_step)
        
        # 交叉注意力生成动态偏置
        attn_bias = self.out_proj(
            cross_attention(q + m + d, k + m + d, v)
        )
        return attn_bias.view(num_heads, seq_len)
```

- 根据**模态类型**和**循环深度**动态调整注意力模式
- 训练初期冻结，后期联合优化

### 4.10 Kimi AttnRes (深度方向残差)

```
标准残差: h_L = h_1 + sum_{i=1}^{L-1} f_i(h_i)
AttnRes:  h_l = sum_{i=0}^{l-1} α_{i→l} · f_i(h_i)

α_{i→l} = exp( w_l^T · RMSNorm(f_i(h_i)) ) / sum_j exp( ... )
```

- Block AttnRes: L层分为N=8个block，block内标准求和，block间注意力
- 训练开销 < 4%，推理延迟 < 2%
- 解决不同层类型接收相同聚合状态的问题

### 4.11 DeepSeek mHC (流形约束超链接)

```python
# 残差连接映射投影到Birkhoff polytope
H_raw = alpha * tanh(theta · x^T) + b
M = exp(H_raw)

# Sinkhorn-Knopp投影 (20次迭代)
for _ in range(20):
    M = M / sum(M, dim=1, keepdim=True)  # 行归一化
    M = M / sum(M, dim=0, keepdim=True)  # 列归一化

# M为双随机矩阵: 每行每列和为1，所有元素≥0
# 谱范数 ≤ 1，保证深层网络信号不爆炸
```

- 与AdamW/Lion完全兼容
- 映射计算保持float32以确保数值稳定性

### 4.12 统一缓存管理器

| 注意力类型 | 缓存形式 | 大小 | 管理方式 |
|-----------|---------|------|---------|
| MLA | 低秩潜变量 c_kv | O(seq_len × d_c) | Paged KV Cache |
| KDA | 矩阵状态 S_t | O(d_k × d_v) | SSM State Cache |
| Lightning | 累积KV | O(d²) | 块级累积缓存 |
| Sliding Window | 窗口内KV | O(W × d) | 循环缓冲区 |
| MoH | 选中头的KV | O(K × d) | 头级别缓存 |

**缓存调度策略**:
- 短序列 (<4K): 全部注意力类型可用，优先MLA
- 中序列 (4K-128K): KDA为主，MLA全局层保留
- 长序列 (>128K): KDA + SSM，MLA仅用于关键全局层
- 循环深度状态: 每Cell维护独立的循环隐藏状态缓存

---

## 5. 模态输入处理

### 5.1 文本Tokenizer

- **基础**: BPE, 150K词表
- **组成**: 128K基础 + 20K代码 + 2K数学符号
- **中英双语**: 中文采用字+词混合粒度
- **特殊Token**:
  - `<|boi|>` / `<|eoi|>`: 图像开始/结束
  - `<|bov|>` / `<|eov|>`: 视频开始/结束
  - `<|bosvg|>` / `<|eosvg|>`: SVG开始/结束
  - `<|botool|>` / `<|eotool|>`: 工具输出开始/结束
  - `<|im_edit|>`: 图像编辑指令

### 5.2 图像处理 (原生任意分辨率)

**不resize！** 直接使用原始分辨率，Native-RoPE编码空间位置。

- 预训练VAE编码器将图像编码至latent空间
- 512×512图像 → 64×64×4 latent → 展平为连续向量
- 三维Native-RoPE (H/W维度) 编码空间位置
- 支持任意宽高比

### 5.3 视频处理 (4fps密集采样)

- **采样**: 4fps密集采样
- **短视频 (<1min)**: 直接处理全部帧
- **长视频 (数小时)**: 压缩记忆机制
  - 时序注意力池化
  - 可学习的记忆token (类似Memory Networks)
  - SSM状态压缩 (Mamba风格)
  - 关键事件摘要 (模型生成文本摘要)

### 5.4 PDF处理 (双路径)

- **主路径**: 整页渲染为图像，端到端理解
- **增强路径**: 数字原生PDF提取结构化文本+坐标
- **融合**: 默认主路径，数字PDF可选并行增强路径

### 5.5 SVG Tokenizer

分层Tokenization：

| 层级 | 内容 | 编码方式 |
|------|------|----------|
| 原子Token | `<path>`, `<circle>`, `M`, `L` | 专用词表 |
| 属性Token | `fill="red"`, `stroke-width="2"` | 键值对编码 |
| 坐标Token | 数值坐标 | 对数刻度bin离散化 |
| 几何Token | `parallel`, `perpendicular` | 专用词表 |
| 结构Token | `<g>`, `</g>` | XML层级标记 |

目标压缩率: 相比原始SVG文本压缩60%+

### 5.6 Tool Channel (动态维度)

- **SymPy通道**: AST Tree Transformer编码表达式树
- **Lean/Coq通道**: Goal-Context Transformer编码证明状态
- **Python通道**: 多类型编码器处理stdout/stderr/DataFrame/plot
- **维度**: 动态自适应工具输出长度

---

## 6. 输出模态与解码

### 6.1 文本输出

- 自回归解码
- Temperature Scaling + top-k
- Contrastive Search/Decoding (减少重复)
- Speculative Decoding (推测解码加速2-3倍)
- 流式输出 (token-by-token)

### 6.2 图像编辑 (非扩散！)

**核心**: 模型决策路由外部工具，扩散仅在简单情况局部使用。

**工具链**:
1. **OpenCV + PIL**: 几何变换、滤波、基础处理
2. **SAM2 + LaMa**: 精确分割 + 图像修复
3. **ComfyUI/Stable Diffusion**: 风格重绘、复杂编辑
4. **自研SVG渲染**: 几何辅助线、结构化标注

**编排**: 模型先分析编辑需求，决策调用哪个/哪些工具，再执行。

### 6.3 SVG输出

- 自回归生成SVG token序列
- 后处理: 解析为有效SVG XML，语法校验
- 与图像编辑协同: 先生成SVG辅助线 → 渲染到图像

### 6.4 结构化输出

- JSON: `<|json_start|>` / `<|json_end|>` 包裹
- 表格: Markdown/HTML格式
- 标注框: `[x1, y1, x2, y2, label]`

---

## 7. 特化推理引擎

### 7.1 Tool Channel 架构

```
SymPy Channel ──→ Tree Transformer ──→ 符号计算结果向量
Lean Channel ──→ Goal-Context Transformer ──→ 证明状态向量
Python Channel ──→ 多类型编码器 ──→ 执行结果向量
         │
         ▼
Cross-Attention Fusion (Layer 8/16/24/32)
         │
    Tool Gating (动态门控)
         │
    统一Transformer骨干
```

### 7.2 错误感知训练 (三阶段)

**阶段一: SFT**
- 数据: 正确工具调用示例
- 目标: 学会正确格式

**阶段二: 错误对比学习**
- 数据: 正确 vs 错误调用 (语法/类型/逻辑/冗余)
- 目标: 区分正确与错误

**阶段三: GRPO/DPO**
- 奖励设计:
  - 基础正确性: +1
  - 工具调用效率: -0.05/次
  - 分层错误惩罚: 语法-0.2, 类型-0.15, 逻辑-0.1
  - 验证器奖励: Lean完成+0.5, 数值自洽+0.2
  - 错误修复奖励: 错误→修复→成功 +0.3

### 7.3 Tool Coordinator

- **Tool Memory**: 跨工具变量共享命名空间
- **条件执行**: `[IF:tool_failed]` 分支
- **并行调用**: 独立子问题同时调用多工具
- **流式返回**: 工具输出实时流式返回模型

---

## 8. 推理引擎架构

> **完整设计文档**: 见 [inference_engine.md](inference_engine.md)

λ-Oracle推理引擎采用**7层分层架构**，核心组件包括：

| 层级 | 组件 | 说明 |
|------|------|------|
| L7 | API网关 | gRPC/HTTP2, 限流/认证 |
| L6 | 请求调度器 | 连续批处理 + P0-P4优先级队列 + 抢占换出 |
| L5 | 统一注意力内核层 | MLA/KDA/Lightning/Sliding/SSM动态调度 |
| L4 | 嵌套MoE推理优化 | Top-4×Top-4路由, 专家预加载, All-to-All优化 |
| L3 | 工具与图像执行层 | SymPy/Lean/Python + OpenCV/SAM2/ComfyUI/SVG |
| L2 | 量化与压缩引擎 | FP16/BF16/INT8/W4A16/W8A8 + 动态精度切换 |
| L1 | 显存管理 | Paged KV Cache + SSM State Swap + 选择性激活重计算 |

**显存预算**: FP16 ~37GB / INT8 ~22GB，目标单卡可运行。

---

## 9. 训练引擎架构

> **完整设计文档**: 见 [training_engine.md](training_engine.md)

λ-Oracle训练引擎采用**4层分层架构**，核心组件包括：

| 层级 | 组件 | 说明 |
|------|------|------|
| L4 | 数据引擎 | 多模态DataLoader, 动态比例调度, 合成数据Pipeline |
| L3 | 分布式训练核心 | TP+ZeRO-3+EP+DP+PP, 嵌套MoE All-to-All优化 |
| L2 | 训练监控与编排器 | 啊哈时刻检测, 自动恢复, 检查点引擎, 三阶段编排 |
| L1 | CANN适配层 | 图编译, 算子融合, BF16混合精度, 内存复用 |

**计算预算**: Phase1 预训练 (3T-5T tokens, 8M batch, 2-3个月) → Phase2 循环微调 (500B, 4M, 3-4周) → Phase3 对齐 (100B, 2M, 4-6周)。

---

## 10. CANN适配与优化

### 10.1 算子兼容性

| 算子 | 状态 | 措施 |
|------|------|------|
| MLA注意力 | 需验证 | 低秩KV压缩在ATB中的支持 |
| 嵌套MoE路由 | 需自定义 | Ascend C实现双层Softmax+TopK |
| Sinkhorn (mHC) | 需自定义 | 20次迭代行/列归一化 |
| AttnRes Block注意力 | 支持 | 标准Attention算子组合 |
| 混合掩码 | 待验证 | causal+bidirectional组合 |
| GroupNorm | 支持 | num_groups设为2的幂 |

### 10.2 分阶段适配

**Phase 1**: 文本分支 + MLA + 循环深度
**Phase 2**: 引入嵌套MoE + 视觉分支
**Phase 3**: AttnRes + mHC + 混合掩码调优
**Phase 4**: 整网图编译 + 量化推理

### 10.3 通信优化

- 嵌套MoE Expert Parallelism: 8卡，每卡2 Meta-Expert
- All-to-All优化: 主NPU聚合策略
- 重计算块大小: L_r = int(sqrt(4L/6)) ≈ 5 (L=32)

---

## 11. 评估体系

### 11.1 触发策略

**智能触发**: 监控内部指标，检测到涌现迹象时立即评估。

监控指标:
- 梯度范数突变
- 注意力熵变化
- 损失下降速率
- 验证集 perplexity 突变

### 11.2 评估基准

| 基准 | 目标 (8B/14B) |
|------|--------------|
| MMLU | >70% / >75% |
| GSM8K | >85% |
| HumanEval | >70% |
| MATH | >50% |
| MMMU | >50% |
| MMBench | >75% |
| DocVQA | >80% |
| MathVista | >55% |
| GeoQA | >85% |
| LiveCodeBench | Pass@1 >60% |

---

## 12. 部署架构

### 12.1 混合部署

| 模式 | 硬件 | 模型 | 显存 | 说明 |
|------|------|------|------|------|
| 云端高性能 | 8×910B | 14B FP16 | ~40GB | 最严格约束: 单卡/少量卡可运行 |
| 企业私有 | 2×910B | 14B INT8 | ~20GB | 数据不出域 |
| 边缘 | 1×310P | 8B INT4 | ~10GB | 轻量化 |
| Mac/PC | CPU/APU | 3B INT4 | ~8GB | 本地不限安全 |

### 12.2 安全对齐

- **本地部署**: 不限制，用户想生成什么不归管（只要不上网）
- **线上部署**: 跟随行业主流标准（基础过滤+价值观对齐）
- **可解释**: 每次拒绝给出理由

---

## 13. 附录: 设计决策汇总

### 13.1 17轮问答决策矩阵

| 维度 | 决策 | 轮次 |
|------|------|------|
| 模型类型 | AI/机器学习多模态 | 1 |
| 架构范式 | Transfusion/Neo-Unify + OpenMythos | 3, 17 |
| 参数规模 | 14B深配置 (48层/4096维) | 3, 12 |
| 输入模态 | 文本+图像+视频+PDF | 3 |
| 输出模态 | 文本+图像编辑+结构化数据+SVG | 3, 5 |
| 循环块结构 | 参考OpenMythos (Prelude→Recurrent Block→Coda) | 17 |
| 嵌套MoE | 16 Meta × 16 Sub, Top-4×Top-4 | 14, 16 |
| 注意力机制 | DeepSeek MLA | 11 |
| 动态注意力 | 小型交叉注意力网络 | 8 |
| 位置编码 | 三维Native-RoPE + ALiBi/xPos | 9 |
| 归一化 | RMSNorm + 可学习门控 | 9 |
| 损失函数 | 不确定性加权 (同方差不确定性) | 8 |
| 初始化 | 完全从头随机初始化 | 8 |
| 优化器 | AdamW + Lion + WSD + 分层LR | 10 |
| 增强技术 | Kimi AttnRes + DeepSeek mHC | 10, 11 |
| 正则化 | 模态差异化Dropout + GradClip + WD + LayerDrop | 10, 13 |
| 检查点 | 动态策略 (中频+啊哈时刻高频) | 10, 13 |
| 并行策略 | TP + ZeRO-3/FSDP | 10 |
| Batch Size | 全局8M tokens | 13 |
| 学习率 | 分层 (1e-3/3e-4/1e-4) | 13 |
| 数据比例 | 动态调整 (每周根据评估反馈) | 13 |
| 推理采样 | Temperature + top-k + Contrastive + Speculative | 11 |
| 混合精度 | FP16/BF16 (看CANN支持) | 11 |
| 图像分辨率 | 原生任意分辨率 | 12 |
| 视频采样 | 4fps密集采样 + 时序聚合 | 12 |
| 视频记忆 | 可学习记忆token + SSM + 关键事件摘要 | 15 |
| 图像编辑 | 模型决策路由外部工具 (OpenCV/SAM2/ComfyUI/SVG) | 14, 15 |
| 工具调用 | 流式返回 | 14 |
| 评估策略 | 智能触发 (涌现检测) | 16 |
| 安全对齐 | 本地不限，线上跟随主流 | 17 |
| 部署约束 | 显存限制 (单卡/少量卡可运行) | 17 |
