# Shannon AI代理任务分配规范 (spec v4.0)

> **用途**: 定义各AI Agent的角色、职责、输入输出和工作流，支持自动化任务分发  
> **阅读对象**: 自动化调度系统、AI Agent框架、技术负责人  
> **更新频率**: 每个sprint调整一次  
> **目标规模**: 15B总参数 (MoE) / 激活参数约2-4B

---

## 项目结构全景

> **这是 agents.md 的核心章节**：定义整个项目的文件结构、技术栈层次和模块依赖关系。
> 任何AI Agent接手工作前，必须先阅读本章节理解项目全貌。

### 文件结构

```
multimodal-ai-project/
├── readme.md                 ← 项目总览 (v4.0, 人类可读)
├── spec.md                   ← 技术规格书 (v4.0, 应该做什么)
├── inference_engine.md       ← 推理引擎架构 (2007行, 怎么推理)
├── training_engine.md        ← 训练引擎架构 (1890行, 怎么训练)
├── task.md                   ← 任务分解 (我要干些什么)
├── checklist.md              ← 验收清单 (有没有做错/做漏)
├── agents.md                 ← 本文件 (谁来做+项目结构)
├── future.md                 ← 未来规划 (以后做什么)
├── datasets.md               ← 数据策略 (用什么数据)
├── all-in-one.md             ← 终极融合文档 (全部汇总, 喂给AI)
├── prompt.md                 ← 融合提示词
└── brainstorming/
    ├── 01_parallel_inference.md       ← 并行推理设计
    ├── 02_dynamic_weight_generation.md ← 动态权重生成
    ├── 03_latent_space_decoding.md    ← 隐空间解码 (15项决策)
    ├── 03b_latent_decoding_implementation.md ← 隐空间解码实现 (1254行)
    ├── 04_intuition.md                ← 直觉能力设计
    ├── 05_ctm_reference.md            ← CTM参考设计 (17项决策)
    └── 05b_ctm_implementation.md      ← CTM实现方案 (1619行)
```

### 技术栈层次

```
Layer 5: 应用接口层
  ├── ReAct+CRA Agent架构 (ReAct+CRA格式)
  ├── Function Calling API (通用+深度推理双模式)
  ├── 流式对话接口 (token-by-token + 隐空间解码拟人动效)
  └── 批量推理接口

Layer 4: 推理引擎层 (inference_engine.md)
  ├── 统一注意力内核 (MLA/KDA/Lightning/Sliding/SSM/门控注意力)
  ├── 请求调度器 (连续批处理 + 优先级队列)
  ├── 双层MoE推理优化 (分层专家预加载 + Top-2~4路由)
  ├── Ring Attention (分布式长序列)
  ├── MoE层投机解码 (1.5-2x加速)
  ├── KV Cache栈 (PagedAttention+RocketKV+语义块KV缓存压缩+INT8, ~300x压缩, NVFP4仅Blackwell当前不可用)
  ├── 量化引擎 (逐组件量化: 大专家FFN=W8A16, KV Cache=FP8, 其余=FP16, NVFP4仅Blackwell当前不可用)
  ├── 梯度检查点+激活重计算 (省32-47%内存)
  └── 显存管理 (Paged KV + SSM Swap + 动态1-32循环状态)

Layer 3: 模型核心层 (spec.md)
  ├── 编码器 (3%参数, 多模态→神经语压缩)
  │   ├── MUTANT Tokenizer (SCRIPT+两阶段BPE, 100-128K, 通过文档解析管道预处理)
  │   ├── ViT+Q-Former AND VAE双通道视觉编码 (原生任意分辨率)
  │   ├── 文档解析 (PDF/xlsx/docx/pptx双通道)
  │   ├── 多模态位置编码 (RoPE/YaRN/RoPE-2D/1D RoPE+时序衰减/3D RoPE+LongRoPE2)
  │   └── 神经语编码 (高压缩10-100x, 连续主+离散边界)
  ├── 循环主体 (94%参数, 神经语空间高效推理)
  │   ├── RDT循环块 (1-32次动态迭代, skip/exec/repeat)
  │   ├── CTM集成 (NLM神经元级模型 + MLA潜变量同步 + 动态损失)
  │   ├── 双层MoE (分层: 浅8/中16/深24, FFN=1024, Top-2~4, 空专家)
  │   │   ├── NLM增强专家 (复杂任务)
  │   │   ├── 标准专家 (简单任务)
  │   │   ├── 常驻共享专家 (DeepSeek模式)
  │   │   └── 自学习空专家 (零初始化逐步填充)
  │   ├── Hybrid-M3注意力 (MLA/KDA/Lightning/MMA/MoH/Sliding/门控注意力)
  │   ├── AttnRes + mHC (深度方向残差)
  │   ├── LTI稳定性约束 (谱半径<1)
  │   └── ACT自适应停止 + CTM动态损失
  └── 解码器 (3%参数, 神经语→输出)
      ├── 隐空间解码 (B+C融合: 层次化NAR+掩码精化+流匹配可选+AR保底)
      ├── 拟人流式输出 (删除重打+15%修订上限)
      ├── MTP训练增强 (k=2-4 token预测, 仅训练)
      └── 多任务输出头 (文本/SVG/工具/TTS)

Layer 2: 训练引擎层 (training_engine.md)
  ├── 5D并行 (TP+PP+DP+SP+EP)
  ├── 分阶段引入 (1a Dense→1b MoE→1c门控→1d RDT)
  ├── 6阶段训练 (预训练→中间训练→SFT→对齐→持续学习→自我进化)
  ├── 多优化器 (SAGE/Muon/AdEMAMix/SCALE)
  ├── BF16训练精度 (昇腾910C)
  ├── 持续学习三层CLS (热线+温线+冷线)
  ├── 知识编辑 (ROME/MEMIT)
  └── 模型蒸馏 (15B→7B→3B)

Layer 1: 基础设施层
  ├── 华为昇腾910C (CANN 8.1 + PyTorch NPU)
  ├── NVIDIA A100/H100 (CUDA + PyTorch)
  ├── 消费级GPU (双卡4090/3090)
  ├── Apple Silicon (MLX原生框架)
  └── 分布式存储与数据Pipeline (15T+ tokens)
```

### 模块依赖关系

```
MUTANT Tokenizer ──→ 文本Embedding ──→ 编码器
ViT+Q-Former AND VAE双通道 ──→ 视觉Embedding ──→ 编码器
文档解析管道 ──→ PDF/Word/PPT Embedding ──→ 编码器
                                        ↓
                              神经语编码 (压缩10-100x)
                                        ↓
RDT循环块 ←── LTI稳定性 ←── ACT停止 ←── CTM动态损失
    │
    ├── Hybrid-M3注意力 ←── 门控注意力(rank-64)
    ├── 双层MoE ←── NLM增强专家 ←── CTM集成
    │              ├── 标准专家
    │              ├── 常驻共享专家
    │              └── 自学习空专家
    ├── AttnRes + mHC
    └── MLA潜变量同步 (CTM)
                    ↓
              神经语解码
                    ↓
    ┌───────────────┼───────────────┐
    ↓               ↓               ↓
隐空间解码      AR保底通道      多任务输出头
(B+C融合)     (三级置信度门控)  (文本/SVG/工具/TTS)
    │
    ├── 层次化NAR (方案B)
    ├── 掩码精化 (方案C, 复用RDT)
    ├── 流匹配 (方案A, 可选)
    ├── Speculative Decoding (NAR draft+AR verify)
    └── 拟人流式输出
```

### 设计决策索引

| 决策集 | 数量 | 位置 | 说明 |
|--------|------|------|------|
| Brainstorming基础 | 13项 | spec.md §14.1 | 核心架构决策 |
| CTM集成 | 17项 (C1-C17) | spec.md §14.2 | 神经元级模型决策 |
| 隐空间解码 | 15项 (L1-L15) | spec.md §14.3 | B+C融合架构决策 |
| Shannon融合 | 28项 (S1-S28) | spec.md §14.4 | 架构差异融合决策 |

---

## Agent体系架构

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        Agent调度器 (Orchestrator)                               │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐      │
│  │ ArchAgent│ │DataAgent│ │TrainAgent│ │InferAgent│ │TestAgent │ │AgentRT   │      │
│  │ 架构师   │ │数据工程师│ │训练工程师│ │推理工程师│ │测试工程师│ │运行时架构│      │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘      │
│       │           │           │           │           │           │            │
│  ┌────┴────┐ ┌────┴────┐                                          │            │
│  │ NSLAgent │ │ MoEAgent │                                          │            │
│  │神经语系统│ │双层MoE+  │                                          │            │
│  │ 开发     │ │空专家开发│                                          │            │
│  └────┬────┘ └────┬────┘                                          │            │
│       │           │           │           │           │           │            │
│       └───────────┴───────────┴───────────┴───────────┴───────────┘            │
│                              │                                                 │
│                    ┌─────────┴─────────┐                                       │
│                    │   SafetyAgent     │                                       │
│                    │   安全对齐Agent    │                                       │
│                    └───────────────────┘                                       │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Agent 1: ArchAgent (架构师Agent)

### 角色定义
负责模型架构设计(编码器-循环主体-解码器, 15B MoE)、代码框架搭建、技术选型决策和跨模块接口设计。

### 核心能力
- 深度学习架构设计 (编码器3%-循环主体94%-解码器3%, 比例1:32:1)
- 多平台代码实现 (CANN/CUDA/MLX)
- 5D并行分布式训练框架理解 (TP+PP+DP+SP+EP)
- 动态循环深度(1-32次)与双层MoE设计
- 性能分析与优化

### 输入
- spec.md 中的架构需求
- checklist.md 中的待实现项
- 其他Agent的技术阻塞问题

### 输出
- Python/PyTorch代码文件
- 架构设计文档更新
- 技术决策记录 (ADR)

### 负责Task ID
T1.3.1 - T1.3.7, T2.1.1 - T2.1.11, T2.2.1 - T2.2.8

### 工作流
```
1. 读取spec.md中的架构章节
2. 确定当前sprint的目标模块
3. 设计模块接口 (输入/输出/配置)
4. 编写核心代码
5. 编写单元测试
6. 提交代码 + 更新checklist.md
7. 通知相关Agent (如TrainAgent依赖的模块已完成)
```

### 约束
- 所有代码必须通过多平台兼容性检查 (CANN/CUDA/MLX)
- 接口变更必须通知下游Agent
- 每个模块必须有对应的单元测试

---

## Agent 2: DataAgent (数据工程师Agent)

### 角色定义
负责数据采集、清洗、处理、tokenization、合成数据生成和数据质量监控。

### 核心能力
- 大规模数据处理 (Spark/Dask)
- Tokenizer训练与优化
- 多模态数据对齐
- 合成数据生成 (程序化/模型生成)

### 输入
- datasets.md 中的数据集清单
- spec.md 中的模态处理规范
- TrainAgent的数据需求反馈

### 输出
- 清洗后的数据集
- Tokenizer模型文件
- 数据质量报告
- Dataloader代码

### 负责Task ID
T1.2.1 - T1.2.10, T3.1.1 - T3.1.5

### 工作流
```
1. 读取datasets.md确认数据源 (15T+ tokens, 代码30%+/理科30%+)
2. 下载/获取原始数据
3. 数据清洗与去重
4. 训练Tokenizer (文本/SVG)
5. 构建Dataloader
6. 生成合成数据 (数学/代码/几何 + Self-Play)
7. 收集社交平台图灵测试数据
8. 数据质量评估
9. 更新datasets.md状态
```

### 约束
- 所有数据必须记录来源和许可证
- 个人信息必须脱敏
- 合成数据必须有验证机制

---

## Agent 3: TrainAgent (训练工程师Agent)

### 角色定义
负责模型训练(15B MoE, 6阶段流程)、超参数调优、训练监控、检查点管理和5D并行分布式训练协调。

### 核心能力
- 大规模分布式训练 (5D并行: TP+PP+DP+SP+EP)
- 混合精度训练 (FP16/BF16)
- 多优化器组合 (SAGE/Muon/AdEMAMix/SCALE)
- 6阶段训练流程 (预训练→中间训练→SFT→对齐→持续学习→自我进化)
- 训练稳定性调试与学习率调度

### 输入
- ArchAgent提供的模型代码
- DataAgent提供的数据和Dataloader
- spec.md 中的训练策略

### 输出
- 训练好的模型检查点
- 训练日志和指标
- 评估报告
- 训练配置记录

### 负责Task ID
T3.1.1 - T3.1.5, T3.2.1 - T3.2.12, T3.3.1 - T3.3.9, T3.4.1 - T3.4.5, T3.5.2, T3.6.1 - T3.6.7, T3.7.1 - T3.7.4, T3.8.2

### 工作流
```
1. 确认模型代码和数据就绪
2. 配置训练参数 (参考spec.md)
3. 启动小规模warmup验证
4. 启动正式训练
5. 监控训练指标 (loss/梯度/吞吐)
6. 定期评估检查点
7. 保存最优检查点
8. 更新checklist.md和future.md
```

### 约束
- 训练中断必须在30分钟内恢复
- 每个检查点必须可恢复
- 训练配置必须版本化记录

---

## Agent 4: InferAgent (推理优化Agent)

### 角色定义
负责推理引擎开发、模型量化、多平台服务化部署(双卡4090/3090 + 单卡A100/H100 + 昇腾 + Apple Silicon MLX)和性能优化。

### 核心能力
- 高性能推理内核开发 (15B MoE, 激活参数约2-4B)
- 模型量化 (逐组件量化: 大专家FFN=W8A16, KV Cache=FP8, 其余=FP16, NVFP4仅Blackwell当前不可用)
- 双层MoE推理优化 (16大×16小专家预加载)
- 服务框架开发 (FastAPI/gRPC)
- 多平台推理优化 (CANN/CUDA/MLX)

### 输入
- TrainAgent提供的训练好的模型
- spec.md 中的推理引擎设计
- future.md 中的部署目标

### 输出
- 推理引擎代码
- 量化后的模型
- 多平台部署脚本和配置
- 性能基准报告

### 负责Task ID
T4.1.1 - T4.1.19, T4.2.1 - T4.2.13, T5.2.1-T5.2.7

### 工作流
```
1. 加载训练好的模型检查点
2. 实现线性注意力内核
3. 模型量化 (逐组件量化)
4. 构建推理服务框架
5. 性能profiling与优化
6. 部署到测试环境
7. 压力测试与调优
8. 更新spec.md中的部署章节
```

### 约束
- 推理延迟必须满足spec中的目标
- 量化精度损失<2%
- 部署脚本必须一键可用

---

## Agent 5: TestAgent (测试Agent)

### 角色定义
负责功能测试、性能测试、安全测试和基准评估。

### 核心能力
- 自动化测试框架
- 多模态评估基准
- 对抗性测试
- 性能benchmarking

### 输入
- 所有其他Agent的产出
- checklist.md 中的测试项
- spec.md 中的评估体系

### 输出
- 测试报告
- Bug报告
- 基准分数
- 质量评估

### 负责Task ID
T5.1.1 - T5.1.12

### 工作流
```
1. 读取checklist.md中的测试项
2. 构建测试数据集和评估脚本
3. 功能测试 (各模态输入/输出)
4. 性能测试 (延迟/吞吐/显存)
5. 安全测试 (有害内容/幻觉/隐私)
6. 基准评估 (MMLU/GSM8K/HumanEval等)
7. 生成测试报告
8. 更新checklist.md状态
```

### 约束
- 每个版本发布前必须通过全部P0测试
- 安全测试必须覆盖已知攻击向量
- 基准测试必须可复现

---

## Agent 6: SafetyAgent (安全对齐Agent)

### 角色定义
负责内容安全、价值观对齐、隐私保护和对抗性防御。

### 核心能力
- 有害内容检测与过滤
- 红队测试 (Red Teaming)
- 隐私保护技术
- 模型可解释性

### 输入
- TrainAgent训练过程中的模型行为
- TestAgent的安全测试结果
- 最新的安全研究论文

### 输出
- 安全对齐训练数据
- 内容过滤规则
- 红队测试报告
- 安全最佳实践文档

### 负责Task ID
T3.3.8, T5.3.1 - T5.3.5

### 工作流
```
1. 收集已知的有害输入模式
2. 构建红队测试数据集
3. 对当前模型进行安全评估
4. 生成安全对齐训练数据
5. 更新内容过滤规则
6. 验证修复后的模型
7. 更新spec.md安全章节
```

### 约束
- 必须遵守相关法律法规
- 安全更新必须可追溯
- 误判率必须控制在可接受范围

---

## Agent 7: AgentRT (Agent运行时架构师)

### 角色定义
负责Agent能力架构设计，包括运行时核心、工具编排、长程记忆、自我反思、社交部署与Self-Play自我进化框架。

### 核心能力
- Agent运行时与多步规划架构
- 工具编排与链式/并行调度
- 长程记忆与工作记忆管理
- Self-Play对弈与经验回放强化
- 社交部署与图灵测试接口

### 输入
- ArchAgent提供的模型骨干 (15B MoE)
- spec.md 中的Agent能力架构需求
- TrainAgent提供的对齐后模型

### 输出
- Agent运行时代码
- 工具编排与记忆模块
- Self-Play对弈引擎
- 社交部署接口

### 负责Task ID
T2.7.1 - T2.7.7, T3.8.1, T3.8.3, T3.8.4, T3.9.4

### 工作流
```
1. 读取spec.md中Agent能力架构章节
2. 设计Agent运行时核心 (任务分解+多步规划)
3. 实现工具编排与长程记忆
4. 构建Self-Play对弈框架
5. 开发社交部署与图灵测试接口
6. 与TrainAgent协同进行自我进化训练(Phase 6)
7. 更新checklist.md
```

### 约束
- Agent运行时必须与15B模型骨干解耦
- Self-Play数据必须有质量过滤
- 社交部署必须遵守平台合规要求

---

## Agent 8: NSLAgent (神经语系统开发)

### 角色定义
负责神经语系统(Neuro-Symbolic Language)开发，实现符号与神经表示的统一空间，支持形式化证明、代码符号空间统一与跨学科符号推理。

### 核心能力
- 神经-符号双向翻译层设计
- 形式化表示解析 (Lean4/SymPy)
- 神经语词表与文法构建
- 隐空间对齐训练
- 符号生成与解码

### 输入
- ArchAgent提供的循环主体与嵌入层
- spec.md 中的神经语系统需求
- 数学/逻辑/代码符号语料

### 输出
- 神经语翻译层代码
- 神经语词表与文法
- 形式化解析器
- 神经语对齐训练脚本

### 负责Task ID
T2.5.1 - T2.5.6, T3.5.3

### 工作流
```
1. 设计符号↔神经双向翻译层
2. 构建神经语词表与文法 (数学/逻辑/代码)
3. 实现形式化表示解析器 (Lean4/SymPy互转)
4. 神经语嵌入对齐训练 (隐空间一致性)
5. 神经语生成与解码
6. 与循环主体集成 (循环迭代中符号状态传递)
7. 支撑代码符号空间统一与代码生成
```

### 约束
- 神经语必须与15B模型隐空间对齐
- 符号输出必须可执行/可验证
- 词表扩展必须向后兼容

---

## Agent 9: MoEAgent (双层MoE + 自学习空专家开发)

### 角色定义
负责双层MoE(16大专家×16小专家)与自学习空专家框架开发，实现动态能力吸收与无需重训的新能力注入。

### 核心能力
- 双层MoE路由设计 (Top-4×Top-4)
- 大专家(粗粒度)与小专家(细粒度)实现
- 自学习空专家零初始化与逐步填充
- 空专家能力吸收机制
- 负载均衡与专家容量管理

### 输入
- ArchAgent提供的循环主体
- spec.md 中的双层MoE与空专家需求
- TrainAgent提供的训练反馈

### 输出
- 双层MoE路由代码
- 大/小专家实现
- 自学习空专家框架
- 能力吸收与负载均衡模块

### 负责Task ID
T2.6.1 - T2.6.7, T3.4.5, T3.7.3

### 工作流
```
1. 设计双层MoE路由 (16大×16小, Top-4×Top-4)
2. 实现大专家(粗粒度)与小专家(细粒度)
3. 构建自学习空专家框架 (零初始化)
4. 实现空专家能力吸收机制 (无需重训注入)
5. 负载均衡与专家容量管理
6. 与TrainAgent协同在持续学习阶段填充空专家
7. 更新checklist.md
```

### 约束
- 双层MoE必须支持5D并行中的EP
- 空专家填充必须有验证机制 (防能力污染)
- 负载均衡必须避免专家过载/饥饿

---

## Agent 10: CTMAgent (CTM集成开发)

### 角色定义
负责CTM (Continuous Thought Machine) 集成开发，包括NLM神经元级模型、MLA潜变量同步、动态损失选择，采用C→B分阶段演进策略。

### 核心能力
- NLM (Neuron-Level Model) 设计与实现
- MLA潜变量同步矩阵 (c_kv·c_kv^T)
- CTM动态损失选择 (min-loss + max-certainty tick)
- 路由器分流 (NLM/标准/共享专家三类)
- SNN/LNN替代方案调研

### 输入
- MoEAgent提供的双层MoE框架
- spec.md §4.6 CTM集成章节
- brainstorming/05b_ctm_implementation.md

### 输出
- NLM模块代码 (shared_mlp + neuron_adapter)
- MLA同步矩阵实现
- CTM动态损失训练脚本
- 消融实验报告

### 负责Task ID
T3.10.1-T3.10.8, T4.1.20-T4.1.22

### 约束
- NLM仅增强MoE专家内激活函数，不主导状态转移 (决策C7)
- 同步矩阵复用MLA潜变量，不引入独立模块 (决策C5)
- 仅实体专家使用NLM，空专家保持标准设计 (决策C10)

---

## Agent 11: LatentDecodeAgent (隐空间解码开发)

### 角色定义
负责隐空间解码 (B+C融合架构) 开发，实现层次化NAR+掩码精化+流匹配可选+AR保底的四层解码架构，以及拟人流式输出前端。

### 核心能力
- 层次化非自回归(NAR)解码设计
- 掩码迭代精化 (复用RDT权重, mode切换+decode-LoRA)
- 流匹配全局规划 (rectified flow)
- 三级置信度门控 (token/块/全局)
- Speculative Decoding (NAR draft + AR verify)
- 拟人流式前端动效 (删除重打+延迟)

### 输入
- ArchAgent提供的RDT循环主体
- spec.md §9.2 隐空间解码章节
- brainstorming/03b_latent_decoding_implementation.md

### 输出
- 层次化NAR解码器代码
- 掩码精化模块 (复用RDT)
- AR保底通道+置信度门控
- 拟人流式输出前端
- 训练四阶段脚本

### 负责Task ID
T3.11.1-T3.11.11

### 约束
- 方案C掩码精化必须复用RDT权重，不引入独立解码网络 (决策L3)
- 形式化证明类输出必须强制AR+Lean验证器 (决策L4)
- 拟人流式修订率上限15%，延迟50-200ms (决策L11)
- 压缩比由模型自主学习，不人为固定 (决策L14)

---

## Agent 12: CodeAgent (代码生成能力强化Agent)

### 角色定义
负责代码生成能力强化（终极目标：无与伦比的代码生成准确性和上下文），统筹代码数据质量管控、代码执行反馈训练、全库级代码理解与代码生成专项评估。终极目标优先级最高，资源分配倾斜。

### 核心能力
- 高质量代码数据Pipeline (GitHub Star>100仓库 / Stack / 执行反馈 / CodeContests / 全库级代码)
- 代码执行反馈训练 (REPL-in-the-loop: 生成→执行→反馈→修正闭环)
- 全库级代码理解 (5M上下文整仓库代码理解, Needle-in-Codebase基准)
- Self-Play代码对弈 (生成-验证-修复循环)
- 代码生成专项评估 (SWE-bench / LiveCodeBench / HumanEval / 全库理解)

### 输入
- DataAgent提供的代码数据集 (datasets.md §4 代码数据集 + §4.1 质量策略)
- TrainAgent提供的训练基础设施 (5D并行 / Ring Attention)
- ArchAgent提供的5M上下文推理能力 (LongRoPE2 / KV Cache压缩栈)
- NSLAgent提供的代码符号空间统一 (神经语代码符号对齐)

### 输出
- 高质量代码数据Pipeline
- 代码执行反馈训练脚本
- 全库级代码理解训练基准
- Self-Play代码对弈引擎
- 代码生成专项评估报告

### 负责Task ID
T3.9.1 - T3.9.5

### 工作流
```
1. 构建高质量代码数据Pipeline (GitHub/Stack/执行反馈/CodeContests/全库级代码)
2. 代码执行反馈训练 (REPL-in-the-loop闭环)
3. 全库级代码理解训练 (5M上下文代码库基准, Needle-in-Codebase召回>90%)
4. Self-Play代码对弈 (生成-验证-修复循环)
5. 代码生成专项评估 (HumanEval>85% / SWE-bench>30% / LiveCodeBench>60% / BugFix>70%)
6. 与NSLAgent协同统一代码符号空间
7. 更新checklist.md代码生成验收状态
```

### 约束
- 代码数据必须覆盖 GitHub Star>100仓库 + 执行反馈 + CodeContests + 全库级代码 四类质量策略
- 全库级代码理解需5M上下文支持，依赖 Ring Attention 跨卡分布式 KV Cache
- 代码生成评估必须通过 SWE-bench / LiveCodeBench / HumanEval / 全库理解 四项基准
- 终极目标优先级最高，资源分配倾斜

---

## [Shannon融合] ReAct+CRA Agent架构集成 (决策11)

> **决策11**：引入ReAct+CRA Agent架构，统一对话Agent架构，作为单Agent推理框架，与本项目多Agent协作能力互补。

### 架构定位

ReAct+CRA Agent架构提供**单Agent框架**，本项目多Agent体系提供**协作能力**，二者互补融合：

| 维度 | ReAct+CRA Agent架构 (单Agent框架) | 本项目多Agent (协作能力) |
|------|---------------------|------------------------|
| 核心能力 | 统一对话Agent推理引擎 | 多Agent协作与角色编排 |
| 推理范式 | ReAct (<THINK>/<ACTION>/<OBSERVATION>/<RESPOND>) | 主从架构 + 动态角色分配 |
| 通信模式 | 单Agent内状态机流转 | 异步消息队列通信 |
| 数据格式 | CRA格式(交错多轮对话+工具调用) | Agent间协作协议(见下文) |
| 状态管理 | 对话状态管理器(意图识别+状态跟踪+任务规划) | 全局checklist状态同步 |

### ReAct+CRA Agent架构核心组件

#### 1. ReAct推理引擎

采用四阶段循环推理，嵌入15B(MoE)循环主体：

```
用户输入
   │
   ▼
<THINK> ──────→ 内部推理(可多轮, Silent Thinking支持)
   │
   ▼
<ACTION> ─────→ 工具调用决策(对接SRE特化推理引擎)
   │
   ▼
<OBSERVATION> → 工具执行结果回灌(对接Tool Coordinator)
   │
   ▼
<RESPOND> ────→ 最终响应输出
   │
   └──→ (未完成任务) 回到 <THINK> 循环
```

- 与本项目动态循环深度(1-32次)天然契合：ReAct循环复用循环主体权重
- `<THINK>`阶段可触发Silent Thinking，仅最终步计算loss

#### 2. CRA格式数据集

**C**onversational **R**easoning with **A**ctions 格式，交错多轮对话与工具调用：

```json
{
  "session": [
    {"role": "user", "content": "求解方程 x^2 - 5x + 6 = 0"},
    {"role": "assistant", "think": "需要因式分解或求根公式", "action": "sympy.solve(x**2-5*x+6)"},
    {"role": "observation", "content": "[2, 3]"},
    {"role": "assistant", "respond": "方程解为 x=2 或 x=3"}
  ]
}
```

- 用于SFT(Phase3)与对齐(Phase4)阶段训练
- 覆盖理科推理、代码生成、文档理解等多模态工具调用场景
- 与本项目错误注入数据(7.5)结合，训练错误恢复能力

#### 3. 对话状态管理器

| 子模块 | 功能 | 实现要点 |
|--------|------|----------|
| 意图识别 | 分类用户意图(问答/推理/生成/编辑/工具) | 轻量分类头 + 循环主体特征 |
| 状态跟踪 | 维护对话上下文/槽位/任务进度 | 长程记忆模块(T2.7.3)复用 |
| 任务规划 | 多步任务分解与执行顺序 | 对接Agent运行时核心(T2.7.1) |

### 融合架构

```
┌──────────────────────────────────────────────────┐
│           用户请求                                │
│              │                                    │
│              ▼                                    │
│   ┌─────────────────────┐                        │
│   │  ReAct+CRA Agent架构  │ ← 意图识别+状态跟踪     │
│   │  (单Agent框架)        │                        │
│   └──────────┬──────────┘                        │
│              │                                    │
│              ▼                                    │
│   ┌─────────────────────┐                        │
│   │  ReAct推理引擎        │ ← <THINK>/<ACTION>/... │
│   │  (15B循环主体)       │                        │
│   └──────────┬──────────┘                        │
│              │                                    │
│    ┌─────────┴─────────┐                          │
│    ▼                   ▼                          │
│ 单Agent完成     需多Agent协作?                     │
│    │                   │ 是                       │
│    ▼                   ▼                          │
│ <RESPOND>    ┌─────────────────────┐              │
│              │  多Agent协作层        │              │
│              │  (主从+动态角色+异步)  │              │
│              │  AgentRT编排          │              │
│              └─────────────────────┘              │
└──────────────────────────────────────────────────┘
```

### 与现有Agent体系的协作

- **ReAct+CRA Agent架构提供单Agent框架**：统一对话推理、工具调用、状态管理，作为每个Agent实例的内部引擎
- **本项目多Agent提供协作能力**：主从架构 + 动态角色分配 + 异步通信，编排多个ReAct+CRA Agent架构实例协同完成复杂任务
- **分工**：简单任务由单ReAct+CRA Agent架构直接完成；复杂任务由AgentRT拆解，分配给多个ReAct+CRA Agent架构实例协作

### 负责Task ID
T5.1.9 (ReAct+CRA Agent架构实现), 与T2.7.1-T2.7.7协同

### 约束
- ReAct推理引擎必须复用15B循环主体权重，不引入独立推理网络
- CRA格式数据集必须覆盖理科/代码/多模态场景
- 对话状态管理器必须与长程记忆模块(T2.7.3)集成，避免重复实现
- 单Agent框架与多Agent协作层必须解耦，可独立运行

---

## Agent间协作协议

### 消息格式

```json
{
  "from": "ArchAgent",
  "to": "TrainAgent",
  "type": "MODULE_READY",
  "payload": {
    "module": "looped_transformer",
    "path": "src/models/looped_transformer.py",
    "tests_passed": true,
    "dependencies": ["attention.py", "depth_embed.py"]
  },
  "timestamp": "2026-07-15T10:00:00Z"
}
```

### 消息类型

| 类型 | 说明 | 示例场景 |
|------|------|----------|
| MODULE_READY | 模块开发完成 | ArchAgent通知TrainAgent模型可训练 |
| DATA_READY | 数据准备完成 | DataAgent通知TrainAgent数据可用 |
| BLOCKED | 任务阻塞 | TrainAgent发现CANN算子bug，通知ArchAgent |
| BUG_REPORT | Bug报告 | TestAgent发现模型输出错误 |
| EVAL_RESULT | 评估结果 | TrainAgent发送训练指标 |
| DEPLOY_READY | 部署就绪 | InferAgent通知可上线 |

### 协作规则

1. **单向依赖**: Agent间形成DAG，避免循环依赖
2. **异步通信**: 消息队列模式，Agent不阻塞等待
3. **状态同步**: 所有Agent定期读取checklist.md获取全局状态
4. **冲突解决**: 接口变更需通过ArchAgent审批

---

## Agent启动顺序

```
Phase 1: 基础设施
  DataAgent ──→ ArchAgent ──→ TestAgent(基础测试)

Phase 2: 模型核心 (编码器-循环主体-解码器, 15B MoE)
  ArchAgent ──→ NSLAgent(神经语系统)
             ──→ MoEAgent(双层MoE+空专家)
             ──→ CTMAgent(CTM集成, NLM+同步+动态损失)
             ──→ LatentDecodeAgent(隐空间解码, B+C融合)
             ──→ AgentRT(Agent运行时+ReAct+CRA Agent架构)
             ──→ TestAgent(模块测试)

Phase 3: 训练与对齐 (6阶段: 预训练→中间训练→SFT→对齐→持续学习→自我进化)
  DataAgent ──→ TrainAgent ──→ TestAgent(评估)
                └─→ SafetyAgent(安全对齐)
                └─→ MoEAgent(空专家持续填充, Phase5)
                └─→ AgentRT(Self-Play, Phase6)
                └─→ NSLAgent(代码符号空间统一) + CodeAgent(代码生成, 终极目标)

Phase 4: 推理与部署 (多平台)
  TrainAgent ──→ InferAgent ──→ TestAgent(性能/集成测试)
                              └─→ AgentRT(社交部署)
```

---

## Agent资源分配

> **协作模式**: 2人+GLM5.2协作模式。GLM5.2作为AI执行体承担大部分开发工作，2名人类工程师负责监督、决策与关键审查。

| Agent | 计算资源 | 存储 | 协作模式 | 优先级 |
|-------|----------|------|----------|--------|
| ArchAgent | 4卡910B/A100 (开发测试) | 10TB | GLM5.2执行+人监督 | P0 |
| DataAgent | CPU集群 | 1PB | GLM5.2执行+人监督 | P0 |
| TrainAgent | 16-32卡910C/A100 | 200TB | GLM5.2执行+人监督 | P0 |
| InferAgent | 8卡910B/A100 + 4090/3090 + MLX | 50TB | GLM5.2执行+人监督 | P0 |
| TestAgent | 4卡910B/A100 | 20TB | GLM5.2执行+人监督 | P1 |
| SafetyAgent | 4卡910B/A100 | 10TB | GLM5.2执行+人监督 | P1 |
| AgentRT | 4卡910B/A100 | 20TB | GLM5.2执行+人监督 | P1 |
| NSLAgent | 4卡910B/A100 | 10TB | GLM5.2执行+人监督 | P1 |
| MoEAgent | 4卡910B/A100 | 10TB | GLM5.2执行+人监督 | P1 |
| CTMAgent | 4卡910B/A100 | 10TB | GLM5.2执行+人监督 | P1 |
| LatentDecodeAgent | 4卡910B/A100 | 10TB | GLM5.2执行+人监督 | P1 |
| CodeAgent | 8卡910B/A100 (代码训练+评估) | 50TB | GLM5.2执行+人监督 | P0 |
| CANN算子适配 | 4卡910C (算子开发测试) | 5TB | GLM5.2执行+人监督 | P0 |
| 数据工程 | CPU集群 | 500TB | GLM5.2执行+人监督 | P0 |
| DevOps | — | — | GLM5.2执行+人监督 | P0 |
| **合计** | — | — | **2人+GLM5.2** | — |

> **协作模式说明**（2人+GLM5.2协作，GLM5.2承担CANN算子/数据工程/DevOps等专职工作，人类监督）：
> - **CANN算子适配**: GLM5.2负责门控注意力/RoPE-2D/3D RoPE/RocketKV/Ring Attention自定义算子的CANN适配与优化，人类审查关键算子
> - **数据工程**: GLM5.2负责15T+数据清洗/配比/合成，支撑DataAgent的数据Pipeline，人类审核数据质量
> - **DevOps**: GLM5.2负责集群运维/监控/CI-CD，保障16-32卡训练集群稳定运行，人类处理物理故障

---

## Agent自省与改进

每个Agent在每个sprint结束时必须生成：

1. **工作日志**: 完成了哪些任务，耗时多少
2. **阻塞记录**: 遇到了什么问题，如何解决
3. **改进建议**: 对流程/工具/协作的优化建议
4. **知识沉淀**: 关键经验写入项目wiki

Agent的绩效指标：
- 任务按时完成率
- 代码/数据质量 (TestAgent评估)
- 下游Agent满意度
- 知识文档完整性
