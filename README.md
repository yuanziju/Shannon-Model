# 原生统一多模态AI模型 (spec v4.0)

> **代号**: Shannon (不再使用 λ-Oracle 代号)  
> **定位**: 面向理科极客的超强多模态理解与编辑AI  
> **架构范式**: 编码器(3%) - 循环主体(94%) - 解码器(3%) × 双层MoE × 原生统一  
> **目标规模**: 15B总参数 (MoE) / 15B MoE（激活参数约2-4B）  
> **部署平台**: 双卡4090/3090 + 单卡A100/H100 + 华为昇腾910C + Apple Silicon(MLX)  
> **核心创新**: 双层MoE + 动态循环深度(1-32次) + CTM神经元级模型 + 隐空间解码 + ReAct+CRA Agent架构  
> **融合来源**: Shannon架构 (28项差异决策) + CTM (17项决策) + 隐空间解码 (15项决策)

---

## 项目愿景

构建一个**真正统一**的多模态AI模型——从token层面让文本、图像、视频、PDF、SVG、代码、数学符号在同一个Transformer空间中共存、交互、推理。

它应该像一个**强的可怕的理科生**：
- 看一眼几何图，就能画出辅助线并给出严谨的证明
- 读一段代码需求，就能生成可运行的完整软件（对标Claude Fable5）
- 打开一份PDF论文，就能提取公式、图表并回答细节问题
- 给它一张图，它能智能修图、换背景、加标注、画辅助线
- 给它一道数学题，它能用自然语言、符号计算、形式化证明三种方式解答

---

## 核心能力矩阵

| 能力域 | 输入 | 输出 | 技术特色 |
|--------|------|------|----------|
| 文本理解与生成 | 文本/PDF/Word/PPT/xlsx | 文本/代码/JSON | MUTANT Tokenizer通过文档解析管道预处理支持 |
| 图像理解 | 静态图像/截图(任意分辨率) | 文本描述/结构化数据 | ViT+Q-Former AND VAE双通道, OCR-free |
| 视频理解 | 短视频/长视频(数小时) | 文本摘要/时序分析 | 密集采样+压缩记忆机制 |
| 图像编辑 | 图像 + 文本指令 | 编辑后图像 | 模型决策路由外部工具(OpenCV/SAM2/ComfyUI) |
| 代码生成 | 自然语言/注释 | 可执行代码 | 对标Claude Fable5，双层MoE+动态循环深度增强 |
| 数学证明 | 文本命题/几何图 | 证明过程/SVG图形 | 特化推理引擎，工具神经元集成 |
| SVG生成 | 文本描述/几何命题 | SVG矢量图 | 分层几何tokenization |
| 工具调用 | 用户请求 | 工具执行结果 | ReAct+CRA Agent架构 (ReAct+CRA格式) + 流式返回 |
| 隐空间解码 | 神经语 | 高质量文本 | B+C融合: 层次化NAR+掩码精化+流匹配可选+AR保底 |

---

## 架构总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Shannon 架构全景 (v4.0)                                │
│              编码器(3%) - 循环主体(94%) - 解码器(3%)  比例 1:32:1             │
│              [Shannon融合] + [CTM集成] + [隐空间解码]                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  输入层                                                                      │
│  ├── 文本Tokenizer (MUTANT: SCRIPT预分词+两阶段BPE, 100-128K, 文档解析预处理) │
│  ├── 视觉编码器 (ViT + Q-Former AND VAE双通道, 原生任意分辨率, RoPE-2D位置编码)  │
│  ├── 视频编码器 (4fps密集采样 → 时序注意力聚合, 3D RoPE位置编码)               │
│  ├── 文档解析 (PDF/xlsx/docx/pptx 双通道: 视觉渲染+结构提取)                  │
│  ├── SVGTokenizer (分层几何tokenization, 坐标对数离散化)                       │
│  ├── 神经语系统 (Neuro-Symbolic Language, 符号↔神经双向翻译)                   │
│  ├── Tool Channel (SymPy/Lean/Python 编码器, 动态维度自适应)                   │
│  └── 多模态位置编码 (RoPE/YaRN/RoPE-2D/1D RoPE+时序衰减/3D RoPE + LongRoPE2) │
├─────────────────────────────────────────────────────────────────────────────┤
│  统一Transformer骨干 (编码器-循环主体-解码器 × 双层MoE)                        │
│  ├── 编码器 (~3% 参数, 1份)                                                   │
│  │   └── 标准Transformer, 运行一次, 生成模态锚点与上下文初始化                  │
│  ├── 循环主体 (~94% 参数, 动态迭代1-32次)                                      │
│  │   ├── Hybrid-M3注意力 (MLA/KDA/Lightning/MMA/MoH/Sliding + 门控注意力)     │
│  │   │   └── 门控注意力 (rank-64低秩, 27M参数, 消除Attention Sink)             │
│  │   ├── 动态注意力权重生成 (小型交叉注意力网络)                                │
│  │   ├── 双层MoE FFN (分层: 浅8/中16/深24, 细粒度FFN=1024, Top-2~4路由)        │
│  │   │   ├── NLM增强专家 (复杂任务, CTM神经元级模型)                           │
│  │   │   ├── 标准专家 (简单任务)                                               │
│  │   │   ├── 常驻共享专家 (DeepSeek模式)                                      │
│  │   │   └── 自学习空专家: 初始零参数, 训练中逐步填充新能力                     │
│  │   ├── CTM集成: NLM + MLA潜变量同步 + 动态损失选择                           │
│  │   ├── LTI稳定性约束 (谱半径<1, 防止残差爆炸)                                │
│  │   ├── 循环索引嵌入 (正弦深度位置编码, 1-32步)                               │
│  │   ├── 深度LoRA适配器 (逐循环轻量适配)                                       │
│  │   ├── ACT自适应停止 + CTM动态损失(min-loss+max-certainty)                  │
│  │   └── 动态循环深度控制 (1-32次, skip/exec/repeat编解码路由)                │
│  ├── 解码器 (~3% 参数, 1份)                                                   │
│  │   └── 标准Transformer, 运行一次, 映射到输出空间 / 隐空间解码(B+C融合)       │
│  ├── 三维Native-RoPE (T/H/W解耦) + 1D RoPE+时序衰减/xPos/LongRoPE2外推       │
│  ├── RMSNorm + 可学习门控 (DeepNorm风格)                                      │
│  ├── Kimi AttnRes (深度方向注意力残差)                                        │
│  └── DeepSeek mHC (流形约束超链接, Birkhoff polytope)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  特化推理引擎 (Specialized Reasoning Engine)                                  │
│  ├── Tool Coordinator: 多工具调度与变量共享命名空间                             │
│  ├── Cross-Attention Fusion: 工具输出作为神经元输入 (动态层插入)               │
│  ├── Tool Gating: 动态门控机制                                                │
│  ├── 错误感知训练: 工具语法/逻辑错误纳入GRPO/DPO损失                            │
│  └── 流式返回: 工具输出实时流式返回模型                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  ReAct+CRA Agent架构                                                         │
│  ├── Agent运行时: ReAct+CRA格式 (任务分解/推理/行动/观察/反思)                 │
│  ├── 长程记忆: 外部记忆检索 + 工作记忆管理                                     │
│  ├── 自我进化: Self-Play 对弈 + 经验回放强化                                  │
│  └── 社交部署: 多Agent协作 / 角色扮演 / 图灵测试接口                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  推理引擎架构 (Inference Engine)                                             │
│  ├── 统一注意力内核层: MLA/KDA/Lightning/Sliding/SSM/门控注意力 动态调度       │
│  ├── 请求调度器: 连续批处理 + 优先级队列(P0-P4) + 多模态预处理流水线            │
│  ├── 双层MoE推理优化: 双层路由 + 专家预加载 + All-to-All通信优化               │
│  ├── Ring Attention: 分布式长序列注意力 (GPU环拓扑)                            │
│  ├── 工具与图像执行层: SymPy/Lean/Python + OpenCV/SAM2/ComfyUI/SVG            │
│  ├── 量化与压缩: 逐组件量化(大专家FFN=W8A16,KV Cache=FP8,其余=FP16) (NVFP4仅Blackwell,当前不可用) + 动态精度切换   │
│  ├── KV Cache栈: PagedAttention + RocketKV + 语义块压缩 + INT8 (NVFP4仅Blackwell,当前不可用) (~300x压缩)   │
│  ├── MoE层投机解码 (1.5-2x加速)                                              │
│  └── 显存管理: Paged KV + SSM State Swap + 梯度检查点+激活重计算 + 选择性重计算│
├─────────────────────────────────────────────────────────────────────────────┤
│  输出层 (含隐空间解码, B+C融合架构)                                          │
│  ├── 文本Decoder (自回归生成 + MoE层投机解码)                                  │
│  ├── 图像编辑 (模型决策路由 → OpenCV/SAM2/ComfyUI/SVG渲染)                      │
│  ├── SVGDecoder (自回归SVG token生成)                                         │
│  ├── 隐空间解码 (B+C融合: 层次化NAR+掩码精化+流匹配可选+AR保底)                  │
│  │   └── 拟人流式: 已发送token可"删除重打", 像真人发现输错                     │
│  ├── MTP训练增强 (k=2-4 token预测, 仅训练阶段)                                 │
│  └── 结构化输出 (JSON/表格/标注框)                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 核心技术栈

| 层级 | 技术选型 | 说明 |
|------|----------|------|
| 硬件 | 华为昇腾910C + NVIDIA A100/H100 + 消费级4090/3090 | 训练用昇腾/A100；推理支持多平台 |
| 计算框架 | CANN 8.1 + PyTorch NPU / CUDA + PyTorch / MLX (Apple Silicon) | 多后端适配，昇腾原生优先 |
| 模型框架 | 自研 (编码器-循环主体-解码器 × 双层MoE) | 动态循环深度1-32次 + 原生统一 |
| Tokenizer | MUTANT (SCRIPT预分词 + 两阶段BPE) | 100-128K词表, 通过文档解析管道预处理支持(PDF/Word/PPT) |
| 视觉编码 | ViT + Q-Former AND VAE双通道 | 原生任意分辨率, 可学习查询压缩 |
| 注意力 | Hybrid-M3 (MLA+KDA+Lightning+MMA+MoH+门控注意力) | 混合注意力融合, rank-64低秩门控 |
| 位置编码 | 多模态感知 (RoPE/YaRN/RoPE-2D/1D RoPE+时序衰减/3D RoPE) | + LongRoPE2长上下文外推 |
| KV Cache | PagedAttention + RocketKV + 语义块KV缓存压缩 + INT8 (NVFP4仅Blackwell,当前不可用) | ~300x压缩栈(MLA 4x×RocketKV 10x×语义块压缩×FP8 2x) |
| 训练框架 | 5D并行 (TP+PP+DP+SP+EP) | 张量/流水线/数据/序列/专家并行 |
| 训练精度 | BF16(昇腾910C，不支持FP8) + BF16(其他) | 融合量化策略 |
| 推理引擎 | 多层架构 (统一注意力内核+调度器+双层MoE优化+量化) | Ring Attention + MoE层投机解码 |
| 推理量化 | 逐组件量化: 大专家FFN=W8A16, KV Cache=FP8, 其余=FP16 (NVFP4仅Blackwell,当前不可用) | 融合双方量化策略 |
| 优化器 | SAGE(主力) + Muon + AdEMAMix + SCALE | 多优化器组合，分层LR调度 |
| 增强技术 | Kimi AttnRes + DeepSeek mHC | 深度注意力残差 + 流形约束超连接 |
| CTM集成 | NLM(神经元级模型) + MLA潜变量同步 + 动态损失选择 | C→B分阶段演进，路由器分流NLM/标准专家 |
| 隐空间解码 | B+C融合: 层次化NAR+掩码精化复用RDT+流匹配可选+AR保底 | 四层架构, MoE层投机解码, 前端拟人动效 |
| MTP | 多Token预测 (k=2-4) | 训练增强, 推理用隐空间解码 |
| 神经语系统 | 神经-符号双向翻译层 | 符号推理与神经表示统一空间 |
| 自学习空专家 | 零初始化专家逐步填充 | 训练中动态吸收新能力，无需重训 |
| Agent架构 | ReAct+CRA Agent架构 (ReAct+CRA格式) | 单Agent+多Agent协作 |
| 工具集成 | SymPy + Lean4 + Python REPL + OpenCV/SAM2/ComfyUI | 特化工具通道 + 图像编辑工具链 |
| 投机解码 | MoE层投机解码 + NAR draft | MoE层1.5-2x加速 + 解码层NAR-AR互补 |
| 持续学习 | 梯度检查点+激活重计算 + 三层CLS | 省32-47%内存, 分层知识管理 |
| 知识编辑 | ROME/MEMIT + 三层CLS协同 | 热层可用知识编辑替代LoRA |
| 部署平台 | 双卡4090/3090 + 单卡A100/H100 + 昇腾910C + Apple Silicon(MLX) | 全平台覆盖，消费级GPU可运行 |
| 上下文长度 | 分阶段: 32K→128K→512K→2M→5M | LongRoPE2 + Ring Attention + KDA/SSM互补 |
| 模型蒸馏 | 15B→7B→3B | 逻辑+特征+MoE蒸馏 (future.md) |

---

## 关键参数规格

| 参数 | 目标值 | 备注 |
|------|--------|------|
| 总参数量 | ~15B (MoE) | 双层MoE大专家 + 小专家 + 注意力/嵌入/投影 |
| 激活参数量 (推理) | 15B MoE（激活参数约2-4B） | Shannon融合 |
| 架构比例 | 编码器3% : 循环主体94% : 解码器3% | 比例1:32:1 |
| 模型结构 | 编码器-循环主体-解码器 / 高维 / 多头 / 大FFN(MoE) | 配合动态循环深度 |
| 循环深度 | 1-32次动态迭代 | 循环主体整体循环; 编解码逐层skip/exec/repeat |
| 双层MoE | 分层: 浅8/中16/深24, 细粒度FFN=1024 | Top-2~4路由, 保留自学习空专家 |
| 文本词表 | 100-128K | MUTANT (SCRIPT+两阶段BPE), 通过文档解析管道预处理支持 |
| 上下文长度 | 分阶段: 32K→128K→512K→2M→5M | LongRoPE2+Ring Attention+KDA/SSM |
| 训练数据 | 15T+ tokens | 代码30%/理科30%/中文25%/英文10%/多语言5%, 代码生成为终极目标 |
| Batch Size | 全局8M tokens | 超大batch |
| 学习率 | 分层 (1e-3/3e-4/1e-4) | 嵌入/深层/输出层递减 |
| 训练精度 | BF16(昇腾910C，不支持FP8) + BF16(其他) | 融合量化策略 |
| 推理量化 | 逐组件量化: 大专家FFN=W8A16, KV Cache=FP8, 其余=FP16 (NVFP4仅Blackwell,当前不可用) | 融合双方量化策略 |
| KV Cache压缩 | PagedAttention+RocketKV+语义块KV缓存压缩+INT8 (NVFP4仅Blackwell,当前不可用) | ~300x压缩栈(MLA 4x×RocketKV 10x×语义块压缩×FP8 2x) |
| 门控注意力 | rank-64低秩, 27M参数 | 消除Attention Sink (第8种头类型) |
| 图像分辨率 | 原生任意分辨率 | ViT+Q-Former AND VAE双通道, RoPE-2D编码位置 |
| 视频采样 | 4fps密集采样 | 时序注意力聚合 |
| 推理显存 | ~24-48GB (FP16, 量化) | 目标双卡4090/3090或单卡A100/H100可运行 |
| 训练阶段 | 预训练→中间训练→SFT→对齐→持续学习→自我进化 | 6阶段 + 分阶段引入(1a-1d) |
| 终极目标 | 无与伦比的代码生成准确性和上下文 | 代码能力终极应用 |

---

## 项目文件结构

```
multimodal-ai-project/
├── readme.md                ← 本文件，人类可读的项目总览 (v4.0)
├── spec.md                  ← 模型架构核心规格 (v4.0, 含Shannon融合28项决策)
├── inference_engine.md      ← 推理引擎架构 (2007行, 含13个Shannon融合章节)
├── training_engine.md       ← 训练引擎架构 (含10个Shannon融合章节)
├── checklist.md             ← 开发检查清单 (199项)
├── task.md                  ← 任务分解与分配 (含Shannon融合15项任务)
├── future.md                ← 未来发展规划 (含模型蒸馏/上下文分阶段)
├── agents.md                ← AI代理任务分配规范 (含ReAct+CRA Agent架构)
├── datasets.md              ← 开源数据集与合成数据策略 (含MUTANT/文档解析)
├── all-in-one.md            ← 终极融合文档（全部内容汇总）
└── brainstorming/
    ├── 01_parallel_inference.md       ← 并行推理设计
    ├── 02_dynamic_weight_generation.md ← 动态权重生成
    ├── 03_latent_space_decoding.md    ← 隐空间解码 (15项决策)
    ├── 03b_latent_decoding_implementation.md ← 隐空间解码实现方案 (1254行)
    ├── 04_intuition.md                ← 直觉能力设计
    ├── 05_ctm_reference.md            ← CTM参考设计 (17项决策)
    └── 05b_ctm_implementation.md      ← CTM实现方案 (1619行)
```

---

## 快速导航

- **想理解模型架构?** → 阅读 `spec.md` 第1-7章 (核心) + 第14章 (决策汇总)
- **想看Shannon融合决策?** → 阅读 `spec.md` 第14.4节 (28项决策表)
- **想看CTM集成设计?** → 阅读 `spec.md` 第4.6节 + 第14.2节 + `brainstorming/05b_ctm_implementation.md`
- **想看隐空间解码?** → 阅读 `spec.md` 第9.2节 + 第14.3节 + `brainstorming/03b_latent_decoding_implementation.md`
- **想深入推理引擎设计?** → 阅读 `inference_engine.md`
- **想深入训练引擎设计?** → 阅读 `training_engine.md`
- **想知道要做什么任务?** → 阅读 `task.md`
- **想检查进度?** → 阅读 `checklist.md`
- **想了解数据需求?** → 阅读 `datasets.md`
- **是AI Agent要接手工作?** → 阅读 `agents.md`

---

## 当前状态

- [x] 需求收集与架构方向确定 (17轮深度问答, 68+问题)
- [x] 核心技术调研 (循环深度/双层MoE/AttnRes/mHC/神经语系统/自学习空专家)
- [x] CTM集成决策 (17项决策, C→B分阶段演进)
- [x] 隐空间解码决策 (15项决策, B+C融合架构)
- [x] Shannon架构融合 (28项差异决策, 逐项对比选择)
- [x] 详细架构设计冻结 (spec v4.0)
- [ ] 数据pipeline搭建 (15T+ tokens, 理科主导, MUTANT Tokenizer)
- [ ] 预训练 (Phase 1: 1a Dense → 1b MoE → 1c 门控 → 1d RDT)
- [ ] 中间训练 (Phase 2)
- [ ] SFT (Phase 3)
- [ ] 对齐训练 (Phase 4, DPO/GRPO)
- [ ] 持续学习 (Phase 5, 梯度检查点+激活重计算 + 三层CLS)
- [ ] 自我进化 / Self-Play (Phase 6)
- [ ] 推理引擎架构开发 (统一注意力内核/调度器/双层MoE优化/量化/MoE层投机解码)
- [ ] 多平台部署 (双卡4090/3090 + 单卡A100/H100 + 昇腾910C + Apple Silicon)
- [ ] 无与伦比的代码生成准确性和上下文 (终极目标)

---

## 终极目标: 无与伦比的代码生成准确性和上下文

Shannon的终极派生目标是**无与伦比的代码生成准确性和上下文**：

- 在代码生成任务上达到超越人类的准确性与工程可用性，对标Claude Fable5
- Shannon借助循环深度(1-32次动态迭代) + 双层MoE + 神经语系统 + CTM + Self-Play自我进化
- 超长上下文(5M)支撑全库级代码理解、跨文件推理与大型工程重构
- 这是理科极客导向架构选择的代码能力终极验证场景

---

## 设计假设与待确认

以下维度基于最佳实践假设，如需调整请反馈：

1. **安全对齐**: 本地部署不限制，线上跟随行业主流
2. **开源策略**: 待用户最终确认
3. **项目时间线**: 最短工期40-52周 (10-13个月)
4. **团队规模**: 2人+GLM5.2
5. **商业模式**: 三档定价+开源策略+私有化部署 (详见 future.md)
