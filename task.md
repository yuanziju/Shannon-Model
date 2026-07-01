# Shannon 任务分解与分配 (spec v4.0)

> **用途**: 将项目拆分为可执行的具体任务，明确依赖关系和交付物  
> **阅读对象**: 项目管理AI Agent、开发团队负责人  
> **更新频率**: 每两周review一次  
> **目标规模**: 15B总参数 (MoE) / 激活参数约2-4B

---

## 任务分配原则

1. **模块化**: 每个任务有明确的输入、输出和边界
2. **可并行**: 无依赖的任务尽可能并行执行
3. **可验证**: 每个任务有明确的验收标准
4. **可追踪**: 每个任务有唯一ID、负责人、截止日期

---

## 任务总览图

```
[T1.1] 环境搭建 ──→ [T1.2] 数据Pipeline ──→ [T1.3] 代码框架
    │                                        │
    └────────────────────────────────────────┘
                      ↓
[T2.1] Transformer骨干(编码器-循环主体-解码器) ──→ [T2.2] 模态接口 ──→ [T2.3] SRE引擎 ──→ [T2.4] CANN适配
    │                                                                                          │
    ├──→ [T2.5] 神经语系统 ──┐                                                                │
    ├──→ [T2.6] 双层MoE+自学习空专家 ──┤                                                      │
    └──→ [T2.7] Agent运行时架构 ──┘                                                          │
                      ↓
[T3.1] 数据引擎 ──→ [T3.2] 分布式训练核心(5D并行) ──→ [T3.3] 预训练(Phase1)
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘
                      ↓
[T3.4] 中间训练(Phase2) ──→ [T3.5] SFT(Phase3) ──→ [T3.6] 对齐训练(Phase4, DPO/GRPO)
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘
                      ↓
[T3.7] 持续学习(Phase5) ──→ [T3.8] 自我进化/Self-Play(Phase6) ──→ [T3.9] 代码生成能力强化
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘
                      ↓
[T4.1] 推理引擎架构 ──→ [T4.2] 服务部署(多平台)
    │
    ↓
[T5.1] 测试验收
```

---

## 第一阶段: 基础设施 (Week 1-6)

### T1.1 训练环境搭建

| 任务ID | 任务名 | 负责人 | 工期 | 依赖 | 交付物 | 验收标准 |
|--------|--------|--------|------|------|--------|----------|
| T1.1.1 | 昇腾集群网络配置 | Infra | 3d | - | 网络拓扑图 | 16-32卡AllReduce带宽≥200Gbps |
| T1.1.2 | CANN 8.1安装 | Infra | 2d | T1.1.1 | 安装脚本 | 所有节点`npu-smi info`正常 |
| T1.1.3 | PyTorch NPU验证 | Infra | 2d | T1.1.2 | 测试报告 | 基础训练脚本跑通 |
| T1.1.4 | 分布式存储挂载 | Infra | 3d | T1.1.1 | 存储配置文档 | fio测试吞吐≥50GB/s |
| T1.1.5 | Docker镜像制作 | Infra | 2d | T1.1.3 | Dockerfile | 镜像内跑通训练demo |
| T1.1.6 | 监控告警系统 | Infra | 3d | T1.1.2 | Grafana dashboard | GPU/显存/温度可监控 |
| T1.1.7 | 实验管理工具 | Infra | 2d | T1.1.5 | 配置文档 | 可记录loss/指标 |

### T1.2 数据Pipeline

| 任务ID | 任务名 | 负责人 | 工期 | 依赖 | 交付物 | 验收标准 |
|--------|--------|--------|------|------|--------|----------|
| T1.2.1 | 数据集清单确认 | Data | 3d | - | datasets.md | 覆盖所有模态 |
| T1.2.2 | 数据下载清洗 | Data | 5d | T1.2.1 | 清洗脚本 | 去重率>80% |
| T1.2.3 | MUTANT Tokenizer训练 (SCRIPT预分词+两阶段BPE, 100-128K词表, 文档解析管道预处理后编码) | Data | 5d | T1.2.2 | mutant_tokenizer.py | 100-128K词表, PDF/Word/PPT通过文档解析管道预处理 |
| T1.2.4 | SVG Tokenizer | Data | 7d | - | svg_tokenizer.py | 分层编码，序列压缩60%+ |
| T1.2.5 | ViT+Q-Former AND VAE双通道视觉编码器准备 | Data | 3d | - | vit_qformer_encoder.py | ViT patch提取+Q-Former查询压缩 |
| T1.2.6 | 文档解析管道(PDF/xlsx/docx/pptx双通道) | Data | 3d | - | doc_parser.py | 4格式双通道(视觉渲染+结构提取) |
| T1.2.7 | 视频抽帧编码 | Data | 4d | - | video_encoder.py | 16-32帧均匀采样 |
| T1.2.8 | 合成数据Pipeline | Data | 7d | T1.2.3 | synth_pipeline.py | 数学/代码/几何数据生成 |
| T1.2.9 | 质量评估脚本 | Data | 3d | T1.2.2 | quality_checker.py | 自动过滤低质量 |
| T1.2.10 | 数据版本管理 | Data | 2d | T1.2.2 | DVC配置 | 数据可追溯 |

### T1.3 代码框架

| 任务ID | 任务名 | 负责人 | 工期 | 依赖 | 交付物 | 验收标准 |
|--------|--------|--------|------|------|--------|----------|
| T1.3.1 | 项目目录结构 | Arch | 1d | - | 目录树 | 符合readme.md规范 |
| T1.3.2 | Config类实现 | Arch | 2d | - | config.py | 支持YAML/JSON加载 |
| T1.3.3 | 分布式训练框架 | Arch | 5d | T1.1.3 | distributed.py | 数据/模型并行 |
| T1.3.4 | 多模态Dataloader | Arch | 4d | T1.2.3,T1.2.5 | dataloader.py | 混批支持 |
| T1.3.5 | 训练循环 | Arch | 3d | T1.3.3,T1.3.4 | trainer.py | 混合精度/梯度累积 |
| T1.3.6 | 检查点管理 | Arch | 2d | T1.3.5 | checkpoint.py | 保存/恢复验证 |
| T1.3.7 | 日志系统 | Arch | 2d | T1.3.5 | logger.py | loss/lr/throughput记录 |

---

## 第二阶段: 模型核心 (Week 4-12, 与T1部分并行)

### T2.1 Transformer骨干 (编码器-循环主体-解码器)

| 任务ID | 任务名 | 负责人 | 工期 | 依赖 | 交付物 | 验收标准 |
|--------|--------|--------|------|------|--------|----------|
| T2.1.1 | RMSNorm + SwiGLU | Core | 2d | T1.3.2 | layers.py | 数值与PyTorch一致 |
| T2.1.2 | 三维Native-RoPE | Core | 3d | T2.1.1 | rope.py | T/H/W位置正确编码 |
| T2.1.3 | 因果注意力 (ATB) | Core | 3d | T1.1.3 | attention.py | CANN上跑通 |
| T2.1.4 | 双向注意力 | Core | 2d | T2.1.3 | attention.py | 图像块内全连接 |
| T2.1.5 | 混合掩码切换 | Core | 3d | T2.1.3,T2.1.4 | mask_utils.py | 动态掩码生成 |
| T2.1.6 | 编码器/解码器实现 (3%/3%) | Core | 3d | T2.1.1 | encoder_decoder.py | 比例1:32:1, 运行一次 |
| T2.1.7 | 循环主体 + 动态1-32次迭代 | Core | 7d | T2.1.6 | recurrent_body.py | 权重共享, 自适应深度1-32步 |
| T2.1.8 | Silent Thinking | Core | 2d | T2.1.7 | recurrent_body.py | 仅最终步算loss |
| T2.1.9 | 动态注意力控制器 | Core | 5d | T2.1.3 | dynamic_attn.py | 辅助网络<1%参数 |
| T2.1.10 | 模态感知深度嵌入 | Core | 3d | T2.1.7 | depth_embed.py | 模态×循环步(1-32)联合 |
| T2.1.11 | 梯度检查点 | Core | 2d | T2.1.1 | checkpoint_utils.py | 显存节省验证 |

### T2.2 模态接口

| 任务ID | 任务名 | 负责人 | 工期 | 依赖 | 交付物 | 验收标准 |
|--------|--------|--------|------|------|--------|----------|
| T2.2.1 | 文本Embedding | Modal | 2d | T1.2.3 | embed.py | 含特殊token |
| T2.2.2 | ViT+Q-Former AND VAE双通道视觉编码 | Modal | 3d | T1.2.5 | image_encoder.py | ViT patch→Q-Former压缩→d_model投影 |
| T2.2.3 | 图像扩散解码 | Modal | 5d | T2.2.2 | image_decoder.py | 模型决策路由外部工具(非扩散) |
| T2.2.4 | 视频时序编码 | Modal | 5d | T1.2.7 | video_encoder.py | 3D卷积+时序注意力 |
| T2.2.5 | 文档解析(PDF/xlsx/docx/pptx双通道) | Modal | 4d | T1.2.6 | pdf_processor.py | 渲染+提取双路径 |
| T2.2.6 | SVG Tokenizer | Modal | 3d | T1.2.4 | svg_tokenizer.py | 分层编码实现 |
| T2.2.7 | SVG Decoder | Modal | 3d | T2.2.6 | svg_decoder.py | 自回归SVG生成 |
| T2.2.8 | 模态类型Embedding | Modal | 2d | T2.2.1 | modality_embed.py | 标识token模态 |

### T2.3 特化推理引擎 (SRE)

| 任务ID | 任务名 | 负责人 | 工期 | 依赖 | 交付物 | 验收标准 |
|--------|--------|--------|------|------|--------|----------|
| T2.3.1 | SymPy通道 | SRE | 4d | T2.1.1 | sympy_channel.py | AST编码+计算 |
| T2.3.2 | Lean/Coq通道 | SRE | 5d | T2.1.1 | lean_channel.py | 证明状态编码 |
| T2.3.3 | Python通道 | SRE | 3d | T2.1.1 | python_channel.py | 执行结果编码 |
| T2.3.4 | Cross-Attention Fusion | SRE | 4d | T2.1.3,T2.3.1-3 | fusion_layer.py | Layer 8/16/24插入 |
| T2.3.5 | Tool Gating | SRE | 3d | T2.3.4 | tool_gating.py | 动态权重门控 |
| T2.3.6 | Tool Coordinator | SRE | 5d | T2.3.4 | coordinator.py | 链式调度+变量共享 |
| T2.3.7 | Tool Memory | SRE | 3d | T2.3.6 | tool_memory.py | 跨工具命名空间映射 |
| T2.3.8 | 错误注入数据 | SRE | 4d | T1.2.8 | error_dataset.py | 语法/类型/逻辑错误集 |
| T2.3.9 | GRPO奖励函数 | SRE | 3d | T2.3.8 | reward.py | 分层奖励实现 |
| T2.3.10 | Function Calling接口 | SRE | 3d | T2.3.6 | tool_api.py | 统一路由 |

### T2.4 CANN适配优化

| 任务ID | 任务名 | 负责人 | 工期 | 依赖 | 交付物 | 验收标准 |
|--------|--------|--------|------|------|--------|----------|
| T2.4.1 | 文本分支CANN跑通 | Opt | 3d | T2.1.1,T1.1.3 | cann_text.py | 前向/反向正确 |
| T2.4.2 | ATB FlashAttention | Opt | 4d | T2.1.3 | atb_attn.py | 替换标准attention |
| T2.4.3 | 循环深度CANN验证 | Opt | 3d | T2.1.6,T2.4.1 | cann_loop.py | 权重共享稳定 |
| T2.4.4 | VAE编解码CANN | Opt | 4d | T2.2.2,T2.2.3 | cann_vae.py | 算子精度验证 |
| T2.4.5 | 混合掩码CANN测试 | Opt | 3d | T2.1.5,T2.4.2 | cann_mask.py | causal+bidirectional |
| T2.4.6 | 算子融合配置 | Opt | 3d | T2.4.1 | fusion_config.json | auto_fusion生效 |
| T2.4.7 | 图编译验证 | Opt | 3d | T2.4.6 | ge_compile.py | 整网编译成功 |
| T2.4.8 | FP16混合精度 | Opt | 2d | T2.4.1 | amp_config.py | 精度损失<1% |
| T2.4.9 | 内存复用优化 | Opt | 2d | T2.4.1 | memory_opt.py | 峰值显存降低 |
| T2.4.10 | 性能Profiling | Opt | 2d | T2.4.7 | perf_report.md | tokens/sec基线 |

### T2.5 神经语系统 (Neuro-Symbolic Language)

| 任务ID | 任务名 | 负责人 | 工期 | 依赖 | 交付物 | 验收标准 |
|--------|--------|--------|------|------|--------|----------|
| T2.5.1 | 符号↔神经双向翻译层 | NSL | 5d | T2.1.6 | symbol_neural_bridge.py | 符号与隐空间对齐 |
| T2.5.2 | 神经语词表与文法 | NSL | 4d | T2.5.1 | neuro_grammar.json | 覆盖数学/逻辑/代码符号 |
| T2.5.3 | 形式化表示解析器 | NSL | 4d | T2.5.2 | formal_parser.py | Lean/SymPy表达式互转 |
| T2.5.4 | 神经语嵌入对齐训练 | NSL | 5d | T2.5.1,T2.2.1 | nsl_align_trainer.py | 隐空间一致性损失收敛 |
| T2.5.5 | 神经语生成与解码 | NSL | 4d | T2.5.4 | nsl_decoder.py | 符号输出可执行 |
| T2.5.6 | 神经语与循环主体集成 | NSL | 3d | T2.5.5,T2.1.7 | nsl_integration.py | 循环迭代中符号状态传递 |

### T2.6 双层MoE + 自学习空专家

| 任务ID | 任务名 | 负责人 | 工期 | 依赖 | 交付物 | 验收标准 |
|--------|--------|--------|------|------|--------|----------|
| T2.6.1 | 双层MoE路由 (分层: 浅8/中16/深24, 细粒度FFN=1024, Top-2~4路由) | MoE | 6d | T2.1.7 | dual_moe.py | Top-4×Top-4路由正确 |
| T2.6.2 | 大专家(分层配置: 浅8/中16/深24)实现 | MoE | 4d | T2.6.1 | big_experts.py | 16大专家加载/切换 |
| T2.6.3 | 小专家(FFN=1024, 细粒度)实现 | MoE | 4d | T2.6.1 | small_experts.py | 16×16小专家稀疏激活 |
| T2.6.4 | 自学习空专家框架 | MoE | 5d | T2.6.1 | empty_expert.py | 零初始化, 训练中逐步填充 |
| T2.6.5 | 空专家能力吸收机制 | MoE | 5d | T2.6.4 | expert_absorb.py | 新能力无需重训即可注入 |
| T2.6.6 | 负载均衡与专家容量 | MoE | 3d | T2.6.2,T2.6.3 | load_balancer.py | 无专家过载/饥饿 |
| T2.6.7 | 双层MoE All-to-All通信 | MoE | 4d | T2.6.1 | moe_all2all.py | EP并行通信优化 |

### T2.7 Agent运行时架构

| 任务ID | 任务名 | 负责人 | 工期 | 依赖 | 交付物 | 验收标准 |
|--------|--------|--------|------|------|--------|----------|
| T2.7.1 | Agent运行时核心 | Agent | 5d | T2.1.7 | agent_runtime.py | 任务分解+多步规划 |
| T2.7.2 | 工具编排与调度 | Agent | 4d | T2.7.1,T2.3.6 | tool_orchestrator.py | 多工具链式/并行编排 |
| T2.7.3 | 长程记忆模块 | Agent | 5d | T2.7.1 | long_term_memory.py | 外部记忆检索+工作记忆 |
| T2.7.4 | 自我反思与纠错 | Agent | 4d | T2.7.1 | self_reflect.py | 失败重试+经验沉淀 |
| T2.7.5 | 社交部署接口 | Agent | 4d | T2.7.1 | social_deploy.py | 多Agent协作/角色扮演 |
| T2.7.6 | 图灵测试接口 | Agent | 3d | T2.7.5 | turing_test_api.py | 社交平台对接 |
| T2.7.7 | Self-Play对弈框架 | Agent | 5d | T2.7.4 | self_play.py | 自我对弈+经验回放 |

---

## 第三阶段: 训练引擎与对齐 (Week 10-24, 6阶段训练流程)

> 训练流程扩展为6阶段: 预训练(Phase1) → 中间训练(Phase2) → SFT(Phase3) → 对齐(Phase4) → 持续学习(Phase5) → 自我进化/Self-Play(Phase6)

### T3.1 数据引擎

| 任务ID | 任务名 | 负责人 | 工期 | 依赖 | 交付物 | 验收标准 |
|--------|--------|--------|------|------|--------|----------|
| T3.1.1 | 多模态DataLoader实现 | Train | 4d | T1.3.4 | multimodal_dataloader.py | 文本/图像/视频/PDF/SVG混批 |
| T3.1.2 | 动态比例调度器 | Train | 3d | T3.1.1 | ratio_scheduler.py | 基于评估反馈自动调整比例 |
| T3.1.3 | 合成数据Pipeline | Train | 5d | T1.2.8 | synth_pipeline.py | Self-Instruct+代码验证+RAG |
| T3.1.4 | 数据质量过滤 | Train | 3d | T3.1.1 | quality_filter.py | 困惑度/长度/格式评分 |
| T3.1.5 | 数据版本管理 | Train | 2d | T3.1.1 | data_versioning.py | DVC集成, 数据可追溯 |

### T3.2 分布式训练核心 (5D并行 + 多优化器)

| 任务ID | 任务名 | 负责人 | 工期 | 依赖 | 交付物 | 验收标准 |
|--------|--------|--------|------|------|--------|----------|
| T3.2.1 | 张量并行(TP)框架 | Train | 4d | T1.3.3 | tensor_parallel.py | 多卡单节点内并行 |
| T3.2.2 | 序列并行(SP)实现 | Train | 4d | T3.2.1 | sequence_parallel.py | 长序列分片, 5D并行组成 |
| T3.2.3 | 专家并行(EP)实现 | Train | 5d | T3.2.1 | expert_parallel.py | 双层MoE 16大专家并行 |
| T3.2.4 | 流水线并行(PP)实现 | Train | 4d | T3.2.1 | pipeline_parallel.py | 多 stage inter-node |
| T3.2.5 | 双层MoE All-to-All优化 | Train | 5d | T3.2.3 | moe_all2all.py | 双缓冲流水线通信 |
| T3.2.6 | 梯度累积与超大Batch | Train | 3d | T3.2.2 | mega_batch.py | 8M+ tokens稳定训练 |
| T3.2.7 | 5D并行(TP+PP+DP+SP+EP)整合 | Train | 5d | T3.2.1-T3.2.5 | parallel_5d.py | 五维并行协同 |
| T3.2.8 | SAGE优化器实现(主力) | Train | 4d | T1.3.5 | sage_optimizer.py | 主力优化器收敛验证 |
| T3.2.9 | Muon + AdEMAMix + SCALE | Train | 5d | T3.2.8 | multi_optimizer.py | 多优化器分层组合 |
| T3.2.10 | 动态损失加权 | Train | 3d | T1.3.5 | dynamic_loss.py | 不确定性估计加权 |
| T3.2.11 | CANN图编译适配 | Train | 5d | T2.4.7 | cann_compile.py | 静态图编译成功 |
| T3.2.12 | BF16混合精度训练 | Train | 3d | T2.4.8 | amp_bf16.py | 精度损失<1% |

### T3.3 预训练 (Phase 1)

| 任务ID | 任务名 | 负责人 | 工期 | 依赖 | 交付物 | 验收标准 |
|--------|--------|--------|------|------|--------|----------|
| T3.3.1 | 小规模warmup | Train | 5d | T3.2.6,T3.2.12 | warmup_report.md | 小模型训练稳定 |
| T3.3.2 | 15B(MoE)初始化 | Train | 3d | T3.3.1 | init_checkpoint | 双层MoE权重分布正常 |
| T3.3.3 | 预训练启动 (Phase 1) | Train | 12-16w | T3.3.2 | phase1_checkpoint | 15T+ tokens, 16-32卡910C, 分阶段引入: 1a Dense→1b MoE→1c门控→1d RDT |
| T3.3.4 | 训练监控仪表盘 | Train | 持续 | T3.3.3 | dashboard | loss/梯度/吞吐量实时 |
| T3.3.5 | 啊哈时刻检测系统 | Train | 3d | T3.3.3 | aha_detector.py | loss突变自动检测 |
| T3.3.6 | 检查点动态策略 | Train | 3d | T3.3.3 | checkpoint_engine.py | 中频+高频自动切换 |
| T3.3.7 | 中间评估 | Train | 持续 | T3.3.3 | eval_reports/ | 每1000步评估 |
| T3.3.8 | 数据混合调优 | Train | 持续 | T3.3.7 | mix_config.yaml | 代码30%+/理科30%+ |
| T3.3.9 | 预训练基线测试 | Train | 3d | T3.3.3 | baseline_report.md | MMLU/MMBench达标 |

### T3.4 中间训练 (Phase 2)

| 任务ID | 任务名 | 负责人 | 工期 | 依赖 | 交付物 | 验收标准 |
|--------|--------|--------|------|------|--------|----------|
| T3.4.1 | 中间训练数据 | Train | 5d | T1.2.8,T3.3.3 | midtrain_data/ | 代码+数学+推理 |
| T3.4.2 | Silent Thinking训练 | Train | 3w | T3.4.1,T2.1.8 | phase2_checkpoint | 推理能力增强 |
| T3.4.3 | 动态循环深度实验 | Train | 5d | T3.4.2 | depth_exp_report.md | 1-32次迭代对比 |
| T3.4.4 | Relaxed LoRA训练 | Train | 5d | T3.4.2 | lora_adapters/ | 逐循环适配器收敛 |
| T3.4.5 | 自学习空专家激活 | Train | 5d | T3.4.2,T2.6.4 | empty_expert_ckpt | 空专家开始填充新能力 |

### T3.5 SFT (Phase 3)

| 任务ID | 任务名 | 负责人 | 工期 | 依赖 | 交付物 | 验收标准 |
|--------|--------|--------|------|------|--------|----------|
| T3.5.1 | SFT指令数据 | Align | 7d | T1.2.8 | sft_data.jsonl | 多模态对话+工具+编辑 |
| T3.5.2 | SFT训练 | Align | 2w | T3.5.1,T3.3.3 | sft_checkpoint | 指令遵循能力提升 |
| T3.5.3 | 神经语SFT对齐 | Align | 5d | T3.5.2,T2.5.4 | nsl_sft_ckpt | 符号输出对齐 |
| T3.5.4 | SFT评估 | Align | 3d | T3.5.2 | sft_eval.md | 指令基准达标 |

### T3.6 对齐训练 (Phase 4, DPO/GRPO)

| 任务ID | 任务名 | 负责人 | 工期 | 依赖 | 交付物 | 验收标准 |
|--------|--------|--------|------|------|--------|----------|
| T3.6.1 | DPO偏好数据 | Align | 5d | T3.5.2 | dpo_data.jsonl | 人工+模型标注 |
| T3.6.2 | DPO训练 | Align | 1w | T3.6.1 | dpo_checkpoint | 偏好对齐 |
| T3.6.3 | 错误感知数据 | Align | 5d | T2.3.8 | error_train_data/ | 错误注入完整 |
| T3.6.4 | 错误对比学习 | Align | 1w | T3.6.3,T3.6.2 | contrast_checkpoint | 错误区分能力 |
| T3.6.5 | GRPO训练 | Align | 2w | T3.6.4,T2.3.9 | grpo_checkpoint | 工具使用优化 |
| T3.6.6 | 安全对齐 | Align | 1w | T3.6.2 | safety_checkpoint | 内容过滤有效 |
| T3.6.7 | 最终评估 | Align | 3d | T3.6.5 | final_eval.md | 全基准测试 |

### T3.7 持续学习 (Phase 5)

| 任务ID | 任务名 | 负责人 | 工期 | 依赖 | 交付物 | 验收标准 |
|--------|--------|--------|------|------|--------|----------|
| T3.7.1 | 持续学习数据流 | Train | 5d | T3.6.7 | cl_data_stream/ | 在线数据接入 |
| T3.7.2 | 增量训练框架 | Train | 5d | T3.7.1 | incremental_trainer.py | 灾难性遗忘抑制 |
| T3.7.3 | 自学习空专家持续填充 | Train | 持续 | T3.4.5,T2.6.5 | expert_growth_log | 新能力动态吸收 |
| T3.7.4 | 知识巩固与回放 | Train | 4d | T3.7.2 | replay_buffer.py | 旧任务性能保持 |

### T3.8 自我进化 / Self-Play (Phase 6)

| 任务ID | 任务名 | 负责人 | 工期 | 依赖 | 交付物 | 验收标准 |
|--------|--------|--------|------|------|--------|----------|
| T3.8.1 | Self-Play对弈引擎 | Agent | 5d | T3.7.2,T2.7.7 | self_play_engine.py | 自我对弈生成数据 |
| T3.8.2 | 经验回放强化训练 | Train | 1w | T3.8.1 | rl_checkpoint | 强化学习收敛 |
| T3.8.3 | 自我评估与提升循环 | Agent | 持续 | T3.8.2 | self_improve_loop.py | 能力自动提升 |
| T3.8.4 | 社交图灵测试数据收集 | Agent | 持续 | T2.7.6 | turing_data/ | 社交平台交互数据 |

### T3.9 代码生成能力强化 (终极目标)

> **终极目标对齐**：无与伦比的代码生成准确性和上下文。本任务集聚焦高质量代码数据、执行反馈训练、全库级代码理解、Self-Play代码对弈与专项评估。

| 任务ID | 任务名 | 负责人 | 工期 | 依赖 | 交付物 | 验收标准 |
|--------|--------|--------|------|------|--------|----------|
| T3.9.1 | 高质量代码数据Pipeline (GitHub/Stack/执行反馈) | Data | 2w | T1.2.8 | code_data_pipeline.py | GitHub Star>100仓库 + 执行反馈 + CodeContests + 全库级代码，覆盖datasets.md §4.1质量策略 |
| T3.9.2 | 代码执行反馈训练 (REPL-in-the-loop) | Train | 2w | T3.9.1 | repl_train.py | 生成→执行→反馈→修正闭环收敛，代码可执行率提升 |
| T3.9.3 | 全库级代码理解训练 (5M上下文代码库基准) | Train | 3w | T3.9.2,T5.1.3 | codebase_understanding.py | 5M上下文 Needle-in-Codebase 召回>90% |
| T3.9.4 | Self-Play代码对弈 (生成-验证-修复循环) | Agent | 持续 | T3.8.2,T3.9.2 | code_self_play.py | 生成-验证-修复闭环跑通，代码对弈样本持续产出 |
| T3.9.5 | 代码生成专项评估 (SWE-bench/LiveCodeBench/全库理解) | QA | 1w | T3.9.3 | code_eval_report.md | HumanEval pass@1>85% / SWE-bench Verified>30% / LiveCodeBench>60% / BugFix修复率>70% |

### T3.10 CTM集成训练 (分阶段, spec 14.2决策)

| 任务ID | 任务名 | 负责人 | 工期 | 依赖 | 交付物 | 验收标准 |
|--------|--------|--------|------|------|--------|----------|
| T3.10.1 | CTM阶段0试点: 直觉模块 | Train | 2w | T3.4.2 | intuition_ctm.py | NLM+同步小规模验证 (决策C9) |
| T3.10.2 | NLM训练warmup策略 | Train | 3d | T3.10.1 | nlm_warmup.py | 前10%冻结NLM, 稳定收敛 |
| T3.10.3 | CTM动态损失选择训练 | Train | 4d | T3.10.2 | ctm_loss.py | min-loss+max-certainty tick (决策C8) |
| T3.10.4 | NLM参数5D并行切分 | Train | 3d | T3.2.5 | nlm_parallel.py | 跟随EP切分 (决策C13) |
| T3.10.5 | NLM vs 深度LoRA消融 | Train | 1w | T3.10.3 | ablation_report.md | 数据决定是否冗余 (决策C17) |
| T3.10.6 | MLA潜变量同步训练 | Train | 5d | T3.10.3 | mla_sync_train.py | c_kv·c_kv^T同步, 代码任务验证 (决策C5/C14) |
| T3.10.7 | Spiking NN+Liquid NN调研 | Train | 2w | - | alt_report.md | 替代方案对比报告 (决策C16) |
| T3.10.8 | CTM回退评估 | Train | 3d | T3.10.6 | ctm_eval.md | 决定保留NLM+动态损失或全保留 (决策C12) |

### T3.11 隐空间解码实现 (B+C融合架构, spec 14.3决策)

| 任务ID | 任务名 | 负责人 | 工期 | 依赖 | 交付物 | 验收标准 |
|--------|--------|--------|------|------|--------|----------|
| T3.11.1 | 神经语连续主+离散边界码簿 | NSL | 5d | T2.5.1 | neuro_codebook.py | 连续向量+5类离散边界标记 (决策L2) |
| T3.11.2 | 推理/解码模式切换+decode-LoRA | Core | 5d | T2.1.7 | mode_switch.py | mode='reasoning'/'decoding'+独立LoRA (决策L3) |
| T3.11.3 | 层次化NAR展开(方案B) | Core | 7d | T3.11.1 | hierarchical_nar.py | 段落→句子→token三级展开+多目标长度预测 (决策L6) |
| T3.11.4 | 掩码迭代精化(方案C, 复用RDT) | Core | 7d | T3.11.2 | mask_refine.py | 全MASK→并行精化, decode-LoRA+mode切换 (决策L3) |
| T3.11.5 | 流匹配全局规划层(方案A, 可选) | Core | 5d | T3.11.3 | flow_planner.py | >512token+批量+写作类时启用 (决策L5) |
| T3.11.6 | AR保底通道+三级置信度门控 | Core | 4d | T3.11.3,T3.11.4 | ar_fallback.py | token<0.55/块<0.70/全局<0.75触发回退 (决策L7) |
| T3.11.7 | Speculative Decoding(NAR draft+AR verify) | Infer | 5d | T3.11.4,T3.11.6 | spec_decode.py | NAR作draft+AR作verify, 共享RDT主体 (决策L13) |
| T3.11.8 | 拟人流式输出前端 | Frontend | 5d | T3.11.6 | human_stream.py | 已发送token可"删除重打", 15%修订率上限, 50-200ms延迟 (决策L11) |
| T3.11.9 | 形式化证明强制AR+Lean验证器 | Core | 4d | T3.11.6 | lean_verifier.py | 检测证明类输出跳过NAR, 强制AR (决策L4) |
| T3.11.10 | 隐空间解码训练四阶段 | Train | 3w | T3.11.3,T3.11.4 | latent_train.py | 重建→层次展开→掩码精化→联合微调 (决策L9) |
| T3.11.11 | 层级数据无监督学习 | Train | 2w | T3.11.10 | hierarchy_learn.py | 对比学习/自编码器/注意力分析 (决策L10/L15) |

---

## 第四阶段: 推理与部署 (Week 20-28)

### T4.1 推理引擎架构 (15B MoE / 激活参数约2-4B)

| 任务ID | 任务名 | 负责人 | 工期 | 依赖 | 交付物 | 验收标准 |
|--------|--------|--------|------|------|--------|----------|
| T4.1.1 | API网关与负载均衡 | Infer | 3d | T4.2.1 | api_gateway.py | gRPC/HTTP2, 限流生效 |
| T4.1.2 | 请求调度器(连续批处理) | Infer | 5d | T4.1.1 | scheduler.py | P0-P4优先级, 抢占换出 |
| T4.1.3 | 多模态预处理流水线 | Infer | 4d | T2.2.x | preprocessor.py | 图像/视频/PDF/SVG并行处理 |
| T4.1.4 | 统一注意力内核调度器 | Infer | 7d | T2.1.3,T2.1.5 | unified_attn.py | MLA/KDA/Lightning/SSM动态切换+门控注意力(rank-64低秩) |
| T4.1.5 | DeepSeek MLA推理实现 | Infer | 5d | T4.1.4 | mla_kernel.py | KV Cache压缩75%+ |
| T4.1.6 | Kimi KDA推理实现 | Infer | 5d | T4.1.4 | kda_kernel.py | 线性注意力, 128K+序列 |
| T4.1.7 | Lightning Attention推理 | Infer | 4d | T4.1.4 | lightning_kernel.py | 块级累积, 恒定速度 |
| T4.1.8 | 滑动窗口注意力推理 | Infer | 3d | T4.1.4 | sliding_kernel.py | 图像patch局部感知 |
| T4.1.9 | 分层缓存管理器 | Infer | 4d | T4.1.5-T4.1.8 | cache_manager.py | KV/SSM/循环状态统一管理 |
| T4.1.10 | 双层MoE推理优化 | Infer | 6d | T2.6.1 | moe_inference.py | Top-2~4路由, 分层专家预加载 |
| T4.1.11 | All-to-All通信优化 | Infer | 4d | T4.1.10 | all2all_opt.py | 双缓冲流水线 |
| T4.1.12 | 工具调用执行器 | Infer | 4d | T2.3.x | tool_executor.py | SymPy/Lean/Python流式执行 |
| T4.1.13 | 图像编辑工具链 | Infer | 4d | T2.2.3 | image_toolchain.py | OpenCV/SAM2/ComfyUI/SVG编排 |
| T4.1.14 | 量化引擎 | Infer | 5d | T4.1.4 | quantizer.py | 逐组件量化: 大专家FFN=W8A16, KV Cache=FP8, 其余=FP16 (NVFP4仅Blackwell,当前不可用) |
| T4.1.15 | 动态精度切换 | Infer | 3d | T4.1.14 | dynamic_precision.py | 层敏感度感知切换 |
| T4.1.16 | 显存管理器 | Infer | 4d | T4.1.9 | memory_manager.py | Paged KV + SSM Swap + 重计算+梯度检查点+RocketKV+语义块KV缓存压缩+FP8 (NVFP4仅Blackwell,当前不可用) (15B显存优化) |
| T4.1.17 | 流式解码输出 | Infer | 3d | T4.1.2 | stream_decoder.py | token-by-token SSE |
| T4.1.18 | 推理性能基线 | Infer | 2d | T4.1.16 | perf_baseline.md | TTFT/TPOT/吞吐达标 (15B MoE, 激活参数约2-4B) |
| T4.1.19 | 多平台适配 | Infer | 3w | T4.1.14 | multi_backend/ | 双卡4090/3090 + 单卡A100/H100 + 昇腾910C + Apple Silicon(MLX) |
| T4.1.20 | CTM路由器分流实现 | Infer | 4d | T4.1.10 | ctm_router.py | NLM/标准/共享专家三类分流 (决策C6) |
| T4.1.21 | NLM增强专家推理路径 | Infer | 5d | T4.1.20 | nlm_expert.py | 共享模板+低秩适配, 历史维护 (决策C2/C3) |
| T4.1.22 | MLA潜变量同步推理 | Infer | 3d | T4.1.5 | mla_sync.py | c_kv·c_kv^T同步矩阵, 按任务激活 (决策C5) |

### T4.2 服务化部署

| 任务ID | 任务名 | 负责人 | 工期 | 依赖 | 交付物 | 验收标准 |
|--------|--------|--------|------|------|--------|----------|
| T4.2.1 | API服务框架 | Deploy | 3d | T4.1.6 | api_server.py | RESTful API |
| T4.2.2 | 多模态输入服务 | Deploy | 3d | T2.2.5,T2.2.7 | upload_service.py | 图像/视频/PDF上传 |
| T4.2.3 | 认证鉴权 | Deploy | 3d | T4.2.1 | auth_module.py | API Key/OAuth2 |
| T4.2.4 | 限流配额 | Deploy | 2d | T4.2.1 | rate_limiter.py | RPM/TPM限制 |
| T4.2.5 | 负载均衡 | Deploy | 2d | T4.2.1 | nginx.conf | 多实例分发 |
| T4.2.6 | 日志审计 | Deploy | 3d | T4.2.1 | logging_service.py | 全链路追踪 |
| T4.2.7 | 监控告警 | Deploy | 2d | T4.2.5 | prometheus_rules.yml | QPS/延迟/错误率 |
| T4.2.8 | 私有化部署脚本 | Deploy | 3d | T4.2.1 | deploy_private.sh | Docker Compose/K8s |
| T4.2.9 | 消费级GPU部署 (双卡4090/3090) | Deploy | 2w | T4.1.19 | consumer_gpu_pkg/ | 双卡量化推理可用 |
| T4.2.10 | 单卡A100/H100部署 | Deploy | 1w | T4.1.19 | a100_pkg/ | 单卡高性能推理 |
| T4.2.11 | 昇腾910C部署包 (CANN 8.1, 算子融合, HCCL) | Deploy | 1w | T4.1.19 | ascend_pkg/ | CANN原生推理 |
| T4.2.12 | Apple Silicon (MLX) 部署 | Deploy | 2w | T4.1.19 | mlx_pkg/ | MLX后端推理可用 |
| T4.2.13 | 边缘设备适配 (Ascend 310) | Deploy | 1w | T4.1.19 | edge_model/ | 轻量化模型 |

---

## 第五阶段: 测试验收 (Week 24+, 持续)

### T5.1 测试验收

| 任务ID | 任务名 | 负责人 | 工期 | 依赖 | 交付物 | 验收标准 |
|--------|--------|--------|------|------|--------|----------|
| T5.1.1 | 文本生成测试 | QA | 3d | T3.3.9 | text_test_report.md | 多主题/多语言 |
| T5.1.2 | 图像理解测试 | QA | 3d | T3.3.9 | image_test_report.md | DocVQA/MMBench |
| T5.1.3 | 视频理解测试 | QA | 3d | T3.3.9 | video_test_report.md | 时序准确性 |
| T5.1.4 | PDF问答测试 | QA | 3d | T3.3.9 | pdf_test_report.md | 复杂版面 |
| T5.1.5 | 代码生成测试 | QA | 5d | T3.3.9 | code_test_report.md | HumanEval+自建 |
| T5.1.6 | 数学证明测试 | QA | 5d | T3.3.9 | math_test_report.md | 竞赛题+Lean |
| T5.1.7 | 图像编辑测试 | QA | 3d | T3.3.9 | edit_test_report.md | 主观+A/B |
| T5.1.8 | SVG生成测试 | QA | 3d | T3.3.9 | svg_test_report.md | 语法+几何约束 |
| T5.1.9 | 工具调用测试 | QA | 3d | T3.3.9 | tool_test_report.md | 错误恢复率 |
| T5.1.10 | 长上下文测试 | QA | 3d | T3.3.9 | longctx_test.md | 128K+Needle |
| T5.1.11 | 性能基准测试 | QA | 3d | T4.1.9 | perf_test_report.md | 吞吐/延迟达标 |
| T5.1.12 | 安全测试 | QA | 5d | T3.3.8 | safety_test.md | 有害内容过滤 |

---

## [Shannon融合] Shannon融合专项: 28项差异决策实现 (跨阶段执行)

> **说明**：本节为Shannon架构融合的差异决策实现任务集，共28项差异决策，下表列出15项核心实现子任务（覆盖架构/数据/训练/推理/Agent/部署全栈）。任务跨阶段并行执行，依赖关系标注于各子任务。本节任务编号 `T5.1.1`–`T5.1.15` 为Shannon融合专项编号，与第五阶段"测试验收"任务集属不同轨道，以本节标题区分。

### T5.1 Shannon融合: 28项差异决策实现

| 任务ID | 任务名 | 负责人 | 工期 | 依赖 | 交付物 | 验收标准 |
|--------|--------|--------|------|------|--------|----------|
| T5.1.1 | 门控注意力实现 (Hybrid-M3第8种头) | Arch | 5d | T2.1.3,T2.1.9 | gated_attn.py | 第8种门控头融合, 精度对齐基线 |
| T5.1.2 | KV Cache四层压缩栈实现 | Infer | 6d | T4.1.5 | kv_compress_stack.py | 四层压缩(量化+低秩+池化+截断), KV压缩>85% |
| T5.1.3 | Ring Attention分布式实现 | Train | 7d | T3.2.2 | ring_attn.py | 跨节点环形通信, 支持128K+序列 |
| T5.1.4 | MUTANT Tokenizer实现 | Data | 7d | T1.2.3 | mutant_tokenizer.py | SCRIPT预分词+两阶段BPE, 100-128K词表, 支持PDF/Word/PPT |
| T5.1.5 | 文档解析管道 (PDF/xlsx/docx/pptx) | Data | 6d | T1.2.6 | doc_parser.py | 双通道(原生+OCR+版面分析), 全格式结构化输出 |
| T5.1.6 | MoE层投机解码实现 | Infer | 6d | T4.1.10 | moesd_decode.py | MoE专家投机解码, 推理加速1.5x+ |
| T5.1.7 | MTP训练增强实现 | Train | 5d | T3.3.3 | mtp_trainer.py | 多Token预测(Multi-Token Prediction)训练, 吞吐提升验证 |
| T5.1.8 | 梯度检查点+激活重计算实现 | Train | 5d | T2.6.1 | alchemist.py | 激活缓存复用, 训练显存降低验证 |
| T5.1.9 | ReAct+CRA Agent架构实现 | Agent | 7d | T2.7.1 | coalm_agent.py | ReAct引擎+CRA数据集+对话状态管理器 (决策11) |
| T5.1.10 | 知识编辑 (ROME/MEMIT)实现 | Train | 5d | T3.7.2 | knowledge_edit.py | ROME/MEMIT知识编辑, 局部更新无需重训 |
| T5.1.11 | 量化策略融合 (逐组件量化) | Infer | 5d | T4.1.14 | quant_fusion.py | 逐组件量化融合, 精度损失<2% (NVFP4仅Blackwell,当前不可用) |
| T5.1.12 | 性能量化分析 | QA | 3d | T5.1.11 | quant_analysis.md | 各量化策略吞吐/显存/精度对比矩阵 |
| T5.1.13 | 昇腾910C详细适配 | Opt | 2w | T2.4.x,T4.1.19 | ascend910c_pkg/ | 910C算子/通信/图编译全适配, 性能达标 |
| T5.1.14 | 模型蒸馏路线 | Train | 3w | T3.6.7 | distill_pipeline.py | 三档蒸馏(15B→7B→3B), 逻辑+特征+MoE蒸馏, 能力保留>85% (决策22) |
| T5.1.15 | 评估矩阵+人格测试 | QA | 5d | T5.1.9 | eval_matrix.md | 理科/代码/多模态/Agent评估矩阵 + 人格一致性测试 |

### 任务说明

- **跨阶段并行**：T5.1.1-T5.1.5 可在第二/三阶段并行启动；T5.1.6-T5.1.11 依赖推理/训练基础模块；T5.1.13-T5.1.15 为后期集成与验证。
- **决策映射**：T5.1.4→决策6, T5.1.5→决策13, T5.1.9→决策11, T5.1.14→决策22；其余对应Hybrid-M3/MoE层投机解码/MTP/梯度检查点+激活重计算/ROME-MEMIT/量化/910C等Shannon差异决策。
- **与现有任务关系**：部分子任务升级/扩展现有任务（如T5.1.4升级T1.2.3, T5.1.5升级T1.2.6/T2.2.5），不删除原有任务，标注升级关系。

### T5.2 Shannon融合补充任务

| 任务ID | 任务名 | 负责人 | 工期 | 依赖 | 交付物 | 验收标准 |
|--------|--------|--------|------|------|--------|----------|
| T5.2.1 | ViT+Q-Former AND VAE双通道视觉编码器实现 | Modal | 7d | T1.2.5 | vit_qformer.py | patch提取+查询压缩+d_model投影 (决策S20) |
| T5.2.2 | 多模态位置编码系统 | Core | 5d | T2.1.2 | multi_pos_enc.py | RoPE/YaRN/RoPE-2D/1D RoPE+时序衰减/3D RoPE+LongRoPE2 (决策S4) |
| T5.2.3 | 分阶段训练策略(1a-1d) | Train | 2w | T3.3.1 | phased_train.py | 1a Dense→1b MoE→1c门控→1d RDT (决策S12) |
| T5.2.4 | 上下文长度分阶段扩展 | Train | 3w | T5.1.3 | ctx_length_expand.py | 32K→128K→512K→2M→5M (决策S19) |
| T5.2.5 | 持续学习三层CLS实现 | Train | 1w | T3.7.2,T5.1.8 | cls_three_tier.py | 热线(RDT状态)+温线(空专家缓存)+冷线(LoRA巩固) (决策S10) |
| T5.2.6 | 故障恢复融合策略 | Infra | 5d | T3.2.7 | fault_recovery.py | 心跳+故障转移+Checkpoint+弹性+监控+自动重启 (决策S26) |
| T5.2.7 | 数据配比融合实现 | Data | 3d | T3.1.2 | data_ratio.py | 代码30%/理科30%/中文25%/英文10%/多语言5% (决策S25) |

---

## 关键路径分析

**关键路径** (决定项目最短工期的任务链):

```
T1.1.1 → T1.1.2 → T1.1.3 → T1.3.3 → T2.1.1 → T2.1.3 → T2.4.1 → T2.4.2
→ T3.1.1 → T3.1.2 → T3.1.3 → T3.3.2 → T3.3.3 (12-16w, 16-32卡910C, 15T+) → T3.3.7 → T3.3.9
→ T3.4.2 → T3.5.2 → T3.6.5 → T3.7.2 → T3.8.2 → T3.9.5 (代码生成评估)
→ T4.1.1 → T4.1.19 → T4.2.1 → T5.1.1 → T3.11.10 (隐空间解码训练) → T5.2.3 (分阶段训练)
```

**预计最短工期**: 40-52周 (10-13个月，含6阶段训练+CTM+隐空间解码+Shannon融合)

> **并行优化说明**：部分训练阶段可并行：CTM训练(阶段1-2)可与SFT(Phase 3)并行；隐空间解码训练(阶段1-2)可与对齐训练(Phase 4)并行。但核心串行路径仍需40-52周。

---

## 资源需求汇总

| 资源类型 | 数量 | 使用阶段 | 说明 |
|----------|------|----------|------|
| Ascend 910C | 16-32卡 (5D并行配置: TP=2×PP=2×DP=2×EP=4=32卡 (16-32卡范围)) | T1-T3 | 训练主集群 |
| NVIDIA A100/H100 | 8卡+ | T1-T4 | 训练/推理优化测试 |
| 消费级 4090/3090 | 2-4卡 | T4 | 双卡部署测试 |
| Apple Silicon (M系列) | 若干 | T4 | MLX后端测试 |
| Ascend 310P | 2卡 | T4 | 边缘部署测试 |
| 存储 | 1PB+ | T1-T3 | 15T+原始数据+15B检查点 |
| 人力 | 2人+GLM5.2 | 全程 | 见agents.md分工 |
