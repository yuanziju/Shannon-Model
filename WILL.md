# Shannon 项目遗言 (WILL.md)

> **用途**: 防止上下文窗口丢失导致工作丢失。每次完成任务后更新此文件,与代码一起提交。
> **规则**: 遗言随代码在同一提交中,不单独提交。

---

## 项目概况

- **项目名**: Shannon AI Model Project
- **仓库**: https://github.com/yuanziju/Shannon-Model
- **工作区**: /workspace
- **工作流**: dev{hash}分支工作 → 验证不崩溃 → 合并main。沙箱崩溃则基于main重建dev{hash}。
- **当前分支**: dev_a654009 (从 main a654009 fork)

## 当前状态 (2026-07-09 重建)

### 沙箱重置历史
- 第3次P0事故: common/(78文件)+Shannon(29文件)+MathMaster(24文件)全部丢失
- main分支只有初始提交 a654009,之前所有提交未保存到git
- 在 dev_a654009 分支上从零重建

### 代码统计
| 模块 | 文件数 | 代码行数 |
|------|--------|----------|
| common/ | 76 | 15,193 |
| models/Shannon/ | 30 | 5,868 |
| models/MathMaster/ | 27 | 9,194 |
| **合计** | **133** | **30,255** |

### 模块清单

#### common/ (76文件) - 跨模型共享基础设施
- `device.py` - Backend枚举(CANN/CUDA/MLX/CPU)+DeviceManager
- `distributed.py` - 5D并行(TP+PP+DP+SP+EP)+DistributedManager
- `checkpoint.py` - 检查点管理(分片/异步/啊哈时刻)
- `metrics.py` - 指标追踪(10基准+啊哈时刻检测)
- `data_utils.py` - 数据配比(代码30%/理科30%/中文25%/英文10%/多语言5%)
- `quantization.py` - 逐组件量化(大专家W8A16/KV=FP8/其余FP16)
- `attention_utils.py` - 掩码+RoPE+QKNorm+SDPA
- `layers/` - RMSNorm/GatedRMSNorm/SwiGLU/RoPE(1D/2D/3D)/YaRN/LongRoPE2/AttnRes/mHC/GradientCheckpoint
- `attention/` - Hybrid-M3 8种: MLA/KDA/Lightning/Sliding/MMA/MoH/Gated/Dynamic+Unified调度器
- `ctm/` - NLM神经元级模型/MLASync同步/CTMDynamicLoss/CTMRouter
- `nsl/` - 神经语系统: SymbolNeuralBridge/NSLGrammar/FormalParser/NSLDecoder
- `latent_decode/` - B+C融合: NeuroCodebook/ModeSwitch/HierarchicalNAR/MaskRefinement/FlowPlanner/ARFallback/SpeculativeDecoder/HumanStream/LeanVerifier
- `sre/` - 特化推理引擎: SymPyChannel/LeanChannel/PythonChannel/CrossAttentionFusion/ToolGating/ToolCoordinator/ToolMemory
- `agent/` - AgentRuntime/ToolOrchestrator/LongTermMemory/SelfReflect/SocialDeploy/SelfPlay/ReActCRAAgent
- `inference/` - RequestScheduler/CacheManager/Quantizer/MemoryManager/StreamDecoder/MoESpeculativeDecoder
- `training/` - ShannonTrainer/TrainingCheckpoint/DynamicLossWeighter/MoEBalanceLoss/MTPLoss/MultiOptimizer(SAGE/Muon/AdEMAMix)/PhasedTrainer/Evaluator

#### models/Shannon/ (30文件) - Shannon 15B MoE 模型
- `src/config/` - ShannonConfig
- `src/model/` - ShannonModel(Encoder3%+Body94%+Decoder3%)
- `src/moe/` - 双层MoE: DualLayerRouter/Experts/EmptyExpert/ExpertAbsorber/LoadBalancer/AllToAll
- `src/recurrent/` - 循环主体: RecurrentBody/DepthEmbed/LTI/DepthLoRA/ACTStop
- `src/encoder/` - TextEmbed/ViTQFormer+VAE/Video/DocParser/SVGTokenizer/ModalityEmbed
- `src/decoder/` - Decoder(B+C融合)/SVGDecoder/StructuredOutput/ImageEditRouter

#### models/MathMaster/ (27文件) - 数学专精模型 (30-70B MoE, 新底子)
- `src/config/` - MathConfig (70B MoE, 10M上下文, 10AB/5路/5子agent/6常驻+16×16双层MoE)
- `src/model/` - MathModel (新底子架构)
- `src/proof/` - Lean4Prover/SymPySolver/NumberTheoryVerifier(反例发现强化)/CoqProver/Isabelle/Metamath/ProofRouter
- `src/reasoning/` - LongReasoningEngine/SelfPlayDebate(多Agent辩论)/CoTDistillation/SelfPlaySolver/MathToT/ConjectureGenerator/ReasoningCheckpoint
- `src/training/` - MathRLHF(5层)/MathDataGenerator(9领域)/MathTrainer(5阶段)/MathEvaluator(8基准)/MathCurriculum(4级)

## MathMaster 新底子架构 (用户确认)

```
输入 → 神经语编码(NSL)
 ↓
[Looped 循环主体] (1-32次动态迭代)
  每轮迭代包含4部分:
    1. 残差池(ResidualPool): AttnRes+mHC + attention检索"有用笔记" + 每轮删除非AB残差+压缩 + 每3轮top-k
    2. 直觉层(IntuitionLayer): 基础版(轻量MLP+隐变量采样),待后续完善(参考Shannon)
    3. AB堆叠(10个AB固定堆叠):
       每个AB内部:
         - 5路Attention(A1-A5) [Hybrid-M3 8种]
         - 动态路由(1对1置换,"电线盒",元路由器,Sinkhorn/Gumbel-Softmax)
         - 5个子agent(G1-G5),每个=不同路由策略
         - 共享专家池:
           * 6常驻专家(4固定+2可学习,密集,不受路由)
           * 16大×16小双层MoE(256小+16大,稀疏,路由选择)
         - MoE汇总 → 5路输出
       AB之间: 固定1对1(编号对编号,不做路由)
    4. 循环控制: AB输出直接传递 + 残差池管理 + 每3轮top-k + 深度嵌入
 ↓
[解码器] → 输出(text/lean4/sympy/conjecture/proof_step/confidence)
```

## 验证结果 (最严格标准)
- py_compile: 133个.py文件全部通过
- common/ 全模块导入: PASS
- Shannon Model: 19.8M params(mini), 前向+反向通过
- MathMaster Model: 20.4M params(mini), 前向+反向通过, 11个输出头
- Proof: SymPy solve/simplify/diff正确, 数论验证器prime/goldbach正确
- Reasoning: ToT搜索+多Agent辩论+CoT蒸馏正确
- Training: 5阶段+9领域合成+MathRLHF分层奖励正确

## 待完善 (下轮对话)
- 直觉层(IntuitionLayer): 参考Shannon设计,待专门讨论
- AB堆叠的"学习路径+伪MoE密集型": 用户说后续展开
- 残差池历史存储的"巧妙工程方案": 用户说后续讲
- 推理层改底子的上下文依赖细节: 待逐层确认

## Git 工作流
1. 不直接编辑main分支
2. 创建dev{hash}分支(基于main的commit hash)
3. 在dev上工作,验证不崩溃
4. 不崩溃→合并main; 崩溃→基于main重建dev{hash2}
5. 子代理多建分支避免并行提交冲突
6. 提交后clone到temp验证远程一致性
