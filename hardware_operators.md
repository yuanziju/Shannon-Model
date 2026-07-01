# Shannon 硬件算子参考代码 (Hardware Operator Reference)

> **用途**: 本文件存放 Shannon 模型 3 个 Hybrid 级算子的硬件相关参考实现代码。
> 包含: Python 逻辑验证脚本 + Triton 参考实现 + CUDA C++ 框架 + 昇腾 CANN 适配要点。
> 关联文档: spec.md §15 (算子融合策略) + 附录K/L/M (计算流程分析)。

---

## 1. Python 逻辑验证脚本

> 验证 3 个 Hybrid 算子的朴素实现 vs 融合实现数值一致性。
> 运行: `python3 verify_operators.py`

### 验证结果

```
Test 1: HybridSeeDNorm (Naive vs Fused)     ✓ PASS  (rel_diff=6.46e-07)
Test 2: HybridMoE (Naive vs Fused)           ✓ PASS  (rel_diff=0.00e+00)
Test 3: HybridMLA (Naive vs Fused)           ✓ PASS  (rel_diff=4.17e-07)
Test 4: Operator Pipeline (SeeDNorm→MLA→SeeDNorm→MoE)  ✓ PASS
Test 5: Numerical Stability (大/小/零输入+QK-Clip极端)  ✓ PASS
总计: 5/5 通过
```

### 关键验证点

| 算子 | 验证内容 | 结果 |
|------|---------|------|
| HybridSeeDNorm | rms_norm + alpha_net + gamma 融合 vs 分步 | max_diff=4.29e-06 |
| HybridSeeDNorm | alpha 正性 (Softplus 保证) | min=0.5887 > 0 ✓ |
| HybridSeeDNorm | 反向传播梯度 | grad norm=119.69 ✓ |
| HybridMoE | 双层路由 + 专家GEMM + 空专家 融合 vs 分步 | max_diff=0.00 (完全一致) |
| HybridMoE | 路由稀疏性 | Top-2/4 = 50% active ✓ |
| HybridMoE | 空专家梯度 detach | backward pass OK ✓ |
| HybridMLA | QKV+RoPE+DoPE+QK-Clip+SDPA+O_proj 融合 vs 分步 | max_diff=1.49e-08 |
| HybridMLA | QK-Clip logit 裁剪 | raw max=24203 → clipped=30.00 ✓ |
| HybridMLA | DoPE 高频衰减 | gate min=0.0, max=1.0 ✓ |
| Pipeline | SeeDNorm→MLA→SeeDNorm→MoE 协同 | no NaN/Inf, std=0.87 ✓ |
| Stability | 大输入 (σ=1000) | output std=19.57, no NaN ✓ |
| Stability | 零输入 | output max=0.0, no NaN ✓ |

---

## 2. HybridSeeDNorm — Triton 参考实现

```python
import triton
import triton.language as tl

@triton.jit
def hybrid_seednorm_kernel(
    x_ptr, gamma_ptr,
    w1_ptr, b1_ptr, w2_ptr, b2_ptr,  # alpha_net weights
    out_ptr,
    d_model: tl.constexpr,
    alpha_dim: tl.constexpr,  # 64
    eps: tl.constexpr,         # 1e-6
    BLOCK_M: tl.constexpr,     # token tile size, 16 or 32
):
    """
    HybridSeeDNorm fused kernel (Triton)
    融合: rms_norm + alpha_net(Linear→GELU→Linear→Softplus) + γ·scale
    HBM 流量: 2X (读x, 写out), 中间张量全部在SRAM
    """
    pid = tl.program_id(0)
    # 行索引: 每个program处理 BLOCK_M 个 token
    row_start = pid * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)

    # ========== Pass 1: 加载 x + 计算 rms + alpha_net ==========

    # 加载 x tile: (BLOCK_M, d_model)
    x = tl.load(x_ptr + rows[:, None] * d_model + tl.arange(0, d_model)[None, :],
                mask=rows[:, None] < N_TOKENS)

    # RMS (FP32 accumulate)
    sum_sq = tl.sum(x.to(tl.float32) * x.to(tl.float32), axis=1)  # (BLOCK_M,)
    rms = tl.sqrt(sum_sq / d_model + eps)  # (BLOCK_M,)
    rms_inv = 1.0 / rms  # (BLOCK_M,)

    # Alpha net: Linear1 (d_model → 64)
    # W1: (64, d_model), 分块加载
    h = tl.zeros([BLOCK_M, alpha_dim], dtype=tl.float32)
    for k in range(0, d_model, BLOCK_K):
        x_block = tl.load(x_ptr + rows[:, None] * d_model + k + tl.arange(0, BLOCK_K)[None, :])
        w1_block = tl.load(w1_ptr + tl.arange(0, alpha_dim)[:, None] * d_model + k + tl.arange(0, BLOCK_K)[None, :])
        h += tl.dot(x_block.to(tl.float32), w1_block.T.to(tl.float32))

    # GELU
    h = h * 0.5 * (1.0 + tl.tanh(0.7978845608 * (h + 0.044715 * h * h * h)))

    # Linear2 (64 → 1)
    w2 = tl.load(w2_ptr + tl.arange(0, alpha_dim))  # (64,)
    b2 = tl.load(b2_ptr)
    alpha = tl.sum(h * w2[None, :], axis=1) + b2  # (BLOCK_M,)

    # Softplus
    alpha = tl.log(1.0 + tl.exp(alpha))  # (BLOCK_M,)

    # ========== Pass 2: 归一化 + 缩放 (寄存器内) ==========

    # 加载 gamma
    gamma = tl.load(gamma_ptr + tl.arange(0, d_model))  # (d_model,)

    # output = gamma * (x / rms) * alpha
    out = gamma[None, :] * (x.to(tl.float32) * rms_inv[:, None]) * alpha[:, None]

    # 写回
    tl.store(out_ptr + rows[:, None] * d_model + tl.arange(0, d_model)[None, :],
             out.to(tl.bfloat16),
             mask=rows[:, None] < N_TOKENS)


def hybrid_seednorm_triton(x, gamma, w1, b1, w2, b2):
    """Python wrapper for HybridSeeDNorm Triton kernel"""
    B, S, D = x.shape
    N_TOKENS = B * S
    x_flat = x.reshape(N_TOKENS, D).contiguous()
    out = torch.empty_like(x_flat)

    BLOCK_M = 32
    grid = (triton.cdiv(N_TOKENS, BLOCK_M),)
    hybrid_seednorm_kernel[grid](
        x_flat, gamma, w1, b1, w2, b2, out,
        D, 64, 1e-6, BLOCK_M,
        num_warps=4, num_stages=2,
    )
    return out.reshape(B, S, D)
```

---

## 3. HybridSeeDNorm — CUDA C++ 框架

```cpp
// hybrid_seednorm.cu
// Fused: RMSNorm + AlphaNet(Linear→GELU→Linear→Softplus) + GammaScale
// Target: NVIDIA GPU (sm_80+) / 昇腾910C (via CANN AscendC)

#include <cuda_runtime.h>
#include <cuda_bf16.h>

// ========== Kernel 配置 ==========
#define BLOCK_M 32        // token tile
#define WARP_SIZE 32
#define NUM_WARPS 4       // 128 threads per block
#define ALPHA_DIM 64

// ========== Shared Memory 布局 ==========
// 总计 ~92 KB (适配 910C Unified Buffer)
// | x_tile[BLOCK_M][D]     | 32 * 8192 * 2 = 512 KB → 分块, 只存 BLOCK_M * BLOCK_K
// | rms[BLOCK_M]           | 32 * 4 = 128 B
// | alpha[BLOCK_M]         | 32 * 4 = 128 B
// | W1_tile[ALPHA_DIM][BK] | 64 * 256 * 2 = 32 KB
// | W2[ALPHA_DIM]          | 64 * 2 = 128 B
// | gamma[D]               | 8192 * 2 = 16 KB (常驻)

template<int D_MODEL, int ALPHA_DIM>
__global__ void hybrid_seednorm_kernel(
    const __nv_bfloat16* __restrict__ x,    // (N, D)
    const __nv_bfloat16* __restrict__ gamma,// (D,)
    const __nv_bfloat16* __restrict__ W1,   // (ALPHA_DIM, D)
    const __nv_bfloat16* __restrict__ b1,   // (ALPHA_DIM,)
    const __nv_bfloat16* __restrict__ W2,   // (1, ALPHA_DIM)
    const float* __restrict__ b2,           // (1,)
    __nv_bfloat16* __restrict__ out,        // (N, D)
    int N, float eps)
{
    int block_id = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    // Shared memory
    __shared__ float s_rms[BLOCK_M];           // rms per token
    __shared__ float s_alpha[BLOCK_M];         // alpha per token
    __shared__ __nv_bfloat16 s_gamma[D_MODEL]; // gamma (常驻)

    // 加载 gamma 到 shared memory (cooperative)
    for (int i = tid; i < D_MODEL; i += blockDim.x)
        s_gamma[i] = gamma[i];
    __syncthreads();

    // ========== Pass 1: RMS + AlphaNet ==========

    // 每个 warp 处理 BLOCK_M/NUM_WARPS = 8 个 token
    int tokens_per_warp = BLOCK_M / NUM_WARPS;
    int row_start = block_id * BLOCK_M + warp_id * tokens_per_warp;

    for (int t = 0; t < tokens_per_warp; t++) {
        int row = row_start + t;
        if (row >= N) continue;

        // --- RMS (warp shuffle reduction) ---
        float sum_sq = 0.0f;
        for (int d = lane; d < D_MODEL; d += WARP_SIZE) {
            float val = __bfloat162float(x[row * D_MODEL + d]);
            sum_sq += val * val;
        }
        // Warp reduce
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_sq += __shfl_xor_sync(0xFFFFFFFF, sum_sq, offset);

        if (lane == 0) {
            s_rms[warp_id * tokens_per_warp + t] =
                sqrtf(sum_sq / D_MODEL + eps);
        }

        // --- AlphaNet: Linear1 (D → 64) ---
        // 每个 lane 负责几个 alpha_dim 输出
        // ... (GEMM via mma instructions or cooperative load)

        // --- GELU ---
        // --- Linear2 (64 → 1) ---
        // --- Softplus ---
        // 存入 s_alpha[warp_id * tokens_per_warp + t]
    }
    __syncthreads();

    // ========== Pass 2: Normalize + Scale ==========
    for (int t = 0; t < tokens_per_warp; t++) {
        int row = row_start + t;
        if (row >= N) continue;

        float rms_val = s_rms[warp_id * tokens_per_warp + t];
        float alpha_val = s_alpha[warp_id * tokens_per_warp + t];
        float rms_inv = 1.0f / rms_val;

        for (int d = lane; d < D_MODEL; d += WARP_SIZE) {
            float x_val = __bfloat162float(x[row * D_MODEL + d]);
            float g = __bfloat162float(s_gamma[d]);
            float out_val = g * (x_val * rms_inv) * alpha_val;
            out[row * D_MODEL + d] = __float2bfloat16(out_val);
        }
    }
}

// ========== Host launcher ==========
void hybrid_seednorm_launch(
    const void* x, const void* gamma,
    const void* W1, const void* b1, const void* W2, const void* b2,
    void* out, int N, int D, cudaStream_t stream)
{
    int grid = (N + BLOCK_M - 1) / BLOCK_M;
    hybrid_seednorm_kernel<8192, 64><<<grid, NUM_WARPS * WARP_SIZE, 0, stream>>>(
        (const __nv_bfloat16*)x, (const __nv_bfloat16*)gamma,
        (const __nv_bfloat16*)W1, (const __nv_bfloat16*)b1,
        (const __nv_bfloat16*)W2, (const float*)b2,
        (__nv_bfloat16*)out, N, 1e-6f);
}
```

---

## 4. HybridMoE — Triton 参考实现 (Grouped GEMM)

```python
import triton
import triton.language as tl

@triton.jit
def hybrid_moe_routing_kernel(
    x_ptr, meta_router_ptr, sub_router_ptr,
    meta_idx_ptr, meta_weight_ptr,
    sub_idx_ptr, sub_weight_ptr,
    N: tl.constexpr, D: tl.constexpr,
    NUM_META: tl.constexpr, NUM_SUB: tl.constexpr,
    TOP_K_META: tl.constexpr, TOP_K_SUB: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """
    Phase 1: Fused meta + sub routing
    输出: meta_idx (N, TOP_K_META), meta_weight (N, TOP_K_META)
          sub_idx (N, TOP_K_SUB), sub_weight (N, TOP_K_SUB)
    """
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = rows < N

    # 加载 x tile
    x = tl.load(x_ptr + rows[:, None] * D + tl.arange(0, D)[None, :], mask=mask[:, None])

    # Meta routing: logits = x @ W_meta^T
    meta_logits = tl.dot(x.to(tl.float32),
                         meta_router_ptr + tl.arange(0, NUM_META)[:, None] * D + tl.arange(0, D)[None, :])
    meta_weights = tl.softmax(meta_logits, axis=1)

    # Top-K meta
    # (Triton没有原生Top-K, 需要循环或近似实现)
    # 简化: 选择最大的 TOP_K_META 个
    for k in range(TOP_K_META):
        # 找最大值
        max_val = tl.max(meta_weights, axis=1)
        max_idx = tl.argmax(meta_weights, axis=1)
        tl.store(meta_idx_ptr + rows * TOP_K_META + k, max_idx, mask=mask)
        tl.store(meta_weight_ptr + rows * TOP_K_META + k, max_val, mask=mask)
        # 置零以便下一轮选次大
        meta_weights = meta_weights * (meta_weights != max_val[:, None])

    # Sub routing (类似, 对选中的 meta expert 做 sub routing)
    # ...


@triton.jit
def hybrid_moe_grouped_gemm_kernel(
    x_ptr, expert_weights_ptr,
    sorted_idx_ptr,  # token按expert分组后的排序索引
    expert_offsets_ptr,  # 每个expert的token起始位置
    out_ptr,
    N: tl.constexpr, D: tl.constexpr, FFN_DIM: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Phase 3: Grouped GEMM for MoE experts
    每个 program block 处理一个 expert 的一批 tokens
    三明治: pre_ffn(D→FFN) + sub_expert(FFN→FFN, NLM激活) + post_ffn(FFN→D)
    """
    expert_id = tl.program_id(0)
    pid_m = tl.program_id(1)

    # 该 expert 的 token 范围
    token_start = tl.load(expert_offsets_ptr + expert_id)
    token_end = tl.load(expert_offsets_ptr + expert_id + 1)
    num_tokens = token_end - token_start

    if pid_m * BLOCK_M >= num_tokens:
        return

    # 加载 sorted token indices
    token_rows = token_start + pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    token_mask = token_rows < token_end
    sorted_idx = tl.load(sorted_idx_ptr + token_rows, mask=token_mask)

    # 加载 x (通过 gather)
    x = tl.load(x_ptr + sorted_idx[:, None] * D + tl.arange(0, D)[None, :],
                mask=token_mask[:, None])

    # === 三明治 GEMM ===
    # 1. pre_ffn: x @ W_pre^T  (D → FFN_DIM)
    # 2. NLM激活: tanh / GELU
    # 3. sub_expert: h @ W_expert^T  (FFN_DIM → FFN_DIM)
    # 4. NLM激活
    # 5. post_ffn: h @ W_post^T  (FFN_DIM → D)
    # (实际实现中用 tl.dot 分块计算)

    # 写回 (通过 scatter)
    tl.store(out_ptr + sorted_idx[:, None] * D + tl.arange(0, D)[None, :],
             result, mask=token_mask[:, None])
```

---

## 5. HybridMLA — Triton 参考实现 (FlashAttention 扩展)

```python
import triton
import triton.language as tl

@triton.jit
def hybrid_mla_kernel(
    # 输入
    q_ptr, k_ptr, v_ptr,           # Q, K, V (已投影)
    rope_cos_ptr, rope_sin_ptr,    # RoPE cos/sin
    dope_gate_ptr,                 # DoPE 门控 (d_head,)
    w_o_ptr,                       # O_proj 权重 (D, D)
    # 输出
    out_ptr,
    # 维度
    N_Q: tl.constexpr, N_KV: tl.constexpr,
    D_HEAD: tl.constexpr, D_MODEL: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    # 参数
    QK_CLIP_THRESHOLD: tl.constexpr,  # 30.0
    SCALE: tl.constexpr,               # 1/sqrt(d_head)
    # Tiling
    BLOCK_M: tl.constexpr,  # Q tile
    BLOCK_N: tl.constexpr,  # K/V tile
):
    """
    HybridMLA fused kernel (Triton)
    融合: RoPE + DoPE + QK-Clip + SDPA(online softmax) + O_proj
    基于 FlashAttention 的 tiling + online softmax 扩展
    """
    pid_m = tl.program_id(0)  # Q tile index
    pid_h = tl.program_id(1)  # head index

    # ========== Q tile 加载 + RoPE + DoPE ==========
    q_rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    q_mask = q_rows < N_Q

    q = tl.load(q_ptr + pid_h * N_Q * D_HEAD + q_rows[:, None] * D_HEAD + tl.arange(0, D_HEAD)[None, :],
                mask=q_mask[:, None])

    # RoPE (element-wise, 在 SRAM 中完成)
    cos = tl.load(rope_cos_ptr + q_rows[:, None] * (D_HEAD // 2) + tl.arange(0, D_HEAD // 2)[None, :],
                  mask=q_mask[:, None])
    sin = tl.load(rope_sin_ptr + q_rows[:, None] * (D_HEAD // 2) + tl.arange(0, D_HEAD // 2)[None, :],
                  mask=q_mask[:, None])

    q_even = q[:, ::2]
    q_odd = q[:, 1::2]
    q_rot_even = q_even * cos - q_odd * sin
    q_rot_odd = q_even * sin + q_odd * cos
    q = tl.cat(q_rot_even, q_rot_odd, can_reorder=False)  # interleave

    # DoPE (element-wise 门控, 与 RoPE 融合)
    dope_gate = tl.load(dope_gate_ptr + tl.arange(0, D_HEAD))
    q = q * dope_gate[None, :]

    # ========== Online Softmax 初始化 ==========
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)

    # ========== K/V tile 内循环 ==========
    for kv_start in range(0, N_KV, BLOCK_N):
        kv_rows = kv_start + tl.arange(0, BLOCK_N)
        kv_mask = kv_rows < N_KV

        # 加载 K, V
        k = tl.load(k_ptr + pid_h * N_KV * D_HEAD + kv_rows[:, None] * D_HEAD + tl.arange(0, D_HEAD)[None, :],
                    mask=kv_mask[:, None])
        v = tl.load(v_ptr + pid_h * N_KV * D_HEAD + kv_rows[:, None] * D_HEAD + tl.arange(0, D_HEAD)[None, :],
                    mask=kv_mask[:, None])

        # RoPE + DoPE on K (同 Q)
        # ...
        k = k * dope_gate[None, :]  # DoPE

        # QK^T
        scores = tl.dot(q, k.T) * SCALE  # (BLOCK_M, BLOCK_N)

        # QK-Clip: 软裁剪 (τ·tanh(S/τ))
        scores = QK_CLIP_THRESHOLD * tl.tanh(scores / QK_CLIP_THRESHOLD)

        # Online softmax
        m_ij = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)

        m_i = m_new

    # ========== 归一化 ==========
    acc = acc / l_i[:, None]

    # ========== O_proj (在 SRAM 中完成) ==========
    # W_o: (D_MODEL, D_HEAD * NUM_HEADS)
    # 每个 head 的 O_proj 分块
    # out = acc @ W_o_head^T
    out = tl.dot(acc, tl.load(w_o_ptr + pid_h * D_HEAD * D_MODEL + tl.arange(0, D_HEAD)[:, None] * D_MODEL + tl.arange(0, D_MODEL)[None, :]))

    # 写回
    tl.store(out_ptr + pid_h * N_Q * D_MODEL + q_rows[:, None] * D_MODEL + tl.arange(0, D_MODEL)[None, :],
             out, mask=q_mask[:, None])


def hybrid_mla_triton(q, k, v, rope_cos, rope_sin, dope_gate, w_o):
    """
    Python wrapper for HybridMLA
    q, k, v: (B, H, S, D_HEAD)
    w_o: (D_MODEL, D_MODEL)
    """
    B, H, S, D = q.shape
    D_MODEL = H * D

    out = torch.empty(B, H, S, D_MODEL, dtype=q.dtype, device=q.device)

    BLOCK_M = 64
    BLOCK_N = 64

    grid = (triton.cdiv(S, BLOCK_M), H, B)
    hybrid_mla_kernel[grid](
        q, k, v, rope_cos, rope_sin, dope_gate, w_o, out,
        S, S, D, D_MODEL, H,
        30.0, 1.0 / (D ** 0.5),
        BLOCK_M, BLOCK_N,
        num_warps=4, num_stages=2,
    )
    return out
```

---

## 6. 昇腾 910C 适配要点

### 6.1 HybridSeeDNorm on 910C

```cpp
// AscendC 参考框架 (概念性)
// 910C: Cube单元(矩阵乘) + Vector单元(向量运算) 分离

// === Cube 计算: AlphaNet Linear ===
// W1 (64×8192) 矩阵乘 → 用 Cube 单元 (mmad 指令)
// 8192 维太宽, 需要分块: 8192/256 = 32 次 mmad
// 每次 mmad: (BLOCK_M, 256) × (256, 64) → (BLOCK_M, 64)
// 结果在 Unified Buffer (UB) 中

// === Vector 计算: RMS + GELU + Softplus ===
// vReduceSum: reduction over D (需要多次, 因为D=8192 > 单次向量长度)
// vExp, vAdd, vSqrt: RMS 计算
// vTanh, vMul: GELU 近似
// vExp, vLog: Softplus

// === 关键: Cube 和 Vector 的流水 ===
// Cube算Linear1 → Vector算RMS+GELU → Cube算Linear2 → Vector算Softplus+Scale
// 双缓冲: Cube算第i块时, Vector算第i-1块
```

### 6.2 HybridMoE on 910C

```cpp
// 910C 的挑战: 细粒度专家 FFN=1024 的小 GEMM

// === 策略: Grouped GEMM via Cube ===
// 将多个小专家的GEMM打包成一个大GEMM
// 16个小专家 × (N_tokens, 1024) × (1024, D) → (16*N_tokens, 1024) × (1024, D)
// 使用 Cube 单元的批量矩阵乘 (mmad with batch dim)

// === NLM激活: Vector epilogue ===
// tanh / NLM 在 Cube 输出后, 由 Vector 单元处理
// 与下一批 Cube 计算流水重叠

// === All-to-All: HCCL ===
// 910C 使用 HCCL (Huawei Collective Communication Library)
// 分桶通信 + 双缓冲 overlap
// copy engine 卸载通信, Cube/Vector 专注计算
```

### 6.3 HybridMLA on 910C

```cpp
// 910C FlashAttention 适配

// === QK^T: Cube 单元 ===
// (BLOCK_M, D_HEAD) × (D_HEAD, BLOCK_N) → (BLOCK_M, BLOCK_N)
// QK-Clip 在 Cube 输出后由 Vector 处理 (tanh + scale)

// === Softmax: Vector 单元 ===
// vMax, vSub, vExp, vSum, vDiv
// Online softmax 需要跨 tile 的 reduction → UB 中的双缓冲

// === AV: Cube 单元 ===
// (BLOCK_M, BLOCK_N) × (BLOCK_N, D_HEAD) → (BLOCK_M, D_HEAD)

// === RoPE/DoPE: Vector 单元 ===
// element-wise, 在 Q/K 从 HBM 加载到 UB 时顺便完成

// === O_proj: Cube 单元 ===
// 最终 (BLOCK_M, D_HEAD) × (D_HEAD, D_MODEL) → (BLOCK_M, D_MODEL)

// === 流水设计 ===
// Cube(QK^T) → Vector(QK-Clip + Softmax) → Cube(AV) → Vector(RoPE/DoPE for next tile)
//                    ↓ overlap
//               Cube(O_proj for completed tiles)
```

---

## 7. 性能基准测试框架

```python
import torch
import time
import json

def benchmark_operator(name, naive_fn, fused_fn, args, warmup=10, iters=100):
    """基准测试: 朴素 vs 融合"""
    # Warmup
    for _ in range(warmup):
        naive_fn(*args)
        fused_fn(*args)
    torch.cuda.synchronize()

    # Naive
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        naive_fn(*args)
    torch.cuda.synchronize()
    naive_time = (time.time() - t0) / iters * 1000  # ms

    # Fused
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fused_fn(*args)
    torch.cuda.synchronize()
    fused_time = (time.time() - t0) / iters * 1000

    speedup = naive_time / fused_time
    result = {
        "operator": name,
        "naive_ms": round(naive_time, 3),
        "fused_ms": round(fused_time, 3),
        "speedup": round(speedup, 2),
    }
    print(f"  {name}: naive={naive_time:.3f}ms, fused={fused_time:.3f}ms, speedup={speedup:.2f}x")
    return result

def run_benchmarks():
    """运行所有算子基准测试"""
    print("\n=== Shannon Hybrid 算子基准测试 ===\n")

    results = []

    # SeeDNorm
    D = 4096
    x = torch.randn(4, 1024, D, device='cuda')
    gamma = torch.ones(D, device='cuda')
    # ... (初始化其他参数)
    # results.append(benchmark_operator("HybridSeeDNorm", ...))

    # MoE
    # results.append(benchmark_operator("HybridMoE", ...))

    # MLA
    # results.append(benchmark_operator("HybridMLA", ...))

    # 保存结果
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to benchmark_results.json")
```

---

## 8. 文件清单

| 文件 | 路径 | 说明 |
|------|------|------|
| Python验证脚本 | `/workspace/hardware_operators/verify_operators.py` | 5/5 测试通过 |
| Triton参考实现 | 本文件 §2, §4, §5 | SeeDNorm/MoE/MLA 的 Triton kernel |
| CUDA C++框架 | 本文件 §3 | HybridSeeDNorm 的 CUDA kernel 框架 |
| 昇腾910C适配 | 本文件 §6 | Cube/Vector 分离映射 + 流水设计 |
| 性能基准框架 | 本文件 §7 | 朴素 vs 融合 benchmark 模板 |

> **注意**: Triton 和 CUDA 代码为参考框架, 非可直接运行的实现。
> 实际部署需要: (1) 补充完整的 tiling 逻辑; (2) 添加反向传播; (3) 平台特定调优。
