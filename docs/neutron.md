# Neutron 编译器使用说明

> **版本**: v0.1.0
> **平台**: x86_64-linux (Ubuntu 20.04+)
> **二进制路径**: `bin/neutron`

## 简介

Neutron 是一个算子编译器，将 ONNX 模型编译为目标平台的 GPU/NPU kernel 源码。

支持的后端：
- **CUDA** (NVIDIA Hopper/Blackwell/Ampere) → 生成 CUDA C++ kernel
- **CANN** (华为 Ascend 910B1/B3/310P3) → 生成 AscendC C++ kernel
- **Triton** (跨平台) → 生成 Triton Python kernel
- **Metal** (Apple Silicon) → 生成 Metal Shading Language kernel

编译流程：`ONNX → 前端解析 → 架构无关优化 → Lowering → 指令选择 → 寄存器分配 → 后端代码生成`

## 用法

```bash
./bin/neutron <input.onnx> [--target cuda|npu|cpu] [--opt 0|1|2|3] [--dump]
```

### 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `<input.onnx>` | 输入的 ONNX 模型文件路径（必填） | — |
| `--target` | 目标平台：`cuda`（NVIDIA GPU）、`npu`（华为昇腾）、`cpu` | `cuda` |
| `--opt` | 优化等级：`0`（无优化）、`1`、`2`、`3`（最高） | `2` |
| `--dump` | 输出编译中间 IR（调试用） | 关闭 |

### 输出

编译成功后，stdout 输出：
- `target`: 目标平台名
- `instructions`: 生成的指令数量
- （`--dump` 模式下）完整的中间 IR、寄存器分配结果、后端 kernel 源码

## 使用示例

### 编译为 CUDA kernel

```bash
./bin/neutron model.onnx --target cuda --opt 2
```

### 编译为昇腾 CANN kernel

```bash
./bin/neutron model.onnx --target npu --opt 2
```

### 调试模式（输出完整编译 IR）

```bash
./bin/neutron model.onnx --target cuda --opt 2 --dump
```

### 在 Python 中调用

```python
import subprocess

result = subprocess.run(
    ["./bin/neutron", "model.onnx", "--target", "npu", "--opt", "2", "--dump"],
    capture_output=True, text=True
)

if result.returncode == 0:
    print(result.stdout)  # 编译输出
else:
    print(f"编译失败: {result.stderr}")
```

## 与 Shannon 项目的集成

Neutron 用于将 Shannon 模型的 Hybrid 算子（HybridSeeDNorm / HybridMoE / HybridMLA）从 ONNX 格式编译为目标平台 kernel：

1. **导出算子为 ONNX**：使用 PyTorch 将算子导出为 ONNX 格式
2. **编译**：用 neutron 编译为目标平台 kernel 源码
3. **集成**：将生成的 kernel 源码集成到推理引擎中

```bash
# 示例：将 HybridSeeDNorm 算子编译为昇腾 CANN kernel
python -c "
import torch
import torchvision

# 导出算子为 ONNX
class HybridSeeDNorm(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.normalize(x, dim=-1)

model = HybridSeeDNorm()
dummy = torch.randn(1, 1024, 4096)
torch.onnx.export(model, dummy, 'hybrid_seednorm.onnx')
"

./bin/neutron hybrid_seednorm.onnx --target npu --opt 2 --dump
```

## 注意事项

- 此二进制为 x86_64-linux 平台预编译版本，需在 Linux 环境运行
- 输入必须为合法的 ONNX 格式文件
- `--target npu` 对应华为昇腾 CANN 后端（Ascend 910B1 微架构）
- `--target cuda` 默认使用 Hopper90 微架构
- `--dump` 模式会输出大量调试信息，建议重定向到文件：`./bin/neutron model.onnx --dump > compile_log.txt 2>&1`
