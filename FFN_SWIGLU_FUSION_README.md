# FFN SwiGLU Fusion Implementation

## 概述 (Overview)

本实现为大模型推理中的FFN层添加了SwiGLU激活融合支持，提供了AMX矩阵乘法优化的接口。用户可以在指定位置添加汇编级别的AMX优化代码。

This implementation adds FFN layer fusion with SwiGLU activation for large model inference, providing an interface for AMX matrix multiplication optimization. Users can add assembly-level AMX optimization code at the designated locations.

## 文件结构 (File Structure)

### Python接口层 (Python Interface Layer)
1. **intel_extension_for_pytorch/llm/functional/fusions.py**
   - 新增函数: `ffn_swiglu_fusion()`
   - 用户调用的主接口
   - Added function: `ffn_swiglu_fusion()`
   - Main interface for users

2. **intel_extension_for_pytorch/llm/functional/__init__.py**
   - 导出 `ffn_swiglu_fusion` 函数
   - Exports `ffn_swiglu_fusion` function

3. **intel_extension_for_pytorch/_meta_registrations.py**
   - 新增: `meta_tpp_ffn_swiglu()` 元函数注册
   - 支持 `torch.compile` 编译
   - Added: `meta_tpp_ffn_swiglu()` meta function registration
   - Supports `torch.compile`

### C++实现层 (C++ Implementation Layer)
4. **csrc/cpu/aten/TPPGEMM.h**
   - 新增前向声明: `tpp_ffn_swiglu_forward_cpu()`
   - 新增类型定义: `tpp_ffn_swiglu_kernel_impl_fn`
   - 新增调度声明: `tpp_ffn_swiglu_kernel_stub`
   - Added forward declaration, type definition, and dispatch declaration

5. **csrc/cpu/aten/TPPGEMM.cpp**
   - 新增调度定义: `IPEX_DEFINE_DISPATCH(tpp_ffn_swiglu_kernel_stub)`
   - 新增前向函数实现: `tpp_ffn_swiglu_forward_cpu()`
   - 新增算子注册: `TORCH_LIBRARY_FRAGMENT` for `tpp_ffn_swiglu`
   - Added dispatch definition, forward function, and operator registration

6. **csrc/cpu/aten/kernels/TPPGEMMKrnl.cpp**
   - 新增内核实现: `tpp_ffn_swiglu_kernel_impl()`
   - 支持 FP32, BF16, FP16 数据类型
   - 新增调度注册: `IPEX_REGISTER_DISPATCH`
   - Added kernel implementation supporting FP32, BF16, FP16

7. **csrc/cpu/tpp/kernels/TPPGEMMKrnl.h**
   - 新增模板函数: `tpp_ffn_swiglu<T>()`
   - **AMX优化插入点在此文件中**
   - Template function with **AMX optimization insertion points**

### 文档和示例 (Documentation and Examples)
8. **csrc/cpu/tpp/kernels/FFN_SWIGLU_AMX_GUIDE.md**
   - AMX汇编级优化指南
   - 详细说明优化位置和方法
   - AMX assembly optimization guide
   - Detailed instructions for optimization locations

9. **examples/cpu/inference/python/llm/ffn_swiglu_fusion_example.py**
   - Python使用示例
   - 性能测试代码
   - Python usage example
   - Performance benchmark code

## 主要功能 (Key Features)

### 算法流程 (Algorithm Flow)
```
input → gate_proj → silu → 
              ↓              ↘
            up_proj  →  multiply  →  down_proj  →  output
```

完整公式 (Complete formula):
```
output = (silu(input @ gate_weight + gate_bias) * (input @ up_weight + up_bias)) @ down_weight + down_bias
```

### 优化点 (Optimization Points)
1. **Gate和Up投影融合** (Gate and Up Projection Fusion)
   - 位置: `TPPGEMMKrnl.h` 第850-930行
   - 可用AMX TMUL指令替换BRGEMM
   - Location: Lines 850-930 in `TPPGEMMKrnl.h`
   - Replace BRGEMM with AMX TMUL instructions

2. **SwiGLU激活融合** (SwiGLU Activation Fusion)
   - 将 silu + multiply 融入矩阵乘法循环
   - Fuse silu + multiply into matrix multiplication loop

3. **Down投影优化** (Down Projection Optimization)
   - 位置: `TPPGEMMKrnl.h` 第950-1000行
   - 使用AMX优化最终投影
   - Location: Lines 950-1000 in `TPPGEMMKrnl.h`
   - Optimize final projection with AMX

## 使用方法 (Usage)

### Python调用 (Python Usage)
```python
from intel_extension_for_pytorch.llm.functional import ffn_swiglu_fusion

output = ffn_swiglu_fusion(
    input,
    gate_weight, gate_bias,
    up_weight, up_bias,
    down_weight, down_bias
)
```

### 运行示例 (Run Example)
```bash
python examples/cpu/inference/python/llm/ffn_swiglu_fusion_example.py
```

## AMX优化集成 (AMX Optimization Integration)

### 步骤 (Steps)
1. 阅读 `csrc/cpu/tpp/kernels/FFN_SWIGLU_AMX_GUIDE.md`
2. 在 `TPPGEMMKrnl.h` 的 `tpp_ffn_swiglu` 函数中添加AMX代码
3. 使用AMX内置函数 (intrinsics) 而非内联汇编
4. 配置tile寄存器 (TMM0-TMM7)
5. 测试不同数据类型 (FP32, BF16, FP16)

### AMX模板 (AMX Template)
```cpp
// 配置tile
_tile_loadconfig(&tile_config);

// 加载数据到tile寄存器
_tile_loadd(0, input_ptr, stride);
_tile_loadd(1, weight_ptr, stride);

// 矩阵乘法
_tile_dpbf16ps(2, 0, 1);  // BF16类型

// 存储结果
_tile_stored(2, output_ptr, stride);

// 释放tile
_tile_release();
```

## 测试 (Testing)

参考现有测试: `tests/cpu/test_tpp_linear.py`
- 可以添加类似 `test_tpp_fused_gate_up_proj` 的测试用例
- 验证正确性和性能

Reference existing tests: `tests/cpu/test_tpp_linear.py`
- Can add test cases similar to `test_tpp_fused_gate_up_proj`
- Verify correctness and performance

## 注意事项 (Notes)

1. **当前实现是基础版本**: 使用TPP操作，预留了AMX优化接口
2. **AMX优化需要手动添加**: 按照指南在指定位置添加汇编代码
3. **权重格式**: 权重需要被预处理为TPP格式（blocking）以获得最佳性能
4. **数据类型支持**: FP32, BF16, FP16（需要AMX FP16支持）

1. **Current implementation is baseline**: Uses TPP operations with AMX optimization interface
2. **AMX optimization needs manual addition**: Add assembly code at designated locations
3. **Weight format**: Weights need to be preprocessed to TPP format (blocking) for best performance
4. **Data type support**: FP32, BF16, FP16 (requires AMX FP16 support)

## 参考资料 (References)

- Intel AMX Programming Guide
- TPP (Tensor Processing Primitives)
- SwiGLU Paper: https://arxiv.org/abs/2002.05202
- Existing implementations: `tpp_fused_gate_up_proj`, `tpp_linear_silu`

## 构建 (Build)

由于这是接口层代码，需要完整构建IPEX来测试:
```bash
python setup.py develop
```

Since this is interface layer code, full IPEX build is needed for testing:
```bash
python setup.py develop
```

## 联系方式 (Contact)

如有问题，请查看:
- FFN_SWIGLU_AMX_GUIDE.md 详细文档
- ffn_swiglu_fusion_example.py 示例代码
- test_tpp_linear.py 测试用例

For questions, refer to:
- FFN_SWIGLU_AMX_GUIDE.md for detailed documentation
- ffn_swiglu_fusion_example.py for example code
- test_tpp_linear.py for test cases
