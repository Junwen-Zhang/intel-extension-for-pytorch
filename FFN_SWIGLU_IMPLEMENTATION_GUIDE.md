# FFN SwiGLU Fusion - 文件定位指南 (File Location Guide)

## 快速定位 (Quick Reference)

### 需要添加AMX汇编代码的位置 (Where to Add AMX Assembly Code)

**主要文件**: `csrc/cpu/tpp/kernels/TPPGEMMKrnl.h`

**函数**: `tpp_ffn_swiglu<T>()` (大约在第770-1010行)

**具体优化位置**:

#### 1. Gate和Up投影 + SwiGLU激活 (约第850-930行)
```cpp
// 当前代码 (Current Code):
brgemm_gate_tpp(in[s1][nc], wt_gate_V[nk][nc], gate_tmp[s1][nk], count, true);
brgemm_gate_tpp(in[s1][nc], wt_up_V[nk][nc], swiglu_out[s1][nk], count, true);
silu_fwd_tpp(gate_tmp[s1][nk], gate_tmp[s1][nk]);
mul_tpp(gate_tmp[s1][nk], swiglu_out[s1][nk], swiglu_out[s1][nk]);

// 在这里添加AMX优化代码 (Add AMX optimization here):
// 1. 用AMX TMUL替换brgemm_gate_tpp
// 2. 融合SwiGLU激活到循环中
// 3. 使用tile寄存器优化
```

#### 2. Down投影 (约第950-1000行)
```cpp
// 当前代码 (Current Code):
brgemm_down_tpp(swiglu_out[s1][nc], wt_down_V[nk][nc], out[s1][nk], count, true);

// 在这里添加AMX优化代码 (Add AMX optimization here):
// 1. 用AMX TMUL替换brgemm_down_tpp
// 2. 优化tile配置
// 3. 减少内存访问
```

## 完整文件列表 (Complete File List)

### 1. Python用户接口 (Python User Interface)
```
intel_extension_for_pytorch/llm/functional/
├── fusions.py              # ffn_swiglu_fusion() 函数定义
└── __init__.py             # 导出接口
```

### 2. Python元数据注册 (Python Meta Registration)
```
intel_extension_for_pytorch/
└── _meta_registrations.py  # meta_tpp_ffn_swiglu() 支持torch.compile
```

### 3. C++头文件 (C++ Header Files)
```
csrc/cpu/aten/
└── TPPGEMM.h               # 前向声明、类型定义、调度声明
```

### 4. C++实现文件 (C++ Implementation Files)
```
csrc/cpu/aten/
├── TPPGEMM.cpp             # 前向函数、调度定义、算子注册
└── kernels/
    └── TPPGEMMKrnl.cpp     # 内核实现入口（调用TPPGEMMKrnl.h中的模板）
```

### 5. TPP内核实现 (TPP Kernel Implementation)
```
csrc/cpu/tpp/kernels/
└── TPPGEMMKrnl.h           # ⭐ tpp_ffn_swiglu<T>() 模板函数
                            # ⭐ AMX优化代码应该加在这里！
```

### 6. 文档 (Documentation)
```
csrc/cpu/tpp/kernels/
└── FFN_SWIGLU_AMX_GUIDE.md # 详细的AMX优化指南

FFN_SWIGLU_FUSION_README.md # 整体说明文档（项目根目录）
```

### 7. 示例代码 (Example Code)
```
examples/cpu/inference/python/llm/
└── ffn_swiglu_fusion_example.py  # Python使用和性能测试示例
```

## 代码调用流程 (Code Call Flow)

```
Python层 (Python Layer):
ffn_swiglu_fusion() in fusions.py
    ↓
    调用 torch.ops.torch_ipex.tpp_ffn_swiglu()
    
C++算子注册 (C++ Operator Registration):
TORCH_LIBRARY_FRAGMENT in TPPGEMM.cpp
    ↓
    绑定到 tpp_ffn_swiglu_forward_cpu()
    
C++前向函数 (C++ Forward Function):
tpp_ffn_swiglu_forward_cpu() in TPPGEMM.cpp
    ↓
    调用 tpp_ffn_swiglu_kernel_stub()
    
内核分发 (Kernel Dispatch):
tpp_ffn_swiglu_kernel_stub in TPPGEMMKrnl.cpp
    ↓
    调用 tpp_ffn_swiglu_kernel_impl()
    
内核实现 (Kernel Implementation):
tpp_ffn_swiglu_kernel_impl() in TPPGEMMKrnl.cpp
    ↓
    调用 torch_ipex::tpp::tpp_ffn_swiglu<T>()
    
⭐ 实际计算 (Actual Computation):
tpp_ffn_swiglu<T>() in TPPGEMMKrnl.h
    ↓
    这里是添加AMX汇编代码的地方！
    (This is where to add AMX assembly code!)
```

## AMX优化步骤概览 (AMX Optimization Steps Overview)

### 第1步: 理解现有实现
- 文件: `csrc/cpu/tpp/kernels/TPPGEMMKrnl.h`
- 查看: `tpp_ffn_swiglu<T>()` 函数
- 理解: 当前使用BRGEMM TPP操作的实现逻辑

### 第2步: 添加AMX内置函数
在文件顶部添加:
```cpp
#ifdef __x86_64__
#include <immintrin.h>
#include <amxintrin.h>
#endif
```

### 第3步: 创建AMX辅助函数
```cpp
// 在tpp_ffn_swiglu<T>()函数前添加
template <typename T>
inline void amx_gemm_swiglu(...) {
    // AMX矩阵乘法 + SwiGLU融合
}
```

### 第4步: 替换计算核心
找到注释说明的位置（约850-930行和950-1000行），用AMX实现替换BRGEMM调用

### 第5步: 测试和优化
- 使用 `examples/cpu/inference/python/llm/ffn_swiglu_fusion_example.py` 测试
- 对比性能和正确性
- 调优tile配置和数据布局

## 相关参考实现 (Related Reference Implementations)

在同一文件 `TPPGEMMKrnl.h` 中可以参考:
- `tpp_fused_gate_up_proj<T>()` (第646-770行) - Gate和Up投影融合的实现
- `tpp_linear_silu<T>()` - SiLU激活的实现
- `tpp_linear_mul<T>()` - 乘法融合的实现

这些函数展示了如何:
- 使用BRGEMM TPP操作
- 处理不同的批次大小
- 应用激活函数
- 管理中间张量

## 常见问题 (FAQ)

### Q1: 我应该从哪里开始？
**A**: 从 `FFN_SWIGLU_AMX_GUIDE.md` 开始阅读，然后查看 `TPPGEMMKrnl.h` 中的 `tpp_ffn_swiglu<T>()` 函数。

### Q2: 需要修改多少个文件？
**A**: 只需要修改 `csrc/cpu/tpp/kernels/TPPGEMMKrnl.h` 这一个文件即可添加AMX优化。其他所有接口代码已经准备好。

### Q3: 如何测试我的修改？
**A**: 运行 `python examples/cpu/inference/python/llm/ffn_swiglu_fusion_example.py`

### Q4: 是否需要支持所有数据类型？
**A**: 建议先实现BF16（最常用），然后考虑FP32和FP16。

### Q5: 如何确保正确性？
**A**: 对比AMX实现和原始BRGEMM实现的输出，确保数值误差在可接受范围内（BF16约1e-2，FP16约1e-3）。

## 进一步帮助 (Further Help)

1. **详细AMX指南**: 查看 `csrc/cpu/tpp/kernels/FFN_SWIGLU_AMX_GUIDE.md`
2. **使用示例**: 运行 `examples/cpu/inference/python/llm/ffn_swiglu_fusion_example.py`
3. **整体文档**: 阅读根目录的 `FFN_SWIGLU_FUSION_README.md`
4. **参考测试**: 查看 `tests/cpu/test_tpp_linear.py` 中的 `test_tpp_fused_gate_up_proj`

## 重要提醒 (Important Notes)

⚠️ **核心优化位置**:
- 文件: `csrc/cpu/tpp/kernels/TPPGEMMKrnl.h`
- 函数: `tpp_ffn_swiglu<T>()`
- 行号: 约770-1010行

✅ **所有其他文件已经准备就绪**，你只需要在上述位置添加AMX汇编级别的优化代码！

✅ **All other files are ready**, you only need to add AMX assembly-level optimization code at the above location!
