# FFN SwiGLU Fusion with AMX Optimization Guide

## Overview

This guide explains how to add custom AMX (Advanced Matrix Extensions) assembly-level optimizations for the FFN (Feed-Forward Network) layer with SwiGLU activation fusion.

## Architecture

The FFN SwiGLU fusion combines three matrix multiplications with SwiGLU activation:
1. **Gate Projection**: `gate = input @ gate_weight + gate_bias`
2. **Up Projection**: `up = input @ up_weight + up_bias`
3. **SwiGLU Activation**: `swiglu_out = silu(gate) * up`
4. **Down Projection**: `output = swiglu_out @ down_weight + down_bias`

The complete operation: `output = (silu(input @ gate_weight + gate_bias) * (input @ up_weight + up_bias)) @ down_weight + down_bias`

## Files Structure

### Python Interface
- **File**: `intel_extension_for_pytorch/llm/functional/fusions.py`
- **Function**: `ffn_swiglu_fusion()`
- **Usage**: User-facing API for calling the optimized FFN layer

### C++ Registration
- **File**: `csrc/cpu/aten/TPPGEMM.h` - Forward declarations
- **File**: `csrc/cpu/aten/TPPGEMM.cpp` - CPU dispatcher implementation
- **File**: `csrc/cpu/aten/kernels/TPPGEMMKrnl.cpp` - Kernel implementation stub

### Kernel Implementation
- **File**: `csrc/cpu/tpp/kernels/TPPGEMMKrnl.h`
- **Function**: `tpp_ffn_swiglu<T>()` template function
- **Location**: Lines ~770-1010 (after `tpp_fused_gate_up_proj`)

## Where to Add AMX Assembly Code

The main optimization opportunities are in the `tpp_ffn_swiglu` function in `TPPGEMMKrnl.h`:

### 1. Gate and Up Projections with SwiGLU (Lines ~850-930)
```cpp
// Current implementation uses TPP BRGEMM operations
// AMX optimization can be added here for:
// - Matrix multiplication (gate and up projections)
// - SwiGLU activation (silu(gate) * up)

// Look for this section:
brgemm_gate_tpp(in[s1][nc], wt_gate_V[nk][nc], gate_tmp[s1][nk], count, true);
brgemm_gate_tpp(in[s1][nc], wt_up_V[nk][nc], swiglu_out[s1][nk], count, true);
silu_fwd_tpp(gate_tmp[s1][nk], gate_tmp[s1][nk]);
mul_tpp(gate_tmp[s1][nk], swiglu_out[s1][nk], swiglu_out[s1][nk]);
```

**AMX Assembly Integration Points:**
- Replace `brgemm_gate_tpp` calls with AMX TMUL instructions
- Fuse SwiGLU activation (silu + multiply) into the TMUL loop
- Use AMX tile registers (TMM0-TMM7) for intermediate results

### 2. Down Projection (Lines ~950-1000)
```cpp
// Current implementation uses TPP BRGEMM operations
// AMX optimization can be added here for matrix multiplication

// Look for this section:
brgemm_down_tpp(swiglu_out[s1][nc], wt_down_V[nk][nc], out[s1][nk], count, true);
```

**AMX Assembly Integration Points:**
- Replace `brgemm_down_tpp` with AMX TMUL instructions
- Optimize tile configuration for the down projection dimensions

## AMX Assembly Template

Here's a basic template for adding AMX assembly:

```cpp
// Example: Replace BRGEMM with AMX assembly
#ifdef __x86_64__
#include <immintrin.h>

// Configure AMX tiles
_tile_loadconfig(&tile_config);

// Load input and weight into tile registers
_tile_loadd(0, input_ptr, stride);      // TMM0 = input
_tile_loadd(1, weight_ptr, stride);     // TMM1 = weight

// Perform matrix multiplication
_tile_dpbf16ps(2, 0, 1);                // TMM2 = TMM0 * TMM1 (for BF16)

// Store result
_tile_stored(2, output_ptr, stride);    // output = TMM2

// Release tiles
_tile_release();
#endif
```

## Integration Steps

1. **Add AMX intrinsic headers** at the top of `TPPGEMMKrnl.h`:
   ```cpp
   #ifdef __x86_64__
   #include <immintrin.h>
   #include <amxintrin.h>
   #endif
   ```

2. **Create helper functions** for AMX operations:
   - Tile configuration setup
   - Matrix multiplication with AMX
   - SwiGLU activation with AMX

3. **Replace TPP calls** in `tpp_ffn_swiglu` with AMX implementations:
   - Replace `brgemm_gate_tpp` with AMX GEMM
   - Fuse SwiGLU activation into AMX loop
   - Replace `brgemm_down_tpp` with AMX GEMM

4. **Add runtime checks** for AMX support:
   ```cpp
   if (has_amx_support()) {
       // Use AMX implementation
   } else {
       // Fall back to TPP implementation
   }
   ```

## Performance Considerations

1. **Tile Size**: Configure AMX tiles (16x16 for BF16/FP16, 16x64 for INT8) to match your data dimensions
2. **Memory Layout**: Ensure data is properly blocked for AMX tile sizes
3. **Prefetching**: Add software prefetch instructions for better cache utilization
4. **Fusion Opportunities**: 
   - Fuse bias addition into matrix multiplication
   - Fuse SwiGLU activation (silu + multiply) into a single operation
   - Minimize intermediate memory writes

## Testing

After implementing AMX optimizations, test with:

```python
import torch
import intel_extension_for_pytorch as ipex

# Your model with FFN layer
model = YourModel()
model.eval()

# Optimize with IPEX
model = ipex.optimize(model, dtype=torch.bfloat16)

# Use the FFN SwiGLU fusion
from intel_extension_for_pytorch.llm.functional import ffn_swiglu_fusion

output = ffn_swiglu_fusion(
    input,
    gate_weight, gate_bias,
    up_weight, up_bias,
    down_weight, down_bias
)
```

## References

- Intel AMX Programming Guide: https://www.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/intrinsics/intrinsics-for-amx-instructions.html
- TPP (Tensor Processing Primitives): https://github.com/libxsmm/tpp-pytorch-extension
- SwiGLU Paper: https://arxiv.org/abs/2002.05202

## Notes

- The current implementation provides a baseline using TPP operations
- AMX assembly code should be added incrementally and benchmarked
- Consider creating separate implementations for different data types (FP32, BF16, FP16)
- Use compiler intrinsics rather than inline assembly for better portability
