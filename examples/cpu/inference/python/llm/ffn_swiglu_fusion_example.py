"""
Example: Using FFN SwiGLU Fusion with IPEX

This example demonstrates how to use the FFN layer fusion with SwiGLU activation
that can be optimized with AMX instructions.
"""

import torch
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.llm.functional import ffn_swiglu_fusion


class FFNWithSwiGLU(torch.nn.Module):
    """
    Standard FFN layer with SwiGLU activation used in LLMs like LLaMA.
    """
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        # Standard implementation: gate = silu(gate_proj(x)) * up_proj(x)
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class OptimizedFFNWithSwiGLU(torch.nn.Module):
    """
    Optimized FFN layer using IPEX fusion for better performance.
    """
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        # Use fused implementation with AMX optimization
        return ffn_swiglu_fusion(
            x,
            self.gate_proj.weight,
            self.gate_proj.bias if self.gate_proj.bias is not None else torch.empty(0),
            self.up_proj.weight,
            self.up_proj.bias if self.up_proj.bias is not None else torch.empty(0),
            self.down_proj.weight,
            self.down_proj.bias if self.down_proj.bias is not None else torch.empty(0),
        )


def benchmark_example():
    """
    Example of benchmarking standard vs optimized FFN with SwiGLU.
    """
    import time
    
    # Configuration (typical for LLaMA models)
    batch_size = 1
    seq_len = 128
    hidden_size = 4096
    intermediate_size = 11008  # 2.7 * hidden_size
    dtype = torch.bfloat16
    
    # Create input
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype)
    
    # Standard FFN
    print("=" * 60)
    print("Standard FFN with SwiGLU")
    print("=" * 60)
    model_standard = FFNWithSwiGLU(hidden_size, intermediate_size).eval()
    model_standard = model_standard.to(dtype)
    
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model_standard(x)
        
        # Benchmark
        start = time.time()
        num_iters = 100
        for _ in range(num_iters):
            output_standard = model_standard(x)
        end = time.time()
        time_standard = (end - start) / num_iters * 1000  # ms
        print(f"Average time: {time_standard:.3f} ms")
    
    # Optimized FFN with IPEX
    print("\n" + "=" * 60)
    print("Optimized FFN with SwiGLU (IPEX + AMX)")
    print("=" * 60)
    model_optimized = OptimizedFFNWithSwiGLU(hidden_size, intermediate_size).eval()
    model_optimized = model_optimized.to(dtype)
    
    # Note: For actual AMX optimization, weights need to be in TPP format
    # This will be handled by ipex.optimize() or weight preprocessing
    
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model_optimized(x)
        
        # Benchmark
        start = time.time()
        for _ in range(num_iters):
            output_optimized = model_optimized(x)
        end = time.time()
        time_optimized = (end - start) / num_iters * 1000  # ms
        print(f"Average time: {time_optimized:.3f} ms")
    
    # Check correctness
    print("\n" + "=" * 60)
    print("Correctness Check")
    print("=" * 60)
    print(f"Output shape: {output_standard.shape}")
    print(f"Max difference: {(output_standard - output_optimized).abs().max().item():.6f}")
    print(f"Mean difference: {(output_standard - output_optimized).abs().mean().item():.6f}")
    
    if time_optimized < time_standard:
        speedup = time_standard / time_optimized
        print(f"\n✓ Speedup: {speedup:.2f}x")
    else:
        print(f"\n✗ No speedup yet (optimization may not be active)")


def usage_example():
    """
    Simple usage example.
    """
    print("=" * 60)
    print("Simple Usage Example")
    print("=" * 60)
    
    # Create a simple FFN module
    hidden_size = 256
    intermediate_size = 1024
    
    model = OptimizedFFNWithSwiGLU(hidden_size, intermediate_size).eval()
    model = model.to(torch.bfloat16)
    
    # Create input
    batch_size = 2
    seq_len = 32
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("✓ Successfully ran FFN with SwiGLU fusion!")


if __name__ == "__main__":
    print("FFN SwiGLU Fusion Example\n")
    
    # Run simple usage example
    usage_example()
    
    # Uncomment to run benchmark
    # Note: Actual performance gains depend on AMX assembly implementation
    # print("\n\n")
    # benchmark_example()
    
    print("\n" + "=" * 60)
    print("Note: To see actual AMX speedup, implement the assembly-level")
    print("optimizations as described in FFN_SWIGLU_AMX_GUIDE.md")
    print("=" * 60)
