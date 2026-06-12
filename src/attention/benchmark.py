import mlx.core as mx
import time

def benchmark_attention(fn, B=1, H=8, seq_lens=[128, 256, 512, 1024], d= 64, n_runs=20):
          results={}
          for N in seq_lens:
                    Q = mx.random.normal((B, H, N, d))
                    K = mx.random.normal((B, H, N, d))
                    V = mx.random.normal((B, H, N, d))
                    mx.eval(Q, K, V)
                    times = []
                    for _ in range(n_runs):
                              t0 = time.perf_counter()
                              out = fn(Q,K,V)
                              mx.eval(out)
                              times.append(time.perf_counter()-t0)
                    avg_ms = sum(times[3:])/len(times[3:])*1000
                    memory_mb= (B*H*N*N*2)/1e6
                    results[N] = {'time_ms': avg_ms, 'memory_mb': memory_mb}
                    print(f"N={N:5d} | {avg_ms:.2f} ms | {memory_mb:.1f} MB")
          return results

def describe_ratio(reference_ms, candidate_ms):
    if candidate_ms < reference_ms:
        return f"{reference_ms / candidate_ms:.2f}x faster"
    return f"{candidate_ms / reference_ms:.2f}x slower"

if __name__ == "__main__":
    from src.attention.standard_attention import standard_attention
    from src.attention.flash_attention import flash_attention_forward
    from src.metal.metal_wrapper import metal_flash_attention
    from src.metal.simd_wrapper import metal_flash_attention_simd
    from src.metal.hybrid_wrapper import metal_flash_attention_hybrid
    from src.metal.hybrid_shuffle_wrapper import metal_flash_attention_hybrid_shuffle
    from src.metal.hybrid_shuffle_parallel_o_wrapper import metal_flash_attention_hybrid_shuffle_parallel_o
    from src.metal.hybrid_shuffle_parallel_o_cached_p_wrapper import metal_flash_attention_hybrid_shuffle_parallel_o_cached_p
    from src.metal.hybrid_shuffle_parallel_o_cached_p_float4_wrapper import metal_flash_attention_hybrid_shuffle_parallel_o_cached_p_float4
    
    print("=== Standard Attention ===")
    std_results = benchmark_attention(standard_attention)
    
    print("\n=== Flash Attention ===")
    flash_results = benchmark_attention(flash_attention_forward)
    print("\n=== Metal Kernel ===")
    metal_results = benchmark_attention(metal_flash_attention)
    print("\n=== SIMD Metal Kernel ===")
    simd_results = benchmark_attention(metal_flash_attention_simd)
    print("\n=== Hybrid Metal Kernel ===")
    hybrid_results = benchmark_attention(metal_flash_attention_hybrid)
    print("\n=== Hybrid Metal Shuffle Kernel ===")
    hybrid_shuffle_results = benchmark_attention(metal_flash_attention_hybrid_shuffle)
    print("\n=== Hybrid Metal Shuffle Parallel O Kernel ===")
    hybrid_shuffle_parallel_o_results = benchmark_attention(metal_flash_attention_hybrid_shuffle_parallel_o)
    print("\n=== Hybrid Metal Shuffle Parallel O Cached P Kernel ===")
    hybrid_shuffle_parallel_o_cached_p_results = benchmark_attention(metal_flash_attention_hybrid_shuffle_parallel_o_cached_p)
    print("\n=== Hybrid Metal Shuffle Parallel O Cached P Float4 Kernel ===")
    hybrid_shuffle_parallel_o_cached_p_float4_results = benchmark_attention(metal_flash_attention_hybrid_shuffle_parallel_o_cached_p_float4)
    
    sequence_lengths = sorted(flash_results.keys())

    print("=== Compared to Standard Attention ===")
    for N in std_results:
        print(
            f"N={N:5d} | Cached P is "
            f"{describe_ratio(std_results[N]['time_ms'], hybrid_shuffle_parallel_o_cached_p_results[N]['time_ms'])} than standard"
        )

    print("=== Custom Kernel Progress ===")
    for N in sequence_lengths:
        print(
            f"N={N:5d} | Cached P is "
            f"{describe_ratio(hybrid_results[N]['time_ms'], hybrid_shuffle_parallel_o_cached_p_results[N]['time_ms'])} than hybrid"
        )