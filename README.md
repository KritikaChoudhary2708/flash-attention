# Flash Attention on Apple Silicon using MLX and Metal

This project implements and optimizes Flash Attention on Apple Silicon using MLX and custom Metal kernels.

The goal was not only to implement attention, but to understand how memory movement, tiling, online softmax, SIMD lanes, reductions, broadcasts, and GPU thread mapping affect real performance.

I started with standard attention in MLX, then implemented tiled Flash Attention with online softmax, moved the computation into custom Metal kernels, and progressively optimized the kernel through measured experiments.

The best custom kernel improved from 874.56 ms to 88.47 ms at sequence length 1024, giving around a 9.89x speedup over my original hybrid Metal kernel.

The custom kernel is still slower than MLX standard attention, which is expected because MLX uses highly optimized backend kernels. This project focuses on learning and demonstrating the internal mechanics of Flash Attention and GPU optimization.

## Quick Start

This project is designed for Apple Silicon because it uses MLX and custom Metal kernels.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 -m src.attention.validate
python3 -m src.attention.benchmark
```

## Why Flash Attention?

Standard attention computes:

```text
QK^T
```

This creates an `N x N` score matrix.

For long sequences, this matrix becomes memory-heavy. The GPU spends a lot of time moving the score matrix between memory levels instead of only computing.

Flash Attention avoids materializing the full score matrix. Instead, it processes keys and values in tiles and updates the softmax online.

This reduces memory movement and makes attention more hardware-aware.

## Concepts Covered

- Scaled dot-product attention
- Softmax scaling
- Memory-bound vs compute-bound workloads
- DRAM vs SRAM/threadgroup memory
- Online softmax
- Tiling
- Custom Metal kernels
- SIMD lanes
- Reduction
- Broadcast
- Shuffle-based lane communication
- Benchmark interpretation

## Repository Structure

```text
src/attention/
  standard_attention.py      # MLX standard attention baseline
  online_softmax.py          # Online softmax implementation
  flash_attention.py         # Python tiled Flash Attention
  validate.py                # Correctness validation
  benchmark.py               # Runtime benchmark comparison

src/metal/
  *.metal                    # Custom Metal kernels
  *_wrapper.py               # MLX wrappers for Metal kernels

results/
  benchmark charts and visual outputs
```

## Implementation Journey

| Version | Idea | Result |
|---|---|---|
| Standard Attention | MLX baseline implementation | Fastest due to optimized backend |
| Python Flash Attention | Tiled attention with online softmax | Correct but slower due to Python-level loops |
| Basic Metal Kernel | Move computation into Metal | Correct but still scalar/manual |
| Hybrid Metal Kernel | Use 4 lanes per query token | Correct but reduction and output update bottlenecks remained |
| Hybrid Shuffle Kernel | Replace SRAM partial reduction with shuffle communication | Faster than hybrid |
| Parallel O Kernel | Split output vector update across lanes | Large speedup |
| Cached P Kernel | Cache `exp(score - max)` per tile | Best custom kernel |
| Float4 Kernel | Test vectorized K/V loading | Similar performance, not clearly better |

## Current Best Custom Kernel

The best custom kernel is:

`Hybrid Metal Shuffle Parallel O Cached P`

This version combines:

- shuffle-based reduction
- parallel output-vector update
- cached softmax probabilities

It is still slower than MLX standard attention, but it is around 9.89x faster than my original hybrid Metal kernel at `N=1024`.

## Final Benchmark

Latest benchmark on Apple Silicon M2 with 8GB RAM:

| Kernel | N=128 | N=256 | N=512 | N=1024 |
|---|---:|---:|---:|---:|
| Standard Attention | 0.34 ms | 0.55 ms | 1.74 ms | 3.45 ms |
| Flash Attention Python | 1.76 ms | 5.96 ms | 23.05 ms | 92.91 ms |
| Basic Metal Kernel | 3.93 ms | 14.75 ms | 50.02 ms | 178.78 ms |
| SIMD Metal Kernel | 86.21 ms | 334.88 ms | 1301.41 ms | 5095.45 ms |
| Hybrid Metal Kernel | 16.14 ms | 59.82 ms | 228.57 ms | 874.56 ms |
| Hybrid Shuffle Kernel | 13.51 ms | 47.31 ms | 180.59 ms | 688.58 ms |
| Parallel O Kernel | 4.74 ms | 15.91 ms | 59.32 ms | 229.37 ms |
| Cached P Kernel | 2.20 ms | 6.40 ms | 23.07 ms | 88.47 ms |
| Float4 Kernel | 2.15 ms | 6.43 ms | 23.09 ms | 88.83 ms |

## Benchmark Interpretation

Compared to MLX standard attention, the best custom kernel is still slower:

| N | Cached P vs Standard |
|---|---:|
| 128 | 6.51x slower |
| 256 | 11.69x slower |
| 512 | 13.23x slower |
| 1024 | 25.66x slower |

Compared to my original hybrid Metal kernel, the best custom kernel is much faster:

| N | Cached P vs Hybrid |
|---|---:|
| 128 | 7.32x faster |
| 256 | 9.35x faster |
| 512 | 9.91x faster |
| 1024 | 9.89x faster |

## Key Optimizations

### 1. Shuffle Reduction

The first hybrid kernel used threadgroup memory to store partial dot products. This required extra memory traffic and barriers.

I replaced this with shuffle-based lane communication, allowing lanes to exchange register values directly.

This reduced synchronization and improved performance.

### 2. Parallel Output Update

In the earlier hybrid kernel, one leader lane computed the softmax state and updated all 64 output dimensions.

I changed the mapping so 4 lanes split the output vector update.

For `d = 64`:

- Lane 0 updates dimensions 0, 4, 8, ...
- Lane 1 updates dimensions 1, 5, 9, ...
- Lane 2 updates dimensions 2, 6, 10, ...
- Lane 3 updates dimensions 3, 7, 11, ...

This removed a major serial bottleneck.

### 3. Cached Softmax Probability

The output update repeatedly computed:

```text
exp(score - max)
```

inside the inner loop.

I cached this value as:

```text
p[j] = exp(score[j] - max)
```

Then reused `p[j]` across output dimensions.

This avoided repeated expensive exponential calls and became the best custom kernel.

### 4. Float4 Loading Experiment

I also tested `float4` vectorized loading for K/V tiles.

The idea was to load 4 floats at once instead of scalar values.

However, because the kernel immediately unpacked `float4` values back into scalar threadgroup memory, the full data path was not vectorized.

As a result, `float4` was not clearly better than Cached P.

## Why MLX Standard Attention Is Still Faster

Flash Attention is a better memory-aware algorithm, but algorithmic improvement alone does not guarantee faster runtime.

MLX standard attention is backed by highly optimized kernels for matrix multiplication and tensor operations.

My custom kernel is educational and still uses:

- manual scalar loops
- custom softmax logic
- threadgroup memory loading
- explicit synchronization
- limited vectorized compute
- no production-level hardware tuning

So the correct conclusion is not that Flash Attention is bad.

The correct conclusion is:

My educational Metal implementation demonstrates the algorithm and improves significantly through optimization, but it is not yet as optimized as MLX backend kernels.

## Limitations

- The custom Metal kernels currently assume `d = 64`.
- Benchmarks are run on Apple Silicon M2 with 8GB RAM.
- Results may vary depending on thermals and background load.
- The goal is educational GPU kernel understanding, not production replacement for MLX kernels.

## What I Learned

- A better algorithm can still lose to a better implementation.
- Memory movement can dominate GPU runtime.
- Correctness is only the first step in GPU programming.
- Reduction can become a bottleneck.
- Thread mapping strongly affects performance.
- One overloaded lane can slow the whole kernel.
- Expensive operations should not be repeated inside inner loops.
- Every optimization must be measured.
- Benchmark wording should be honest and clear.

## Next Steps

Possible future improvements:

- Profile kernels using Xcode GPU tools
- Test float16
- Reduce threadgroup memory pressure
- Explore Metal matrix primitives
- Improve vectorized compute path
- Experiment with larger sequence lengths carefully within 8GB memory limits
