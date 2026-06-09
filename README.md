# FlashAttention on Apple Silicon (MLX & Metal)

This repository contains custom implementations of **FlashAttention** and **Online Softmax** on Apple Silicon GPUs, built using Apple's [MLX library](https://github.com/ml-explore/mlx) and custom **Metal (MSL)** compute shaders. 

It contains standard dot-product attention, online softmax attention, a python-based block FlashAttention, and three custom Metal shaders showing different levels of GPU optimization (Naive, SIMD-group, and Hybrid).

---

## 📂 Project Structure

- 💾 `src/`
  - 📂 `attention/`
    - [standard_attention.py](src/attention/standard_attention.py): Baseline attention implementation using standard matrix multiplications and a global softmax.
    - [online_softmax.py](src/attention/online_softmax.py): Incremental online softmax attention that avoids materializing the full $N \times N$ attention matrix.
    - [flash_attention.py](src/attention/flash_attention.py): Tiled, block-based Python implementation of FlashAttention.
    - [validate.py](src/attention/validate.py): Correctness validation suite comparing all custom implementations against standard attention.
    - [benchmark.py](src/attention/benchmark.py): Speed and memory benchmarking script.
    - [roofline.py](src/attention/roofline.py): Analytical roofline modeling and visualization scripts.
  - 📂 `metal/`
    - [flash_attention_mlx.metal](src/metal/flash_attention_mlx.metal): Naive custom Metal shader for block-based attention.
    - [metal_wrapper.py](src/metal/metal_wrapper.py): Python loader for the naive Metal kernel using `mlx.fast.metal_kernel`.
    - [flash_attention_simd.metal](src/metal/flash_attention_simd.metal): Metal shader optimized using SIMD-groups (`simd_sum`) for parallelizing query dimension computations.
    - [simd_wrapper.py](src/metal/simd_wrapper.py): Python wrapper for the SIMD-optimized Metal kernel.
    - [flash_attention_hybrid.metal](src/metal/flash_attention_hybrid.metal): Hybrid Metal shader grouping multiple queries per threadgroup for optimal thread occupancy.
    - [hybrid_wrapper.py](src/metal/hybrid_wrapper.py): Python wrapper for the Hybrid Metal kernel.
- 💾 `results/`
  - `speed_benchmark.png`: Plot comparing computation time across attention variants.
  - `memory_benchmark.png`: Plot illustrating memory usage scaling (quadratic vs. linear).
  - `roofline_chart.png`: Roofline model charting arithmetic intensity vs achieved performance.

---

## 🛠 Setup & Installation

Ensure you have a macOS machine with Apple Silicon (M1/M2/M3/M4).

1. Clone or navigate to the repository:
   ```bash
   cd flash-attention
   ```
2. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```
3. Run validations or benchmarks.

---

## 🧪 Verification & Correctness

To run the correctness validations across all implementations:
```bash
PYTHONPATH=. .venv/bin/python3 src/attention/validate.py
```

Expected Output:
```text
Max difference: 8.34e-07
PASS — outputs match
Flash Attention max difference: 4.17e-07
PASS
Metal kernel max difference: 3.87e-07
PASS
SIMD Metal kernel max difference: 6.85e-07
PASS
Hybrid Metal kernel max difference: 3.58e-07
PASS
```

---

## 📊 Performance Benchmark

To run the performance benchmark:
```bash
PYTHONPATH=. .venv/bin/python3 -u src/attention/benchmark.py
```

### Empirical Results (MacBook Air M2 8GB)

Measured execution times (average of 17 runs after 3 warmup runs) for Batch Size $B=1$, Heads $H=8$, Head Dimension $d=64$:

| Sequence Length ($N$) | Standard Attention | Flash Attention (Python) | Naive Metal Kernel | Hybrid Metal Kernel | SIMD Metal Kernel | Memory Usage |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **128** | 0.45 ms | 1.74 ms | 4.61 ms | 16.04 ms | 89.82 ms | 0.3 MB |
| **256** | 0.70 ms | 5.90 ms | 16.52 ms | 58.11 ms | 349.89 ms | 1.0 MB |
| **512** | 1.38 ms | 27.31 ms | 53.65 ms | 229.28 ms | 1321.24 ms | 4.2 MB |
| **1024** | 3.43 ms | 97.65 ms | 190.58 ms | 899.72 ms | 5147.25 ms | 16.8 MB |

#### Insights & Performance Analysis:
1. **Standard MLX Attention** is highly optimized. It uses Apple's native, heavily optimized C++ backend libraries, which compile/fuse operators to run at hardware-peak levels (achieving 3.43 ms at $N=1024$).
2. **Python Flash Attention** incurs considerable loop overhead due to the nested interpreter loops (32 tiles * 32 tiles = 1024 steps per batch/head) in Python, but achieves linear $O(N)$ peak memory scaling.
3. **Naive Metal Kernel** out-performs the other custom Metal kernels. This is because each query is processed independently per threadgroup, minimizing thread synchronization/barriers.
4. **SIMD Metal Kernel** suffers from extreme execution divergence. Specifically, only `lane_id == 0` performs the final softmax calculation and `o_reg` updates, leaving the other 31 lanes in the warp idle. Frequent barriers (`threadgroup_barrier`) also contribute to the overhead.
5. **Hybrid Metal Kernel** groups 8 queries per threadgroup and allocates 4 lanes per query to parallelize dot-products. This reduces the number of threadgroups and achieves a **5.7x speedup** over the SIMD kernel (899.72 ms vs. 5147.25 ms at $N=1024$).

---

## 📈 Roofline Model & Arithmetic Intensity

The roofline model is a visual method to identify whether a kernel is **Memory-Bandwidth Bound** or **Compute-Capacity Bound** based on its **Arithmetic Intensity** ($\text{FLOPs/Byte}$).

To generate the analytical charts:
```bash
PYTHONPATH=. .venv/bin/python3 src/attention/roofline.py
```

### Analysis details (for $N=1024$, $H=8$, $B=1$, $d=64$):
- **Total FLOPs**: $2 \times 2 \times N^2 \times d \times H = 1,073,741,824$ (1.07 GFLOPs)
- **Standard Attention DRAM Traffic**: Scores matrix is read/written multiple times ($4 \times H \times N^2 \times 2 \text{ bytes} = 67,108,864$ bytes)
  - **Arithmetic Intensity**: $32.0 \text{ FLOPs/byte}$
  - **Bottleneck**: Memory-Bound. The large intermediate score matrix forces repeated DRAM round-trips.
- **Flash Attention DRAM Traffic**: Intermediate score matrix is kept entirely in on-chip SRAM; only $Q$, $K$, $V$, and $O$ are read/written once ($4 \times H \times N \times d \times 2 \text{ bytes} = 4,194,304$ bytes)
  - **Arithmetic Intensity**: $512.0 \text{ FLOPs/byte}$ ($16\times$ improvement!)
  - **Bottleneck**: Compute-Bound. The kernel is limited by the processor's raw floating-point throughput, not memory latency.

These charts are saved in the `results/` folder:
- **Speed Comparison Plot**: `results/speed_benchmark.png`
- **Memory Scaling Plot**: `results/memory_benchmark.png`
- **Roofline Boundary Chart**: `results/roofline_chart.png`
