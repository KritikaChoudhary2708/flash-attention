import matplotlib.pyplot as plt
import numpy as np
import math
# M2 MacBook Air 8GB hardware limits
PEAK_BANDWIDTH_GBS = 100      # GB/s
PEAK_COMPUTE_GFLOPS = 3600    # GFLOP/s (FP16, 8-core GPU)

# Measured benchmark results
seq_lens = [128, 256, 512, 1024]

std_times_ms = [0.37, 0.64, 1.67, 3.88]
flash_times_ms = [1.75, 6.14, 24.07, 94.55]
metal_times_ms = [3.90, 15.55, 50.08, 178.67]

memory_mb = [0.3, 1.0, 4.2, 16.8]

# Calculated metrics for N=1024, H=8, B=1, d=64
N, H, B, d = 1024, 8, 1, 64

# FLOPs — same for both implementations
total_flops = 2 * 2 * N * N * d * H  # 2 matmuls, 2 ops each

# DRAM bytes
std_dram_bytes = 4 * H * N * N * 2    # 4 trips × score matrix
flash_dram_bytes = 4 * H * N * d * 2  # read Q,K,V + write O

# Arithmetic intensity
std_intensity = total_flops / std_dram_bytes
flash_intensity = total_flops / flash_dram_bytes

# Achieved performance
std_achieved = (total_flops / 1e9) / (std_times_ms[3] / 1000)
flash_achieved = (total_flops / 1e9) / (flash_times_ms[3] / 1000)

print(f"Standard Attention:")
print(f"  Arithmetic Intensity: {std_intensity:.1f} FLOPs/byte")
print(f"  Achieved: {std_achieved:.1f} GFLOP/s")
print(f"\nFlash Attention:")
print(f"  Arithmetic Intensity: {flash_intensity:.1f} FLOPs/byte")
print(f"  Achieved: {flash_achieved:.1f} GFLOP/s")

def plot_speed_benchmark():
    plt.figure(figsize=(10, 6))
    plt.plot(seq_lens, std_times_ms, 'b-o', label='Standard Attention', linewidth=2)
    plt.plot(seq_lens, flash_times_ms, 'r-o', label='Flash Attention (Python)', linewidth=2)
    plt.plot(seq_lens, metal_times_ms, 'g-o', label='Metal Kernel', linewidth=2)
    plt.xlabel('Sequence Length (N)')
    plt.ylabel('Time (ms)')
    plt.title('Attention Speed Benchmark — MacBook Air M2 8GB')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/speed_benchmark.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: results/speed_benchmark.png")

def plot_memory_benchmark():
    plt.figure(figsize=(10, 6))
    
    # Standard attention — quadratic growth
    plt.plot(seq_lens, memory_mb, 'b-o', label='Standard Attention (N²)', linewidth=2)
    
    # Flash Attention — linear growth
    flash_memory = [8 * N * 64 * 2 / 1e6 for N in seq_lens]
    plt.plot(seq_lens, flash_memory, 'r-o', label='Flash Attention O(N)', linewidth=2)
    
    plt.xlabel('Sequence Length (N)')
    plt.ylabel('Peak Memory (MB)')
    plt.title('Memory Usage — Standard vs Flash Attention')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/memory_benchmark.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: results/memory_benchmark.png")

def plot_roofline():
    plt.figure(figsize=(10, 6))
    
    # Draw roofline ceilings
    intensities = np.logspace(-1, 4, 1000)
    memory_roof = PEAK_BANDWIDTH_GBS * intensities
    compute_roof = np.full_like(intensities, PEAK_COMPUTE_GFLOPS)
    roofline = np.minimum(memory_roof, compute_roof)
    
    plt.loglog(intensities, roofline, 'k-', linewidth=2, label='Roofline ceiling')
    
    # Plot standard attention point
    plt.scatter([std_intensity], [std_achieved], 
                color='blue', s=200, zorder=5, label=f'Standard Attention ({std_intensity:.0f} FLOPs/byte)')
    
    # Plot flash attention point
    plt.scatter([flash_intensity], [flash_achieved],
                color='red', s=200, zorder=5, label=f'Flash Attention ({flash_intensity:.0f} FLOPs/byte)')
    
    # Labels
    plt.xlabel('Arithmetic Intensity (FLOPs/byte)')
    plt.ylabel('Performance (GFLOP/s)')
    plt.title('Roofline Analysis — MacBook Air M2 8GB')
    plt.legend()
    plt.grid(True, alpha=0.3, which='both')
    plt.axhline(y=PEAK_COMPUTE_GFLOPS, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=PEAK_BANDWIDTH_GBS, color='gray', linestyle='--', alpha=0.5)
    plt.savefig('results/roofline_chart.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: results/roofline_chart.png")

    
def main():
    import os
    os.makedirs('results', exist_ok=True)
    
    plot_speed_benchmark()
    plot_memory_benchmark()
    plot_roofline()
    
    print("\nAll charts saved to results/")
    print(f"\nKey findings:")
    print(f"Standard Attention intensity: {std_intensity:.1f} FLOPs/byte → memory bound")
    print(f"Flash Attention intensity:    {flash_intensity:.1f} FLOPs/byte → compute bound")
    print(f"Intensity improvement:        {flash_intensity/std_intensity:.1f}x")

if __name__ == "__main__":
    main()