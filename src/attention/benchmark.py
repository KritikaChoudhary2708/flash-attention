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

if __name__ == "__main__":
          from standard_attention import standard_attention
          results=benchmark_attention(standard_attention)