import mlx.core as mx
from standard_attention import standard_attention
from online_softmax import online_softmax_attention

def validate(N=64, d=64, tolerance=1e-4):
          Q = mx.random.normal((N, d))
          K = mx.random.normal((N, d))
          V = mx.random.normal((N, d))
          std_out = standard_attention(Q[None, None], K[None, None], V[None, None])[0, 0]
          online_out = online_softmax_attention(Q, K, V)
          mx.eval(std_out, online_out)
          diff = float(mx.abs(std_out - online_out).max())
          print(f'Max difference: {diff:.2e}')
          if diff < tolerance:
            print('PASS — outputs match')
          else:
            print('FAIL — outputs differ, check your math')
        

if __name__ == "__main__":
          validate()
          