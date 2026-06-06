import mlx.core as mx
from src.attention.standard_attention import standard_attention
from src.attention.online_softmax import online_softmax_attention
from src.attention.flash_attention import flash_attention_forward
from src.metal.metal_wrapper import metal_flash_attention
from src.metal.simd_wrapper import metal_flash_attention_simd

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


def validate_flash(N=512, d=64, n_heads=8, tol=1e-3):
    Q = mx.random.normal((1, n_heads, N, d))
    K = mx.random.normal((1, n_heads, N, d))
    V = mx.random.normal((1, n_heads, N, d))
    
    std_out = standard_attention(Q, K, V)
    flash_out = flash_attention_forward(Q, K, V)
    mx.eval(std_out, flash_out)
    
    diff = float(mx.abs(std_out - flash_out).max())
    print(f'Flash Attention max difference: {diff:.2e}')
    print('PASS' if diff < tol else 'FAIL')      



def validate_metal(N=512, d=64, n_heads=8, tol=1e-2):
    Q = mx.random.normal((1, n_heads, N, d))
    K = mx.random.normal((1, n_heads, N, d))
    V = mx.random.normal((1, n_heads, N, d))
    
    std_out = standard_attention(Q, K, V)
    metal_out = metal_flash_attention(Q, K, V)
    mx.eval(std_out, metal_out)
    
    diff = float(mx.abs(std_out - metal_out).max())
    print(f'Metal kernel max difference: {diff:.2e}')
    print('PASS' if diff < tol else 'FAIL')
  
def validate_metal_simd(N=512, d=64, n_heads=8, tol=1e-2):
    Q = mx.random.normal((1, n_heads, N, d))
    K = mx.random.normal((1, n_heads, N, d))
    V = mx.random.normal((1, n_heads, N, d))

    std_out = standard_attention(Q, K, V)
    simd_out = metal_flash_attention_simd(Q, K, V)
    mx.eval(std_out, simd_out)

    diff = float(mx.abs(std_out - simd_out).max())
    print(f'SIMD Metal kernel max difference: {diff:.2e}')
    print('PASS' if diff < tol else 'FAIL')

if __name__ == "__main__":
          validate()
          validate_flash()
          validate_metal()
          validate_metal_simd()
          