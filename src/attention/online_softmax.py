import mlx.core as mx

def online_softmax_attention(Q,K,V):
          N, d = Q.shape
          scale = d**(-0.5)
          O = mx.zeros((N, d))
          l = mx.zeros((N,))
          m = mx.full((N,), -1e9)
          for j in range(N):
                    s_j = (Q@K[j]) * scale
                    m_new = mx.maximum(m, s_j)
                    correction = mx.exp(m - m_new)
                    l_new = correction * l + mx.exp(s_j - m_new)
                    v_j = V[j]
                    O = (correction[:, None] * l[:, None] * O + mx.exp(s_j - m_new)[:, None] * v_j[None, :]) / l_new[:, None]
                    m, l = m_new, l_new
          return O