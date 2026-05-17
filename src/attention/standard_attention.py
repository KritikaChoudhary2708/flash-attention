import mlx.core as mx
import math

def standard_attention(Q,K,V, mask =None):
          d_k =  Q.shape[-1]
          scale = 1.0 / math.sqrt(d_k)
          scores = mx.matmul(Q, K.transpose(0,1,3,2))*scale
          if mask is not None:
                    scores = scores+ mask
          weights = mx.softmax(scores, axis =-1)
          output = mx.matmul(weights, V)
          return output