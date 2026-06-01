import mlx.core as mx
import math

def flash_attention_forward(Q, K, V, block_size=32):
    B, H, N, d =  Q.shape
    scale = d**-0.5
    O = mx.zeros_like(Q)
    n_blcks = math.ceil(N/ block_size)
    for i in range(n_blcks):
        i_start = i * block_size
        i_end = min(i_start+block_size, N)
        Qi = Q[:, :, i_start:i_end,:]
        actual_br =  i_end - i_start
        mi = mx.full((B, H, actual_br), -1e9)
        li = mx.zeros((B, H, actual_br))
        Oi = mx.zeros((B, H, actual_br, d))
        for j in range(n_blcks):
            j_start = j* block_size
            j_end = min(j_start +block_size, N)
            Kj = K[:, : , j_start: j_end, :]
            Vj = V[:, : , j_start: j_end, :]
            Sij = mx.matmul(Qi, Kj.transpose(0,1,3,2))* scale
            mij = Sij.max(axis=-1)
            mi_new = mx.maximum(mi, mij)
            alpha = mx.exp(mi - mi_new)
            Pij = mx.exp(Sij-mi_new[:, :, :, None])
            li_new = alpha * li + Pij.sum(axis=-1)
            Oi = (alpha[:, :, :, None] * li[:, :, :, None] * Oi + mx.matmul(Pij, Vj)) / li_new[:, :, :, None]
            mi, li = mi_new, li_new
        O[:, :, i_start:i_end, :] = Oi
    return O




