uint q_idx = thread_position_in_grid.x;
uint h_idx = thread_position_in_grid.y;
uint tid = thread_position_in_threadgroup.x;

if (q_idx >= uint(N)) return;
threadgroup float K_tile[32][64];
threadgroup float V_tile[32][64];

float q_reg[64];
for (uint i = 0; i < uint(d); i++)
    q_reg[i] = Q[h_idx * N * d + q_idx * d + i];

float m = -INFINITY;
float l = 0.0f;
float o_reg[64] = {0};
for (uint jb = 0; jb < (N + 32 - 1) / 32; jb++) {
    uint j0 = jb * 32;
    
    if (tid < 32 && j0 + tid < uint(N)) {
        for (uint i = 0; i < uint(d); i++) {
            K_tile[tid][i] = K[h_idx * N * d + (j0 + tid) * d + i];
            V_tile[tid][i] = V[h_idx * N * d + (j0 + tid) * d + i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    uint bend = min(32u, uint(N) - j0);
    float s[32];
    float m_block = -INFINITY;
    for (uint j = 0; j < bend; j++) {
        float dot = 0.0f;
        for (uint i = 0; i < uint(d); i++)
            dot += q_reg[i] * K_tile[j][i];
        s[j] = dot * scale;
        m_block = max(m_block, s[j]);
    }
    
    float m_new = max(m, m_block);
    float alpha = exp(m - m_new);
    float l_block = 0.0f;
    for (uint j = 0; j < bend; j++)
        l_block += exp(s[j] - m_new);
    float l_new = alpha * l + l_block;
    
    for (uint i = 0; i < uint(d); i++) {
        float pv = 0.0f;
        for (uint j = 0; j < bend; j++)
            pv += exp(s[j] - m_new) * V_tile[j][i];
        o_reg[i] = (alpha * l * o_reg[i] + pv) / l_new;
    }
    
    m = m_new;
    l = l_new;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
for (uint i = 0; i < uint(d); i++)
    O[h_idx * N * d + q_idx * d + i] = o_reg[i];