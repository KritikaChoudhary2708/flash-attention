uint global_tid = thread_position_in_grid.x;
uint h_idx = thread_position_in_grid.y;
uint b_idx = thread_position_in_grid.z;

uint lane_id = global_tid % 32;
uint q_idx = global_tid / 32;

if (q_idx >= uint(N)) return;

uint base = (b_idx * uint(H) + h_idx) * uint(N) * uint(d);
threadgroup float K_tile[32][64];
threadgroup float V_tile[32][64];
float m = -INFINITY;
float l = 0.0f;
float o_reg[64] = {0};
float s[32];

for (uint jb = 0; jb < (uint(N) + 32 - 1) / 32; jb++) {
    uint j0 = jb * 32;
        if (lane_id < 32 && j0 + lane_id < uint(N)) {
        for (uint dim = 0; dim < uint(d); dim++) {
            K_tile[lane_id][dim] = K[base + (j0 + lane_id) * uint(d) + dim];
            V_tile[lane_id][dim] = V[base + (j0 + lane_id) * uint(d) + dim];
          }
          }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint bend = min(32u, uint(N) - j0);
    for (uint j = 0; j < bend; j++) {
        float partial = 0.0f;

        partial += Q[base + q_idx * uint(d) + lane_id] * K_tile[j][lane_id];
        partial += Q[base + q_idx * uint(d) + lane_id + 32] * K_tile[j][lane_id + 32];

        float dot = simd_sum(partial);

        if (lane_id == 0) {
            s[j] = dot * scale;
        }
    }
    if (lane_id == 0) {
        float m_block = -INFINITY;
        for (uint j = 0; j < bend; j++) {
            m_block = max(m_block, s[j]);
        }

        float m_new = max(m, m_block);
        float alpha = exp(m - m_new);

        float l_block = 0.0f;
        for (uint j = 0; j < bend; j++) {
            l_block += exp(s[j] - m_new);
        }

        float l_new = alpha * l + l_block;

        for (uint dim = 0; dim < uint(d); dim++) {
            float pv = 0.0f;
            for (uint j = 0; j < bend; j++) {
                pv += exp(s[j] - m_new) * V_tile[j][dim];
            }
            o_reg[dim] = (alpha * l * o_reg[dim] + pv) / l_new;
        }

        m = m_new;
        l = l_new;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane_id == 0) {
    for (uint dim = 0; dim < uint(d); dim++) {
        O[base + q_idx * uint(d) + dim] = o_reg[dim];
    }
}