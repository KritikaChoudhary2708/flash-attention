const uint QUERIES_PER_GROUP = 8;
const uint LANES_PER_QUERY = 4;

uint global_tid = thread_position_in_grid.x;
uint global_group = global_tid / 32;

uint h_idx = thread_position_in_grid.y;
uint b_idx = thread_position_in_grid.z;
uint tid = thread_position_in_threadgroup.x;

uint query_in_group = tid / LANES_PER_QUERY;
uint lane_in_query = tid % LANES_PER_QUERY;
uint q_idx = global_group * QUERIES_PER_GROUP + query_in_group;

if (q_idx >= uint(N)) return;

uint base = (b_idx * uint(H) + h_idx) * uint(N) * uint(d);

threadgroup float K_tile[32][64];
threadgroup float V_tile[32][64];
threadgroup float partials[8][4];

float m = -INFINITY;
float l = 0.0f;
float o_reg[64] = {0};
float s[32];

for (uint jb = 0; jb < (uint(N) + 32 - 1) / 32; jb++) {
    uint j0 = jb * 32;
        if (tid < 32 && j0 + tid < uint(N)) {
        for (uint dim = 0; dim < uint(d); dim++) {
            K_tile[tid][dim] = K[base + (j0 + tid) * uint(d) + dim];
            V_tile[tid][dim] = V[base + (j0 + tid) * uint(d) + dim];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint bend = min(32u, uint(N) - j0);

    for (uint j = 0; j < bend; j++) {
        float partial = 0.0f;

        for (uint dim = lane_in_query; dim < uint(d); dim += LANES_PER_QUERY) {
            partial += Q[base + q_idx * uint(d) + dim] * K_tile[j][dim];
        }

        partials[query_in_group][lane_in_query] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane_in_query == 0) {
            float dot = 0.0f;
            for (uint r = 0; r < LANES_PER_QUERY; r++) {
                dot += partials[query_in_group][r];
            }
            s[j] = dot * scale;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
        if (lane_in_query == 0) {
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

if (lane_in_query == 0) {
    for (uint dim = 0; dim < uint(d); dim++) {
        O[base + q_idx * uint(d) + dim] = o_reg[dim];
    }
}