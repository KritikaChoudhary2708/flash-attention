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

float m = -INFINITY;
float l = 0.0f;
float o_reg[64] = {0};
float s[32];

uint leader_lane = query_in_group * LANES_PER_QUERY;

for (uint jb = 0; jb < (uint(N) + 32 - 1) / 32; jb++) {
    uint j0 = jb * 32;

    if (tid < 32 && j0 + tid < uint(N)) {
        uint row_base = base + (j0 + tid) * uint(d);

        for (uint dim = 0; dim < uint(d); dim += 4) {
            float4 k_vec = *((device const float4*)(K + row_base + dim));
            float4 v_vec = *((device const float4*)(V + row_base + dim));

            K_tile[tid][dim + 0] = k_vec.x;
            K_tile[tid][dim + 1] = k_vec.y;
            K_tile[tid][dim + 2] = k_vec.z;
            K_tile[tid][dim + 3] = k_vec.w;

            V_tile[tid][dim + 0] = v_vec.x;
            V_tile[tid][dim + 1] = v_vec.y;
            V_tile[tid][dim + 2] = v_vec.z;
            V_tile[tid][dim + 3] = v_vec.w;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint bend = min(32u, uint(N) - j0);

    for (uint j = 0; j < bend; j++) {
        float partial = 0.0f;

        for (uint dim = lane_in_query; dim < uint(d); dim += LANES_PER_QUERY) {
            partial += Q[base + q_idx * uint(d) + dim] * K_tile[j][dim];
        }

        float dot = partial;
        dot += simd_shuffle(partial, query_in_group * 4 + 1);
        dot += simd_shuffle(partial, query_in_group * 4 + 2);
        dot += simd_shuffle(partial, query_in_group * 4 + 3);

        float score = dot * scale;
        score = simd_shuffle(score, leader_lane);
        s[j] = score;
    }

    float m_new = m;
    float alpha = 1.0f;
    float l_new = l;

    if (lane_in_query == 0) {
        float m_block = -INFINITY;

        for (uint j = 0; j < bend; j++) {
            m_block = max(m_block, s[j]);
        }

        m_new = max(m, m_block);
        alpha = exp(m - m_new);

        float l_block = 0.0f;

        for (uint j = 0; j < bend; j++) {
            l_block += exp(s[j] - m_new);
        }

        l_new = alpha * l + l_block;
    }

    m_new = simd_shuffle(m_new, leader_lane);
    alpha = simd_shuffle(alpha, leader_lane);
    l_new = simd_shuffle(l_new, leader_lane);
    float p[32];

    for (uint j = 0; j < bend; j++) {
        p[j] = exp(s[j] - m_new);
    }

    for (uint dim = lane_in_query; dim < uint(d); dim += LANES_PER_QUERY) {
    float pv = 0.0f;

        for (uint j = 0; j < bend; j++) {
            pv += p[j] * V_tile[j][dim];
        }

        o_reg[dim] = (alpha * l * o_reg[dim] + pv) / l_new;
    }

    m = m_new;
    l = l_new;

    threadgroup_barrier(mem_flags::mem_threadgroup);
}

for (uint dim = lane_in_query; dim < uint(d); dim += LANES_PER_QUERY) {
    O[base + q_idx * uint(d) + dim] = o_reg[dim];
}