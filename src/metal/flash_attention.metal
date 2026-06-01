#include <metal_stdlib>
using namespace metal;

constant int BLOCK_SIZE = 32;
kernel void flash_attention_kernel(
          device const float* Q [[buffer(0)]],
          device const float* K [[buffer(1)]],
          device const float* V [[buffer(2)]],
          device float* O_out [[buffer(3)]],
          constant int& N [[buffer(4)]],
          constant int& d [[buffer(5)]],
          constant float& scale [[buffer(6)]],
          uint2 gid [[thread_position_in_grid]],
          uint tid [[thread_index_in_threadgroup]]
) {
          threadgroup float K_tile[32][64];
          threadgroup float V_tile[32][64];
          int q_idx = gid.x;
          int h_idx = gid.y;
          if ( q_idx >=N) return;
          float q_reg[64];
          for (int i = 0; i < d; i++)
                    q_reg[i] = Q[h_idx*N*d +q_idx*d+i];
          
          float m = -INFINITY;
          float l = 0.0f;
          float o_reg[64] = {0};
          // tiling loop 
          for (int jb = 0; jb < (N+ BLOCK_SIZE-1)/BLOCK_SIZE; jb++){
                    int j0 = jb* BLOCK_SIZE;
                    // STEP 1: cooperative loading into SRAM
                    if (tid < BLOCK_SIZE && j0+tid <N){
                              for (int i = 0; i < d; i++) {
                                        K_tile[tid][i] = K[h_idx *N*d+(j0+tid)*d+i];
                                        V_tile[tid][i] = V[h_idx *N*d+(j0+tid)*d+i];
                              }
                    }
          
                    threadgroup_barrier(mem_flags::mem_threadgroup); // barrier 1
                    // STEP 2: score computation
                    int bend = min(BLOCK_SIZE, N - j0);
                    float s[32];
                    float m_block = -INFINITY;
                    for (int j = 0; j < bend; j++) {
                              float dot = 0.0f;
                              for (int i = 0; i < d; i++)
                              dot += q_reg[i] * K_tile[j][i];
                              s[j] = dot * scale;
                              m_block = max(m_block, s[j]);
                    }
          
                    // STEP 3: online softmax update
                    float m_new = max(m, m_block);
                    float alpha = exp(m - m_new);
                    float l_block = 0.0f;
                    for (int j = 0; j < bend; j++)
                              l_block += exp(s[j] - m_new);
                    float l_new = alpha * l + l_block;
          
                    // Step 4: output update
                    for (int i = 0; i <d; i++){
                              float pv = 0.0f;
                              for (int j = 0; j < bend; j++){
                                        pv += exp(s[j]-m_new)*V_tile[j][i];
                              }
                              o_reg[i] = (alpha*l*o_reg[i]+pv)/l_new;
                    }

                    // Step 5: Update running variables
                    m = m_new;
                    l= l_new;
                    threadgroup_barrier(mem_flags::mem_threadgroup); // barrier 2
          }
          // STEP 6: write output
          for (int i = 0; i < d; i++)
                    O_out[h_idx * N * d + q_idx * d + i] = o_reg[i];


}