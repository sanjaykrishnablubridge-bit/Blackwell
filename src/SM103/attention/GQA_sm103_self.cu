#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>
#include <cassert>
#include <cmath>
#include <cuda_bf16.h>
#include "utils/kernelUtils.cuh"

using namespace nvcuda;

// =================================
//  V1 : wmma + two pass softmax
// =================================
template<int Br, int Bc, int D>
__device__ inline void computeScores(
    const __nv_bfloat16 *sQ,
    const __nv_bfloat16 *sK,
    float *sS,
    float scale
){
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> qf;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> kf;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;

  for(int nt = 0; nt < Bc / 16; ++nt){
    wmma::fill_fragment(acc, 0.0f);
    for(int kt = 0; kt < D/16; ++kt){
      wmma::load_matrix_sync(qf, sQ + kt * 16, D);
      wmma::load_matrix_sync(kf, sK + nt * 16 * D + kt * 16, D);
      wmma::mma_sync(acc, qf, kf, acc);
    }
    for(int t = 0; t < acc.num_elements; ++t){
      acc.x[t] *= scale;
    }
    wmma::store_matrix_sync(sS + nt * 16, acc, Bc, wmma::mem_row_major);
  }
}

template<int Br, int Bc, int D>
__global__ void gqa_v1(
  __nv_bfloat16 *d_Q,
  __nv_bfloat16 *d_K,
  __nv_bfloat16 *d_V,
  __nv_bfloat16 *d_O,
  float *d_LSE,
  int B,
  int Hq,
  int Hkv,
  int G,
  int S,
  float scale
){
  const int b = blockIdx.x;
  const int hq = blockIdx.y;
  const int q_tile = blockIdx.z;
  const int hkv  = hq/G;
  const int lane = threadIdx.x;

  const int q_row0 = q_tile * Br;
  const int nKeyTiles = S/Bc;

  const long qBase = ((b * Hq + hq) * S + q_row0) * D;
  const long kvBase = ((b * Hkv + hkv) * S) * D;
  const long lBase = (b * Hq + hq) * S + q_row0;

  __shared__ __align__(16) __nv_bfloat16 sQ[Br * D];
  __shared__ __align__(16) __nv_bfloat16 sK[Bc * D];
  __shared__ __align__(16) __nv_bfloat16 sV[Bc * D];
  __shared__ __align__(16) float         sS[Br * Bc];
  __shared__ __align__(16) __nv_bfloat16 sP[Br * Bc];
  __shared__ __align__(16) float         sO[Br * D];
  __shared__ float sm[Br];
  __shared__ float sl[Br];

  // load Q tile
  for(int i = lane; i < Br * D; i+=32)
    sQ[i] = d_Q[qBase + i];
  if(lane < Br) sm[lane] = -INFINITY;
  __syncwarp();

  for(int kc = 0; kc < nKeyTiles; ++kc){
    long kBase = kvBase + (long)kc * Bc * D;
    for(int i = lane; i < Bc * D; i+=32){
      sK[i] = d_K[kBase + i];
    }
    __syncwarp();

    computeScores<Br, Bc, D>(sQ, sK, sS, scale);
__syncwarp();

    if(lane < Br){
      float max = sm[lane];
      for(int j = 0; j < Bc; ++j){
        max = fmaxf(max, sS[lane * Bc + j]);
      }
      sm[lane] = max;
    }
    __syncwarp();
  }
  
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> oacc[D/16];
  for(int nt = 0; nt < D/16; ++nt){
    wmma::fill_fragment(oacc[nt], 0.0f);
  } 
  if(lane < Br) sl[lane] = 0.0f;
  __syncwarp();

  for(int kc = 0; kc < nKeyTiles; ++kc){
    const long kBase = kvBase + (long)kc * Bc * D;
    for(int i = lane; i < Bc * D; i+=32){
      sK[i] = d_K[kBase + i];
      sV[i] = d_V[kBase + i];
    }
    __syncwarp();
    // why do we have this here?
    computeScores<Br, Bc, D>(sQ, sK, sS, scale);
    __syncwarp();

    if(lane < Br){
      float mrow = sm[lane];
      float s = 0.0f;
      for(int j = 0; j < Bc; ++j){
        float e = expf(sS[lane * Bc + j] - mrow);
        sP[lane * Bc + j] = __float2bfloat16(e);
        s += e;
      }
      sl[lane] += s;
    }
    __syncwarp();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> pf;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> vf;
    for(int nt = 0; nt < D/16; ++nt){
      for(int kt = 0; kt < Bc/16; ++kt){
        wmma::load_matrix_sync(pf, sP + kt * 16, Bc);
        wmma::load_matrix_sync(vf, sV + kt * 16 * D + nt * 16, D);
        wmma::mma_sync(oacc[nt], pf, vf, oacc[nt]);
      }
    }
    __syncwarp();
  }

  for(int nt = 0; nt < D/16; ++nt){
    wmma::store_matrix_sync(sO + nt * 16, oacc[nt], D, wmma::mem_row_major);
  }
  __syncwarp();

  if(lane < Br){
    float denom = sl[lane];
    float inv = 1.0f / denom;
    for(int d = 0; d < D; ++d){
      d_O[qBase + lane * D + d] = __float2bfloat16(sO[lane * D + d] * inv);
    }
    d_LSE[lBase + lane] = sm[lane] + logf(denom);
  }
}

// =================================
//  V2 : wmma + online softmax
// =================================
template<int Br, int Bc, int D>
__global__ void gqa_v2(
  __nv_bfloat16 *d_Q,
  __nv_bfloat16 *d_K,
  __nv_bfloat16 *d_V,
  __nv_bfloat16 *d_O,
  float *d_LSE,
  int B,
  int Hq,
  int Hkv,
  int G,
  int S,
  float scale
){
  const int b      = blockIdx.x;
  const int hq     = blockIdx.y;
  const int q_tile = blockIdx.z;
  const int hkv    = hq / G;
  const int lane   = threadIdx.x;

  const int q_row0    = q_tile * Br;
  const int nKVTiles = S / Bc;

  /*
   * Q is of shape [B, Hq, S, D] 
   * to reach the [b, hq, q_row0, 0], the stride is
   * b * Hq * S * D + hq * S * D + q_row0 * D + 0
   * => ((b * Hq + hq) * S + q_row0) * D 
  */
  const long qBase  = ((long)(b * Hq  + hq)  * S  + q_row0) * D;
  const long kvBase = ((long)(b * Hkv + hkv) * S) * D;
  const long lBase  = ((long)(b * Hq  + hq)  * S) + q_row0;

  __shared__ __align__(16) __nv_bfloat16 sQ[Br * D];
  __shared__ __align__(16) __nv_bfloat16 sK[Bc * D];
  __shared__ __align__(16) __nv_bfloat16 sV[Bc * D];
  __shared__ __align__(16) __nv_bfloat16 sP[Br * Bc];
  __shared__ __align__(16) float         sS[Br * Bc];   // scores for the current KV tile
  __shared__ __align__(16) float         sO[Br * D];    // running (unnormalized) output
  __shared__ __align__(16) float         sPV[Br * D];   // P@V for the current KV tile
  __shared__ float sm[Br];                              // running row max
  __shared__ float sl[Br];                              // running row denominator
  __shared__ float sCorr[Br];                           // per-row rescale factor exp(m_old - m_new)

  // Load Q tile, zero the running output, init running stats.
  for(int i = lane; i < Br * D; i += 32){
    sQ[i] = d_Q[qBase + i];
    sO[i] = 0.0f;
  }
  if(lane < Br){
    sm[lane] = -INFINITY;
    sl[lane] = 0.0f;
  }
  __syncwarp();

  // Fused online-softmax loop over KV tiles:
  //   m_new = max(m, rowmax(S_tile))
  //   corr  = exp(m - m_new)
  //   P     = exp(S_tile - m_new)                  (unnormalized)
  //   l     = l*corr + rowsum(P)
  //   O     = O*corr + P @ V                        (running, unnormalized)
  for(int kc = 0; kc < nKVTiles; ++kc){
    const long kBase = kvBase + (long)kc * Bc * D;
    for(int i = lane; i < Bc * D; i += 32){
      sK[i] = d_K[kBase + i];
      sV[i] = d_V[kBase + i];
    }
    __syncwarp();

    // S = (Q @ K^T) * scale  ->  sS [Br, Bc]
    {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> qf;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> kf;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;

      for(int nt = 0; nt < Bc / 16; ++nt){
        wmma::fill_fragment(acc, 0.0f);
        for(int kt = 0; kt < D / 16; ++kt){
          wmma::load_matrix_sync(qf, sQ + kt * 16, D);
          wmma::load_matrix_sync(kf, sK + nt * 16 * D + kt * 16, D);
          wmma::mma_sync(acc, qf, kf, acc);
        }
        for(int t = 0; t < acc.num_elements; ++t){
          acc.x[t] *= scale;
        }
        wmma::store_matrix_sync(sS + nt * 16, acc, Bc, wmma::mem_row_major);
      }
    }
    __syncwarp();

    // Online-softmax stats update + unnormalized P, one lane per query row.
    if(lane < Br){
      const float m_old = sm[lane];
      const float l_old = sl[lane];

      float tile_max = -INFINITY;
      for(int j = 0; j < Bc; ++j){
        tile_max = fmaxf(tile_max, sS[lane * Bc + j]);
      }

      const float m_new = fmaxf(m_old, tile_max);
      const float corr  = __expf(m_old - m_new);   // 0 on the first tile (m_old = -inf)

      float p_sum = 0.0f;
      for(int j = 0; j < Bc; ++j){
        const float p = __expf(sS[lane * Bc + j] - m_new);
        sP[lane * Bc + j] = __float2bfloat16(p);
        p_sum += p;
      }

      sm[lane]    = m_new;
      sl[lane]    = l_old * corr + p_sum;
      sCorr[lane] = corr;
    }
    __syncwarp();

    // Rescale the running output by corr before accumulating this tile.
    for(int i = lane; i < Br * D; i += 32){
      sO[i] *= sCorr[i / D];
    }
    __syncwarp();

    // PV = P @ V  ->  sPV [Br, D]
    {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> pf;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> vf;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> oacc[D/16];

      for(int nt = 0; nt < D/16; ++nt){
        wmma::fill_fragment(oacc[nt], 0.0f);
        for(int vt = 0; vt < Bc/16; ++vt){
          wmma::load_matrix_sync(pf, sP + vt * 16, Bc);
          wmma::load_matrix_sync(vf, sV + vt * 16 * D + nt * 16, D);
          wmma::mma_sync(oacc[nt], pf, vf, oacc[nt]);
        }
        wmma::store_matrix_sync(sPV + nt * 16, oacc[nt], D, wmma::mem_row_major);
      }
    }
    __syncwarp();

    // Accumulate this tile's P@V into the running output.
    for(int i = lane; i < Br * D; i += 32){
      sO[i] += sPV[i];
    }
    __syncwarp();
  } // end of the fused KV loop

  // Finalize: normalize by the running denominator and write LSE.
  for(int i = lane; i < Br * D; i += 32){
    d_O[qBase + i] = __float2bfloat16(sO[i] / sl[i / D]);
  }
  if(lane < Br){
    d_LSE[lBase + lane] = sm[lane] + logf(sl[lane]);
  }

} // end of the v2 kernel

// =================================
//  V3 : mma + online softmax
// =================================
__device__ uint64_t make_smem_desc(
  void* smem_ptr, // pointer to the start of the matrix in the shared memory
  int ld_bytes // leading dimension in bytes = no: of bytes per row for row-major
){
  uint64_t addr = (uint64_t)__cvta_generic_to_shared(smem_ptr);
  // Address must be 16-byte aligned 
  // addr >> 4 gives address in 16-byte units, fits in 14 bits
  uint64_t desc = 0;
  desc |= (addr >> 4) & 0x3FFFull;
  desc |= ((uint64_t)(ld_bytes / 16)) << 16;
  desc |= ((uint64_t)(ld_bytes / 16)) << 32;
  return desc;
}

__device__ uint32_t make_idesc_m64_bf16(int N){
  uint32_t idesc = 0;
  // sparsity : dense = 0
  idesc |= 0u;
  // N encoding: (N/8 - 1) in bits
  uint32_t n_enc = (uint32_t)(N / 8-1);
  idesc |= (n_enc & 0x3Fu) << 2;
  // K encoding: K = 16 -> 0 in bits
  idesc |= (0u) << 8;
  // atype: bf16 = 1 in bits
  idesc |= (1u) << 10;
  // btype: bf16 = 1 in bits
  idesc |= (1u) << 12;
  // M encoding: M =64 -> 0 in bits
  idesc |= (0u) << 16;

  return idesc;
}

__device__ uint64_t advance_desc_col(uint64_t desc, int col_step_elements){
  uint64_t addr_delta = (uint64_t)(col_step_elements) / 8ULL;
  uint64_t base_addr = desc & 0x3FFFull;
  uint64_t new_addr = (base_addr + addr_delta) & 0x3FFFull;
  return (desc & ~0x3FFFull) | new_addr;
} 

__device__ uint64_t advance_desc_row(uint64_t desc, int num_rows, int row_stride_bytes){
  uint64_t byte_offset = (uint64_t)num_rows * (uint64_t)row_stride_bytes;
  uint64_t addr_delta = byte_offset / 16ULL;
  uint64_t base_addr = desc & 0x3FFFull;
  uint64_t new_addr = (base_addr + addr_delta) & 0x3FFFull;
  return (desc & ~03FFFull) | new_addr;
}

__device__ void tmem_readout_to_smem(
  float* smem_out,
  uint32_t tmem_addr,
  int N,
  int smem_stride,
  float scale
){
  int warp_id = threadIdx.x / 32;
  int lane = threadIdx.x % 32;

  int row = warp_id * 16 + (lane / 4);

  int cols_per_lane = N / 32;
  int col_base = (lane % 4) * cols_per_lane;

  for(int col_off = 0; col_off < cols_per_lane; ++col_off){
    int col = col_base + col_off;
    uint32_t raw;

    asm volatile(
      "tcgen05.ld.sync.aligned.32b.x1 {%0}, [%1];"
      : "=r"(raw)
      : "r"(tmem_addr + (uint32_t)col)
      : "memory"
    );

    float val = reinterpret_cast<float&>(raw) * scale;
    smem_out[row * smem_stride + col] = val;
  }
}

template<int Br, int Bc, int D>
__global__ void gqa_v3(
  __nv_bfloat16 *d_Q,
  __nv_bfloat16 *d_K,
  __nv_bfloat16 *d_V,
  __nv_bfloat16 *d_O,
  float *d_LSE,
  int B,
  int Hq,
  int Hkv,
  int G,
  int S,
  float scale
){
  const int b      = blockIdx.x;
  const int hq     = blockIdx.y;
  const int q_tile = blockIdx.z;
  const int hkv    = hq / G;
  const int tid   = threadIdx.x;

  const int q_row0   = q_tile * Br;
  const int nKVTiles = S / Bc;

  const long qBase  = ((long)(b * Hq + hq) * S + q_row0) * D;
  const long kvBase = ((long)(b * Hkv + hkv) * S) * D;
  const long lBase  = ((long)(b * Hq + hq) * S + q_row0);

  __shared__ __align__(16) __nv_bfloat16 sQ[Br * D];
  __shared__ __align__(16) __nv_bfloat16 sK[Bc * D];
  __shared__ __align__(16) __nv_bfloat16 sV[Bc * D];
  __shared__ __align__(16) __nv_bfloat16 sP[Br * Bc];
  __shared__ __align__(16) float         sS[Br * Bc];   // scores for the current KV tile
  __shared__ __align__(16) float         sO[Br * D];    // running (unnormalized) output
  __shared__ __align__(16) float         sPV[Br * D];   // P@V for the current KV tile
  __shared__ float sm[Br];                              // running row max
  __shared__ float sl[Br];                              // running row denominator
  __shared__ float sCorr[Br];                           // per-row rescale factor exp(m_old - m_new)
  
  for(int i = tid; i < Br * D; i += blockDim.x){
    sQ[i] = d_Q[qBase + i];
    sO[i] = 0.0f;
  }

  if(tid < Br){
    sm[tid] = -INFINITY;
    sl[tid] = 0.0f;
  }
  __syncthreads();

  // Allocate Tensor Memory
  uint32_t tmem_addr;
  {
    __shared__ uint32_t s_tmem_addr;
    if(tid == 0){
      asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned [%0], %1;"
        : "=r"(tmem_addr)
        : "r"((uint32_t)Bc)
      );
      s_tmem_addr = tmem_addr;
    }
    __syncthreads();
    tmem_addr = s_tmem_addr;
  }

  const uint64_t descQ_base = make_smem_desc(sQ, D * (int)sizeof(__nv_bfloatt16));

  for(int kc = 0; kc < nKVTiles; ++kc){
    const long kBase = kvBase + (long)kc * Bc * D;
    for(int i = tid; i < Bc * D; i += blockDim.x){
      sK[i] = d_K[kBase + i];
      sV[i] = d_V[kBase + i];
    }
    __syncthreads();

    // S = (Q @ K^T) * scale -> sS[Br, Bc]
    {
      const uint64_t descK_base = make_smem_desc(sK, D * (int)sizeof(__nv_bfloat16));
      const uint32_t idesc      = make_idesc_m64_bf16(Bc);
      const uint32_t mask[4]    = {0u, 0u, 0u, 0u};

      if(tid == 0){
        for(int kt = 0; kt < D/16; ++kt){
          uint64_t descQ = advance_desc_col(descQ_base, kt * 16);
          uint64_t descK = advance_desc_col(descK_base, kt * 16);
          uint32_t pred  = (kt > 0) ? 1u : 0u;

          asm volatile(
            "tcgen05.mma.cta_group::1.kind::f16 "
            "[%0], %1, %2, %3, {%4, %5, %6, %7}, %8;"
            :
            : "r"(tmem_addr),
              "l"(descQ), "l"(descK),
              "r"(idesc),
              "r"(mask[0]), "r"(mask[1]),
              "r"(mask[2]), "r"(mask[3]),
              "r"(pred)
            : "memory"
          );
        }
      }
      __syncthreads();

      tmem_readout_to_smem(sS, tmem_addr, Bc, Bc, scale);
      __syncthreads();

      if(tid < Br){
        const float m_old = sm[tid];
        const float l_old = sl[tid];

        float tile_max = -INFINITY;
        for(int j = 0; j < Bc; ++j){
          tile_max = fmaxf(tile_max, sS[tid * Bc +j]);
        }

        const float m_new = fmaxf(m_old, tile_max);
        const float corr  = __expf(m_old - m_new);

        float p_sum = 0.0f;
        for(int j = 0; j < Bc; ++j){
          const float p = __expf(sS[tid * Bc +j] - m_new);
          sP[tid * Bc +j] = __float2bfloat16(p);
          p_sum += p;
        }

        sm[tid]    = m_new;
        sl[tid]    = l_old * corr + p_sum;
        sCorr[tid] = corr;
      }
      __syncthreads();

      for(int i = tid; i < Br * D; i += blockDim.x){
        sO[i] *= sCorr[i/D];
      }
      __syncthreads();

      // P @ V
      {
        // sP: [Br, Bc] row-major, ld = Bc * sizeof(bf16)
        const uint64_t descP_base = make_smem_desc(sP, Bc * (int)sizeof(__nv_bfloat16));

        // sV: [Bc, D] row-major, ld = D * sizeof(bf16)
        const uint64_t descV_base = make_smem_desc(sV, D * (int)sizeof(__nv_bfloat16));

        // Same idesc as QK^T — M=64, N=64, bf16 inputs, fp32 accum
        // N here is D=64, same as Bc=64 so idesc is identical
        const uint32_t idesc   = make_idesc_m64_bf16(D);
        const uint32_t mask[4] = {0u, 0u, 0u, 0u};

        // K reduction loop: Bc/16 = 64/16 = 4 steps
        if(tid == 0){
            for(int kt = 0; kt < Bc / 16; ++kt){
                // Advance P along its K dimension (columns of P)
                // Each step: 16 bf16 elements = 32 bytes along the row
                uint64_t descP = advance_desc_col(descP_base, kt * 16);

                // Advance V along its K dimension (rows of V)
                // Each step: 16 rows × D columns
                // Row stride of V = D * sizeof(bf16) bytes
                // So advancing 16 rows = 16 * D * sizeof(bf16) bytes
                // In descriptor address units: >> 4
                uint64_t descV = advance_desc_row(descV_base, kt * 16, D * (int)sizeof(__nv_bfloat16));

                uint32_t pred = (kt > 0) ? 1u : 0u;

                asm volatile(
                    "tcgen05.mma.cta_group::1.kind::f16 "
                    "[%0], %1, %2, %3, {%4, %5, %6, %7}, %8;"
                    :
                    : "r"(tmem_addr),
                      "l"(descP), "l"(descV),
                      "r"(idesc),
                      "r"(mask[0]), "r"(mask[1]),
                      "r"(mask[2]), "r"(mask[3]),
                      "r"(pred)
                    : "memory"
                );
            }
        }
        __syncthreads();

        // Read tmem → sPV (no scale needed here)
        tmem_readout_to_smem(sPV, tmem_addr, D, D, 1.0f);
        __syncthreads();
      }
      for(int i = tid; i < Br * D; i += blockDim.x)
        sO[i] += sPV[i];
      __syncthreads();
  } // end of kv tile loop

  // Deallocate tensor memory after kv loop
  if(tid == 0){
    asm volatile(
      "tcgen05.dealloc.cta_group::1.sync.aligned [%0], %1;"
      : 
      : "r"(tmem_addr), "r"((uint32_t)Bc)
    );
  }
  __syncthreads();

  for(int i = tid; i < Br * D; i += blockDim.x)
      d_O[qBase + i] = __float2bfloat16(sO[i] / sl[i / D]);

  if(tid < Br)
      d_LSE[lBase + tid] = sm[tid] + logf(sl[tid]);

} // end of v3

//* Kernel Launch Function
// V1
template<int Br, int Bc, int D>
void launch_gqa_v1(
  __nv_bfloat16 *d_Q,
  __nv_bfloat16 *d_K,
  __nv_bfloat16 *d_V,
  __nv_bfloat16 *d_O,
  float *d_LSE,
  int B,
  int Hq,
  int Hkv,
  int G,
  int S,
  float scale
){
  // WMMA does 16x16x16 tiles, so every tiled dimension must be a multiple of 16.
  static_assert(Br % 16 == 0, "Br must be a multiple of 16 (WMMA tile)");
  static_assert(Bc % 16 == 0, "Bc must be a multiple of 16 (WMMA tile)");
  static_assert(D  % 16 == 0, "D  must be a multiple of 16 (WMMA tile)");

  dim3 GRID(B, Hq, S/Br);
  dim3 BLOCK(32);
  gqa_v1<Br, Bc, D><<<GRID, BLOCK>>>(d_Q, d_K, d_V, d_O, d_LSE,
                          B, Hq, Hkv, G, S, scale);
}

// V2
template<int Br, int Bc, int D>
void launch_gqa_v2(
  __nv_bfloat16 *d_Q,
  __nv_bfloat16 *d_K,
  __nv_bfloat16 *d_V,
  __nv_bfloat16 *d_O,
  float *d_LSE,
  int B,
  int Hq,
  int Hkv,
  int G,
  int S,
  float scale
){
  // WMMA does 16x16x16 tiles, so every tiled dimension must be a multiple of 16.
  static_assert(Br % 16 == 0, "Br must be a multiple of 16 (WMMA tile)");
  static_assert(Bc % 16 == 0, "Bc must be a multiple of 16 (WMMA tile)");
  static_assert(D  % 16 == 0, "D  must be a multiple of 16 (WMMA tile)");

  dim3 GRID(B, Hq, S/Br);
  dim3 BLOCK(32);
  gqa_v2<Br, Bc, D><<<GRID, BLOCK>>>(d_Q, d_K, d_V, d_O, d_LSE,
                          B, Hq, Hkv, G, S, scale);
}

// V3
template<int Br, int Bc, int D>
void launch_gqa_v3(
  __nv_bfloat16 *d_Q,
  __nv_bfloat16 *d_K,
  __nv_bfloat16 *d_V,
  __nv_bfloat16 *d_O,
  float *d_LSE,
  int B,
  int Hq,
  int Hkv,
  int G,
  int S,
  float scale
){
  // WMMA does 16x16x16 tiles, so every tiled dimension must be a multiple of 16.
  static_assert(Br % 64 == 0, "Br must be a multiple of 64 for tcgen05 M = 64");
  static_assert(Bc % 8 == 0, "Bc must be a multiple of 8 for tcgen05 N = 8");
  static_assert(D  % 16 == 0, "D  must be a multiple of 16 for tcgen05 dense");

  dim3 GRID(B, Hq, S/Br);
  dim3 BLOCK(128); // 4 warps
  gqa_v3<Br, Bc, D><<<GRID, BLOCK>>>(d_Q, d_K, d_V, d_O, d_LSE,
                          B, Hq, Hkv, G, S, scale);
  cudaError_t err = cuda
}

int main(){
  constexpr int B   = 16;
  constexpr int Hq  = 12;
  constexpr int Hkv = 4;
  constexpr int G   = Hq/Hkv;
  constexpr int S   = 4096;
  constexpr int D   = 64;
  constexpr int Br  = 64;
  constexpr int Bc  = 64;

  static_assert(Hq % Hkv == 0, "Hq must be divisible by Hkv");
  static_assert(S % Br   == 0, "S must be divisible by Br");
  static_assert(S % Bc   == 0, "S must be divisible by Bc");

  const float scale = 1.0f / sqrtf((float)D);

  const size_t Nq   = (size_t)B * Hq * S * D;
  const size_t Nkv  = (size_t)B * Hkv * S * D;
  const size_t Nlse = (size_t)B * Hq * S;

  __nv_bfloat16 *h_Q = new __nv_bfloat16[Nq];
  __nv_bfloat16 *h_K = new __nv_bfloat16[Nkv];
  __nv_bfloat16 *h_V = new __nv_bfloat16[Nkv];
  __nv_bfloat16 *h_O = new __nv_bfloat16[Nq];
  float *h_LSE       = new float[Nlse]{};
   
  initPtr(h_Q, (int)Nq);
  initPtr(h_K, (int)Nkv);
  initPtr(h_V, (int)Nkv);
  
  __nv_bfloat16 *d_Q, *d_K, *d_V, *d_O;
  float *d_LSE;
  CUDA_CHECK(cudaMalloc(&d_Q,   Nq   * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_K,   Nkv  * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_V,   Nkv  * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_O,   Nq   * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_LSE, Nlse * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_Q, h_Q, Nq  * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_K, h_K, Nkv * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_V, h_V, Nkv * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

  launch_gqa_v1<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, G, S, scale);
  
  CUDA_CHECK(cudaMemcpy(h_O, d_O, Nq  * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_LSE, d_LSE, Nlse * sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_Q));
  CUDA_CHECK(cudaFree(d_K));
  CUDA_CHECK(cudaFree(d_V));
  CUDA_CHECK(cudaFree(d_O));
  CUDA_CHECK(cudaFree(d_LSE));
  delete(h_Q);
  delete(h_K);
  delete(h_V);
  delete(h_O);
  delete(h_LSE);

  return 0;
}
