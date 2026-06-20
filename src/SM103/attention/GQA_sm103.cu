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
  const int lane   = threadIdx.x;

  const int q_row0   = q_tile * Br;
  const int nKVTiles = S / Bc;

  const long qBase  = ((long)(b * Hq + hq) * S + q_row0) * D;
  const long kvBase = ((long)(b * Hkv + hkv) * S) * D;
  const long lBase  = ((long)(b * Hq + hq) * S + q_row0);
}

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
  static_assert(Br % 16 == 0, "Br must be a multiple of 16 (WMMA tile)");
  static_assert(Bc % 16 == 0, "Bc must be a multiple of 16 (WMMA tile)");
  static_assert(D  % 16 == 0, "D  must be a multiple of 16 (WMMA tile)");

  dim3 GRID(B, Hq, S/Br);
  dim3 BLOCK(32);
  gqa_v3<Br, Bc, D><<<GRID, BLOCK>>>(d_Q, d_K, d_V, d_O, d_LSE,
                          B, Hq, Hkv, G, S, scale);
}

int main(){
  constexpr int B   = 16;
  constexpr int Hq  = 12;
  constexpr int Hkv = 4;
  constexpr int G   = Hq/Hkv;
  constexpr int S   = 4096;
  constexpr int D   = 64;
  constexpr int Br  = 16;
  constexpr int Bc  = 32;

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
