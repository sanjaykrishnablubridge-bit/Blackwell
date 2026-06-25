#include <cuda_runtime.h>
#include <cuda.h>
#include <cooperative_groups.h>
#include <cuda/barrier>
#include <mma.h>
#include <stdio.h>
#include <cassert>
#include <cmath>
#include <cuda_bf16.h>
#include "utils/kernelUtils.cuh"
#include "utils/kernelBench.cuh"
  
namespace cg = cooperative_groups;
using namespace nvcuda;

// ---------------------------------------------------------------------------
// Helper: scores[Br,Bc] = scale * ( Q[Br,D] @ Kᵀ[D,Bc] )  using bf16 WMMA tiles.
//
// WMMA is a *warp-collective* 16x16x16 matrix multiply: C[16,16] = A[16,16] @ B[16,16].
//   - A = Q,  loaded row-major          -> M=Br rows, K=D contraction
//   - B = Kᵀ, loaded *col-major* from K  -> a col-major view of the [Bc,D] K tile IS Kᵀ
//   - C = scores accumulator (fp32)      -> N=Bc columns
// Br=16 is one M-tile; we sweep Bc/16 N-tiles and contract over D in D/16 K-steps.
// ---------------------------------------------------------------------------
template<int Br, int Bc, int D>
__device__ inline void computeScores(const __nv_bfloat16 *sQ,
                                     const __nv_bfloat16 *sK,
                                     float *sS, float scale)
{
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> qf;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> kf; // col-major => Kᵀ
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;

  for (int nt = 0; nt < Bc / 16; ++nt) {          // sweep N (key) tiles
    wmma::fill_fragment(acc, 0.0f);
    for (int kt = 0; kt < D / 16; ++kt) {         // contract over the head dim
      wmma::load_matrix_sync(qf, sQ + kt * 16,                 D); // Q rows, cols [kt*16 ..)
      wmma::load_matrix_sync(kf, sK + nt * 16 * D + kt * 16,   D); // col-major view of K = Kᵀ
      wmma::mma_sync(acc, qf, kf, acc);
    }
    // apply the softmax scale (1/sqrt(D)) then drop this 16x16 block into sS
    for (int t = 0; t < acc.num_elements; ++t) acc.x[t] *= scale;
    wmma::store_matrix_sync(sS + nt * 16, acc, Bc, wmma::mem_row_major);
  }
}

// ---------------------------------------------------------------------------
// GQA forward, one block = one (batch, query-head, Br-row tile). One warp/block.
//
// Numerically-stable, NON-online softmax (two passes over the keys):
//   Pass 1: stream all keys, find the true per-row max  m[r]      (no exp yet)
//   Pass 2: stream all keys again, accumulate l[r]=Σexp(s-m) and O=Σ exp(s-m)·V
//   Finally: O /= l   and   LSE = m + log(l)
// (The online/flash version fuses these with a running rescale; we keep them
//  separate so the math is easy to follow. Cost: Q·Kᵀ is computed twice.)
// ---------------------------------------------------------------------------
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
  int S,
  int G,
  float scale
){
  // WMMA only does 16x16x16 tiles, so every tiled dimension must be a multiple of 16.
  static_assert(Br % 16 == 0, "Br must be a multiple of 16 (WMMA tile)");
  static_assert(Bc % 16 == 0, "Bc must be a multiple of 16 (WMMA tile)");
  static_assert(D  % 16 == 0, "D  must be a multiple of 16 (WMMA tile)");

  // ---- 1. What does this block own? -------------------------------------
  const int b      = blockIdx.x;   // which batch          (0 .. B-1)
  const int hq     = blockIdx.y;   // which query head     (0 .. Hq-1)
  const int q_tile = blockIdx.z;   // which Br-row tile    (0 .. S/Br-1)
  const int hkv    = hq / G;       // GQA: query head hq shares this kv head
  const int lane   = threadIdx.x;  // 0..31 — single warp per block

  const int q_row0    = q_tile * Br;  // first query row this block computes
  const int nKeyTiles = S / Bc;       // number of Bc-wide key tiles to stream

  // Flat offsets into the [B,H,S,D] (and [B,H,S] for LSE) row-major arrays.
  const long qBase  = ((long)(b * Hq  + hq ) * S + q_row0) * D; // Q / O tile start
  const long kvBase = ((long)(b * Hkv + hkv) * S         ) * D; // K / V head start
  const long lBase  =  (long)(b * Hq  + hq ) * S + q_row0;      // LSE tile start

  // ---- 2. Shared-memory tiles (sizes are compile-time -> static smem) ----
  __shared__ __align__(16) __nv_bfloat16 sQ[Br * D];  // query  tile [Br, D]
  __shared__ __align__(16) __nv_bfloat16 sK[Bc * D];  // key    tile [Bc, D]
  __shared__ __align__(16) __nv_bfloat16 sV[Bc * D];  // value  tile [Bc, D]
  __shared__ __align__(16) float         sS[Br * Bc]; // scores tile [Br, Bc] (fp32)
  __shared__ __align__(16) __nv_bfloat16 sP[Br * Bc]; // weights tile[Br, Bc] (bf16 for WMMA)
  __shared__ __align__(16) float         sO[Br * D];  // O accum     [Br, D]  (fp32)
  __shared__ float sm[Br];                            // per-row running max
  __shared__ float sl[Br];                            // per-row exp-sum (denominator)

  // Load the Q tile once — it is reused for every key tile and both passes.
  for (int i = lane; i < Br * D; i += 32) sQ[i] = d_Q[qBase + i];
  if (lane < Br) sm[lane] = -INFINITY;
  __syncwarp();

  // ===== PASS 1 : true per-row max over ALL keys =========================
  for (int kc = 0; kc < nKeyTiles; ++kc) {
    const long kBase = kvBase + (long)kc * Bc * D;
    for (int i = lane; i < Bc * D; i += 32) sK[i] = d_K[kBase + i];
    __syncwarp();

    computeScores<Br, Bc, D>(sQ, sK, sS, scale);   // sS = scaled Q·Kᵀ for this tile
    __syncwarp();

    if (lane < Br) {                               // one lane per query row
      float mx = sm[lane];
      for (int j = 0; j < Bc; ++j) mx = fmaxf(mx, sS[lane * Bc + j]);
      sm[lane] = mx;
    }
    __syncwarp();
  }

  // ===== PASS 2 : denominator + weighted sum of V ========================
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> oacc[D / 16];
  for (int nt = 0; nt < D / 16; ++nt) wmma::fill_fragment(oacc[nt], 0.0f);
  if (lane < Br) sl[lane] = 0.0f;
  __syncwarp();

  for (int kc = 0; kc < nKeyTiles; ++kc) {
    const long kBase = kvBase + (long)kc * Bc * D;
    for (int i = lane; i < Bc * D; i += 32) {      // load K and V tiles together
      sK[i] = d_K[kBase + i];
      sV[i] = d_V[kBase + i];
    }
    __syncwarp();

    computeScores<Br, Bc, D>(sQ, sK, sS, scale);   // recompute the same scaled scores
    __syncwarp();

    // P = exp(scores - rowmax); also accumulate the running denominator l[r].
    if (lane < Br) {
      const float mrow = sm[lane];
      float s = 0.0f;
      for (int j = 0; j < Bc; ++j) {
        float e = expf(sS[lane * Bc + j] - mrow);  // safe: argument <= 0
        sP[lane * Bc + j] = __float2bfloat16(e);   // cast to bf16 for the P·V WMMA
        s += e;
      }
      sl[lane] += s;
    }
    __syncwarp();

    // O[Br,D] += P[Br,Bc] @ V[Bc,D]   (accumulates across key tiles in oacc)
    //   A = P  (row-major, M=Br, K=Bc) ; B = V (row-major, K=Bc, N=D)
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> pf;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> vf;
    for (int nt = 0; nt < D / 16; ++nt) {          // sweep N (head-dim) tiles
      for (int kt = 0; kt < Bc / 16; ++kt) {       // contract over the key tile
        wmma::load_matrix_sync(pf, sP + kt * 16,               Bc);
        wmma::load_matrix_sync(vf, sV + kt * 16 * D + nt * 16, D);
        wmma::mma_sync(oacc[nt], pf, vf, oacc[nt]);
      }
    }
    __syncwarp();
  }

  // Spill the (still un-normalised) O accumulator to shared memory.
  for (int nt = 0; nt < D / 16; ++nt)
    wmma::store_matrix_sync(sO + nt * 16, oacc[nt], D, wmma::mem_row_major);
  __syncwarp();

  // ===== 3. normalise, write O and LSE ===================================
  if (lane < Br) {
    const float denom = sl[lane];
    const float inv   = 1.0f / denom;
    for (int d = 0; d < D; ++d)
      d_O[qBase + lane * D + d] = __float2bfloat16(sO[lane * D + d] * inv);
    d_LSE[lBase + lane] = sm[lane] + logf(denom);  // log-sum-exp of the logits
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
// Matrix-descriptor field encoding: keep the 18 low bits of the byte value and drop
// the bottom 4 (everything in the descriptor is in 16-byte units).
__device__ __forceinline__ uint64_t desc_encode(uint64_t x){
  return (x & 0x3FFFFull) >> 4;
}

// 64-bit shared-memory matrix descriptor for tcgen05.mma, NON-swizzled layout.
//   bits[13:0]  : matrix start address      (16-byte units)
//   bits[29:16] : leading byte offset (LBO)
//   bits[45:32] : stride  byte offset (SBO)
//   bit  46     : descriptor "valid" flag = 1   (REQUIRED; omitting it faults)
//   bits[63:61] : swizzle mode = 0 (none)
// `lead_rows` is the tile's row count (its leading dimension), used to form the LBO
// the way the canonical non-swizzled bf16 layout expects (LBO = rows*16, SBO = 8*16).
__device__ uint64_t make_smem_desc(void* smem_ptr, int lead_rows){
  uint64_t addr = (uint64_t)__cvta_generic_to_shared(smem_ptr);
  uint64_t LBO  = (uint64_t)lead_rows * 16ull;
  uint64_t SBO  = 8ull * 16ull;
  uint64_t desc = desc_encode(addr)
                | (desc_encode(LBO) << 16)
                | (desc_encode(SBO) << 32)
                | (1ull << 46);                 // valid bit; swizzle bits [63:61] stay 0
  return desc;
}

// 32-bit tcgen05.mma instruction descriptor, kind::f16 (covers bf16), fp32 accumulator.
//   bit  4      : D dtype = fp32 (1)
//   bit  7      : A dtype = bf16 (1)
//   bit  10     : B dtype = bf16 (1)
//   bits[23:17] : MMA_N = N / 8   (8-column units)
//   bits[30:24] : MMA_M = M / 16  (16-row units)
__device__ uint32_t make_idesc_bf16(int M, int N){
  uint32_t idesc = 0;
  idesc |= (1u << 4);                            // D = fp32
  idesc |= (1u << 7);                            // A = bf16
  idesc |= (1u << 10);                           // B = bf16
  idesc |= ((uint32_t)(N >> 3) << 17);           // N in 8-element units
  idesc |= ((uint32_t)(M >> 4) << 24);           // M in 16-element units
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
  return (desc & ~0x3FFFull) | new_addr;
}

// Read an [M, N] fp32 accumulator out of tensor memory into row-major shared memory.
//
// tcgen05 TMEM access is *per-warp*: warp w owns tmem lanes [w*32, w*32+32) and nothing
// else. The TMEM address packs the lane band in bits [31:16] and the column in bits
// [15:0]:  addr = base + (row_base << 16) + col.  The .32x32b.x1 shape has every lane of
// the warp read one 32-bit column-word from its own lane (row), so M must be a multiple
// of 32 and we need exactly M/32 warps; any extra warps in the block sit this out.
__device__ void tmem_readout_to_smem(
  float* smem_out,
  uint32_t tmem_addr,
  int M,            // number of valid accumulator rows (= Br)
  int N,            // number of columns to read back
  int smem_stride,
  float scale
){
  const int warp_id = threadIdx.x / 32;
  const int lane    = threadIdx.x % 32;

  const uint32_t row_base = (uint32_t)warp_id * 32u;   // first tmem lane this warp owns
  if ((int)row_base >= M) return;                      // only M/32 warps hold valid rows
  const int row = (int)row_base + lane;                // tmem lane this thread reads back

  for(int col = 0; col < N; ++col){
    uint32_t raw;
    asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];"
      : "=r"(raw)
      : "r"(tmem_addr + (row_base << 16) + (uint32_t)col)
      : "memory"
    );
    // tcgen05.ld is async; the destination register is only valid after wait::ld.
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    smem_out[row * smem_stride + col] = reinterpret_cast<float&>(raw) * scale;
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

  // tcgen05 needs enough TMEM columns for the largest accumulator we read back:
  // QK^T uses N = Bc columns, P@V uses N = D columns. Allocate the max
  // (the column count must be a power of two and >= 32).
  constexpr uint32_t NCOLS = (Bc > D) ? (uint32_t)Bc : (uint32_t)D;
  static_assert(NCOLS >= 32 && (NCOLS & (NCOLS - 1)) == 0,
                "tcgen05 column count must be a power of two >= 32");

  // Allocate Tensor Memory
  uint32_t tmem_addr;
  {
    __shared__ uint32_t s_tmem_addr;
    // tcgen05.alloc / relinquish are .sync.aligned WARP-COLLECTIVE ops: all 32 lanes of
    // the warp must execute them together. Running them under `tid == 0` masks 31 lanes
    // and the in-instruction warp barrier deadlocks. Use the whole first warp.
    if(tid < 32){
      // tcgen05.alloc writes the TMEM base address into a *shared* location ([%0]),
      // it is NOT a register result — pass the shared-state-space address of s_tmem_addr.
      uint32_t s_addr = (uint32_t)__cvta_generic_to_shared(&s_tmem_addr);
      asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(s_addr), "r"(NCOLS) : "memory"
      );
      // We allocate exactly once per block, so release the alloc permit now.
      asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
    __syncthreads();
    tmem_addr = s_tmem_addr;
  }

  const uint64_t descQ_base = make_smem_desc(sQ, Br);   // sQ is [Br, D] row-major

  for(int kc = 0; kc < nKVTiles; ++kc){
    const long kBase = kvBase + (long)kc * Bc * D;
    for(int i = tid; i < Bc * D; i += blockDim.x){
      sK[i] = d_K[kBase + i];
      sV[i] = d_V[kBase + i];
    }
    __syncthreads();

    // S = (Q @ K^T) * scale -> sS[Br, Bc]
    {
      const uint64_t descK_base = make_smem_desc(sK, Bc);   // sK is [Bc, D] row-major
      const uint32_t idesc      = make_idesc_bf16(Br, Bc);  // M = Br, N = Bc

      if(tid == 0){
        for(int kt = 0; kt < D/16; ++kt){
          uint64_t descQ = advance_desc_col(descQ_base, kt * 16);
          uint64_t descK = advance_desc_col(descK_base, kt * 16);
          // enable-input-d (accumulate): false on the first K-step so it overwrites
          // the accumulator, true afterwards so it accumulates A·B + D.
          uint32_t accumulate = (kt > 0) ? 1u : 0u;

          asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t"
            "}\n"
            :
            : "r"(tmem_addr), "l"(descQ), "l"(descK), "r"(idesc), "r"(accumulate)
            : "memory"
          );
        }
      }
      __syncthreads();

      tmem_readout_to_smem(sS, tmem_addr, Br, Bc, Bc, scale);
      __syncthreads();
    } // end QK^T scope

    // ---- online-softmax stats update + unnormalized P -----------------
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
        // sP: [Br, Bc] row-major
        const uint64_t descP_base = make_smem_desc(sP, Br);

        // sV: [Bc, D] row-major
        const uint64_t descV_base = make_smem_desc(sV, Bc);

        // P@V accumulator is M = Br, N = D (bf16 inputs, fp32 accum).
        const uint32_t idesc   = make_idesc_bf16(Br, D);

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

                uint32_t accumulate = (kt > 0) ? 1u : 0u;

                asm volatile(
                    "{\n\t"
                    ".reg .pred p;\n\t"
                    "setp.ne.b32 p, %4, 0;\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t"
                    "}\n"
                    :
                    : "r"(tmem_addr), "l"(descP), "l"(descV), "r"(idesc), "r"(accumulate)
                    : "memory"
                );
            }
        }
        __syncthreads();

        // Read tmem → sPV (no scale needed here)
        tmem_readout_to_smem(sPV, tmem_addr, Br, D, D, 1.0f);
        __syncthreads();
      }
      for(int i = tid; i < Br * D; i += blockDim.x)
        sO[i] += sPV[i];
      __syncthreads();
  } // end of kv tile loop

  // Deallocate tensor memory after kv loop (address is a plain .b32 register, not [addr]).
  // dealloc is .sync.aligned warp-collective — the whole first warp must execute it.
  if(tid < 32){
    asm volatile(
      "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
      :
      : "r"(tmem_addr), "r"(NCOLS)
      : "memory"
    );
  }
  __syncthreads();

  for(int i = tid; i < Br * D; i += blockDim.x)
      d_O[qBase + i] = __float2bfloat16(sO[i] / sl[i / D]);

  if(tid < Br)
      d_LSE[lBase + tid] = sm[tid] + logf(sl[tid]);

} // end of v3

//* ============================
//* Kernel Launcher
//* ============================
//* Tiled mapping: one block per (batch, query-head, query-row-tile).
//*   blockIdx.x = b       (0 .. B-1)
//*   blockIdx.y = hq      (0 .. Hq-1)   -> kv head = hq / G
//*   blockIdx.z = q_tile  (0 .. S/Br-1) -> query rows [q_tile*Br : q_tile*Br + Br)
//* One warp per block because WMMA is a warp-collective op. Tiles live in static
//* __shared__ memory (sizes are compile-time via Br/Bc/D), so no dynamic smem.
template<int Br, int Bc, int D>
void launch_gqa_v1(
  __nv_bfloat16 *d_Q, __nv_bfloat16 *d_K, __nv_bfloat16 *d_V,
  __nv_bfloat16 *d_O, float *d_LSE,
  int B, int Hq, int Hkv, int S, int G, float scale
){
  dim3 GRID(B, Hq, S / Br);   // (16, 12, 256) = 49,152 blocks
  dim3 BLOCK(32);             // ONE warp per block
  gqa_v1<Br, Bc, D><<<GRID, BLOCK>>>(d_Q, d_K, d_V, d_O, d_LSE,
                                     B, Hq, Hkv, S, G, scale);
}

// V2 — same signature as launch_gqa_v1 so callers are interchangeable.
template<int Br, int Bc, int D>
void launch_gqa_v2(
  __nv_bfloat16 *d_Q, __nv_bfloat16 *d_K, __nv_bfloat16 *d_V,
  __nv_bfloat16 *d_O, float *d_LSE,
  int B, int Hq, int Hkv, int S, int G, float scale
){
  // WMMA does 16x16x16 tiles, so every tiled dimension must be a multiple of 16.
  static_assert(Br % 16 == 0, "Br must be a multiple of 16 (WMMA tile)");
  static_assert(Bc % 16 == 0, "Bc must be a multiple of 16 (WMMA tile)");
  static_assert(D  % 16 == 0, "D  must be a multiple of 16 (WMMA tile)");

  dim3 GRID(B, Hq, S / Br);
  dim3 BLOCK(32);                // ONE warp per block (WMMA is warp-collective)
  // NOTE: gqa_v2's kernel param order is (Hkv, G, S), so pass G before S here.
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
  int S,
  int G,
  float scale
){
  // tcgen05 MMA is M=64 / N (multiple of 8) / K=16 per step.
  static_assert(Br % 64 == 0, "Br must be a multiple of 64 for tcgen05 M = 64");
  static_assert(Bc % 8 == 0, "Bc must be a multiple of 8 for tcgen05 N = 8");
  static_assert(D  % 16 == 0, "D  must be a multiple of 16 for tcgen05 dense");

  dim3 GRID(B, Hq, S/Br);
  dim3 BLOCK(128); // 4 warps
  // NOTE: gqa_v3's kernel param order is (Hkv, G, S), so forward G before S.
  gqa_v3<Br, Bc, D><<<GRID, BLOCK>>>(d_Q, d_K, d_V, d_O, d_LSE,
                          B, Hq, Hkv, G, S, scale);
}


int main(){
  std::cout << "Benchmarking Grouped-Query Attention kernels — Blackwell SM_120\n";

  constexpr int B   = 16;     //! batch - later try 32
  constexpr int Hq  = 12;     // number of query heads
  constexpr int Hkv = 4;      // number of key/value heads
  constexpr int G   = Hq/Hkv; // groups per KV head
  constexpr int S   = 4096;   // context length
  constexpr int D   = 64;     // head dimension
  constexpr int Br  = 16;     // tile size along the query sequence dimension (v1/v2, one warp)
  constexpr int Bc  = 32;     // tile size along the key/value sequence dimension
  constexpr int Br_v3 = 64;   // v3 (tcgen05) requires M = 64 → Br = 64, 4 warps/block

  static_assert(Hq % Hkv == 0, "Hq must be divisible by Hkv");
  static_assert(S  % Br    == 0, "S must be divisible by Br");
  static_assert(S  % Br_v3 == 0, "S must be divisible by Br_v3");
  static_assert(S  % Bc    == 0, "S must be divisible by Bc");

  const float scale = 1.0f / sqrtf((float)D);

  const size_t Nq   = (size_t)B * Hq  * S * D;  // Q and O   [B, Hq,  S, D]
  const size_t Nkv  = (size_t)B * Hkv * S * D;  // K and V   [B, Hkv, S, D]
  const size_t Nlse = (size_t)B * Hq  * S;      // LSE       [B, Hq,  S]

  // Host buffers: bf16 inputs/output + fp32 LSE, plus fp32 reference targets.
  std::vector<__nv_bfloat16> h_Q(Nq), h_K(Nkv), h_V(Nkv), h_O(Nq);
  std::vector<float>         h_LSE(Nlse);
  std::vector<float>         h_O_ref(Nq), h_LSE_ref(Nlse);

  //* ── Load PyTorch reference data (falls back to random + benchmark-only) ──
  auto fileMatchesSize = [](const std::string &p, size_t n_floats) -> bool {
    FILE *f = fopen(p.c_str(), "rb");
    if(!f) return false;
    fseek(f, 0, SEEK_END);
    size_t bytes = (size_t)ftell(f);
    fclose(f);
    return bytes == n_floats * sizeof(float);
  };
  //* the .bin files are float32; narrow to bf16 so the kernel sees the same bits
  //* PyTorch rounded to when it generated the reference.
  auto loadBinBF16 = [](const char *path, std::vector<__nv_bfloat16> &dst, size_t n){
    std::vector<float> tmp(n);
    loadBin(path, tmp.data(), n);
    for(size_t i = 0; i < n; ++i) dst[i] = __float2bfloat16(tmp[i]);
  };

  bool has_ref = fileMatchesSize("data/gqa_q.bin", Nq);
  if(has_ref){
    loadBinBF16("data/gqa_q.bin", h_Q, Nq);
    loadBinBF16("data/gqa_k.bin", h_K, Nkv);
    loadBinBF16("data/gqa_v.bin", h_V, Nkv);
    loadBin("data/gqa_o.bin",   h_O_ref.data(),   Nq);
    loadBin("data/gqa_lse.bin", h_LSE_ref.data(), Nlse);
    std::cout << "\nLoaded PyTorch reference from data/gqa_*.bin\n";
  } else {
    initPtr(h_Q.data(), (int)Nq);
    initPtr(h_K.data(), (int)Nkv);
    initPtr(h_V.data(), (int)Nkv);
    std::cout << "\nNo reference files found — using random data (benchmarks only)\n";
  }

  __nv_bfloat16 *d_Q, *d_K, *d_V, *d_O;
  float *d_LSE;
  CUDA_CHECK(cudaMalloc(&d_Q,   Nq   * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_K,   Nkv  * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_V,   Nkv  * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_O,   Nq   * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_LSE, Nlse * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), Nq  * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), Nkv * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), Nkv * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

  //* ── Correctness check (bf16 → loose tolerance vs PyTorch bf16 SDPA) ──
  if(has_ref){
    launch_gqa_v1<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_O.data(),   d_O,   Nq   * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_LSE.data(), d_LSE, Nlse * sizeof(float),         cudaMemcpyDeviceToHost));

    // widen the bf16 output to fp32 so checkResult / reportPrecision can compare it
    std::vector<float> h_O_f32(Nq);
    for(size_t i = 0; i < Nq; ++i) h_O_f32[i] = __bfloat162float(h_O[i]);

    // bf16 attention: ~2^-8 relative precision, so use bf16-scale tolerances.
    std::cout << "\nCorrectness V1 (WMMA two-pass vs PyTorch bf16 SDPA):\n";
    reportPrecision("  output O ", h_O_ref.data(),   h_O_f32.data(), Nq);
    reportPrecision("  lse      ", h_LSE_ref.data(), h_LSE.data(),   Nlse);
    std::cout << "  O   : "; checkResult(h_O_ref.data(),   h_O_f32.data(), Nq,   2e-2f, 2e-2f);
    std::cout << "  LSE : "; checkResult(h_LSE_ref.data(), h_LSE.data(),   Nlse, 2e-2f, 2e-2f);

    // ── V2 : WMMA online softmax — same bf16 in / bf16 out comparison path ──
    launch_gqa_v2<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_O.data(),   d_O,   Nq   * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_LSE.data(), d_LSE, Nlse * sizeof(float),         cudaMemcpyDeviceToHost));

    // widen the bf16 output to fp32 so checkResult / reportPrecision can compare it
    for(size_t i = 0; i < Nq; ++i) h_O_f32[i] = __bfloat162float(h_O[i]);

    std::cout << "\nCorrectness V2 (WMMA online softmax vs PyTorch bf16 SDPA):\n";
    reportPrecision("  output O ", h_O_ref.data(),   h_O_f32.data(), Nq);
    reportPrecision("  lse      ", h_LSE_ref.data(), h_LSE.data(),   Nlse);
    std::cout << "  O   : "; checkResult(h_O_ref.data(),   h_O_f32.data(), Nq,   2e-2f, 2e-2f);
    std::cout << "  LSE : "; checkResult(h_LSE_ref.data(), h_LSE.data(),   Nlse, 2e-2f, 2e-2f);
  }

  //* ── Benchmark ──────────────────────────────────────────────────────────
  //* Attention FLOPs (algorithmic): 4 * B * Hq * S * S * D  (QKᵀ + P·V, ×2 for MAC).
  //* NOTE: the two-pass kernel computes QKᵀ twice, so its real FLOPs are higher;
  //* we report the standard algorithmic count so it is comparable to PyTorch.
  long long flops = 4LL * B * Hq * (long long)S * S * D;
  //* Algorithmic-minimum traffic: read Q,K,V once + write O,LSE. The tiled kernel
  //* actually re-reads K/V many times, so this GB/s is an ideal, not effective.
  size_t bytes = (2 * Nq + 2 * Nkv) * sizeof(__nv_bfloat16) + Nlse * sizeof(float);

  KernelStats stats_v1 = benchmarkKernel(
    [&](){ launch_gqa_v1<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); },
    100, 25, flops, bytes
  );
  displayStats("V1 — WMMA two-pass (stable softmax, bf16)", stats_v1);

  KernelStats stats_v2 = benchmarkKernel(
    [&](){ launch_gqa_v2<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); },
    100, 25, flops, bytes
  );
  displayStats("V2 — WMMA online softmax (single-pass, bf16)", stats_v2);

  KernelStats stats_v3 = benchmarkKernel(
    [&](){ launch_gqa_v3<Br_v3, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); },
    100, 25, flops, bytes
  );
  displayStats("V3 — tcgen05 online softmax (single-pass, bf16)", stats_v3);

  CUDA_CHECK(cudaFree(d_Q));
  CUDA_CHECK(cudaFree(d_K));
  CUDA_CHECK(cudaFree(d_V));
  CUDA_CHECK(cudaFree(d_O));
  CUDA_CHECK(cudaFree(d_LSE));

  return 0;
}
