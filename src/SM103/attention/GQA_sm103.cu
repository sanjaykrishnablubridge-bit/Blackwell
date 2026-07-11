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

// =================================
//  V0 : wmma + two-pass stable softmax
// =================================
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
__global__ void gqa_v0(
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
} // end of v0


// =================================
//  V1 : wmma + online softmax
// =================================
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

} // end of the v1 kernel

// =================================
//  V2 : mma + online softmax
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

// ---------------------------------------------------------------------------
// tcgen05 non-swizzled operands are NOT plain row-major. make_smem_desc()'s
// LBO=rows*16 / SBO=128 describe the *canonical K-major atom layout*:
//     canon_idx(row, k, rows) = (k/8)*rows*8 + row*8 + (k%8)
// i.e. K is split into 8-wide atoms, each atom stored [rows, 8] row-major, atoms
// concatenated along K. (LBO=rows*16 = bytes between 8-col atoms; SBO=128 = 8 rows
// inside an atom.) All operands fed to tcgen05.mma must be stored this way.
__device__ __forceinline__ int canon_idx(int row, int k, int rows){
  return (k / 8) * rows * 8 + row * 8 + (k % 8);
}

// Advance a descriptor by `katom` MMA-K steps (MMA_K = 16 = two 8-col atoms).
// In the canonical layout one K-atom of 16 spans rows*16 elements = rows*32 bytes,
// so the 16-byte-unit address field advances by katom*rows*2.
__device__ uint64_t advance_desc_katom(uint64_t desc, int katom, int rows){
  uint64_t units    = (uint64_t)katom * (uint64_t)rows * 2ull;
  uint64_t base_addr = desc & 0x3FFFull;
  uint64_t new_addr  = (base_addr + units) & 0x3FFFull;
  return (desc & ~0x3FFFull) | new_addr;
}

// ---------------------------------------------------------------------------
// tcgen05.mma is ASYNC: after issuing it we must wait for completion (via an
// mbarrier that tcgen05.commit arrives on) before reading the accumulator out of
// TMEM. These mirror the reference (learn-cuda 02e_matmul_sm100/common.h).
__device__ __forceinline__ void mbar_init(uint32_t bar, int count){
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(count));
}
// One-shot arrival fired when the preceding tcgen05.mma group completes.
__device__ __forceinline__ void mbar_commit_mma(uint32_t bar){
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
               :: "r"(bar) : "memory");
}
// Spin until the barrier flips out of `phase` (i.e. the MMA has finished).
// __noinline__ so the fixed asm labels are emitted exactly once (called from 2 sites).
__noinline__ __device__ void mbar_wait(uint32_t bar, int phase){
  uint32_t ticks = 0x989680;
  asm volatile(
    "{\n\t.reg .pred P1;\n\t"
    "LAB_WAIT:\n\t"
    "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
    "@P1 bra.uni DONE;\n\t"
    "bra.uni LAB_WAIT;\n\t"
    "DONE:\n\t}"
    :: "r"(bar), "r"(phase), "r"(ticks)
  );
}

// Read an [M, N] fp32 accumulator out of tensor memory into row-major shared memory.
//
// The tcgen05 fp32 accumulator ALWAYS spreads across all 4 TMEM sub-partitions, with
// M/4 rows in each. Sub-partition w is the TMEM lane band [w*32, w*32+32); its lanes
// 0..M/4-1 hold accumulator rows [w*(M/4), w*(M/4)+M/4). So the read needs 4 warps
// (blockDim.x >= 128), and each warp reads rows_per_warp = M/4 rows — NOT 32. (For the
// reference GEMM's M=128 this happens to be 32/warp; for M=64 it is 16/warp, which is
// why a warp*32 mapping corrupts everything past row 15.)
//
// The TMEM address packs the lane band in bits [31:16] and the column in bits [15:0]:
// addr = base + (lane_base << 16) + col. All 32 lanes must issue the .sync ld together;
// only lanes < M/4 carry a valid row, so only they write to smem.
__device__ void tmem_readout_to_smem(
  float* smem_out,
  uint32_t tmem_addr,
  int M,            // number of valid accumulator rows (= Br)
  int N,            // number of columns to read back
  int smem_stride,
  float scale
){
  // Ordering fence required between the MMA-completion wait and tcgen05.ld.
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

  const int warp_id = threadIdx.x / 32;
  const int lane    = threadIdx.x % 32;

  const int rows_per_warp = M / 4;                     // 4 sub-partitions hold M/4 rows each
  const uint32_t lane_base = (uint32_t)warp_id * 32u;  // TMEM lane band for this sub-partition
  const int row = warp_id * rows_per_warp + lane;      // valid only for lane < rows_per_warp

  for(int col = 0; col < N; ++col){
    uint32_t raw;
    asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];"
      : "=r"(raw)
      : "r"(tmem_addr + (lane_base << 16) + (uint32_t)col)
      : "memory"
    );
    // tcgen05.ld is async; the destination register is only valid after wait::ld.
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    if(lane < rows_per_warp)
      smem_out[row * smem_stride + col] = reinterpret_cast<float&>(raw) * scale;
  }
}

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
  __shared__ __align__(8) uint64_t s_mma_bar;           // mbarrier: signals tcgen05.mma completion

  // sQ is the QK^T A-operand [Br, D] (K=D). Store in canonical K-major atom layout.
  // sO is a plain row-major fp32 accumulator, so it keeps a linear index.
  for(int i = tid; i < Br * D; i += blockDim.x){
    const int r = i / D, c = i % D;
    sQ[canon_idx(r, c, Br)] = d_Q[qBase + i];
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

  // mbarrier used to wait for each async tcgen05.mma group. One arrival per commit.
  const uint32_t mma_bar = (uint32_t)__cvta_generic_to_shared(&s_mma_bar);
  if(tid == 0) mbar_init(mma_bar, 1);
  __syncthreads();
  int mbar_phase = 0;   // flips after every MMA we wait on

  const uint64_t descQ_base = make_smem_desc(sQ, Br);   // sQ is [Br, D] row-major

  for(int kc = 0; kc < nKVTiles; ++kc){
    const long kBase = kvBase + (long)kc * Bc * D;
    for(int i = tid; i < Bc * D; i += blockDim.x){
      const int bc = i / D, d = i % D;
      // K is QK^T's B-operand [N=Bc, K=D] (K-major): canonical over (row=bc, k=d).
      sK[canon_idx(bc, d, Bc)] = d_K[kBase + i];
      // V is P@V's B-operand. tcgen05 computes A @ B^T with B K-major, so B must be
      // V^T = [N=D, K=Bc]. Transpose on load: element V[bc,d] -> canon(row=d, k=bc, rows=D).
      sV[canon_idx(d, bc, D)] = d_V[kBase + i];
    }
    __syncthreads();

    // S = (Q @ K^T) * scale -> sS[Br, Bc]
    {
      const uint64_t descK_base = make_smem_desc(sK, Bc);   // sK is [Bc, D] row-major
      const uint32_t idesc      = make_idesc_bf16(Br, Bc);  // M = Br, N = Bc

      if(tid == 0){
        // Order this MMA's TMEM write after the barrier (WAR vs the prior readout).
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        for(int kt = 0; kt < D/16; ++kt){
          // Advance along K=D in the canonical layout: kt*rows*32 bytes per K-atom.
          uint64_t descQ = advance_desc_katom(descQ_base, kt, Br);
          uint64_t descK = advance_desc_katom(descK_base, kt, Bc);
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
        mbar_commit_mma(mma_bar);        // arrive when this MMA group finishes
      }
      // ALL threads wait for the async MMA to complete before reading TMEM.
      mbar_wait(mma_bar, mbar_phase);
      mbar_phase ^= 1;

      tmem_readout_to_smem(sS, tmem_addr, Br, Bc, Bc, scale);
      // Order these TMEM reads before the barrier, so the next MMA that reuses this
      // TMEM region cannot overwrite it while reads are still in flight (WAR hazard).
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
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
          // sP is P@V's A-operand [Br, Bc] (K=Bc): store canonical over (row=tid, k=j).
          sP[canon_idx(tid, j, Br)] = __float2bfloat16(p);
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
        // sP: A-operand [M=Br, K=Bc], canonical layout -> rows = Br.
        const uint64_t descP_base = make_smem_desc(sP, Br);

        // sV: B-operand stored transposed as [N=D, K=Bc], canonical layout -> rows = D.
        const uint64_t descV_base = make_smem_desc(sV, D);

        // P@V accumulator is M = Br, N = D (bf16 inputs, fp32 accum).
        const uint32_t idesc   = make_idesc_bf16(Br, D);

        // K reduction loop over the key tile: Bc/16 steps.
        if(tid == 0){
            // Order this MMA's TMEM write after the barrier (WAR vs the prior readout).
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            for(int kt = 0; kt < Bc / 16; ++kt){
                // Advance both operands along K=Bc in the canonical layout.
                uint64_t descP = advance_desc_katom(descP_base, kt, Br);
                uint64_t descV = advance_desc_katom(descV_base, kt, D);

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
            mbar_commit_mma(mma_bar);    // arrive when this MMA group finishes
        }
        // ALL threads wait for the async MMA to complete before reading TMEM.
        mbar_wait(mma_bar, mbar_phase);
        mbar_phase ^= 1;

        // Read tmem → sPV (no scale needed here)
        tmem_readout_to_smem(sPV, tmem_addr, Br, D, D, 1.0f);
        // Order these TMEM reads before the barrier (WAR hazard vs the next tile's MMA).
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
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

} // end of v2

// =================================
//  V3 : V2 + cp.async double-buffered KV loads (software pipeline)
// =================================
// Same tcgen05 core as V2; only the load path changes. Each KV tile kc+1 is
// prefetched with cp.async while the tensor cores work on tile kc. This matters
// because V2/V3 run 1 CTA/SM (TMEM), so there is no second block to hide global
// latency — software prefetch is the only lever.
//
// cp.async copies contiguous global -> contiguous shared. K stays K-major so its
// canonical [K/8,rows,8] layout is contiguous in 8-element (16-byte) chunks -> direct
// cp.async. V must be transposed (a scatter), which cp.async can't do, so V is
// cp.async'd contiguously into a plain [Bc,D] staging buffer and transposed in shared.
__device__ __forceinline__ void cp_async_16(uint32_t smem_addr, const void* gmem_ptr){
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gmem_ptr) : "memory");
}
__device__ __forceinline__ void cp_async_commit(){ asm volatile("cp.async.commit_group;\n" ::: "memory"); }
// Wait until at most 1 cp.async group is still in flight (keeps the next tile prefetching).
__device__ __forceinline__ void cp_async_wait_keep1(){ asm volatile("cp.async.wait_group 1;\n" ::: "memory"); }

// Prefetch one KV tile: K -> canonical K-major sK buffer; V -> plain [Bc,D] staging.
template<int Bc, int D>
__device__ __forceinline__ void prefetch_kv_tile(
    __nv_bfloat16* sK_dst, __nv_bfloat16* sVstage_dst,
    const __nv_bfloat16* d_K, const __nv_bfloat16* d_V, long kBase){
  const int tid     = threadIdx.x;
  const int nChunks = Bc * D / 8;          // 16-byte (8 bf16) chunks per tile
  for(int ch = tid; ch < nChunks; ch += blockDim.x){
    const int bc = ch / (D / 8);           // which row
    const int c0 = (ch % (D / 8)) * 8;     // 8-wide K-atom start (contiguous in canonical)
    cp_async_16((uint32_t)__cvta_generic_to_shared(&sK_dst[canon_idx(bc, c0, Bc)]),
                &d_K[kBase + (long)bc * D + c0]);
  }
  for(int i16 = tid; i16 < nChunks; i16 += blockDim.x){   // plain contiguous copy
    cp_async_16((uint32_t)__cvta_generic_to_shared(&sVstage_dst[i16 * 8]),
                &d_V[kBase + (long)i16 * 8]);
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
  const int tid    = threadIdx.x;

  const int q_row0   = q_tile * Br;
  const int nKVTiles = S / Bc;

  const long qBase  = ((long)(b * Hq + hq) * S + q_row0) * D;
  const long kvBase = ((long)(b * Hkv + hkv) * S) * D;
  const long lBase  = ((long)(b * Hq + hq) * S + q_row0);

  __shared__ __align__(16) __nv_bfloat16 sQ[Br * D];
  __shared__ __align__(16) __nv_bfloat16 sK[2 * Bc * D];      // double-buffered, canonical K-major
  __shared__ __align__(16) __nv_bfloat16 sVstage[2 * Bc * D]; // double-buffered, plain [Bc,D] staging
  __shared__ __align__(16) __nv_bfloat16 sV[Bc * D];          // canonical [D,Bc] (transposed), single
  __shared__ __align__(16) __nv_bfloat16 sP[Br * Bc];
  __shared__ __align__(16) float         sS[Br * Bc];
  __shared__ __align__(16) float         sO[Br * D];
  __shared__ __align__(16) float         sPV[Br * D];
  __shared__ float sm[Br];
  __shared__ float sl[Br];
  __shared__ float sCorr[Br];
  __shared__ __align__(8) uint64_t s_mma_bar;

  for(int i = tid; i < Br * D; i += blockDim.x){
    const int r = i / D, c = i % D;
    sQ[canon_idx(r, c, Br)] = d_Q[qBase + i];   // Q loaded once (blocking), canonical K-major
    sO[i] = 0.0f;
  }
  if(tid < Br){ sm[tid] = -INFINITY; sl[tid] = 0.0f; }
  __syncthreads();

  constexpr uint32_t NCOLS = (Bc > D) ? (uint32_t)Bc : (uint32_t)D;
  static_assert(NCOLS >= 32 && (NCOLS & (NCOLS - 1)) == 0,
                "tcgen05 column count must be a power of two >= 32");

  uint32_t tmem_addr;
  {
    __shared__ uint32_t s_tmem_addr;
    if(tid < 32){
      uint32_t s_addr = (uint32_t)__cvta_generic_to_shared(&s_tmem_addr);
      asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                   :: "r"(s_addr), "r"(NCOLS) : "memory");
      asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
    __syncthreads();
    tmem_addr = s_tmem_addr;
  }

  const uint32_t mma_bar = (uint32_t)__cvta_generic_to_shared(&s_mma_bar);
  if(tid == 0) mbar_init(mma_bar, 1);
  __syncthreads();
  int mbar_phase = 0;

  const uint64_t descQ_base = make_smem_desc(sQ, Br);

  // Prologue: prefetch tile 0.
  prefetch_kv_tile<Bc, D>(sK, sVstage, d_K, d_V, kvBase);
  cp_async_commit();

  for(int kc = 0; kc < nKVTiles; ++kc){
    const int cur = kc & 1;
    __nv_bfloat16* sKcur  = sK      + cur * Bc * D;
    __nv_bfloat16* sVscur = sVstage + cur * Bc * D;

    // Prefetch next tile (empty commit on the last iter keeps the wait arithmetic uniform).
    if(kc + 1 < nKVTiles){
      const int nxt = (kc + 1) & 1;
      prefetch_kv_tile<Bc, D>(sK + nxt * Bc * D, sVstage + nxt * Bc * D,
                              d_K, d_V, kvBase + (long)(kc + 1) * Bc * D);
    }
    cp_async_commit();
    cp_async_wait_keep1();   // current tile's copies are now complete
    __syncthreads();

    // S = (Q @ K^T) * scale -> sS[Br, Bc]
    {
      const uint64_t descK_base = make_smem_desc(sKcur, Bc);
      const uint32_t idesc      = make_idesc_bf16(Br, Bc);
      if(tid == 0){
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        for(int kt = 0; kt < D/16; ++kt){
          uint64_t descQ = advance_desc_katom(descQ_base, kt, Br);
          uint64_t descK = advance_desc_katom(descK_base, kt, Bc);
          uint32_t accumulate = (kt > 0) ? 1u : 0u;
          asm volatile(
            "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
            :: "r"(tmem_addr), "l"(descQ), "l"(descK), "r"(idesc), "r"(accumulate) : "memory");
        }
        mbar_commit_mma(mma_bar);
      }
      mbar_wait(mma_bar, mbar_phase); mbar_phase ^= 1;
      tmem_readout_to_smem(sS, tmem_addr, Br, Bc, Bc, scale);
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
      __syncthreads();
    }

    // online-softmax stats + unnormalized P (canonical)
    if(tid < Br){
      const float m_old = sm[tid];
      const float l_old = sl[tid];
      float tile_max = -INFINITY;
      for(int j = 0; j < Bc; ++j) tile_max = fmaxf(tile_max, sS[tid * Bc + j]);
      const float m_new = fmaxf(m_old, tile_max);
      const float corr  = __expf(m_old - m_new);
      float p_sum = 0.0f;
      for(int j = 0; j < Bc; ++j){
        const float p = __expf(sS[tid * Bc + j] - m_new);
        sP[canon_idx(tid, j, Br)] = __float2bfloat16(p);
        p_sum += p;
      }
      sm[tid] = m_new; sl[tid] = l_old * corr + p_sum; sCorr[tid] = corr;
    }
    __syncthreads();

    for(int i = tid; i < Br * D; i += blockDim.x) sO[i] *= sCorr[i / D];
    __syncthreads();

    // In-shared transpose: plain [Bc,D] staging -> canonical [D,Bc] sV (no global latency).
    for(int i = tid; i < Bc * D; i += blockDim.x){
      const int bc = i / D, d = i % D;
      sV[canon_idx(d, bc, D)] = sVscur[i];
    }
    __syncthreads();

    // O_tile = P @ V -> sPV[Br, D]
    {
      const uint64_t descP_base = make_smem_desc(sP, Br);
      const uint64_t descV_base = make_smem_desc(sV, D);
      const uint32_t idesc      = make_idesc_bf16(Br, D);
      if(tid == 0){
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        for(int kt = 0; kt < Bc/16; ++kt){
          uint64_t descP = advance_desc_katom(descP_base, kt, Br);
          uint64_t descV = advance_desc_katom(descV_base, kt, D);
          uint32_t accumulate = (kt > 0) ? 1u : 0u;
          asm volatile(
            "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
            :: "r"(tmem_addr), "l"(descP), "l"(descV), "r"(idesc), "r"(accumulate) : "memory");
        }
        mbar_commit_mma(mma_bar);
      }
      mbar_wait(mma_bar, mbar_phase); mbar_phase ^= 1;
      tmem_readout_to_smem(sPV, tmem_addr, Br, D, D, 1.0f);
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
      __syncthreads();
    }

    for(int i = tid; i < Br * D; i += blockDim.x) sO[i] += sPV[i];
    __syncthreads();
  } // end kv loop

  if(tid < 32)
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(tmem_addr), "r"(NCOLS) : "memory");
  __syncthreads();

  for(int i = tid; i < Br * D; i += blockDim.x)
    d_O[qBase + i] = __float2bfloat16(sO[i] / sl[i / D]);
  if(tid < Br)
    d_LSE[lBase + tid] = sm[tid] + logf(sl[tid]);

} // end of v3

// =================================
//  V4 : V3 + vectorized TMEM readout
// =================================
// Identical to V3 except the accumulator readout: instead of one 32x32b.x1 load +
// a wait::ld PER COLUMN (N loads + N waits, run 2x per KV tile), read 8 columns per
// tcgen05.ld.32x32b.x8 with a single wait per group of 8 (matches the reference GEMM).
// Same row/col mapping as tmem_readout_to_smem; requires N % 8 == 0 (Bc=32, D=64 ok).
__device__ void tmem_readout_to_smem_vec(
  float* smem_out,
  uint32_t tmem_addr,
  int M,
  int N,
  int smem_stride,
  float scale
){
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

  const int warp_id = threadIdx.x / 32;
  const int lane    = threadIdx.x % 32;
  const int rows_per_warp = M / 4;
  const uint32_t lane_base = (uint32_t)warp_id * 32u;
  const int row = warp_id * rows_per_warp + lane;

  for(int col = 0; col < N; col += 8){
    uint32_t r0,r1,r2,r3,r4,r5,r6,r7;
    asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
      : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7)
      : "r"(tmem_addr + (lane_base << 16) + (uint32_t)col)
      : "memory"
    );
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");   // one wait per 8 columns
    if(lane < rows_per_warp){
      float* o = &smem_out[row * smem_stride + col];
      o[0] = reinterpret_cast<float&>(r0) * scale;
      o[1] = reinterpret_cast<float&>(r1) * scale;
      o[2] = reinterpret_cast<float&>(r2) * scale;
      o[3] = reinterpret_cast<float&>(r3) * scale;
      o[4] = reinterpret_cast<float&>(r4) * scale;
      o[5] = reinterpret_cast<float&>(r5) * scale;
      o[6] = reinterpret_cast<float&>(r6) * scale;
      o[7] = reinterpret_cast<float&>(r7) * scale;
    }
  }
}

// Same as tmem_readout_to_smem_vec, but stores bf16 (packed 2-per-store via
// __nv_bfloat162) instead of fp32 — halves sS's smem footprint (Br*Bc*2 vs *4
// bytes), which is what makes room for V18's double-buffered TMA staging.
__device__ void tmem_readout_to_smem_bf16_vec(
  __nv_bfloat16* smem_out,
  uint32_t tmem_addr,
  int M,
  int N,
  int smem_stride,
  float scale
){
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

  const int warp_id = threadIdx.x / 32;
  const int lane    = threadIdx.x % 32;
  const int rows_per_warp = M / 4;
  const uint32_t lane_base = (uint32_t)warp_id * 32u;
  const int row = warp_id * rows_per_warp + lane;

  for(int col = 0; col < N; col += 8){
    uint32_t r0,r1,r2,r3,r4,r5,r6,r7;
    asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
      : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7)
      : "r"(tmem_addr + (lane_base << 16) + (uint32_t)col)
      : "memory"
    );
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    if(lane < rows_per_warp){
      __nv_bfloat16* o = &smem_out[row * smem_stride + col];
      *reinterpret_cast<__nv_bfloat162*>(&o[0]) = __floats2bfloat162_rn(
          reinterpret_cast<float&>(r0) * scale, reinterpret_cast<float&>(r1) * scale);
      *reinterpret_cast<__nv_bfloat162*>(&o[2]) = __floats2bfloat162_rn(
          reinterpret_cast<float&>(r2) * scale, reinterpret_cast<float&>(r3) * scale);
      *reinterpret_cast<__nv_bfloat162*>(&o[4]) = __floats2bfloat162_rn(
          reinterpret_cast<float&>(r4) * scale, reinterpret_cast<float&>(r5) * scale);
      *reinterpret_cast<__nv_bfloat162*>(&o[6]) = __floats2bfloat162_rn(
          reinterpret_cast<float&>(r6) * scale, reinterpret_cast<float&>(r7) * scale);
    }
  }
}

template<int Br, int Bc, int D>
__global__ void gqa_v4(
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
  const int tid    = threadIdx.x;

  const int q_row0   = q_tile * Br;
  const int nKVTiles = S / Bc;

  const long qBase  = ((long)(b * Hq + hq) * S + q_row0) * D;
  const long kvBase = ((long)(b * Hkv + hkv) * S) * D;
  const long lBase  = ((long)(b * Hq + hq) * S + q_row0);

  __shared__ __align__(16) __nv_bfloat16 sQ[Br * D];
  __shared__ __align__(16) __nv_bfloat16 sK[2 * Bc * D];
  __shared__ __align__(16) __nv_bfloat16 sVstage[2 * Bc * D];
  __shared__ __align__(16) __nv_bfloat16 sV[Bc * D];
  __shared__ __align__(16) __nv_bfloat16 sP[Br * Bc];
  __shared__ __align__(16) float         sS[Br * Bc];
  __shared__ __align__(16) float         sO[Br * D];
  __shared__ __align__(16) float         sPV[Br * D];
  __shared__ float sm[Br];
  __shared__ float sl[Br];
  __shared__ float sCorr[Br];
  __shared__ __align__(8) uint64_t s_mma_bar;

  for(int i = tid; i < Br * D; i += blockDim.x){
    const int r = i / D, c = i % D;
    sQ[canon_idx(r, c, Br)] = d_Q[qBase + i];
    sO[i] = 0.0f;
  }
  if(tid < Br){ sm[tid] = -INFINITY; sl[tid] = 0.0f; }
  __syncthreads();

  constexpr uint32_t NCOLS = (Bc > D) ? (uint32_t)Bc : (uint32_t)D;
  static_assert(NCOLS >= 32 && (NCOLS & (NCOLS - 1)) == 0,
                "tcgen05 column count must be a power of two >= 32");

  uint32_t tmem_addr;
  {
    __shared__ uint32_t s_tmem_addr;
    if(tid < 32){
      uint32_t s_addr = (uint32_t)__cvta_generic_to_shared(&s_tmem_addr);
      asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                   :: "r"(s_addr), "r"(NCOLS) : "memory");
      asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
    __syncthreads();
    tmem_addr = s_tmem_addr;
  }

  const uint32_t mma_bar = (uint32_t)__cvta_generic_to_shared(&s_mma_bar);
  if(tid == 0) mbar_init(mma_bar, 1);
  __syncthreads();
  int mbar_phase = 0;

  const uint64_t descQ_base = make_smem_desc(sQ, Br);

  prefetch_kv_tile<Bc, D>(sK, sVstage, d_K, d_V, kvBase);
  cp_async_commit();

  for(int kc = 0; kc < nKVTiles; ++kc){
    const int cur = kc & 1;
    __nv_bfloat16* sKcur  = sK      + cur * Bc * D;
    __nv_bfloat16* sVscur = sVstage + cur * Bc * D;

    if(kc + 1 < nKVTiles){
      const int nxt = (kc + 1) & 1;
      prefetch_kv_tile<Bc, D>(sK + nxt * Bc * D, sVstage + nxt * Bc * D,
                              d_K, d_V, kvBase + (long)(kc + 1) * Bc * D);
    }
    cp_async_commit();
    cp_async_wait_keep1();
    __syncthreads();

    // S = (Q @ K^T) * scale -> sS[Br, Bc]
    {
      const uint64_t descK_base = make_smem_desc(sKcur, Bc);
      const uint32_t idesc      = make_idesc_bf16(Br, Bc);
      if(tid == 0){
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        for(int kt = 0; kt < D/16; ++kt){
          uint64_t descQ = advance_desc_katom(descQ_base, kt, Br);
          uint64_t descK = advance_desc_katom(descK_base, kt, Bc);
          uint32_t accumulate = (kt > 0) ? 1u : 0u;
          asm volatile(
            "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
            :: "r"(tmem_addr), "l"(descQ), "l"(descK), "r"(idesc), "r"(accumulate) : "memory");
        }
        mbar_commit_mma(mma_bar);
      }
      mbar_wait(mma_bar, mbar_phase); mbar_phase ^= 1;
      tmem_readout_to_smem_vec(sS, tmem_addr, Br, Bc, Bc, scale);   // vectorized readout
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
      __syncthreads();
    }

    // online-softmax stats + unnormalized P (canonical)
    if(tid < Br){
      const float m_old = sm[tid];
      const float l_old = sl[tid];
      float tile_max = -INFINITY;
      for(int j = 0; j < Bc; ++j) tile_max = fmaxf(tile_max, sS[tid * Bc + j]);
      const float m_new = fmaxf(m_old, tile_max);
      const float corr  = __expf(m_old - m_new);
      float p_sum = 0.0f;
      for(int j = 0; j < Bc; ++j){
        const float p = __expf(sS[tid * Bc + j] - m_new);
        sP[canon_idx(tid, j, Br)] = __float2bfloat16(p);
        p_sum += p;
      }
      sm[tid] = m_new; sl[tid] = l_old * corr + p_sum; sCorr[tid] = corr;
    }
    __syncthreads();

    for(int i = tid; i < Br * D; i += blockDim.x) sO[i] *= sCorr[i / D];
    __syncthreads();

    for(int i = tid; i < Bc * D; i += blockDim.x){
      const int bc = i / D, d = i % D;
      sV[canon_idx(d, bc, D)] = sVscur[i];
    }
    __syncthreads();

    // O_tile = P @ V -> sPV[Br, D]
    {
      const uint64_t descP_base = make_smem_desc(sP, Br);
      const uint64_t descV_base = make_smem_desc(sV, D);
      const uint32_t idesc      = make_idesc_bf16(Br, D);
      if(tid == 0){
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        for(int kt = 0; kt < Bc/16; ++kt){
          uint64_t descP = advance_desc_katom(descP_base, kt, Br);
          uint64_t descV = advance_desc_katom(descV_base, kt, D);
          uint32_t accumulate = (kt > 0) ? 1u : 0u;
          asm volatile(
            "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
            :: "r"(tmem_addr), "l"(descP), "l"(descV), "r"(idesc), "r"(accumulate) : "memory");
        }
        mbar_commit_mma(mma_bar);
      }
      mbar_wait(mma_bar, mbar_phase); mbar_phase ^= 1;
      tmem_readout_to_smem_vec(sPV, tmem_addr, Br, D, D, 1.0f);     // vectorized readout
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
      __syncthreads();
    }

    for(int i = tid; i < Br * D; i += blockDim.x) sO[i] += sPV[i];
    __syncthreads();
  } // end kv loop

  if(tid < 32)
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(tmem_addr), "r"(NCOLS) : "memory");
  __syncthreads();

  for(int i = tid; i < Br * D; i += blockDim.x)
    d_O[qBase + i] = __float2bfloat16(sO[i] / sl[i / D]);
  if(tid < Br)
    d_LSE[lBase + tid] = sm[tid] + logf(sl[tid]);

} // end of v4

// =================================
//  V5 : V4 + TMA KV loads (Path A: TMA -> staging -> in-shared canonicalize)
// =================================
// Same verified tcgen05 core + vectorized readout as V4; the KV load path becomes
// TMA (cp.async.bulk.tensor.2d, SWIZZLE_NONE). TMA lands each tile row-major in a
// staging buffer; an in-shared reorder then produces the canonical K (K-major) and
// transposed V that tcgen05.mma needs. 2-stage prefetch: tile kc+1's TMA overlaps
// the compute on tile kc, gated by a per-buffer load mbarrier (transaction bytes).
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes){
  asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar){
  asm volatile("mbarrier.arrive.shared.b64 _, [%0];" :: "r"(bar) : "memory");
}
// TMA 2D tile load: global tile at (col=c, row=r) -> shared [box_rows, box_cols] row-major;
// completion arrives `box bytes` on `bar` (paired with an earlier mbar_expect_tx).
__device__ __forceinline__ void tma_load_2d(uint32_t smem_addr, const void* tmap, int c, int r, uint32_t bar){
  asm volatile(
    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
    :: "r"(smem_addr), "l"(tmap), "r"(c), "r"(r), "r"(bar) : "memory");
}

template<int Br, int Bc, int D>
__global__ void gqa_v5(
  __nv_bfloat16 *d_Q,
  __nv_bfloat16 *d_O,
  float *d_LSE,
  const __grid_constant__ CUtensorMap Ktmap,   // K as flattened [B*Hkv*S, D]
  const __grid_constant__ CUtensorMap Vtmap,   // V as flattened [B*Hkv*S, D]
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
  const int tid    = threadIdx.x;

  const int q_row0   = q_tile * Br;
  const int nKVTiles = S / Bc;
  const int kvRow0   = (b * Hkv + hkv) * S;   // first K/V row of this head in the flat tensor

  const long qBase  = ((long)(b * Hq + hq) * S + q_row0) * D;
  const long lBase  = ((long)(b * Hq + hq) * S + q_row0);

  __shared__ __align__(16)  __nv_bfloat16 sQ[Br * D];
  __shared__ __align__(128) __nv_bfloat16 sKstage[2 * Bc * D];  // TMA target, row-major, double-buffered
  __shared__ __align__(128) __nv_bfloat16 sVstage[2 * Bc * D];
  __shared__ __align__(16)  __nv_bfloat16 sK[Bc * D];           // canonical K-major (reorder output)
  __shared__ __align__(16)  __nv_bfloat16 sV[Bc * D];           // canonical [D,Bc] transposed
  __shared__ __align__(16)  __nv_bfloat16 sP[Br * Bc];
  __shared__ __align__(16)  float         sS[Br * Bc];
  __shared__ __align__(16)  float         sO[Br * D];
  __shared__ __align__(16)  float         sPV[Br * D];
  __shared__ float sm[Br];
  __shared__ float sl[Br];
  __shared__ float sCorr[Br];
  __shared__ __align__(8) uint64_t s_mma_bar;
  __shared__ __align__(8) uint64_t s_load_bar[2];

  for(int i = tid; i < Br * D; i += blockDim.x){
    const int r = i / D, c = i % D;
    sQ[canon_idx(r, c, Br)] = d_Q[qBase + i];
    sO[i] = 0.0f;
  }
  if(tid < Br){ sm[tid] = -INFINITY; sl[tid] = 0.0f; }

  const uint32_t mma_bar = (uint32_t)__cvta_generic_to_shared(&s_mma_bar);
  const uint32_t lbar[2] = { (uint32_t)__cvta_generic_to_shared(&s_load_bar[0]),
                             (uint32_t)__cvta_generic_to_shared(&s_load_bar[1]) };
  if(tid == 0){ mbar_init(mma_bar, 1); mbar_init(lbar[0], 1); mbar_init(lbar[1], 1); }
  __syncthreads();

  constexpr uint32_t NCOLS = (Bc > D) ? (uint32_t)Bc : (uint32_t)D;
  static_assert(NCOLS >= 32 && (NCOLS & (NCOLS - 1)) == 0,
                "tcgen05 column count must be a power of two >= 32");

  uint32_t tmem_addr;
  {
    __shared__ uint32_t s_tmem_addr;
    if(tid < 32){
      uint32_t s_addr = (uint32_t)__cvta_generic_to_shared(&s_tmem_addr);
      asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                   :: "r"(s_addr), "r"(NCOLS) : "memory");
      asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
    __syncthreads();
    tmem_addr = s_tmem_addr;
  }

  int mbar_phase   = 0;
  int load_phase[2] = {0, 0};
  const uint32_t TX = 2u * (uint32_t)Bc * (uint32_t)D * (uint32_t)sizeof(__nv_bfloat16); // K + V box bytes

  const uint64_t descQ_base = make_smem_desc(sQ, Br);

  // Prologue: TMA tile 0 into buffer 0.
  if(tid == 0){
    mbar_expect_tx(lbar[0], TX);
    tma_load_2d((uint32_t)__cvta_generic_to_shared(sKstage), &Ktmap, 0, kvRow0, lbar[0]);
    tma_load_2d((uint32_t)__cvta_generic_to_shared(sVstage), &Vtmap, 0, kvRow0, lbar[0]);
    mbar_arrive(lbar[0]);
  }

  for(int kc = 0; kc < nKVTiles; ++kc){
    const int cur = kc & 1;
    __nv_bfloat16* sKscur = sKstage + cur * Bc * D;
    __nv_bfloat16* sVscur = sVstage + cur * Bc * D;

    // Prefetch next tile.
    if(kc + 1 < nKVTiles){
      const int nxt = (kc + 1) & 1;
      if(tid == 0){
        const int r = kvRow0 + (kc + 1) * Bc;
        mbar_expect_tx(lbar[nxt], TX);
        tma_load_2d((uint32_t)__cvta_generic_to_shared(sKstage + nxt * Bc * D), &Ktmap, 0, r, lbar[nxt]);
        tma_load_2d((uint32_t)__cvta_generic_to_shared(sVstage + nxt * Bc * D), &Vtmap, 0, r, lbar[nxt]);
        mbar_arrive(lbar[nxt]);
      }
    }

    // Wait for the current tile's TMA, make it visible, then reorder to canonical.
    mbar_wait(lbar[cur], load_phase[cur]); load_phase[cur] ^= 1;
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    __syncthreads();

    for(int i = tid; i < Bc * D; i += blockDim.x){
      const int bc = i / D, d = i % D;
      sK[canon_idx(bc, d, Bc)] = sKscur[i];   // K-major
      sV[canon_idx(d, bc, D)]  = sVscur[i];   // transposed
    }
    __syncthreads();

    // S = (Q @ K^T) * scale -> sS[Br, Bc]
    {
      const uint64_t descK_base = make_smem_desc(sK, Bc);
      const uint32_t idesc      = make_idesc_bf16(Br, Bc);
      if(tid == 0){
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        for(int kt = 0; kt < D/16; ++kt){
          uint64_t descQ = advance_desc_katom(descQ_base, kt, Br);
          uint64_t descK = advance_desc_katom(descK_base, kt, Bc);
          uint32_t accumulate = (kt > 0) ? 1u : 0u;
          asm volatile(
            "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
            :: "r"(tmem_addr), "l"(descQ), "l"(descK), "r"(idesc), "r"(accumulate) : "memory");
        }
        mbar_commit_mma(mma_bar);
      }
      mbar_wait(mma_bar, mbar_phase); mbar_phase ^= 1;
      tmem_readout_to_smem_vec(sS, tmem_addr, Br, Bc, Bc, scale);
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
      __syncthreads();
    }

    // online-softmax stats + unnormalized P (canonical)
    if(tid < Br){
      const float m_old = sm[tid];
      const float l_old = sl[tid];
      float tile_max = -INFINITY;
      for(int j = 0; j < Bc; ++j) tile_max = fmaxf(tile_max, sS[tid * Bc + j]);
      const float m_new = fmaxf(m_old, tile_max);
      const float corr  = __expf(m_old - m_new);
      float p_sum = 0.0f;
      for(int j = 0; j < Bc; ++j){
        const float p = __expf(sS[tid * Bc + j] - m_new);
        sP[canon_idx(tid, j, Br)] = __float2bfloat16(p);
        p_sum += p;
      }
      sm[tid] = m_new; sl[tid] = l_old * corr + p_sum; sCorr[tid] = corr;
    }
    __syncthreads();

    for(int i = tid; i < Br * D; i += blockDim.x) sO[i] *= sCorr[i / D];
    __syncthreads();

    // O_tile = P @ V -> sPV[Br, D]
    {
      const uint64_t descP_base = make_smem_desc(sP, Br);
      const uint64_t descV_base = make_smem_desc(sV, D);
      const uint32_t idesc      = make_idesc_bf16(Br, D);
      if(tid == 0){
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        for(int kt = 0; kt < Bc/16; ++kt){
          uint64_t descP = advance_desc_katom(descP_base, kt, Br);
          uint64_t descV = advance_desc_katom(descV_base, kt, D);
          uint32_t accumulate = (kt > 0) ? 1u : 0u;
          asm volatile(
            "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
            :: "r"(tmem_addr), "l"(descP), "l"(descV), "r"(idesc), "r"(accumulate) : "memory");
        }
        mbar_commit_mma(mma_bar);
      }
      mbar_wait(mma_bar, mbar_phase); mbar_phase ^= 1;
      tmem_readout_to_smem_vec(sPV, tmem_addr, Br, D, D, 1.0f);
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
      __syncthreads();
    }

    for(int i = tid; i < Br * D; i += blockDim.x) sO[i] += sPV[i];
    __syncthreads();
  } // end kv loop

  if(tid < 32)
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(tmem_addr), "r"(NCOLS) : "memory");
  __syncthreads();

  for(int i = tid; i < Br * D; i += blockDim.x)
    d_O[qBase + i] = __float2bfloat16(sO[i] / sl[i / D]);
  if(tid < Br)
    d_LSE[lBase + tid] = sm[tid] + logf(sl[tid]);

} // end of v5

// =================================
//  V7 : V6 + log2-domain (base-2) softmax
// =================================
// Same TMA + tcgen05 core as V6; only the softmax math changes, and it is a
// numerically-equivalent rewrite (the O output is bit-identical to V6):
//   * QK^T scores are prescaled by log2(e) at readout (folded into the existing
//     readout scale, so free), moving softmax into the base-2 domain.
//   * exp(x) -> exp2(x) via ex2.approx.ftz (native SFU op). __expf(x) already
//     lowers to ex2.approx.ftz(x*log2e); prescaling lets us drop that per-element
//     multiply-by-log2e — the "compute exp2 instead of exp" trick.
//   * the running row-max uses a 3-way fmaxf tree (fuses to FMNMX3 on sm_100+).
//   * m/l are now in base-2 units, so the LSE finalize becomes ln2*(m + log2(l)).
__device__ __forceinline__ float ex2_approx(float x){
  float y;
  // NOT volatile: ex2.approx is a pure function of x. Marking it volatile forces
  // every EX2 to run in strict program order, making the ~Bc-wide softmax exp loop
  // latency-bound on the SFU (that regressed V7 18->23ms). Plain asm lets the
  // compiler pipeline them, exactly like the __expf builtin V6 used.
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

template<int Br, int Bc, int D>
__global__ void gqa_v7(
  __nv_bfloat16 *d_Q,
  __nv_bfloat16 *d_O,
  float *d_LSE,
  const __grid_constant__ CUtensorMap Ktmap,   // K as flattened [B*Hkv*S, D]
  const __grid_constant__ CUtensorMap Vtmap,   // V as flattened [B*Hkv*S, D]
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
  const int tid    = threadIdx.x;

  const int q_row0   = q_tile * Br;
  const int nKVTiles = S / Bc;
  const int kvRow0   = (b * Hkv + hkv) * S;   // first K/V row of this head in the flat tensor

  const long qBase  = ((long)(b * Hq + hq) * S + q_row0) * D;
  const long lBase  = ((long)(b * Hq + hq) * S + q_row0);

  // Fold log2(e) into the score scale so QK^T lands in base-2 units (see header).
  const float scale_l2e = scale * 1.4426950408889634f;   // scale * float32 log2(e)

  __shared__ __align__(16)  __nv_bfloat16 sQ[Br * D];
  __shared__ __align__(128) __nv_bfloat16 sKstage[2 * Bc * D];  // TMA target, row-major, double-buffered
  __shared__ __align__(128) __nv_bfloat16 sVstage[2 * Bc * D];
  __shared__ __align__(16)  __nv_bfloat16 sK[Bc * D];           // canonical K-major (reorder output)
  __shared__ __align__(16)  __nv_bfloat16 sV[Bc * D];           // canonical [D,Bc] transposed
  __shared__ __align__(16)  __nv_bfloat16 sP[Br * Bc];
  __shared__ __align__(16)  float         sS[Br * Bc];
  __shared__ __align__(16)  float         sO[Br * D];
  __shared__ __align__(16)  float         sPV[Br * D];
  __shared__ float sm[Br];
  __shared__ float sl[Br];
  __shared__ float sCorr[Br];
  __shared__ __align__(8) uint64_t s_mma_bar;
  __shared__ __align__(8) uint64_t s_load_bar[2];

  for(int i = tid; i < Br * D; i += blockDim.x){
    const int r = i / D, c = i % D;
    sQ[canon_idx(r, c, Br)] = d_Q[qBase + i];
    sO[i] = 0.0f;
  }
  if(tid < Br){ sm[tid] = -INFINITY; sl[tid] = 0.0f; }

  const uint32_t mma_bar = (uint32_t)__cvta_generic_to_shared(&s_mma_bar);
  const uint32_t lbar[2] = { (uint32_t)__cvta_generic_to_shared(&s_load_bar[0]),
                             (uint32_t)__cvta_generic_to_shared(&s_load_bar[1]) };
  if(tid == 0){ mbar_init(mma_bar, 1); mbar_init(lbar[0], 1); mbar_init(lbar[1], 1); }
  __syncthreads();

  constexpr uint32_t NCOLS = (Bc > D) ? (uint32_t)Bc : (uint32_t)D;
  static_assert(NCOLS >= 32 && (NCOLS & (NCOLS - 1)) == 0,
                "tcgen05 column count must be a power of two >= 32");

  uint32_t tmem_addr;
  {
    __shared__ uint32_t s_tmem_addr;
    if(tid < 32){
      uint32_t s_addr = (uint32_t)__cvta_generic_to_shared(&s_tmem_addr);
      asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                   :: "r"(s_addr), "r"(NCOLS) : "memory");
      asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
    __syncthreads();
    tmem_addr = s_tmem_addr;
  }

  int mbar_phase   = 0;
  int load_phase[2] = {0, 0};
  const uint32_t TX = 2u * (uint32_t)Bc * (uint32_t)D * (uint32_t)sizeof(__nv_bfloat16); // K + V box bytes

  const uint64_t descQ_base = make_smem_desc(sQ, Br);

  // Prologue: TMA tile 0 into buffer 0.
  if(tid == 0){
    mbar_expect_tx(lbar[0], TX);
    tma_load_2d((uint32_t)__cvta_generic_to_shared(sKstage), &Ktmap, 0, kvRow0, lbar[0]);
    tma_load_2d((uint32_t)__cvta_generic_to_shared(sVstage), &Vtmap, 0, kvRow0, lbar[0]);
    mbar_arrive(lbar[0]);
  }

  for(int kc = 0; kc < nKVTiles; ++kc){
    const int cur = kc & 1;
    __nv_bfloat16* sKscur = sKstage + cur * Bc * D;
    __nv_bfloat16* sVscur = sVstage + cur * Bc * D;

    // Prefetch next tile.
    if(kc + 1 < nKVTiles){
      const int nxt = (kc + 1) & 1;
      if(tid == 0){
        const int r = kvRow0 + (kc + 1) * Bc;
        mbar_expect_tx(lbar[nxt], TX);
        tma_load_2d((uint32_t)__cvta_generic_to_shared(sKstage + nxt * Bc * D), &Ktmap, 0, r, lbar[nxt]);
        tma_load_2d((uint32_t)__cvta_generic_to_shared(sVstage + nxt * Bc * D), &Vtmap, 0, r, lbar[nxt]);
        mbar_arrive(lbar[nxt]);
      }
    }

    // Wait for the current tile's TMA, make it visible, then reorder to canonical.
    mbar_wait(lbar[cur], load_phase[cur]); load_phase[cur] ^= 1;
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    __syncthreads();

    for(int i = tid; i < Bc * D; i += blockDim.x){
      const int bc = i / D, d = i % D;
      sK[canon_idx(bc, d, Bc)] = sKscur[i];   // K-major
      sV[canon_idx(d, bc, D)]  = sVscur[i];   // transposed
    }
    __syncthreads();

    // S2 = (Q @ K^T) * (scale*log2e) -> sS[Br, Bc]  (scores already in base-2 units)
    {
      const uint64_t descK_base = make_smem_desc(sK, Bc);
      const uint32_t idesc      = make_idesc_bf16(Br, Bc);
      if(tid == 0){
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        for(int kt = 0; kt < D/16; ++kt){
          uint64_t descQ = advance_desc_katom(descQ_base, kt, Br);
          uint64_t descK = advance_desc_katom(descK_base, kt, Bc);
          uint32_t accumulate = (kt > 0) ? 1u : 0u;
          asm volatile(
            "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
            :: "r"(tmem_addr), "l"(descQ), "l"(descK), "r"(idesc), "r"(accumulate) : "memory");
        }
        mbar_commit_mma(mma_bar);
      }
      mbar_wait(mma_bar, mbar_phase); mbar_phase ^= 1;
      tmem_readout_to_smem_vec(sS, tmem_addr, Br, Bc, Bc, scale_l2e);   // prescale by log2e here
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
      __syncthreads();
    }

    // online-softmax stats + unnormalized P, base-2 domain (exp2 via ex2.approx)
    if(tid < Br){
      const float m_old = sm[tid];
      const float l_old = sl[tid];

      // 3-way max tree over the row (fuses to FMNMX3 on sm_100+).
      float tile_max = -INFINITY;
      int j = 0;
      for(; j + 2 < Bc; j += 3)
        tile_max = fmaxf(tile_max,
                         fmaxf(sS[tid * Bc + j],
                               fmaxf(sS[tid * Bc + j + 1], sS[tid * Bc + j + 2])));
      for(; j < Bc; ++j) tile_max = fmaxf(tile_max, sS[tid * Bc + j]);

      const float m_new = fmaxf(m_old, tile_max);
      const float corr  = ex2_approx(m_old - m_new);   // == exp(m_old - m_new)
      float p_sum = 0.0f;
      for(int j2 = 0; j2 < Bc; ++j2){
        const float p = ex2_approx(sS[tid * Bc + j2] - m_new);   // == exp(S - m_new)
        sP[canon_idx(tid, j2, Br)] = __float2bfloat16(p);
        p_sum += p;
      }
      sm[tid] = m_new; sl[tid] = l_old * corr + p_sum; sCorr[tid] = corr;
    }
    __syncthreads();

    for(int i = tid; i < Br * D; i += blockDim.x) sO[i] *= sCorr[i / D];
    __syncthreads();

    // O_tile = P @ V -> sPV[Br, D]
    {
      const uint64_t descP_base = make_smem_desc(sP, Br);
      const uint64_t descV_base = make_smem_desc(sV, D);
      const uint32_t idesc      = make_idesc_bf16(Br, D);
      if(tid == 0){
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        for(int kt = 0; kt < Bc/16; ++kt){
          uint64_t descP = advance_desc_katom(descP_base, kt, Br);
          uint64_t descV = advance_desc_katom(descV_base, kt, D);
          uint32_t accumulate = (kt > 0) ? 1u : 0u;
          asm volatile(
            "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
            :: "r"(tmem_addr), "l"(descP), "l"(descV), "r"(idesc), "r"(accumulate) : "memory");
        }
        mbar_commit_mma(mma_bar);
      }
      mbar_wait(mma_bar, mbar_phase); mbar_phase ^= 1;
      tmem_readout_to_smem_vec(sPV, tmem_addr, Br, D, D, 1.0f);
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
      __syncthreads();
    }

    for(int i = tid; i < Br * D; i += blockDim.x) sO[i] += sPV[i];
    __syncthreads();
  } // end kv loop

  if(tid < 32)
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(tmem_addr), "r"(NCOLS) : "memory");
  __syncthreads();

  for(int i = tid; i < Br * D; i += blockDim.x)
    d_O[qBase + i] = __float2bfloat16(sO[i] / sl[i / D]);
  if(tid < Br)
    // m and l are in base-2 units: LSE_nat = ln2 * (m2 + log2(l)).
    d_LSE[lBase + tid] = 0.6931471805599453f * (sm[tid] + log2f(sl[tid]));

} // end of v7

// =================================
//  V8 : V7 + packed bf16 conversion (F2FP.BF16.F32.PACK_AB)
// =================================
// Same core as V7; only the fp32->bf16 conversions are packed two-at-a-time:
//   * softmax writes the P operand a pair of probabilities per store via
//     __floats2bfloat162_rn (one F2FP.PACK + one 32-bit STS, vs two scalar
//     converts + two 16-bit stores). canon_idx(tid,j) and canon_idx(tid,j+1)
//     are adjacent for even j (same 8-wide K-atom) and the base index is even,
//     so the bf16x2 store is contiguous and 4-byte aligned.
//   * the final O write packs output pairs the same way (D even -> both columns
//     share the row denominator sl[i/D]).
// Numerically identical to V7 (same round-to-nearest), just fewer conversion and
// store instructions in the softmax epilogue and the epilogue O write.
template<int Br, int Bc, int D>
__global__ void gqa_v8(
  __nv_bfloat16 *d_Q,
  __nv_bfloat16 *d_O,
  float *d_LSE,
  const __grid_constant__ CUtensorMap Ktmap,   // K as flattened [B*Hkv*S, D]
  const __grid_constant__ CUtensorMap Vtmap,   // V as flattened [B*Hkv*S, D]
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
  const int tid    = threadIdx.x;

  const int q_row0   = q_tile * Br;
  const int nKVTiles = S / Bc;
  const int kvRow0   = (b * Hkv + hkv) * S;   // first K/V row of this head in the flat tensor

  const long qBase  = ((long)(b * Hq + hq) * S + q_row0) * D;
  const long lBase  = ((long)(b * Hq + hq) * S + q_row0);

  // Fold log2(e) into the score scale so QK^T lands in base-2 units (see V7 header).
  const float scale_l2e = scale * 1.4426950408889634f;   // scale * float32 log2(e)

  __shared__ __align__(16)  __nv_bfloat16 sQ[Br * D];
  __shared__ __align__(128) __nv_bfloat16 sKstage[2 * Bc * D];  // TMA target, row-major, double-buffered
  __shared__ __align__(128) __nv_bfloat16 sVstage[2 * Bc * D];
  __shared__ __align__(16)  __nv_bfloat16 sK[Bc * D];           // canonical K-major (reorder output)
  __shared__ __align__(16)  __nv_bfloat16 sV[Bc * D];           // canonical [D,Bc] transposed
  __shared__ __align__(16)  __nv_bfloat16 sP[Br * Bc];
  __shared__ __align__(16)  float         sS[Br * Bc];
  __shared__ __align__(16)  float         sO[Br * D];
  __shared__ __align__(16)  float         sPV[Br * D];
  __shared__ float sm[Br];
  __shared__ float sl[Br];
  __shared__ float sCorr[Br];
  __shared__ __align__(8) uint64_t s_mma_bar;
  __shared__ __align__(8) uint64_t s_load_bar[2];

  for(int i = tid; i < Br * D; i += blockDim.x){
    const int r = i / D, c = i % D;
    sQ[canon_idx(r, c, Br)] = d_Q[qBase + i];
    sO[i] = 0.0f;
  }
  if(tid < Br){ sm[tid] = -INFINITY; sl[tid] = 0.0f; }

  const uint32_t mma_bar = (uint32_t)__cvta_generic_to_shared(&s_mma_bar);
  const uint32_t lbar[2] = { (uint32_t)__cvta_generic_to_shared(&s_load_bar[0]),
                             (uint32_t)__cvta_generic_to_shared(&s_load_bar[1]) };
  if(tid == 0){ mbar_init(mma_bar, 1); mbar_init(lbar[0], 1); mbar_init(lbar[1], 1); }
  __syncthreads();

  constexpr uint32_t NCOLS = (Bc > D) ? (uint32_t)Bc : (uint32_t)D;
  static_assert(NCOLS >= 32 && (NCOLS & (NCOLS - 1)) == 0,
                "tcgen05 column count must be a power of two >= 32");

  uint32_t tmem_addr;
  {
    __shared__ uint32_t s_tmem_addr;
    if(tid < 32){
      uint32_t s_addr = (uint32_t)__cvta_generic_to_shared(&s_tmem_addr);
      asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                   :: "r"(s_addr), "r"(NCOLS) : "memory");
      asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
    __syncthreads();
    tmem_addr = s_tmem_addr;
  }

  int mbar_phase   = 0;
  int load_phase[2] = {0, 0};
  const uint32_t TX = 2u * (uint32_t)Bc * (uint32_t)D * (uint32_t)sizeof(__nv_bfloat16); // K + V box bytes

  const uint64_t descQ_base = make_smem_desc(sQ, Br);

  // Prologue: TMA tile 0 into buffer 0.
  if(tid == 0){
    mbar_expect_tx(lbar[0], TX);
    tma_load_2d((uint32_t)__cvta_generic_to_shared(sKstage), &Ktmap, 0, kvRow0, lbar[0]);
    tma_load_2d((uint32_t)__cvta_generic_to_shared(sVstage), &Vtmap, 0, kvRow0, lbar[0]);
    mbar_arrive(lbar[0]);
  }

  for(int kc = 0; kc < nKVTiles; ++kc){
    const int cur = kc & 1;
    __nv_bfloat16* sKscur = sKstage + cur * Bc * D;
    __nv_bfloat16* sVscur = sVstage + cur * Bc * D;

    // Prefetch next tile.
    if(kc + 1 < nKVTiles){
      const int nxt = (kc + 1) & 1;
      if(tid == 0){
        const int r = kvRow0 + (kc + 1) * Bc;
        mbar_expect_tx(lbar[nxt], TX);
        tma_load_2d((uint32_t)__cvta_generic_to_shared(sKstage + nxt * Bc * D), &Ktmap, 0, r, lbar[nxt]);
        tma_load_2d((uint32_t)__cvta_generic_to_shared(sVstage + nxt * Bc * D), &Vtmap, 0, r, lbar[nxt]);
        mbar_arrive(lbar[nxt]);
      }
    }

    // Wait for the current tile's TMA, make it visible, then reorder to canonical.
    mbar_wait(lbar[cur], load_phase[cur]); load_phase[cur] ^= 1;
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    __syncthreads();

    for(int i = tid; i < Bc * D; i += blockDim.x){
      const int bc = i / D, d = i % D;
      sK[canon_idx(bc, d, Bc)] = sKscur[i];   // K-major
      sV[canon_idx(d, bc, D)]  = sVscur[i];   // transposed
    }
    __syncthreads();

    // S2 = (Q @ K^T) * (scale*log2e) -> sS[Br, Bc]  (scores already in base-2 units)
    {
      const uint64_t descK_base = make_smem_desc(sK, Bc);
      const uint32_t idesc      = make_idesc_bf16(Br, Bc);
      if(tid == 0){
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        for(int kt = 0; kt < D/16; ++kt){
          uint64_t descQ = advance_desc_katom(descQ_base, kt, Br);
          uint64_t descK = advance_desc_katom(descK_base, kt, Bc);
          uint32_t accumulate = (kt > 0) ? 1u : 0u;
          asm volatile(
            "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
            :: "r"(tmem_addr), "l"(descQ), "l"(descK), "r"(idesc), "r"(accumulate) : "memory");
        }
        mbar_commit_mma(mma_bar);
      }
      mbar_wait(mma_bar, mbar_phase); mbar_phase ^= 1;
      tmem_readout_to_smem_vec(sS, tmem_addr, Br, Bc, Bc, scale_l2e);   // prescale by log2e here
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
      __syncthreads();
    }

    // online-softmax stats + unnormalized P, base-2 domain (exp2 via ex2.approx)
    if(tid < Br){
      const float m_old = sm[tid];
      const float l_old = sl[tid];

      // 3-way max tree over the row (fuses to FMNMX3 on sm_100+).
      float tile_max = -INFINITY;
      int j = 0;
      for(; j + 2 < Bc; j += 3)
        tile_max = fmaxf(tile_max,
                         fmaxf(sS[tid * Bc + j],
                               fmaxf(sS[tid * Bc + j + 1], sS[tid * Bc + j + 2])));
      for(; j < Bc; ++j) tile_max = fmaxf(tile_max, sS[tid * Bc + j]);

      const float m_new = fmaxf(m_old, tile_max);
      const float corr  = ex2_approx(m_old - m_new);   // == exp(m_old - m_new)

      // Pack two probabilities per F2FP (F2FP.BF16.F32.PACK_AB) + one 32-bit STS.
      float p_sum = 0.0f;
      for(int j2 = 0; j2 < Bc; j2 += 2){
        const float p0 = ex2_approx(sS[tid * Bc + j2]     - m_new);   // == exp(S - m_new)
        const float p1 = ex2_approx(sS[tid * Bc + j2 + 1] - m_new);
        *reinterpret_cast<__nv_bfloat162*>(&sP[canon_idx(tid, j2, Br)]) =
            __floats2bfloat162_rn(p0, p1);
        p_sum += p0 + p1;
      }
      sm[tid] = m_new; sl[tid] = l_old * corr + p_sum; sCorr[tid] = corr;
    }
    __syncthreads();

    for(int i = tid; i < Br * D; i += blockDim.x) sO[i] *= sCorr[i / D];
    __syncthreads();

    // O_tile = P @ V -> sPV[Br, D]
    {
      const uint64_t descP_base = make_smem_desc(sP, Br);
      const uint64_t descV_base = make_smem_desc(sV, D);
      const uint32_t idesc      = make_idesc_bf16(Br, D);
      if(tid == 0){
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        for(int kt = 0; kt < Bc/16; ++kt){
          uint64_t descP = advance_desc_katom(descP_base, kt, Br);
          uint64_t descV = advance_desc_katom(descV_base, kt, D);
          uint32_t accumulate = (kt > 0) ? 1u : 0u;
          asm volatile(
            "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
            :: "r"(tmem_addr), "l"(descP), "l"(descV), "r"(idesc), "r"(accumulate) : "memory");
        }
        mbar_commit_mma(mma_bar);
      }
      mbar_wait(mma_bar, mbar_phase); mbar_phase ^= 1;
      tmem_readout_to_smem_vec(sPV, tmem_addr, Br, D, D, 1.0f);
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
      __syncthreads();
    }

    for(int i = tid; i < Br * D; i += blockDim.x) sO[i] += sPV[i];
    __syncthreads();
  } // end kv loop

  if(tid < 32)
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(tmem_addr), "r"(NCOLS) : "memory");
  __syncthreads();

  // Normalize + write O, packing two columns per bf16x2 store (D even -> same row denom).
  for(int i = 2 * tid; i < Br * D; i += 2 * blockDim.x){
    const float denom = sl[i / D];
    *reinterpret_cast<__nv_bfloat162*>(&d_O[qBase + i]) =
        __floats2bfloat162_rn(sO[i] / denom, sO[i + 1] / denom);
  }
  if(tid < Br)
    // m and l are in base-2 units: LSE_nat = ln2 * (m2 + log2(l)).
    d_LSE[lBase + tid] = 0.6931471805599453f * (sm[tid] + log2f(sl[tid]));

} // end of v8

// =================================
//  V11 helper : vectorized TMEM readout that ACCUMULATES into fp32 smem
// =================================
// Same mapping/x8 shape as tmem_readout_to_smem_vec, but does smem += value instead
// of smem = value*scale. Lets the P@V result land directly in the running sO
// accumulator — no sPV staging buffer, no separate sO+=sPV pass, one less barrier.
// Safe: each (row, col) element is read-modify-written by exactly one thread, and the
// preceding __syncthreads (after the sCorr rescale of sO) ordered the prior writers.
__device__ void tmem_readout_accum_vec(
  float* smem_acc,
  uint32_t tmem_addr,
  int M,
  int N,
  int smem_stride
){
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

  const int warp_id = threadIdx.x / 32;
  const int lane    = threadIdx.x % 32;
  const int rows_per_warp = M / 4;
  const uint32_t lane_base = (uint32_t)warp_id * 32u;
  const int row = warp_id * rows_per_warp + lane;

  for(int col = 0; col < N; col += 8){
    uint32_t r0,r1,r2,r3,r4,r5,r6,r7;
    asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
      : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7)
      : "r"(tmem_addr + (lane_base << 16) + (uint32_t)col)
      : "memory"
    );
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    if(lane < rows_per_warp){
      float* o = &smem_acc[row * smem_stride + col];
      o[0] += reinterpret_cast<float&>(r0);
      o[1] += reinterpret_cast<float&>(r1);
      o[2] += reinterpret_cast<float&>(r2);
      o[3] += reinterpret_cast<float&>(r3);
      o[4] += reinterpret_cast<float&>(r4);
      o[5] += reinterpret_cast<float&>(r5);
      o[6] += reinterpret_cast<float&>(r6);
      o[7] += reinterpret_cast<float&>(r7);
    }
  }
}

// =================================
//  V9 : V8 + software-pipelined QK^T (overlap next-tile MMA with this-tile softmax)
// =================================
// V7/V8 showed the softmax epilogue and the loads are NOT the bottleneck — the
// remaining cost is the tensor-core / CUDA-core serialization: the tensor core sits
// idle during softmax+readout and vice-versa, ~128 times (one per KV tile). Fix: issue
// tile kc+1's QK^T MMA BEFORE doing tile kc's softmax, so QK^T(kc+1) runs on the tensor
// core while softmax(kc) runs on the CUDA cores. Same single 128-thread warpgroup and the
// verified V8 sync primitives. Two enabling changes:
//   (a) DISJOINT TMEM buffers — score buffer tmem_S [cols 0,Bc) and PV buffer tmem_PV
//       [cols Bc,Bc+D). V8 reused one region for QK^T then P@V sequentially; here QK^T(kc+1)
//       and P@V(kc) are in flight at the same time, so they must not share columns.
//   (b) DOUBLE-BUFFERED canonical sK/sV — reorder(kc+1) writes buffer nxt while P@V(kc)
//       still reads sV[cur] and the just-issued QK^T(kc+1) reads sK[nxt].
// tmem_S is single-buffered: readout(kc)->sS completes (fence::before + __syncthreads)
// before QK^T(kc+1) overwrites it, and softmax(kc) then reads sS (smem), not TMEM.
// (setmaxnreg / true 2-warpgroup split is deferred to the 2-CTA stage, where it's needed.)
template<int Br, int Bc, int D>
__global__ void gqa_v9(
  __nv_bfloat16 *d_Q,
  __nv_bfloat16 *d_O,
  float *d_LSE,
  const __grid_constant__ CUtensorMap Ktmap,   // K as flattened [B*Hkv*S, D]
  const __grid_constant__ CUtensorMap Vtmap,   // V as flattened [B*Hkv*S, D]
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
  const int tid    = threadIdx.x;

  const int q_row0   = q_tile * Br;
  const int nKVTiles = S / Bc;
  const int kvRow0   = (b * Hkv + hkv) * S;   // first K/V row of this head in the flat tensor

  const long qBase  = ((long)(b * Hq + hq) * S + q_row0) * D;
  const long lBase  = ((long)(b * Hq + hq) * S + q_row0);

  // Fold log2(e) into the score scale so QK^T lands in base-2 units (see V7 header).
  const float scale_l2e = scale * 1.4426950408889634f;   // scale * float32 log2(e)

  __shared__ __align__(16)  __nv_bfloat16 sQ[Br * D];
  __shared__ __align__(128) __nv_bfloat16 sKstage[2 * Bc * D];  // TMA target, row-major, double-buffered
  __shared__ __align__(128) __nv_bfloat16 sVstage[2 * Bc * D];
  __shared__ __align__(16)  __nv_bfloat16 sK[2 * Bc * D];       // canonical K-major, double-buffered
  __shared__ __align__(16)  __nv_bfloat16 sV[2 * Bc * D];       // canonical [D,Bc] transposed, double-buffered
  __shared__ __align__(16)  __nv_bfloat16 sP[Br * Bc];
  __shared__ __align__(16)  float         sS[Br * Bc];
  __shared__ __align__(16)  float         sO[Br * D];
  __shared__ __align__(16)  float         sPV[Br * D];
  __shared__ float sm[Br];
  __shared__ float sl[Br];
  __shared__ float sCorr[Br];
  __shared__ __align__(8) uint64_t s_bar_s;                    // QK^T MMA completion
  __shared__ __align__(8) uint64_t s_bar_pv;                   // P@V MMA completion
  __shared__ __align__(8) uint64_t s_load_bar[2];              // TMA per-buffer completion

  for(int i = tid; i < Br * D; i += blockDim.x){
    const int r = i / D, c = i % D;
    sQ[canon_idx(r, c, Br)] = d_Q[qBase + i];
    sO[i] = 0.0f;
  }
  if(tid < Br){ sm[tid] = -INFINITY; sl[tid] = 0.0f; }

  const uint32_t bar_s   = (uint32_t)__cvta_generic_to_shared(&s_bar_s);
  const uint32_t bar_pv  = (uint32_t)__cvta_generic_to_shared(&s_bar_pv);
  const uint32_t lbar[2] = { (uint32_t)__cvta_generic_to_shared(&s_load_bar[0]),
                             (uint32_t)__cvta_generic_to_shared(&s_load_bar[1]) };
  if(tid == 0){ mbar_init(bar_s, 1); mbar_init(bar_pv, 1); mbar_init(lbar[0], 1); mbar_init(lbar[1], 1); }
  __syncthreads();

  // Disjoint TMEM: score buffer [0,Bc) + PV buffer [Bc,Bc+D), rounded up to a pow2 alloc.
  constexpr uint32_t SPAN  = (uint32_t)Bc + (uint32_t)D;
  constexpr uint32_t NCOLS = (SPAN <= 32) ? 32u : (SPAN <= 64) ? 64u
                           : (SPAN <= 128) ? 128u : 256u;
  static_assert(SPAN <= NCOLS, "TMEM alloc must hold disjoint score + PV buffers");
  static_assert(NCOLS >= 32 && (NCOLS & (NCOLS - 1)) == 0,
                "tcgen05 column count must be a power of two >= 32");

  uint32_t tmem_addr;
  {
    __shared__ uint32_t s_tmem_addr;
    if(tid < 32){
      uint32_t s_addr = (uint32_t)__cvta_generic_to_shared(&s_tmem_addr);
      asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                   :: "r"(s_addr), "r"(NCOLS) : "memory");
      asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
    __syncthreads();
    tmem_addr = s_tmem_addr;
  }
  const uint32_t tmem_S  = tmem_addr;                    // QK^T scores: cols [0, Bc)
  const uint32_t tmem_PV = tmem_addr + (uint32_t)Bc;     // P@V output:  cols [Bc, Bc+D)

  int phase_s = 0, phase_pv = 0, load_phase[2] = {0, 0};
  const uint32_t TX = 2u * (uint32_t)Bc * (uint32_t)D * (uint32_t)sizeof(__nv_bfloat16); // K + V box bytes

  const uint64_t descQ_base = make_smem_desc(sQ, Br);

  // Reorder one TMA-staged tile (buffer `buf`) into canonical sK (K-major) + sV (transposed).
  // (Written inline in prologue + loop; kept as a lambda-free explicit loop for clarity.)

  // ---- Prologue: TMA tile 0, reorder to canonical buffer 0, issue QK^T(0). ----
  if(tid == 0){
    mbar_expect_tx(lbar[0], TX);
    tma_load_2d((uint32_t)__cvta_generic_to_shared(sKstage), &Ktmap, 0, kvRow0, lbar[0]);
    tma_load_2d((uint32_t)__cvta_generic_to_shared(sVstage), &Vtmap, 0, kvRow0, lbar[0]);
    mbar_arrive(lbar[0]);
  }
  mbar_wait(lbar[0], load_phase[0]); load_phase[0] ^= 1;
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  __syncthreads();
  for(int i = tid; i < Bc * D; i += blockDim.x){
    const int bc = i / D, d = i % D;
    sK[canon_idx(bc, d, Bc)] = sKstage[i];
    sV[canon_idx(d, bc, D)]  = sVstage[i];
  }
  __syncthreads();
  if(tid == 0){
    const uint64_t descK_base = make_smem_desc(sK, Bc);
    const uint32_t idesc      = make_idesc_bf16(Br, Bc);
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    for(int kt = 0; kt < D/16; ++kt){
      uint64_t descQ = advance_desc_katom(descQ_base, kt, Br);
      uint64_t descK = advance_desc_katom(descK_base, kt, Bc);
      uint32_t accumulate = (kt > 0) ? 1u : 0u;
      asm volatile(
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
        :: "r"(tmem_S), "l"(descQ), "l"(descK), "r"(idesc), "r"(accumulate) : "memory");
    }
    mbar_commit_mma(bar_s);
  }

  for(int kc = 0; kc < nKVTiles; ++kc){
    const int cur = kc & 1;
    const int nxt = (kc + 1) & 1;

    // Prefetch tile kc+1 (overlaps this tile's QK^T wait + readout).
    if(kc + 1 < nKVTiles && tid == 0){
      const int r = kvRow0 + (kc + 1) * Bc;
      mbar_expect_tx(lbar[nxt], TX);
      tma_load_2d((uint32_t)__cvta_generic_to_shared(sKstage + nxt * Bc * D), &Ktmap, 0, r, lbar[nxt]);
      tma_load_2d((uint32_t)__cvta_generic_to_shared(sVstage + nxt * Bc * D), &Vtmap, 0, r, lbar[nxt]);
      mbar_arrive(lbar[nxt]);
    }

    // Wait QK^T(kc), read scores out of tmem_S -> sS, then free tmem_S for QK^T(kc+1).
    mbar_wait(bar_s, phase_s); phase_s ^= 1;
    tmem_readout_to_smem_vec(sS, tmem_S, Br, Bc, Bc, scale_l2e);
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();

    // Reorder tile kc+1 into canonical buffer nxt and issue QK^T(kc+1) — this MMA runs on
    // the tensor core while the softmax(kc) below runs on the CUDA cores (the overlap).
    if(kc + 1 < nKVTiles){
      mbar_wait(lbar[nxt], load_phase[nxt]); load_phase[nxt] ^= 1;
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      __syncthreads();
      for(int i = tid; i < Bc * D; i += blockDim.x){
        const int bc = i / D, d = i % D;
        sK[nxt * Bc * D + canon_idx(bc, d, Bc)] = sKstage[nxt * Bc * D + i];
        sV[nxt * Bc * D + canon_idx(d, bc, D)]  = sVstage[nxt * Bc * D + i];
      }
      __syncthreads();
      if(tid == 0){
        const uint64_t descK_base = make_smem_desc(sK + nxt * Bc * D, Bc);
        const uint32_t idesc      = make_idesc_bf16(Br, Bc);
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        for(int kt = 0; kt < D/16; ++kt){
          uint64_t descQ = advance_desc_katom(descQ_base, kt, Br);
          uint64_t descK = advance_desc_katom(descK_base, kt, Bc);
          uint32_t accumulate = (kt > 0) ? 1u : 0u;
          asm volatile(
            "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
            :: "r"(tmem_S), "l"(descQ), "l"(descK), "r"(idesc), "r"(accumulate) : "memory");
        }
        mbar_commit_mma(bar_s);
      }
    }

    // online-softmax stats + unnormalized P (base-2, packed) — overlaps QK^T(kc+1).
    if(tid < Br){
      const float m_old = sm[tid];
      const float l_old = sl[tid];

      float tile_max = -INFINITY;
      int j = 0;
      for(; j + 2 < Bc; j += 3)
        tile_max = fmaxf(tile_max,
                         fmaxf(sS[tid * Bc + j],
                               fmaxf(sS[tid * Bc + j + 1], sS[tid * Bc + j + 2])));
      for(; j < Bc; ++j) tile_max = fmaxf(tile_max, sS[tid * Bc + j]);

      const float m_new = fmaxf(m_old, tile_max);
      const float corr  = ex2_approx(m_old - m_new);

      float p_sum = 0.0f;
      for(int j2 = 0; j2 < Bc; j2 += 2){
        const float p0 = ex2_approx(sS[tid * Bc + j2]     - m_new);
        const float p1 = ex2_approx(sS[tid * Bc + j2 + 1] - m_new);
        *reinterpret_cast<__nv_bfloat162*>(&sP[canon_idx(tid, j2, Br)]) =
            __floats2bfloat162_rn(p0, p1);
        p_sum += p0 + p1;
      }
      sm[tid] = m_new; sl[tid] = l_old * corr + p_sum; sCorr[tid] = corr;
    }
    __syncthreads();

    for(int i = tid; i < Br * D; i += blockDim.x) sO[i] *= sCorr[i / D];
    __syncthreads();

    // O_tile = P @ V(cur) -> tmem_PV (disjoint from tmem_S, so it coexists with QK^T(kc+1)).
    if(tid == 0){
      const uint64_t descP_base = make_smem_desc(sP, Br);
      const uint64_t descV_base = make_smem_desc(sV + cur * Bc * D, D);
      const uint32_t idesc      = make_idesc_bf16(Br, D);
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
      for(int kt = 0; kt < Bc/16; ++kt){
        uint64_t descP = advance_desc_katom(descP_base, kt, Br);
        uint64_t descV = advance_desc_katom(descV_base, kt, D);
        uint32_t accumulate = (kt > 0) ? 1u : 0u;
        asm volatile(
          "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
          "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
          :: "r"(tmem_PV), "l"(descP), "l"(descV), "r"(idesc), "r"(accumulate) : "memory");
      }
      mbar_commit_mma(bar_pv);
    }
    mbar_wait(bar_pv, phase_pv); phase_pv ^= 1;
    tmem_readout_to_smem_vec(sPV, tmem_PV, Br, D, D, 1.0f);
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();

    for(int i = tid; i < Br * D; i += blockDim.x) sO[i] += sPV[i];
    __syncthreads();
  } // end kv loop

  if(tid < 32)
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(tmem_addr), "r"(NCOLS) : "memory");
  __syncthreads();

  // Normalize + write O, packing two columns per bf16x2 store (D even -> same row denom).
  for(int i = 2 * tid; i < Br * D; i += 2 * blockDim.x){
    const float denom = sl[i / D];
    *reinterpret_cast<__nv_bfloat162*>(&d_O[qBase + i]) =
        __floats2bfloat162_rn(sO[i] / denom, sO[i + 1] / denom);
  }
  if(tid < Br)
    d_LSE[lBase + tid] = 0.6931471805599453f * (sm[tid] + log2f(sl[tid]));

} // end of v9

// =================================
//  V11 : V10 (V8 kernel @ Bc=64) + P@V readout fused into the O accumulation
// =================================
// Same core as V8/V10; only the P@V epilogue changes. V8 staged the P@V readout in
// sPV, then a separate pass did sO += sPV. Here tmem_readout_accum_vec lands the
// readout directly in sO: drops the 32KB sPV buffer, a full Br*D smem round-trip,
// and one __syncthreads per KV tile. Also step 1 of the smem diet toward Bc=128
// (V8's structure at Bc=128 is ~273KB static — over the ~228KB/SM limit).
// Numerically identical to V10: same values, same fp32 adds, just no staging hop.
template<int Br, int Bc, int D>
__global__ void gqa_v11(
  __nv_bfloat16 *d_Q,
  __nv_bfloat16 *d_O,
  float *d_LSE,
  const __grid_constant__ CUtensorMap Ktmap,   // K as flattened [B*Hkv*S, D]
  const __grid_constant__ CUtensorMap Vtmap,   // V as flattened [B*Hkv*S, D]
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
  const int tid    = threadIdx.x;

  const int q_row0   = q_tile * Br;
  const int nKVTiles = S / Bc;
  const int kvRow0   = (b * Hkv + hkv) * S;   // first K/V row of this head in the flat tensor

  const long qBase  = ((long)(b * Hq + hq) * S + q_row0) * D;
  const long lBase  = ((long)(b * Hq + hq) * S + q_row0);

  // Fold log2(e) into the score scale so QK^T lands in base-2 units (see V7 header).
  const float scale_l2e = scale * 1.4426950408889634f;   // scale * float32 log2(e)

  __shared__ __align__(16)  __nv_bfloat16 sQ[Br * D];
  __shared__ __align__(128) __nv_bfloat16 sKstage[2 * Bc * D];  // TMA target, row-major, double-buffered
  __shared__ __align__(128) __nv_bfloat16 sVstage[2 * Bc * D];
  __shared__ __align__(16)  __nv_bfloat16 sK[Bc * D];           // canonical K-major (reorder output)
  __shared__ __align__(16)  __nv_bfloat16 sV[Bc * D];           // canonical [D,Bc] transposed
  __shared__ __align__(16)  __nv_bfloat16 sP[Br * Bc];
  __shared__ __align__(16)  float         sS[Br * Bc];
  __shared__ __align__(16)  float         sO[Br * D];           // P@V readout accumulates here directly
  __shared__ float sm[Br];
  __shared__ float sl[Br];
  __shared__ float sCorr[Br];
  __shared__ __align__(8) uint64_t s_mma_bar;
  __shared__ __align__(8) uint64_t s_load_bar[2];

  for(int i = tid; i < Br * D; i += blockDim.x){
    const int r = i / D, c = i % D;
    sQ[canon_idx(r, c, Br)] = d_Q[qBase + i];
    sO[i] = 0.0f;
  }
  if(tid < Br){ sm[tid] = -INFINITY; sl[tid] = 0.0f; }

  const uint32_t mma_bar = (uint32_t)__cvta_generic_to_shared(&s_mma_bar);
  const uint32_t lbar[2] = { (uint32_t)__cvta_generic_to_shared(&s_load_bar[0]),
                             (uint32_t)__cvta_generic_to_shared(&s_load_bar[1]) };
  if(tid == 0){ mbar_init(mma_bar, 1); mbar_init(lbar[0], 1); mbar_init(lbar[1], 1); }
  __syncthreads();

  constexpr uint32_t NCOLS = (Bc > D) ? (uint32_t)Bc : (uint32_t)D;
  static_assert(NCOLS >= 32 && (NCOLS & (NCOLS - 1)) == 0,
                "tcgen05 column count must be a power of two >= 32");

  uint32_t tmem_addr;
  {
    __shared__ uint32_t s_tmem_addr;
    if(tid < 32){
      uint32_t s_addr = (uint32_t)__cvta_generic_to_shared(&s_tmem_addr);
      asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                   :: "r"(s_addr), "r"(NCOLS) : "memory");
      asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
    __syncthreads();
    tmem_addr = s_tmem_addr;
  }

  int mbar_phase   = 0;
  int load_phase[2] = {0, 0};
  const uint32_t TX = 2u * (uint32_t)Bc * (uint32_t)D * (uint32_t)sizeof(__nv_bfloat16); // K + V box bytes

  const uint64_t descQ_base = make_smem_desc(sQ, Br);

  // Prologue: TMA tile 0 into buffer 0.
  if(tid == 0){
    mbar_expect_tx(lbar[0], TX);
    tma_load_2d((uint32_t)__cvta_generic_to_shared(sKstage), &Ktmap, 0, kvRow0, lbar[0]);
    tma_load_2d((uint32_t)__cvta_generic_to_shared(sVstage), &Vtmap, 0, kvRow0, lbar[0]);
    mbar_arrive(lbar[0]);
  }

  for(int kc = 0; kc < nKVTiles; ++kc){
    const int cur = kc & 1;
    __nv_bfloat16* sKscur = sKstage + cur * Bc * D;
    __nv_bfloat16* sVscur = sVstage + cur * Bc * D;

    // Prefetch next tile.
    if(kc + 1 < nKVTiles){
      const int nxt = (kc + 1) & 1;
      if(tid == 0){
        const int r = kvRow0 + (kc + 1) * Bc;
        mbar_expect_tx(lbar[nxt], TX);
        tma_load_2d((uint32_t)__cvta_generic_to_shared(sKstage + nxt * Bc * D), &Ktmap, 0, r, lbar[nxt]);
        tma_load_2d((uint32_t)__cvta_generic_to_shared(sVstage + nxt * Bc * D), &Vtmap, 0, r, lbar[nxt]);
        mbar_arrive(lbar[nxt]);
      }
    }

    // Wait for the current tile's TMA, make it visible, then reorder to canonical.
    mbar_wait(lbar[cur], load_phase[cur]); load_phase[cur] ^= 1;
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    __syncthreads();

    for(int i = tid; i < Bc * D; i += blockDim.x){
      const int bc = i / D, d = i % D;
      sK[canon_idx(bc, d, Bc)] = sKscur[i];   // K-major
      sV[canon_idx(d, bc, D)]  = sVscur[i];   // transposed
    }
    __syncthreads();

    // S2 = (Q @ K^T) * (scale*log2e) -> sS[Br, Bc]  (scores already in base-2 units)
    {
      const uint64_t descK_base = make_smem_desc(sK, Bc);
      const uint32_t idesc      = make_idesc_bf16(Br, Bc);
      if(tid == 0){
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        for(int kt = 0; kt < D/16; ++kt){
          uint64_t descQ = advance_desc_katom(descQ_base, kt, Br);
          uint64_t descK = advance_desc_katom(descK_base, kt, Bc);
          uint32_t accumulate = (kt > 0) ? 1u : 0u;
          asm volatile(
            "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
            :: "r"(tmem_addr), "l"(descQ), "l"(descK), "r"(idesc), "r"(accumulate) : "memory");
        }
        mbar_commit_mma(mma_bar);
      }
      mbar_wait(mma_bar, mbar_phase); mbar_phase ^= 1;
      tmem_readout_to_smem_vec(sS, tmem_addr, Br, Bc, Bc, scale_l2e);   // prescale by log2e here
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
      __syncthreads();
    }

    // online-softmax stats + unnormalized P, base-2 domain (exp2 via ex2.approx)
    if(tid < Br){
      const float m_old = sm[tid];
      const float l_old = sl[tid];

      // 3-way max tree over the row (fuses to FMNMX3 on sm_100+).
      float tile_max = -INFINITY;
      int j = 0;
      for(; j + 2 < Bc; j += 3)
        tile_max = fmaxf(tile_max,
                         fmaxf(sS[tid * Bc + j],
                               fmaxf(sS[tid * Bc + j + 1], sS[tid * Bc + j + 2])));
      for(; j < Bc; ++j) tile_max = fmaxf(tile_max, sS[tid * Bc + j]);

      const float m_new = fmaxf(m_old, tile_max);
      const float corr  = ex2_approx(m_old - m_new);   // == exp(m_old - m_new)

      // Pack two probabilities per F2FP (F2FP.BF16.F32.PACK_AB) + one 32-bit STS.
      float p_sum = 0.0f;
      for(int j2 = 0; j2 < Bc; j2 += 2){
        const float p0 = ex2_approx(sS[tid * Bc + j2]     - m_new);   // == exp(S - m_new)
        const float p1 = ex2_approx(sS[tid * Bc + j2 + 1] - m_new);
        *reinterpret_cast<__nv_bfloat162*>(&sP[canon_idx(tid, j2, Br)]) =
            __floats2bfloat162_rn(p0, p1);
        p_sum += p0 + p1;
      }
      sm[tid] = m_new; sl[tid] = l_old * corr + p_sum; sCorr[tid] = corr;
    }
    __syncthreads();

    for(int i = tid; i < Br * D; i += blockDim.x) sO[i] *= sCorr[i / D];
    __syncthreads();

    // O_tile = P @ V, readout ACCUMULATED straight into sO (no sPV staging).
    {
      const uint64_t descP_base = make_smem_desc(sP, Br);
      const uint64_t descV_base = make_smem_desc(sV, D);
      const uint32_t idesc      = make_idesc_bf16(Br, D);
      if(tid == 0){
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        for(int kt = 0; kt < Bc/16; ++kt){
          uint64_t descP = advance_desc_katom(descP_base, kt, Br);
          uint64_t descV = advance_desc_katom(descV_base, kt, D);
          uint32_t accumulate = (kt > 0) ? 1u : 0u;
          asm volatile(
            "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
            :: "r"(tmem_addr), "l"(descP), "l"(descV), "r"(idesc), "r"(accumulate) : "memory");
        }
        mbar_commit_mma(mma_bar);
      }
      mbar_wait(mma_bar, mbar_phase); mbar_phase ^= 1;
      tmem_readout_accum_vec(sO, tmem_addr, Br, D, D);   // sO += P@V, no staging hop
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
      __syncthreads();
    }
  } // end kv loop

  if(tid < 32)
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(tmem_addr), "r"(NCOLS) : "memory");
  __syncthreads();

  // Normalize + write O, packing two columns per bf16x2 store (D even -> same row denom).
  for(int i = 2 * tid; i < Br * D; i += 2 * blockDim.x){
    const float denom = sl[i / D];
    *reinterpret_cast<__nv_bfloat162*>(&d_O[qBase + i]) =
        __floats2bfloat162_rn(sO[i] / denom, sO[i + 1] / denom);
  }
  if(tid < Br)
    // m and l are in base-2 units: LSE_nat = ln2 * (m2 + log2(l)).
    d_LSE[lBase + tid] = 0.6931471805599453f * (sm[tid] + log2f(sl[tid]));

} // end of v11

// =================================
//  V12 : V11 + single-buffered TMA staging (smem diet step 2, enables Bc=128)
// =================================
// Same core as V11; only the KV load pipeline changes: the double-buffered TMA
// staging (+ next-tile prefetch) becomes a single buffer with the TMA issued at the
// top of each iteration. This deliberately gives up the load/compute overlap — V3's
// history says prefetch bought little when loads weren't the bottleneck — in exchange
// for the smem that lets the SAME kernel run at Bc=128 (~209KB static, fits), which is
// V13. Buffer reuse is safe: iteration kc's reorder reads of the staging buffer are
// fenced from iteration kc+1's TMA write by the end-of-loop __syncthreads.
template<int Br, int Bc, int D>
__global__ void gqa_v12(
  __nv_bfloat16 *d_Q,
  __nv_bfloat16 *d_O,
  float *d_LSE,
  const __grid_constant__ CUtensorMap Ktmap,   // K as flattened [B*Hkv*S, D]
  const __grid_constant__ CUtensorMap Vtmap,   // V as flattened [B*Hkv*S, D]
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
  const int tid    = threadIdx.x;

  const int q_row0   = q_tile * Br;
  const int nKVTiles = S / Bc;
  const int kvRow0   = (b * Hkv + hkv) * S;   // first K/V row of this head in the flat tensor

  const long qBase  = ((long)(b * Hq + hq) * S + q_row0) * D;
  const long lBase  = ((long)(b * Hq + hq) * S + q_row0);

  // Fold log2(e) into the score scale so QK^T lands in base-2 units (see V7 header).
  const float scale_l2e = scale * 1.4426950408889634f;   // scale * float32 log2(e)

  __shared__ __align__(16)  __nv_bfloat16 sQ[Br * D];
  __shared__ __align__(128) __nv_bfloat16 sKstage[Bc * D];     // TMA target, row-major, SINGLE buffer
  __shared__ __align__(128) __nv_bfloat16 sVstage[Bc * D];
  __shared__ __align__(16)  __nv_bfloat16 sK[Bc * D];          // canonical K-major (reorder output)
  __shared__ __align__(16)  __nv_bfloat16 sV[Bc * D];          // canonical [D,Bc] transposed
  __shared__ __align__(16)  __nv_bfloat16 sP[Br * Bc];
  __shared__ __align__(16)  float         sS[Br * Bc];
  __shared__ __align__(16)  float         sO[Br * D];          // P@V readout accumulates here directly
  __shared__ float sm[Br];
  __shared__ float sl[Br];
  __shared__ float sCorr[Br];
  __shared__ __align__(8) uint64_t s_mma_bar;
  __shared__ __align__(8) uint64_t s_load_bar;                 // single TMA completion barrier

  for(int i = tid; i < Br * D; i += blockDim.x){
    const int r = i / D, c = i % D;
    sQ[canon_idx(r, c, Br)] = d_Q[qBase + i];
    sO[i] = 0.0f;
  }
  if(tid < Br){ sm[tid] = -INFINITY; sl[tid] = 0.0f; }

  const uint32_t mma_bar = (uint32_t)__cvta_generic_to_shared(&s_mma_bar);
  const uint32_t lbar    = (uint32_t)__cvta_generic_to_shared(&s_load_bar);
  if(tid == 0){ mbar_init(mma_bar, 1); mbar_init(lbar, 1); }
  __syncthreads();

  constexpr uint32_t NCOLS = (Bc > D) ? (uint32_t)Bc : (uint32_t)D;
  static_assert(NCOLS >= 32 && (NCOLS & (NCOLS - 1)) == 0,
                "tcgen05 column count must be a power of two >= 32");

  uint32_t tmem_addr;
  {
    __shared__ uint32_t s_tmem_addr;
    if(tid < 32){
      uint32_t s_addr = (uint32_t)__cvta_generic_to_shared(&s_tmem_addr);
      asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                   :: "r"(s_addr), "r"(NCOLS) : "memory");
      asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
    __syncthreads();
    tmem_addr = s_tmem_addr;
  }

  int mbar_phase = 0;
  int load_phase = 0;
  const uint32_t TX = 2u * (uint32_t)Bc * (uint32_t)D * (uint32_t)sizeof(__nv_bfloat16); // K + V box bytes

  const uint64_t descQ_base = make_smem_desc(sQ, Br);

  for(int kc = 0; kc < nKVTiles; ++kc){
    // TMA this tile into the single staging buffer (no prefetch — see header).
    if(tid == 0){
      const int r = kvRow0 + kc * Bc;
      mbar_expect_tx(lbar, TX);
      tma_load_2d((uint32_t)__cvta_generic_to_shared(sKstage), &Ktmap, 0, r, lbar);
      tma_load_2d((uint32_t)__cvta_generic_to_shared(sVstage), &Vtmap, 0, r, lbar);
      mbar_arrive(lbar);
    }
    mbar_wait(lbar, load_phase); load_phase ^= 1;
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    __syncthreads();

    for(int i = tid; i < Bc * D; i += blockDim.x){
      const int bc = i / D, d = i % D;
      sK[canon_idx(bc, d, Bc)] = sKstage[i];   // K-major
      sV[canon_idx(d, bc, D)]  = sVstage[i];   // transposed
    }
    __syncthreads();

    // S2 = (Q @ K^T) * (scale*log2e) -> sS[Br, Bc]  (scores already in base-2 units)
    {
      const uint64_t descK_base = make_smem_desc(sK, Bc);
      const uint32_t idesc      = make_idesc_bf16(Br, Bc);
      if(tid == 0){
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        for(int kt = 0; kt < D/16; ++kt){
          uint64_t descQ = advance_desc_katom(descQ_base, kt, Br);
          uint64_t descK = advance_desc_katom(descK_base, kt, Bc);
          uint32_t accumulate = (kt > 0) ? 1u : 0u;
          asm volatile(
            "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
            :: "r"(tmem_addr), "l"(descQ), "l"(descK), "r"(idesc), "r"(accumulate) : "memory");
        }
        mbar_commit_mma(mma_bar);
      }
      mbar_wait(mma_bar, mbar_phase); mbar_phase ^= 1;
      tmem_readout_to_smem_vec(sS, tmem_addr, Br, Bc, Bc, scale_l2e);   // prescale by log2e here
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
      __syncthreads();
    }

    // online-softmax stats + unnormalized P, base-2 domain (exp2 via ex2.approx)
    if(tid < Br){
      const float m_old = sm[tid];
      const float l_old = sl[tid];

      // 3-way max tree over the row (fuses to FMNMX3 on sm_100+).
      float tile_max = -INFINITY;
      int j = 0;
      for(; j + 2 < Bc; j += 3)
        tile_max = fmaxf(tile_max,
                         fmaxf(sS[tid * Bc + j],
                               fmaxf(sS[tid * Bc + j + 1], sS[tid * Bc + j + 2])));
      for(; j < Bc; ++j) tile_max = fmaxf(tile_max, sS[tid * Bc + j]);

      const float m_new = fmaxf(m_old, tile_max);
      const float corr  = ex2_approx(m_old - m_new);   // == exp(m_old - m_new)

      // Pack two probabilities per F2FP (F2FP.BF16.F32.PACK_AB) + one 32-bit STS.
      float p_sum = 0.0f;
      for(int j2 = 0; j2 < Bc; j2 += 2){
        const float p0 = ex2_approx(sS[tid * Bc + j2]     - m_new);   // == exp(S - m_new)
        const float p1 = ex2_approx(sS[tid * Bc + j2 + 1] - m_new);
        *reinterpret_cast<__nv_bfloat162*>(&sP[canon_idx(tid, j2, Br)]) =
            __floats2bfloat162_rn(p0, p1);
        p_sum += p0 + p1;
      }
      sm[tid] = m_new; sl[tid] = l_old * corr + p_sum; sCorr[tid] = corr;
    }
    __syncthreads();

    for(int i = tid; i < Br * D; i += blockDim.x) sO[i] *= sCorr[i / D];
    __syncthreads();

    // O_tile = P @ V, readout ACCUMULATED straight into sO (no sPV staging).
    {
      const uint64_t descP_base = make_smem_desc(sP, Br);
      const uint64_t descV_base = make_smem_desc(sV, D);
      const uint32_t idesc      = make_idesc_bf16(Br, D);
      if(tid == 0){
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        for(int kt = 0; kt < Bc/16; ++kt){
          uint64_t descP = advance_desc_katom(descP_base, kt, Br);
          uint64_t descV = advance_desc_katom(descV_base, kt, D);
          uint32_t accumulate = (kt > 0) ? 1u : 0u;
          asm volatile(
            "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
            :: "r"(tmem_addr), "l"(descP), "l"(descV), "r"(idesc), "r"(accumulate) : "memory");
        }
        mbar_commit_mma(mma_bar);
      }
      mbar_wait(mma_bar, mbar_phase); mbar_phase ^= 1;
      tmem_readout_accum_vec(sO, tmem_addr, Br, D, D);   // sO += P@V, no staging hop
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
      __syncthreads();
    }
  } // end kv loop

  if(tid < 32)
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(tmem_addr), "r"(NCOLS) : "memory");
  __syncthreads();

  // Normalize + write O, packing two columns per bf16x2 store (D even -> same row denom).
  for(int i = 2 * tid; i < Br * D; i += 2 * blockDim.x){
    const float denom = sl[i / D];
    *reinterpret_cast<__nv_bfloat162*>(&d_O[qBase + i]) =
        __floats2bfloat162_rn(sO[i] / denom, sO[i + 1] / denom);
  }
  if(tid < Br)
    // m and l are in base-2 units: LSE_nat = ln2 * (m2 + log2(l)).
    d_LSE[lBase + tid] = 0.6931471805599453f * (sm[tid] + log2f(sl[tid]));

} // end of v12

// =================================
//  V14 : V13 (V12 kernel @ Bc=128) + cluster launch scaffolding (Stage 4, step 1/3)
// =================================
// Identical body to gqa_v12; the ONLY change is __cluster_dims__(1,1,2): CTAs pair up
// along gridDim.z, so a cluster = two adjacent q-tiles of the SAME (batch, head) —
// both CTAs of a pair consume the identical K/V stream, which is what V15's multicast
// and V16's cta_group::2 MMA build on. All tcgen05 ops stay cta_group::1 and all
// buffers/sync stay CTA-local, so this isolates the pure cost/effect of clustered
// scheduling (expected ~neutral). Grid divisibility: S/Br = 32 is even.
template<int Br, int Bc, int D>
__global__ void __cluster_dims__(1, 1, 2) gqa_v14(
  __nv_bfloat16 *d_Q,
  __nv_bfloat16 *d_O,
  float *d_LSE,
  const __grid_constant__ CUtensorMap Ktmap,   // K as flattened [B*Hkv*S, D]
  const __grid_constant__ CUtensorMap Vtmap,   // V as flattened [B*Hkv*S, D]
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
  const int tid    = threadIdx.x;

  const int q_row0   = q_tile * Br;
  const int nKVTiles = S / Bc;
  const int kvRow0   = (b * Hkv + hkv) * S;   // first K/V row of this head in the flat tensor

  const long qBase  = ((long)(b * Hq + hq) * S + q_row0) * D;
  const long lBase  = ((long)(b * Hq + hq) * S + q_row0);

  // Fold log2(e) into the score scale so QK^T lands in base-2 units (see V7 header).
  const float scale_l2e = scale * 1.4426950408889634f;   // scale * float32 log2(e)

  __shared__ __align__(16)  __nv_bfloat16 sQ[Br * D];
  __shared__ __align__(128) __nv_bfloat16 sKstage[Bc * D];     // TMA target, row-major, SINGLE buffer
  __shared__ __align__(128) __nv_bfloat16 sVstage[Bc * D];
  __shared__ __align__(16)  __nv_bfloat16 sK[Bc * D];          // canonical K-major (reorder output)
  __shared__ __align__(16)  __nv_bfloat16 sV[Bc * D];          // canonical [D,Bc] transposed
  __shared__ __align__(16)  __nv_bfloat16 sP[Br * Bc];
  __shared__ __align__(16)  float         sS[Br * Bc];
  __shared__ __align__(16)  float         sO[Br * D];          // P@V readout accumulates here directly
  __shared__ float sm[Br];
  __shared__ float sl[Br];
  __shared__ float sCorr[Br];
  __shared__ __align__(8) uint64_t s_mma_bar;
  __shared__ __align__(8) uint64_t s_load_bar;                 // single TMA completion barrier

  for(int i = tid; i < Br * D; i += blockDim.x){
    const int r = i / D, c = i % D;
    sQ[canon_idx(r, c, Br)] = d_Q[qBase + i];
    sO[i] = 0.0f;
  }
  if(tid < Br){ sm[tid] = -INFINITY; sl[tid] = 0.0f; }

  const uint32_t mma_bar = (uint32_t)__cvta_generic_to_shared(&s_mma_bar);
  const uint32_t lbar    = (uint32_t)__cvta_generic_to_shared(&s_load_bar);
  if(tid == 0){ mbar_init(mma_bar, 1); mbar_init(lbar, 1); }
  __syncthreads();

  constexpr uint32_t NCOLS = (Bc > D) ? (uint32_t)Bc : (uint32_t)D;
  static_assert(NCOLS >= 32 && (NCOLS & (NCOLS - 1)) == 0,
                "tcgen05 column count must be a power of two >= 32");

  uint32_t tmem_addr;
  {
    __shared__ uint32_t s_tmem_addr;
    if(tid < 32){
      uint32_t s_addr = (uint32_t)__cvta_generic_to_shared(&s_tmem_addr);
      asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                   :: "r"(s_addr), "r"(NCOLS) : "memory");
      asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
    __syncthreads();
    tmem_addr = s_tmem_addr;
  }

  int mbar_phase = 0;
  int load_phase = 0;
  const uint32_t TX = 2u * (uint32_t)Bc * (uint32_t)D * (uint32_t)sizeof(__nv_bfloat16); // K + V box bytes

  const uint64_t descQ_base = make_smem_desc(sQ, Br);

  for(int kc = 0; kc < nKVTiles; ++kc){
    // TMA this tile into the single staging buffer (per-CTA, as V12/V13).
    if(tid == 0){
      const int r = kvRow0 + kc * Bc;
      mbar_expect_tx(lbar, TX);
      tma_load_2d((uint32_t)__cvta_generic_to_shared(sKstage), &Ktmap, 0, r, lbar);
      tma_load_2d((uint32_t)__cvta_generic_to_shared(sVstage), &Vtmap, 0, r, lbar);
      mbar_arrive(lbar);
    }
    mbar_wait(lbar, load_phase); load_phase ^= 1;
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    __syncthreads();

    for(int i = tid; i < Bc * D; i += blockDim.x){
      const int bc = i / D, d = i % D;
      sK[canon_idx(bc, d, Bc)] = sKstage[i];   // K-major
      sV[canon_idx(d, bc, D)]  = sVstage[i];   // transposed
    }
    __syncthreads();

    // S2 = (Q @ K^T) * (scale*log2e) -> sS[Br, Bc]  (scores already in base-2 units)
    {
      const uint64_t descK_base = make_smem_desc(sK, Bc);
      const uint32_t idesc      = make_idesc_bf16(Br, Bc);
      if(tid == 0){
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        for(int kt = 0; kt < D/16; ++kt){
          uint64_t descQ = advance_desc_katom(descQ_base, kt, Br);
          uint64_t descK = advance_desc_katom(descK_base, kt, Bc);
          uint32_t accumulate = (kt > 0) ? 1u : 0u;
          asm volatile(
            "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
            :: "r"(tmem_addr), "l"(descQ), "l"(descK), "r"(idesc), "r"(accumulate) : "memory");
        }
        mbar_commit_mma(mma_bar);
      }
      mbar_wait(mma_bar, mbar_phase); mbar_phase ^= 1;
      tmem_readout_to_smem_vec(sS, tmem_addr, Br, Bc, Bc, scale_l2e);   // prescale by log2e here
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
      __syncthreads();
    }

    // online-softmax stats + unnormalized P, base-2 domain (exp2 via ex2.approx)
    if(tid < Br){
      const float m_old = sm[tid];
      const float l_old = sl[tid];

      // 3-way max tree over the row (fuses to FMNMX3 on sm_100+).
      float tile_max = -INFINITY;
      int j = 0;
      for(; j + 2 < Bc; j += 3)
        tile_max = fmaxf(tile_max,
                         fmaxf(sS[tid * Bc + j],
                               fmaxf(sS[tid * Bc + j + 1], sS[tid * Bc + j + 2])));
      for(; j < Bc; ++j) tile_max = fmaxf(tile_max, sS[tid * Bc + j]);

      const float m_new = fmaxf(m_old, tile_max);
      const float corr  = ex2_approx(m_old - m_new);   // == exp(m_old - m_new)

      // Pack two probabilities per F2FP (F2FP.BF16.F32.PACK_AB) + one 32-bit STS.
      float p_sum = 0.0f;
      for(int j2 = 0; j2 < Bc; j2 += 2){
        const float p0 = ex2_approx(sS[tid * Bc + j2]     - m_new);   // == exp(S - m_new)
        const float p1 = ex2_approx(sS[tid * Bc + j2 + 1] - m_new);
        *reinterpret_cast<__nv_bfloat162*>(&sP[canon_idx(tid, j2, Br)]) =
            __floats2bfloat162_rn(p0, p1);
        p_sum += p0 + p1;
      }
      sm[tid] = m_new; sl[tid] = l_old * corr + p_sum; sCorr[tid] = corr;
    }
    __syncthreads();

    for(int i = tid; i < Br * D; i += blockDim.x) sO[i] *= sCorr[i / D];
    __syncthreads();

    // O_tile = P @ V, readout ACCUMULATED straight into sO (no sPV staging).
    {
      const uint64_t descP_base = make_smem_desc(sP, Br);
      const uint64_t descV_base = make_smem_desc(sV, D);
      const uint32_t idesc      = make_idesc_bf16(Br, D);
      if(tid == 0){
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        for(int kt = 0; kt < Bc/16; ++kt){
          uint64_t descP = advance_desc_katom(descP_base, kt, Br);
          uint64_t descV = advance_desc_katom(descV_base, kt, D);
          uint32_t accumulate = (kt > 0) ? 1u : 0u;
          asm volatile(
            "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
            :: "r"(tmem_addr), "l"(descP), "l"(descV), "r"(idesc), "r"(accumulate) : "memory");
        }
        mbar_commit_mma(mma_bar);
      }
      mbar_wait(mma_bar, mbar_phase); mbar_phase ^= 1;
      tmem_readout_accum_vec(sO, tmem_addr, Br, D, D);   // sO += P@V, no staging hop
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
      __syncthreads();
    }
  } // end kv loop

  if(tid < 32)
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(tmem_addr), "r"(NCOLS) : "memory");
  __syncthreads();

  // Normalize + write O, packing two columns per bf16x2 store (D even -> same row denom).
  for(int i = 2 * tid; i < Br * D; i += 2 * blockDim.x){
    const float denom = sl[i / D];
    *reinterpret_cast<__nv_bfloat162*>(&d_O[qBase + i]) =
        __floats2bfloat162_rn(sO[i] / denom, sO[i + 1] / denom);
  }
  if(tid < Br)
    // m and l are in base-2 units: LSE_nat = ln2 * (m2 + log2(l)).
    d_LSE[lBase + tid] = 0.6931471805599453f * (sm[tid] + log2f(sl[tid]));

} // end of v14

// =================================
//  V15 : V14 + multicast K/V TMA (Stage 4, step 2/3)
// =================================
// The two CTAs of a cluster process adjacent q-tiles of the SAME head, so they need
// the IDENTICAL K/V tile each iteration. Cluster rank 0 issues ONE multicast TMA per
// K/V tile (ctaMask 0b11): the tile lands at the same smem offset in BOTH CTAs'
// staging, and each CTA's own load mbarrier receives the transaction bytes. K/V HBM
// requests halve. MMAs stay cta_group::1; reorder/compute stay CTA-local.
// Sync protocol per tile:
//   1. each CTA's tid0: expect_tx on its LOCAL barrier
//   2. cluster.sync()  — REQUIRED: (a) WAR — rank0's multicast writes the PEER's
//      staging, so the peer must be done reading it (prev reorder); (b) both expects
//      are set before any completion bytes can arrive
//   3. rank0 tid0 issues the two multicast TMAs
//   4. each CTA's tid0: local arrive (its barrier counts 1 arrival + TX bytes)
//   5. each CTA waits its local barrier, then proceeds CTA-locally as before
__device__ __forceinline__ void tma_load_2d_mc(uint32_t smem_addr, const void* tmap,
                                               int c, int r, uint32_t bar, uint16_t ctaMask){
  asm volatile(
    "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster"
    " [%0], [%1, {%2, %3}], [%4], %5;"
    :: "r"(smem_addr), "l"(tmap), "r"(c), "r"(r), "r"(bar), "h"(ctaMask) : "memory");
}

template<int Br, int Bc, int D>
__global__ void __cluster_dims__(1, 1, 2) gqa_v15(
  __nv_bfloat16 *d_Q,
  __nv_bfloat16 *d_O,
  float *d_LSE,
  const __grid_constant__ CUtensorMap Ktmap,   // K as flattened [B*Hkv*S, D]
  const __grid_constant__ CUtensorMap Vtmap,   // V as flattened [B*Hkv*S, D]
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
  const int tid    = threadIdx.x;

  cg::cluster_group cluster = cg::this_cluster();
  const unsigned int crank  = cluster.block_rank();   // 0 or 1 within the pair

  const int q_row0   = q_tile * Br;
  const int nKVTiles = S / Bc;
  const int kvRow0   = (b * Hkv + hkv) * S;   // identical for both CTAs of the cluster

  const long qBase  = ((long)(b * Hq + hq) * S + q_row0) * D;
  const long lBase  = ((long)(b * Hq + hq) * S + q_row0);

  // Fold log2(e) into the score scale so QK^T lands in base-2 units (see V7 header).
  const float scale_l2e = scale * 1.4426950408889634f;   // scale * float32 log2(e)

  __shared__ __align__(16)  __nv_bfloat16 sQ[Br * D];
  __shared__ __align__(128) __nv_bfloat16 sKstage[Bc * D];     // multicast TMA target (both CTAs)
  __shared__ __align__(128) __nv_bfloat16 sVstage[Bc * D];
  __shared__ __align__(16)  __nv_bfloat16 sK[Bc * D];          // canonical K-major (reorder output)
  __shared__ __align__(16)  __nv_bfloat16 sV[Bc * D];          // canonical [D,Bc] transposed
  __shared__ __align__(16)  __nv_bfloat16 sP[Br * Bc];
  __shared__ __align__(16)  float         sS[Br * Bc];
  __shared__ __align__(16)  float         sO[Br * D];          // P@V readout accumulates here directly
  __shared__ float sm[Br];
  __shared__ float sl[Br];
  __shared__ float sCorr[Br];
  __shared__ __align__(8) uint64_t s_mma_bar;
  __shared__ __align__(8) uint64_t s_load_bar;                 // single TMA completion barrier

  for(int i = tid; i < Br * D; i += blockDim.x){
    const int r = i / D, c = i % D;
    sQ[canon_idx(r, c, Br)] = d_Q[qBase + i];
    sO[i] = 0.0f;
  }
  if(tid < Br){ sm[tid] = -INFINITY; sl[tid] = 0.0f; }

  const uint32_t mma_bar = (uint32_t)__cvta_generic_to_shared(&s_mma_bar);
  const uint32_t lbar    = (uint32_t)__cvta_generic_to_shared(&s_load_bar);
  if(tid == 0){ mbar_init(mma_bar, 1); mbar_init(lbar, 1); }
  __syncthreads();

  constexpr uint32_t NCOLS = (Bc > D) ? (uint32_t)Bc : (uint32_t)D;
  static_assert(NCOLS >= 32 && (NCOLS & (NCOLS - 1)) == 0,
                "tcgen05 column count must be a power of two >= 32");

  uint32_t tmem_addr;
  {
    __shared__ uint32_t s_tmem_addr;
    if(tid < 32){
      uint32_t s_addr = (uint32_t)__cvta_generic_to_shared(&s_tmem_addr);
      asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                   :: "r"(s_addr), "r"(NCOLS) : "memory");
      asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
    __syncthreads();
    tmem_addr = s_tmem_addr;
  }

  int mbar_phase = 0;
  int load_phase = 0;
  const uint32_t TX = 2u * (uint32_t)Bc * (uint32_t)D * (uint32_t)sizeof(__nv_bfloat16); // K + V box bytes

  const uint64_t descQ_base = make_smem_desc(sQ, Br);

  for(int kc = 0; kc < nKVTiles; ++kc){
    // Multicast this K/V tile into BOTH CTAs' staging (see protocol in the header).
    if(tid == 0) mbar_expect_tx(lbar, TX);
    cluster.sync();
    if(crank == 0 && tid == 0){
      const int r = kvRow0 + kc * Bc;
      tma_load_2d_mc((uint32_t)__cvta_generic_to_shared(sKstage), &Ktmap, 0, r, lbar, 0b11);
      tma_load_2d_mc((uint32_t)__cvta_generic_to_shared(sVstage), &Vtmap, 0, r, lbar, 0b11);
    }
    if(tid == 0) mbar_arrive(lbar);
    mbar_wait(lbar, load_phase); load_phase ^= 1;
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    __syncthreads();

    for(int i = tid; i < Bc * D; i += blockDim.x){
      const int bc = i / D, d = i % D;
      sK[canon_idx(bc, d, Bc)] = sKstage[i];   // K-major
      sV[canon_idx(d, bc, D)]  = sVstage[i];   // transposed
    }
    __syncthreads();

    // S2 = (Q @ K^T) * (scale*log2e) -> sS[Br, Bc]  (scores already in base-2 units)
    {
      const uint64_t descK_base = make_smem_desc(sK, Bc);
      const uint32_t idesc      = make_idesc_bf16(Br, Bc);
      if(tid == 0){
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        for(int kt = 0; kt < D/16; ++kt){
          uint64_t descQ = advance_desc_katom(descQ_base, kt, Br);
          uint64_t descK = advance_desc_katom(descK_base, kt, Bc);
          uint32_t accumulate = (kt > 0) ? 1u : 0u;
          asm volatile(
            "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
            :: "r"(tmem_addr), "l"(descQ), "l"(descK), "r"(idesc), "r"(accumulate) : "memory");
        }
        mbar_commit_mma(mma_bar);
      }
      mbar_wait(mma_bar, mbar_phase); mbar_phase ^= 1;
      tmem_readout_to_smem_vec(sS, tmem_addr, Br, Bc, Bc, scale_l2e);   // prescale by log2e here
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
      __syncthreads();
    }

    // online-softmax stats + unnormalized P, base-2 domain (exp2 via ex2.approx)
    if(tid < Br){
      const float m_old = sm[tid];
      const float l_old = sl[tid];

      // 3-way max tree over the row (fuses to FMNMX3 on sm_100+).
      float tile_max = -INFINITY;
      int j = 0;
      for(; j + 2 < Bc; j += 3)
        tile_max = fmaxf(tile_max,
                         fmaxf(sS[tid * Bc + j],
                               fmaxf(sS[tid * Bc + j + 1], sS[tid * Bc + j + 2])));
      for(; j < Bc; ++j) tile_max = fmaxf(tile_max, sS[tid * Bc + j]);

      const float m_new = fmaxf(m_old, tile_max);
      const float corr  = ex2_approx(m_old - m_new);   // == exp(m_old - m_new)

      // Pack two probabilities per F2FP (F2FP.BF16.F32.PACK_AB) + one 32-bit STS.
      float p_sum = 0.0f;
      for(int j2 = 0; j2 < Bc; j2 += 2){
        const float p0 = ex2_approx(sS[tid * Bc + j2]     - m_new);   // == exp(S - m_new)
        const float p1 = ex2_approx(sS[tid * Bc + j2 + 1] - m_new);
        *reinterpret_cast<__nv_bfloat162*>(&sP[canon_idx(tid, j2, Br)]) =
            __floats2bfloat162_rn(p0, p1);
        p_sum += p0 + p1;
      }
      sm[tid] = m_new; sl[tid] = l_old * corr + p_sum; sCorr[tid] = corr;
    }
    __syncthreads();

    for(int i = tid; i < Br * D; i += blockDim.x) sO[i] *= sCorr[i / D];
    __syncthreads();

    // O_tile = P @ V, readout ACCUMULATED straight into sO (no sPV staging).
    {
      const uint64_t descP_base = make_smem_desc(sP, Br);
      const uint64_t descV_base = make_smem_desc(sV, D);
      const uint32_t idesc      = make_idesc_bf16(Br, D);
      if(tid == 0){
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        for(int kt = 0; kt < Bc/16; ++kt){
          uint64_t descP = advance_desc_katom(descP_base, kt, Br);
          uint64_t descV = advance_desc_katom(descV_base, kt, D);
          uint32_t accumulate = (kt > 0) ? 1u : 0u;
          asm volatile(
            "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
            :: "r"(tmem_addr), "l"(descP), "l"(descV), "r"(idesc), "r"(accumulate) : "memory");
        }
        mbar_commit_mma(mma_bar);
      }
      mbar_wait(mma_bar, mbar_phase); mbar_phase ^= 1;
      tmem_readout_accum_vec(sO, tmem_addr, Br, D, D);   // sO += P@V, no staging hop
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
      __syncthreads();
    }
  } // end kv loop

  // The multicast source (rank 0) must not exit while its peer might still have TMA
  // traffic outstanding toward it; a final cluster sync keeps the pair's lifetimes tied.
  cluster.sync();

  if(tid < 32)
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(tmem_addr), "r"(NCOLS) : "memory");
  __syncthreads();

  // Normalize + write O, packing two columns per bf16x2 store (D even -> same row denom).
  for(int i = 2 * tid; i < Br * D; i += 2 * blockDim.x){
    const float denom = sl[i / D];
    *reinterpret_cast<__nv_bfloat162*>(&d_O[qBase + i]) =
        __floats2bfloat162_rn(sO[i] / denom, sO[i + 1] / denom);
  }
  if(tid < Br)
    // m and l are in base-2 units: LSE_nat = ln2 * (m2 + log2(l)).
    d_LSE[lBase + tid] = 0.6931471805599453f * (sm[tid] + log2f(sl[tid]));

} // end of v15

// =================================
//  V16 : V13 core + persistent kernel (Stage 5)
// =================================
// Stage-4 verdict: V14 (clustering) was free but V15 (multicast) regressed — the
// pair's K/V reread was already an L2 hit, so cluster coupling cost more than it
// saved. cta_group::2 was skipped on that evidence. V16 instead removes per-block
// overhead with zero coupling: launch ONE resident CTA per SM and loop over work
// items (b, hq, q_tile) with static striding (uniform cost per item — every item
// sweeps the full S, so a dynamic atomic queue would only add contention).
// Hoisted out of the work loop: TMEM alloc/dealloc, mbarrier inits, Q descriptor —
// ~3072 per-block setups collapse to ~#SM. Work index order is b-fastest, then hq,
// then q_tile, so concurrently-resident SMs share few distinct KV streams (all G
// heads of one KV head adjacent) for L2 reuse. mbarrier parity phases simply keep
// running across work items. One extra __syncthreads at the end of each work item
// keeps the next item's sO=0 from racing the epilogue's sO reads.
template<int Br, int Bc, int D>
__global__ void gqa_v16(
  __nv_bfloat16 *d_Q,
  __nv_bfloat16 *d_O,
  float *d_LSE,
  const __grid_constant__ CUtensorMap Ktmap,   // K as flattened [B*Hkv*S, D]
  const __grid_constant__ CUtensorMap Vtmap,   // V as flattened [B*Hkv*S, D]
  int B,
  int Hq,
  int Hkv,
  int G,
  int S,
  float scale
){
  const int tid       = threadIdx.x;
  const int nKVTiles  = S / Bc;
  const int totalWork = B * Hq * (S / Br);   // one item = one (b, hq, q_tile)

  // Fold log2(e) into the score scale so QK^T lands in base-2 units (see V7 header).
  const float scale_l2e = scale * 1.4426950408889634f;   // scale * float32 log2(e)

  __shared__ __align__(16)  __nv_bfloat16 sQ[Br * D];
  __shared__ __align__(128) __nv_bfloat16 sKstage[Bc * D];     // TMA target, row-major, SINGLE buffer
  __shared__ __align__(128) __nv_bfloat16 sVstage[Bc * D];
  __shared__ __align__(16)  __nv_bfloat16 sK[Bc * D];          // canonical K-major (reorder output)
  __shared__ __align__(16)  __nv_bfloat16 sV[Bc * D];          // canonical [D,Bc] transposed
  __shared__ __align__(16)  __nv_bfloat16 sP[Br * Bc];
  __shared__ __align__(16)  float         sS[Br * Bc];
  __shared__ __align__(16)  float         sO[Br * D];          // P@V readout accumulates here directly
  __shared__ float sm[Br];
  __shared__ float sl[Br];
  __shared__ float sCorr[Br];
  __shared__ __align__(8) uint64_t s_mma_bar;
  __shared__ __align__(8) uint64_t s_load_bar;                 // single TMA completion barrier

  // ---- One-time per-block setup (was per-tile-of-work in V13) ----
  const uint32_t mma_bar = (uint32_t)__cvta_generic_to_shared(&s_mma_bar);
  const uint32_t lbar    = (uint32_t)__cvta_generic_to_shared(&s_load_bar);
  if(tid == 0){ mbar_init(mma_bar, 1); mbar_init(lbar, 1); }
  __syncthreads();

  constexpr uint32_t NCOLS = (Bc > D) ? (uint32_t)Bc : (uint32_t)D;
  static_assert(NCOLS >= 32 && (NCOLS & (NCOLS - 1)) == 0,
                "tcgen05 column count must be a power of two >= 32");

  uint32_t tmem_addr;
  {
    __shared__ uint32_t s_tmem_addr;
    if(tid < 32){
      uint32_t s_addr = (uint32_t)__cvta_generic_to_shared(&s_tmem_addr);
      asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                   :: "r"(s_addr), "r"(NCOLS) : "memory");
      asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
    __syncthreads();
    tmem_addr = s_tmem_addr;
  }

  int mbar_phase = 0;   // parity phases persist across work items
  int load_phase = 0;
  const uint32_t TX = 2u * (uint32_t)Bc * (uint32_t)D * (uint32_t)sizeof(__nv_bfloat16); // K + V box bytes

  const uint64_t descQ_base = make_smem_desc(sQ, Br);   // sQ address is fixed -> hoisted

  // ---- Persistent work loop: b fastest, then hq, then q_tile (L2-friendly) ----
  for(int w = blockIdx.x; w < totalWork; w += gridDim.x){
    const int b      = w % B;
    const int hq     = (w / B) % Hq;
    const int q_tile = w / (B * Hq);
    const int hkv    = hq / G;

    const int q_row0 = q_tile * Br;
    const int kvRow0 = (b * Hkv + hkv) * S;

    const long qBase = ((long)(b * Hq + hq) * S + q_row0) * D;
    const long lBase = ((long)(b * Hq + hq) * S + q_row0);

    // Per-work-item state init (prior item's epilogue finished: barrier at loop end).
    for(int i = tid; i < Br * D; i += blockDim.x){
      const int r = i / D, c = i % D;
      sQ[canon_idx(r, c, Br)] = d_Q[qBase + i];
      sO[i] = 0.0f;
    }
    if(tid < Br){ sm[tid] = -INFINITY; sl[tid] = 0.0f; }
    __syncthreads();

    for(int kc = 0; kc < nKVTiles; ++kc){
      // TMA this tile into the single staging buffer.
      if(tid == 0){
        const int r = kvRow0 + kc * Bc;
        mbar_expect_tx(lbar, TX);
        tma_load_2d((uint32_t)__cvta_generic_to_shared(sKstage), &Ktmap, 0, r, lbar);
        tma_load_2d((uint32_t)__cvta_generic_to_shared(sVstage), &Vtmap, 0, r, lbar);
        mbar_arrive(lbar);
      }
      mbar_wait(lbar, load_phase); load_phase ^= 1;
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      __syncthreads();

      for(int i = tid; i < Bc * D; i += blockDim.x){
        const int bc = i / D, d = i % D;
        sK[canon_idx(bc, d, Bc)] = sKstage[i];   // K-major
        sV[canon_idx(d, bc, D)]  = sVstage[i];   // transposed
      }
      __syncthreads();

      // S2 = (Q @ K^T) * (scale*log2e) -> sS[Br, Bc]  (scores already in base-2 units)
      {
        const uint64_t descK_base = make_smem_desc(sK, Bc);
        const uint32_t idesc      = make_idesc_bf16(Br, Bc);
        if(tid == 0){
          asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
          for(int kt = 0; kt < D/16; ++kt){
            uint64_t descQ = advance_desc_katom(descQ_base, kt, Br);
            uint64_t descK = advance_desc_katom(descK_base, kt, Bc);
            uint32_t accumulate = (kt > 0) ? 1u : 0u;
            asm volatile(
              "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
              "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
              :: "r"(tmem_addr), "l"(descQ), "l"(descK), "r"(idesc), "r"(accumulate) : "memory");
          }
          mbar_commit_mma(mma_bar);
        }
        mbar_wait(mma_bar, mbar_phase); mbar_phase ^= 1;
        tmem_readout_to_smem_vec(sS, tmem_addr, Br, Bc, Bc, scale_l2e);   // prescale by log2e here
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
        __syncthreads();
      }

      // online-softmax stats + unnormalized P, base-2 domain (exp2 via ex2.approx)
      if(tid < Br){
        const float m_old = sm[tid];
        const float l_old = sl[tid];

        // 3-way max tree over the row (fuses to FMNMX3 on sm_100+).
        float tile_max = -INFINITY;
        int j = 0;
        for(; j + 2 < Bc; j += 3)
          tile_max = fmaxf(tile_max,
                           fmaxf(sS[tid * Bc + j],
                                 fmaxf(sS[tid * Bc + j + 1], sS[tid * Bc + j + 2])));
        for(; j < Bc; ++j) tile_max = fmaxf(tile_max, sS[tid * Bc + j]);

        const float m_new = fmaxf(m_old, tile_max);
        const float corr  = ex2_approx(m_old - m_new);   // == exp(m_old - m_new)

        // Pack two probabilities per F2FP (F2FP.BF16.F32.PACK_AB) + one 32-bit STS.
        float p_sum = 0.0f;
        for(int j2 = 0; j2 < Bc; j2 += 2){
          const float p0 = ex2_approx(sS[tid * Bc + j2]     - m_new);   // == exp(S - m_new)
          const float p1 = ex2_approx(sS[tid * Bc + j2 + 1] - m_new);
          *reinterpret_cast<__nv_bfloat162*>(&sP[canon_idx(tid, j2, Br)]) =
              __floats2bfloat162_rn(p0, p1);
          p_sum += p0 + p1;
        }
        sm[tid] = m_new; sl[tid] = l_old * corr + p_sum; sCorr[tid] = corr;
      }
      __syncthreads();

      for(int i = tid; i < Br * D; i += blockDim.x) sO[i] *= sCorr[i / D];
      __syncthreads();

      // O_tile = P @ V, readout ACCUMULATED straight into sO (no sPV staging).
      {
        const uint64_t descP_base = make_smem_desc(sP, Br);
        const uint64_t descV_base = make_smem_desc(sV, D);
        const uint32_t idesc      = make_idesc_bf16(Br, D);
        if(tid == 0){
          asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
          for(int kt = 0; kt < Bc/16; ++kt){
            uint64_t descP = advance_desc_katom(descP_base, kt, Br);
            uint64_t descV = advance_desc_katom(descV_base, kt, D);
            uint32_t accumulate = (kt > 0) ? 1u : 0u;
            asm volatile(
              "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
              "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
              :: "r"(tmem_addr), "l"(descP), "l"(descV), "r"(idesc), "r"(accumulate) : "memory");
          }
          mbar_commit_mma(mma_bar);
        }
        mbar_wait(mma_bar, mbar_phase); mbar_phase ^= 1;
        tmem_readout_accum_vec(sO, tmem_addr, Br, D, D);   // sO += P@V, no staging hop
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
        __syncthreads();
      }
    } // end kv loop

    // Epilogue for this work item: normalize + write O (packed) and LSE.
    for(int i = 2 * tid; i < Br * D; i += 2 * blockDim.x){
      const float denom = sl[i / D];
      *reinterpret_cast<__nv_bfloat162*>(&d_O[qBase + i]) =
          __floats2bfloat162_rn(sO[i] / denom, sO[i + 1] / denom);
    }
    if(tid < Br)
      // m and l are in base-2 units: LSE_nat = ln2 * (m2 + log2(l)).
      d_LSE[lBase + tid] = 0.6931471805599453f * (sm[tid] + log2f(sl[tid]));

    // Next item's sQ/sO writes must not race this epilogue's sO/sm/sl reads.
    __syncthreads();
  } // end persistent work loop

  if(tid < 32)
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(tmem_addr), "r"(NCOLS) : "memory");

} // end of v16

// =================================
//  V17 : V13 core + warp specialization (Stage 6, step 1)
// =================================
// First warp-specialized version. V13 (and every kernel before it) is bulk-synchronous:
// the whole 128-thread block issues the TMA load, then ALL threads sit in mbar_wait for
// it, then ALL threads reorder/MMA/softmax/PV — so the TMA latency for tile kc is fully
// on the critical path every iteration, never overlapped with kc's own compute.
//
// V17 adds a 5th warp (threads 128..159) whose ONLY job is issuing the K/V TMA loads —
// a dedicated "producer" warp, decoupled from the 4 "consumer" warps (threads 0..127,
// unchanged from V13/gqa_v12: same tid, same TMEM-readout warp mapping, same math).
// Producer and consumer talk through two mbarriers instead of a block-wide __syncthreads:
//   s_load_bar : producer -> consumer  ("tile kc's bytes are in sKstage/sVstage")
//   s_free_bar : consumer -> producer  ("sKstage/sVstage are free, load kc+1")
// The consumer arrives s_free_bar right after the reorder copy drains the staging
// buffer into canonical sK/sV — BEFORE the QK^T MMA, softmax, and P@V for that tile even
// start. So the producer's TMA for kc+1 runs concurrently with the consumer's MMA/softmax/
// PV for kc, hiding TMA latency behind tensor-core + SIMT work instead of paying for it
// serially. The staging buffer itself stays SINGLE-buffered (same smem footprint as V13,
// still 1 CTA/SM) — the overlap comes purely from decoupling the wait, not from adding a
// second buffer (that's V18).
//
// Consumer threads (0..127) synchronize with each other via a NAMED barrier (`bar.sync 1,
// 128`) instead of `__syncthreads()`, since __syncthreads() would also block on the
// producer warp (which must be free to run ahead). The producer warp never touches that
// barrier id. Only the one-time setup (smem/mbarrier init, TMEM alloc/dealloc) and the
// final epilogue use a real block-wide __syncthreads() (all 160 threads).
__device__ __forceinline__ void consumer_sync(){
  asm volatile("bar.sync 1, 128;" ::: "memory");
}

template<int Br, int Bc, int D>
__global__ void gqa_v17(
  __nv_bfloat16 *d_Q,
  __nv_bfloat16 *d_O,
  float *d_LSE,
  const __grid_constant__ CUtensorMap Ktmap,   // K as flattened [B*Hkv*S, D]
  const __grid_constant__ CUtensorMap Vtmap,   // V as flattened [B*Hkv*S, D]
  int B,
  int Hq,
  int Hkv,
  int G,
  int S,
  float scale
){
  static_assert(Br == 128, "V17's consumer group is hardwired to 128 threads (TMEM readout needs warps 0-3)");

  const int b      = blockIdx.x;
  const int hq     = blockIdx.y;
  const int q_tile = blockIdx.z;
  const int hkv    = hq / G;
  const int tid    = threadIdx.x;              // 0..159: 0..127 consumer, 128..159 producer

  const int q_row0   = q_tile * Br;
  const int nKVTiles = S / Bc;
  const int kvRow0   = (b * Hkv + hkv) * S;   // first K/V row of this head in the flat tensor

  const long qBase  = ((long)(b * Hq + hq) * S + q_row0) * D;
  const long lBase  = ((long)(b * Hq + hq) * S + q_row0);

  // Fold log2(e) into the score scale so QK^T lands in base-2 units (see V7 header).
  const float scale_l2e = scale * 1.4426950408889634f;   // scale * float32 log2(e)

  __shared__ __align__(16)  __nv_bfloat16 sQ[Br * D];
  __shared__ __align__(128) __nv_bfloat16 sKstage[Bc * D];     // TMA target, row-major, SINGLE buffer
  __shared__ __align__(128) __nv_bfloat16 sVstage[Bc * D];
  __shared__ __align__(16)  __nv_bfloat16 sK[Bc * D];          // canonical K-major (reorder output)
  __shared__ __align__(16)  __nv_bfloat16 sV[Bc * D];          // canonical [D,Bc] transposed
  __shared__ __align__(16)  __nv_bfloat16 sP[Br * Bc];
  __shared__ __align__(16)  float         sS[Br * Bc];
  __shared__ __align__(16)  float         sO[Br * D];          // P@V readout accumulates here directly
  __shared__ float sm[Br];
  __shared__ float sl[Br];
  __shared__ float sCorr[Br];
  __shared__ __align__(8) uint64_t s_mma_bar;
  __shared__ __align__(8) uint64_t s_load_bar;   // producer -> consumer: tile bytes ready
  __shared__ __align__(8) uint64_t s_free_bar;   // consumer -> producer: staging buffer free

  for(int i = tid; i < Br * D; i += blockDim.x){
    const int r = i / D, c = i % D;
    sQ[canon_idx(r, c, Br)] = d_Q[qBase + i];
    sO[i] = 0.0f;
  }
  if(tid < Br){ sm[tid] = -INFINITY; sl[tid] = 0.0f; }

  const uint32_t mma_bar  = (uint32_t)__cvta_generic_to_shared(&s_mma_bar);
  const uint32_t lbar     = (uint32_t)__cvta_generic_to_shared(&s_load_bar);
  const uint32_t fbar     = (uint32_t)__cvta_generic_to_shared(&s_free_bar);
  if(tid == 0){ mbar_init(mma_bar, 1); mbar_init(lbar, 1); mbar_init(fbar, 1); }
  __syncthreads();

  constexpr uint32_t NCOLS = (Bc > D) ? (uint32_t)Bc : (uint32_t)D;
  static_assert(NCOLS >= 32 && (NCOLS & (NCOLS - 1)) == 0,
                "tcgen05 column count must be a power of two >= 32");

  uint32_t tmem_addr;
  {
    __shared__ uint32_t s_tmem_addr;
    if(tid < 32){
      uint32_t s_addr = (uint32_t)__cvta_generic_to_shared(&s_tmem_addr);
      asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                   :: "r"(s_addr), "r"(NCOLS) : "memory");
      asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
    __syncthreads();
    tmem_addr = s_tmem_addr;
  }

  const uint32_t TX = 2u * (uint32_t)Bc * (uint32_t)D * (uint32_t)sizeof(__nv_bfloat16); // K + V box bytes
  const uint32_t sKstage_addr = (uint32_t)__cvta_generic_to_shared(sKstage);
  const uint32_t sVstage_addr = (uint32_t)__cvta_generic_to_shared(sVstage);

  if(tid >= 128){
    // ---- Producer warp: only issues TMA loads, runs ahead of the consumers. ----
    if(tid == 128){
      int free_phase = 0;
      for(int kc = 0; kc < nKVTiles; ++kc){
        if(kc > 0){ mbar_wait(fbar, free_phase); free_phase ^= 1; }
        const int r = kvRow0 + kc * Bc;
        mbar_expect_tx(lbar, TX);
        tma_load_2d(sKstage_addr, &Ktmap, 0, r, lbar);
        tma_load_2d(sVstage_addr, &Vtmap, 0, r, lbar);
        mbar_arrive(lbar);
      }
    }
  } else {
    // ---- Consumer warps (0-3): reorder + MMA-issue + softmax + PV, unchanged from V13. ----
    int mbar_phase = 0;
    int load_phase = 0;
    const uint64_t descQ_base = make_smem_desc(sQ, Br);

    for(int kc = 0; kc < nKVTiles; ++kc){
      mbar_wait(lbar, load_phase); load_phase ^= 1;
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      consumer_sync();

      for(int i = tid; i < Bc * D; i += Br){
        const int bc = i / D, d = i % D;
        sK[canon_idx(bc, d, Bc)] = sKstage[i];   // K-major
        sV[canon_idx(d, bc, D)]  = sVstage[i];   // transposed
      }
      consumer_sync();
      // Staging buffer fully drained into canonical sK/sV — let the producer reuse it
      // for kc+1 while this iteration's MMA/softmax/PV run.
      if(tid == 0) mbar_arrive(fbar);

      // S2 = (Q @ K^T) * (scale*log2e) -> sS[Br, Bc]  (scores already in base-2 units)
      {
        const uint64_t descK_base = make_smem_desc(sK, Bc);
        const uint32_t idesc      = make_idesc_bf16(Br, Bc);
        if(tid == 0){
          asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
          for(int kt = 0; kt < D/16; ++kt){
            uint64_t descQ = advance_desc_katom(descQ_base, kt, Br);
            uint64_t descK = advance_desc_katom(descK_base, kt, Bc);
            uint32_t accumulate = (kt > 0) ? 1u : 0u;
            asm volatile(
              "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
              "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
              :: "r"(tmem_addr), "l"(descQ), "l"(descK), "r"(idesc), "r"(accumulate) : "memory");
          }
          mbar_commit_mma(mma_bar);
        }
        mbar_wait(mma_bar, mbar_phase); mbar_phase ^= 1;
        tmem_readout_to_smem_vec(sS, tmem_addr, Br, Bc, Bc, scale_l2e);   // prescale by log2e here
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
        consumer_sync();
      }

      // online-softmax stats + unnormalized P, base-2 domain (exp2 via ex2.approx)
      {
        const float m_old = sm[tid];
        const float l_old = sl[tid];

        // 3-way max tree over the row (fuses to FMNMX3 on sm_100+).
        float tile_max = -INFINITY;
        int j = 0;
        for(; j + 2 < Bc; j += 3)
          tile_max = fmaxf(tile_max,
                           fmaxf(sS[tid * Bc + j],
                                 fmaxf(sS[tid * Bc + j + 1], sS[tid * Bc + j + 2])));
        for(; j < Bc; ++j) tile_max = fmaxf(tile_max, sS[tid * Bc + j]);

        const float m_new = fmaxf(m_old, tile_max);
        const float corr  = ex2_approx(m_old - m_new);   // == exp(m_old - m_new)

        // Pack two probabilities per F2FP (F2FP.BF16.F32.PACK_AB) + one 32-bit STS.
        float p_sum = 0.0f;
        for(int j2 = 0; j2 < Bc; j2 += 2){
          const float p0 = ex2_approx(sS[tid * Bc + j2]     - m_new);   // == exp(S - m_new)
          const float p1 = ex2_approx(sS[tid * Bc + j2 + 1] - m_new);
          *reinterpret_cast<__nv_bfloat162*>(&sP[canon_idx(tid, j2, Br)]) =
              __floats2bfloat162_rn(p0, p1);
          p_sum += p0 + p1;
        }
        sm[tid] = m_new; sl[tid] = l_old * corr + p_sum; sCorr[tid] = corr;
      }
      consumer_sync();

      for(int i = tid; i < Br * D; i += Br) sO[i] *= sCorr[i / D];
      consumer_sync();

      // O_tile = P @ V, readout ACCUMULATED straight into sO (no sPV staging).
      {
        const uint64_t descP_base = make_smem_desc(sP, Br);
        const uint64_t descV_base = make_smem_desc(sV, D);
        const uint32_t idesc      = make_idesc_bf16(Br, D);
        if(tid == 0){
          asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
          for(int kt = 0; kt < Bc/16; ++kt){
            uint64_t descP = advance_desc_katom(descP_base, kt, Br);
            uint64_t descV = advance_desc_katom(descV_base, kt, D);
            uint32_t accumulate = (kt > 0) ? 1u : 0u;
            asm volatile(
              "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
              "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
              :: "r"(tmem_addr), "l"(descP), "l"(descV), "r"(idesc), "r"(accumulate) : "memory");
          }
          mbar_commit_mma(mma_bar);
        }
        mbar_wait(mma_bar, mbar_phase); mbar_phase ^= 1;
        tmem_readout_accum_vec(sO, tmem_addr, Br, D, D);   // sO += P@V, no staging hop
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
        consumer_sync();
      }
    } // end kv loop
  }

  __syncthreads();   // full-block reconvergence: producer is done, consumers are done

  if(tid < 32)
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(tmem_addr), "r"(NCOLS) : "memory");

  // Normalize + write O, packing two columns per bf16x2 store (D even -> same row denom).
  for(int i = 2 * tid; i < Br * D; i += 2 * blockDim.x){
    const float denom = sl[i / D];
    *reinterpret_cast<__nv_bfloat162*>(&d_O[qBase + i]) =
        __floats2bfloat162_rn(sO[i] / denom, sO[i + 1] / denom);
  }
  if(tid < Br)
    // m and l are in base-2 units: LSE_nat = ln2 * (m2 + log2(l)).
    d_LSE[lBase + tid] = 0.6931471805599453f * (sm[tid] + log2f(sl[tid]));

} // end of v17

// =================================
//  V18 : V17 + double-buffered TMA staging (Stage 6, step 2)
// =================================
// V17 decoupled the TMA wait from the block-wide barrier but the staging buffer was
// still SINGLE-buffered: the producer could only start loading tile kc+1 once the
// consumer finished the *reorder* of tile kc (early in the iteration), giving ~1
// iteration's worth of MMA/softmax/PV time to hide the TMA. V18 gives the producer a
// full second slot (ping-pong sKstage/sVstage) so it can issue kc+1's load the instant
// kc's slot is reordered out, with a whole extra iteration of headroom before the
// consumer can possibly stall on it — a deeper, more forgiving pipeline.
//
// Shared-memory cost: doubling sKstage+sVstage costs +32KB, which does NOT fit next to
// V13/V17's fp32 sS (Br*Bc*4 = 64KB) under sm_103a's static-smem ceiling (measured
// empirically: 0x38c00 = 226KB max per block; V17 already uses ~209KB, leaving ~17KB —
// not enough for +32KB). Fix: sS is stored as bf16 instead of fp32 (32KB instead of
// 64KB, via tmem_readout_to_smem_bf16_vec), freeing exactly enough room —
// measured total ~210.5KB, comfortably under the 226KB ceiling. This trades a little
// precision on the pre-softmax scores (they were already headed into bf16-packed sP
// one step later) for the smem headroom; correctness is checked against PyTorch SDPA
// with the same tolerance as every other version below.
//
// No distributed-shared-memory/cluster fallback was needed for this step — flagging it
// per the plan: DSM (V14/V15-style CTA pairing) is the fallback if a future version
// needs MORE than this and the bf16-sS trick alone doesn't cover it.
template<int Br, int Bc, int D>
__global__ void gqa_v18(
  __nv_bfloat16 *d_Q,
  __nv_bfloat16 *d_O,
  float *d_LSE,
  const __grid_constant__ CUtensorMap Ktmap,   // K as flattened [B*Hkv*S, D]
  const __grid_constant__ CUtensorMap Vtmap,   // V as flattened [B*Hkv*S, D]
  int B,
  int Hq,
  int Hkv,
  int G,
  int S,
  float scale
){
  static_assert(Br == 128, "V18's consumer group is hardwired to 128 threads (TMEM readout needs warps 0-3)");

  const int b      = blockIdx.x;
  const int hq     = blockIdx.y;
  const int q_tile = blockIdx.z;
  const int hkv    = hq / G;
  const int tid    = threadIdx.x;              // 0..159: 0..127 consumer, 128..159 producer

  const int q_row0   = q_tile * Br;
  const int nKVTiles = S / Bc;
  const int kvRow0   = (b * Hkv + hkv) * S;   // first K/V row of this head in the flat tensor

  const long qBase  = ((long)(b * Hq + hq) * S + q_row0) * D;
  const long lBase  = ((long)(b * Hq + hq) * S + q_row0);

  // Fold log2(e) into the score scale so QK^T lands in base-2 units (see V7 header).
  const float scale_l2e = scale * 1.4426950408889634f;   // scale * float32 log2(e)

  __shared__ __align__(16)  __nv_bfloat16 sQ[Br * D];
  __shared__ __align__(128) __nv_bfloat16 sKstage[2][Bc * D];   // TMA target, row-major, DOUBLE buffer
  __shared__ __align__(128) __nv_bfloat16 sVstage[2][Bc * D];
  __shared__ __align__(16)  __nv_bfloat16 sK[Bc * D];           // canonical K-major (reorder output)
  __shared__ __align__(16)  __nv_bfloat16 sV[Bc * D];           // canonical [D,Bc] transposed
  __shared__ __align__(16)  __nv_bfloat16 sP[Br * Bc];
  __shared__ __align__(16)  __nv_bfloat16 sS[Br * Bc];          // bf16, not fp32 — see header
  __shared__ __align__(16)  float         sO[Br * D];           // P@V readout accumulates here directly
  __shared__ float sm[Br];
  __shared__ float sl[Br];
  __shared__ float sCorr[Br];
  __shared__ __align__(8) uint64_t s_mma_bar;
  __shared__ __align__(8) uint64_t s_load_bar[2];   // producer -> consumer: slot's bytes ready
  __shared__ __align__(8) uint64_t s_free_bar[2];   // consumer -> producer: slot is free

  for(int i = tid; i < Br * D; i += blockDim.x){
    const int r = i / D, c = i % D;
    sQ[canon_idx(r, c, Br)] = d_Q[qBase + i];
    sO[i] = 0.0f;
  }
  if(tid < Br){ sm[tid] = -INFINITY; sl[tid] = 0.0f; }

  const uint32_t mma_bar = (uint32_t)__cvta_generic_to_shared(&s_mma_bar);
  const uint32_t lbar0   = (uint32_t)__cvta_generic_to_shared(&s_load_bar[0]);
  const uint32_t lbar1   = (uint32_t)__cvta_generic_to_shared(&s_load_bar[1]);
  const uint32_t fbar0   = (uint32_t)__cvta_generic_to_shared(&s_free_bar[0]);
  const uint32_t fbar1   = (uint32_t)__cvta_generic_to_shared(&s_free_bar[1]);
  if(tid == 0){
    mbar_init(mma_bar, 1);
    mbar_init(lbar0, 1); mbar_init(lbar1, 1);
    mbar_init(fbar0, 1); mbar_init(fbar1, 1);
  }
  __syncthreads();

  constexpr uint32_t NCOLS = (Bc > D) ? (uint32_t)Bc : (uint32_t)D;
  static_assert(NCOLS >= 32 && (NCOLS & (NCOLS - 1)) == 0,
                "tcgen05 column count must be a power of two >= 32");

  uint32_t tmem_addr;
  {
    __shared__ uint32_t s_tmem_addr;
    if(tid < 32){
      uint32_t s_addr = (uint32_t)__cvta_generic_to_shared(&s_tmem_addr);
      asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                   :: "r"(s_addr), "r"(NCOLS) : "memory");
      asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
    __syncthreads();
    tmem_addr = s_tmem_addr;
  }

  const uint32_t TX = 2u * (uint32_t)Bc * (uint32_t)D * (uint32_t)sizeof(__nv_bfloat16); // K + V box bytes
  const uint32_t sKstage_addr[2] = {
    (uint32_t)__cvta_generic_to_shared(sKstage[0]),
    (uint32_t)__cvta_generic_to_shared(sKstage[1])
  };
  const uint32_t sVstage_addr[2] = {
    (uint32_t)__cvta_generic_to_shared(sVstage[0]),
    (uint32_t)__cvta_generic_to_shared(sVstage[1])
  };
  const uint32_t lbar[2] = {lbar0, lbar1};
  const uint32_t fbar[2] = {fbar0, fbar1};

  if(tid >= 128){
    // ---- Producer warp: fills both slots ahead of the consumers. ----
    if(tid == 128){
      int free_phase[2] = {0, 0};
      for(int kc = 0; kc < nKVTiles; ++kc){
        const int slot = kc & 1;
        // Slots 0 and 1 start empty, so the first wait for each is at kc == slot+2.
        if(kc >= 2){ mbar_wait(fbar[slot], free_phase[slot]); free_phase[slot] ^= 1; }
        const int r = kvRow0 + kc * Bc;
        mbar_expect_tx(lbar[slot], TX);
        tma_load_2d(sKstage_addr[slot], &Ktmap, 0, r, lbar[slot]);
        tma_load_2d(sVstage_addr[slot], &Vtmap, 0, r, lbar[slot]);
        mbar_arrive(lbar[slot]);
      }
    }
  } else {
    // ---- Consumer warps (0-3): reorder + MMA-issue + softmax + PV. ----
    int mbar_phase = 0;
    int load_phase[2] = {0, 0};
    const uint64_t descQ_base = make_smem_desc(sQ, Br);

    for(int kc = 0; kc < nKVTiles; ++kc){
      const int slot = kc & 1;
      mbar_wait(lbar[slot], load_phase[slot]); load_phase[slot] ^= 1;
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      consumer_sync();

      for(int i = tid; i < Bc * D; i += Br){
        const int bc = i / D, d = i % D;
        sK[canon_idx(bc, d, Bc)] = sKstage[slot][i];   // K-major
        sV[canon_idx(d, bc, D)]  = sVstage[slot][i];   // transposed
      }
      consumer_sync();
      // This slot is fully drained into canonical sK/sV — let the producer reuse it
      // two tiles from now, while this iteration's MMA/softmax/PV run.
      if(tid == 0) mbar_arrive(fbar[slot]);

      // S2 = (Q @ K^T) * (scale*log2e) -> sS[Br, Bc], bf16 (scores already in base-2 units)
      {
        const uint64_t descK_base = make_smem_desc(sK, Bc);
        const uint32_t idesc      = make_idesc_bf16(Br, Bc);
        if(tid == 0){
          asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
          for(int kt = 0; kt < D/16; ++kt){
            uint64_t descQ = advance_desc_katom(descQ_base, kt, Br);
            uint64_t descK = advance_desc_katom(descK_base, kt, Bc);
            uint32_t accumulate = (kt > 0) ? 1u : 0u;
            asm volatile(
              "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
              "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
              :: "r"(tmem_addr), "l"(descQ), "l"(descK), "r"(idesc), "r"(accumulate) : "memory");
          }
          mbar_commit_mma(mma_bar);
        }
        mbar_wait(mma_bar, mbar_phase); mbar_phase ^= 1;
        tmem_readout_to_smem_bf16_vec(sS, tmem_addr, Br, Bc, Bc, scale_l2e);   // prescale by log2e here
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
        consumer_sync();
      }

      // online-softmax stats + unnormalized P, base-2 domain (exp2 via ex2.approx)
      {
        const float m_old = sm[tid];
        const float l_old = sl[tid];

        // 3-way max tree over the row (fuses to FMNMX3 on sm_100+).
        float tile_max = -INFINITY;
        int j = 0;
        for(; j + 2 < Bc; j += 3)
          tile_max = fmaxf(tile_max,
                           fmaxf(__bfloat162float(sS[tid * Bc + j]),
                                 fmaxf(__bfloat162float(sS[tid * Bc + j + 1]),
                                       __bfloat162float(sS[tid * Bc + j + 2]))));
        for(; j < Bc; ++j) tile_max = fmaxf(tile_max, __bfloat162float(sS[tid * Bc + j]));

        const float m_new = fmaxf(m_old, tile_max);
        const float corr  = ex2_approx(m_old - m_new);   // == exp(m_old - m_new)

        // Pack two probabilities per F2FP (F2FP.BF16.F32.PACK_AB) + one 32-bit STS.
        float p_sum = 0.0f;
        for(int j2 = 0; j2 < Bc; j2 += 2){
          const float p0 = ex2_approx(__bfloat162float(sS[tid * Bc + j2])     - m_new);   // == exp(S - m_new)
          const float p1 = ex2_approx(__bfloat162float(sS[tid * Bc + j2 + 1]) - m_new);
          *reinterpret_cast<__nv_bfloat162*>(&sP[canon_idx(tid, j2, Br)]) =
              __floats2bfloat162_rn(p0, p1);
          p_sum += p0 + p1;
        }
        sm[tid] = m_new; sl[tid] = l_old * corr + p_sum; sCorr[tid] = corr;
      }
      consumer_sync();

      for(int i = tid; i < Br * D; i += Br) sO[i] *= sCorr[i / D];
      consumer_sync();

      // O_tile = P @ V, readout ACCUMULATED straight into sO (no sPV staging).
      {
        const uint64_t descP_base = make_smem_desc(sP, Br);
        const uint64_t descV_base = make_smem_desc(sV, D);
        const uint32_t idesc      = make_idesc_bf16(Br, D);
        if(tid == 0){
          asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
          for(int kt = 0; kt < Bc/16; ++kt){
            uint64_t descP = advance_desc_katom(descP_base, kt, Br);
            uint64_t descV = advance_desc_katom(descV_base, kt, D);
            uint32_t accumulate = (kt > 0) ? 1u : 0u;
            asm volatile(
              "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
              "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
              :: "r"(tmem_addr), "l"(descP), "l"(descV), "r"(idesc), "r"(accumulate) : "memory");
          }
          mbar_commit_mma(mma_bar);
        }
        mbar_wait(mma_bar, mbar_phase); mbar_phase ^= 1;
        tmem_readout_accum_vec(sO, tmem_addr, Br, D, D);   // sO += P@V, no staging hop
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
        consumer_sync();
      }
    } // end kv loop
  }

  __syncthreads();   // full-block reconvergence: producer is done, consumers are done

  if(tid < 32)
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(tmem_addr), "r"(NCOLS) : "memory");

  // Normalize + write O, packing two columns per bf16x2 store (D even -> same row denom).
  for(int i = 2 * tid; i < Br * D; i += 2 * blockDim.x){
    const float denom = sl[i / D];
    *reinterpret_cast<__nv_bfloat162*>(&d_O[qBase + i]) =
        __floats2bfloat162_rn(sO[i] / denom, sO[i + 1] / denom);
  }
  if(tid < Br)
    // m and l are in base-2 units: LSE_nat = ln2 * (m2 + log2(l)).
    d_LSE[lBase + tid] = 0.6931471805599453f * (sm[tid] + log2f(sl[tid]));

} // end of v18

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
void launch_gqa_v0(
  __nv_bfloat16 *d_Q, __nv_bfloat16 *d_K, __nv_bfloat16 *d_V,
  __nv_bfloat16 *d_O, float *d_LSE,
  int B, int Hq, int Hkv, int S, int G, float scale
){
  dim3 GRID(B, Hq, S / Br);   // (16, 12, 256) = 49,152 blocks
  dim3 BLOCK(32);             // ONE warp per block
  gqa_v0<Br, Bc, D><<<GRID, BLOCK>>>(d_Q, d_K, d_V, d_O, d_LSE,
                                     B, Hq, Hkv, S, G, scale);
}

// V1 — same signature as launch_gqa_v0 so callers are interchangeable.
template<int Br, int Bc, int D>
void launch_gqa_v1(
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

  // REQUIRED: pin occupancy to 1 CTA/SM. TMEM is a per-SM resource; this kernel's
  // tcgen05.alloc gives colliding regions when >1 block shares an SM (verified: at
  // 2-3 CTAs/SM the accumulator gets scattered TMEM corruption). We reserve enough
  // (unused) dynamic shared memory that only one block fits per SM. If you later want
  // >1 CTA/SM for perf, TMEM must be partitioned explicitly across resident blocks.
  constexpr int OCC_SMEM = 64 * 1024;   // static(~60KB)+64KB > 114KB => 1 block/SM
  static bool cfgd = false;
  if(!cfgd){
    cudaFuncSetAttribute(gqa_v2<Br, Bc, D>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, OCC_SMEM);
    cfgd = true;
  }
  // NOTE: gqa_v2's kernel param order is (Hkv, G, S), so forward G before S.
  gqa_v2<Br, Bc, D><<<GRID, BLOCK, OCC_SMEM>>>(d_Q, d_K, d_V, d_O, d_LSE,
                          B, Hq, Hkv, G, S, scale);
}

// V3 — same signature/launch as V2; cp.async double-buffered KV loads.
template<int Br, int Bc, int D>
void launch_gqa_v3(
  __nv_bfloat16 *d_Q, __nv_bfloat16 *d_K, __nv_bfloat16 *d_V,
  __nv_bfloat16 *d_O, float *d_LSE,
  int B, int Hq, int Hkv, int S, int G, float scale
){
  static_assert(Br % 64 == 0, "Br must be a multiple of 64 for tcgen05 M = 64");
  static_assert(Bc % 8 == 0, "Bc must be a multiple of 8 for tcgen05 N = 8");
  static_assert(D  % 16 == 0, "D  must be a multiple of 16 for tcgen05 dense");

  dim3 GRID(B, Hq, S/Br);
  dim3 BLOCK(128); // 4 warps

  // Same 1-CTA/SM requirement as V2 (per-SM TMEM). V3's static smem is larger
  // (double-buffered KV + V staging), so reserve enough dynamic smem to still pin
  // occupancy to one block. static(~72KB)+64KB > 114KB => 1 block/SM.
  constexpr int OCC_SMEM = 64 * 1024;
  static bool cfgd = false;
  if(!cfgd){
    cudaFuncSetAttribute(gqa_v3<Br, Bc, D>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, OCC_SMEM);
    cfgd = true;
  }
  gqa_v3<Br, Bc, D><<<GRID, BLOCK, OCC_SMEM>>>(d_Q, d_K, d_V, d_O, d_LSE,
                          B, Hq, Hkv, G, S, scale);
}

// V4 — V3 (cp.async pipeline) + vectorized TMEM readout.
template<int Br, int Bc, int D>
void launch_gqa_v4(
  __nv_bfloat16 *d_Q, __nv_bfloat16 *d_K, __nv_bfloat16 *d_V,
  __nv_bfloat16 *d_O, float *d_LSE,
  int B, int Hq, int Hkv, int S, int G, float scale
){
  static_assert(Br % 64 == 0, "Br must be a multiple of 64 for tcgen05 M = 64");
  static_assert(Bc % 8 == 0, "Bc must be a multiple of 8 for tcgen05 N = 8");
  static_assert(D  % 16 == 0, "D  must be a multiple of 16 for tcgen05 dense");

  dim3 GRID(B, Hq, S/Br);
  dim3 BLOCK(128);

  constexpr int OCC_SMEM = 64 * 1024;   // 1 CTA/SM (per-SM TMEM), same as V2/V3
  static bool cfgd = false;
  if(!cfgd){
    cudaFuncSetAttribute(gqa_v4<Br, Bc, D>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, OCC_SMEM);
    cfgd = true;
  }
  gqa_v4<Br, Bc, D><<<GRID, BLOCK, OCC_SMEM>>>(d_Q, d_K, d_V, d_O, d_LSE,
                          B, Hq, Hkv, G, S, scale);
}

// Build a 2D TMA descriptor for a row-major [rows, cols] bf16 tensor, tile [box_rows, box_cols].
// cuTensorMapEncodeTiled wants the fastest (contiguous) dim first, so dims/box are {cols, rows}.
static CUtensorMap make_tma_2d(__nv_bfloat16* gptr, uint64_t rows, uint64_t cols,
                               uint32_t box_rows, uint32_t box_cols){
  CUtensorMap tmap{};
  uint64_t gdim[2]    = { cols, rows };
  uint64_t gstride[1] = { cols * sizeof(__nv_bfloat16) };   // bytes between rows (must be %16==0)
  uint32_t bdim[2]    = { box_cols, box_rows };
  uint32_t estride[2] = { 1, 1 };
  CUresult res = cuTensorMapEncodeTiled(
    &tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, gptr, gdim, gstride, bdim, estride,
    CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
    CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  assert(res == CUDA_SUCCESS);
  return tmap;
}

// V5 — V4 + TMA KV loads. Same signature as the others so the bench lambda is uniform;
// the K/V tensor maps are built once (they only depend on the fixed d_K/d_V pointers).
template<int Br, int Bc, int D>
void launch_gqa_v5(
  __nv_bfloat16 *d_Q, __nv_bfloat16 *d_K, __nv_bfloat16 *d_V,
  __nv_bfloat16 *d_O, float *d_LSE,
  int B, int Hq, int Hkv, int S, int G, float scale
){
  static_assert(Br % 64 == 0, "Br must be a multiple of 64 for tcgen05 M = 64");
  static_assert(Bc % 8 == 0, "Bc must be a multiple of 8 for tcgen05 N = 8");
  static_assert(D  % 16 == 0, "D  must be a multiple of 16 for tcgen05 dense");

  dim3 GRID(B, Hq, S/Br);
  dim3 BLOCK(128);

  constexpr int OCC_SMEM = 64 * 1024;   // 1 CTA/SM (per-SM TMEM), as V2-V4
  static bool cfgd = false;
  static CUtensorMap Ktmap, Vtmap;
  if(!cfgd){
    const uint64_t kvRows = (uint64_t)B * Hkv * S;   // K/V flattened as [B*Hkv*S, D]
    Ktmap = make_tma_2d(d_K, kvRows, (uint64_t)D, (uint32_t)Bc, (uint32_t)D);
    Vtmap = make_tma_2d(d_V, kvRows, (uint64_t)D, (uint32_t)Bc, (uint32_t)D);
    cudaFuncSetAttribute(gqa_v5<Br, Bc, D>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, OCC_SMEM);
    cfgd = true;
  }
  gqa_v5<Br, Bc, D><<<GRID, BLOCK, OCC_SMEM>>>(d_Q, d_O, d_LSE, Ktmap, Vtmap,
                          B, Hq, Hkv, G, S, scale);
}

// V6 — V5 + Br = 128. Same signature as the others so the bench lambda is uniform;
// the K/V tensor maps are built once (they only depend on the fixed d_K/d_V pointers).
template<int Br, int Bc, int D>
void launch_gqa_v6(
  __nv_bfloat16 *d_Q, __nv_bfloat16 *d_K, __nv_bfloat16 *d_V,
  __nv_bfloat16 *d_O, float *d_LSE,
  int B, int Hq, int Hkv, int S, int G, float scale
){
  static_assert(Br % 64 == 0, "Br must be a multiple of 64 for tcgen05 M = 64");
  static_assert(Bc % 8 == 0, "Bc must be a multiple of 8 for tcgen05 N = 8");
  static_assert(D  % 16 == 0, "D  must be a multiple of 16 for tcgen05 dense");

  dim3 GRID(B, Hq, S/Br);
  dim3 BLOCK(128);

  constexpr int OCC_SMEM = 64 * 1024;   // 1 CTA/SM (per-SM TMEM), as V2-V4
  static bool cfgd = false;
  static CUtensorMap Ktmap, Vtmap;
  if(!cfgd){
    const uint64_t kvRows = (uint64_t)B * Hkv * S;   // K/V flattened as [B*Hkv*S, D]
    Ktmap = make_tma_2d(d_K, kvRows, (uint64_t)D, (uint32_t)Bc, (uint32_t)D);
    Vtmap = make_tma_2d(d_V, kvRows, (uint64_t)D, (uint32_t)Bc, (uint32_t)D);
    cudaFuncSetAttribute(gqa_v5<Br, Bc, D>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, OCC_SMEM);
    cfgd = true;
  }
  gqa_v5<Br, Bc, D><<<GRID, BLOCK, OCC_SMEM>>>(d_Q, d_O, d_LSE, Ktmap, Vtmap,
                          B, Hq, Hkv, G, S, scale);
}

// V7 — V6 (TMA + Br=128) + log2-domain softmax (gqa_v7 kernel). Same signature/launch
// path as V6; only the kernel differs, so the K/V tensor maps are built the same way.
template<int Br, int Bc, int D>
void launch_gqa_v7(
  __nv_bfloat16 *d_Q, __nv_bfloat16 *d_K, __nv_bfloat16 *d_V,
  __nv_bfloat16 *d_O, float *d_LSE,
  int B, int Hq, int Hkv, int S, int G, float scale
){
  static_assert(Br % 64 == 0, "Br must be a multiple of 64 for tcgen05 M = 64");
  static_assert(Bc % 8 == 0, "Bc must be a multiple of 8 for tcgen05 N = 8");
  static_assert(D  % 16 == 0, "D  must be a multiple of 16 for tcgen05 dense");

  dim3 GRID(B, Hq, S/Br);
  dim3 BLOCK(128);

  constexpr int OCC_SMEM = 64 * 1024;   // 1 CTA/SM (per-SM TMEM), as V2-V4
  static bool cfgd = false;
  static CUtensorMap Ktmap, Vtmap;
  if(!cfgd){
    const uint64_t kvRows = (uint64_t)B * Hkv * S;   // K/V flattened as [B*Hkv*S, D]
    Ktmap = make_tma_2d(d_K, kvRows, (uint64_t)D, (uint32_t)Bc, (uint32_t)D);
    Vtmap = make_tma_2d(d_V, kvRows, (uint64_t)D, (uint32_t)Bc, (uint32_t)D);
    cudaFuncSetAttribute(gqa_v7<Br, Bc, D>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, OCC_SMEM);
    cfgd = true;
  }
  gqa_v7<Br, Bc, D><<<GRID, BLOCK, OCC_SMEM>>>(d_Q, d_O, d_LSE, Ktmap, Vtmap,
                          B, Hq, Hkv, G, S, scale);
}

// V8 — V7 + packed bf16 conversion (gqa_v8 kernel). Same signature/launch path as V7;
// only the kernel differs, so the K/V tensor maps are built the same way.
template<int Br, int Bc, int D>
void launch_gqa_v8(
  __nv_bfloat16 *d_Q, __nv_bfloat16 *d_K, __nv_bfloat16 *d_V,
  __nv_bfloat16 *d_O, float *d_LSE,
  int B, int Hq, int Hkv, int S, int G, float scale
){
  static_assert(Br % 64 == 0, "Br must be a multiple of 64 for tcgen05 M = 64");
  static_assert(Bc % 8 == 0, "Bc must be a multiple of 8 for tcgen05 N = 8");
  static_assert(D  % 16 == 0, "D  must be a multiple of 16 for tcgen05 dense");

  dim3 GRID(B, Hq, S/Br);
  dim3 BLOCK(128);

  constexpr int OCC_SMEM = 64 * 1024;   // 1 CTA/SM (per-SM TMEM), as V2-V4
  static bool cfgd = false;
  static CUtensorMap Ktmap, Vtmap;
  if(!cfgd){
    const uint64_t kvRows = (uint64_t)B * Hkv * S;   // K/V flattened as [B*Hkv*S, D]
    Ktmap = make_tma_2d(d_K, kvRows, (uint64_t)D, (uint32_t)Bc, (uint32_t)D);
    Vtmap = make_tma_2d(d_V, kvRows, (uint64_t)D, (uint32_t)Bc, (uint32_t)D);
    cudaFuncSetAttribute(gqa_v8<Br, Bc, D>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, OCC_SMEM);
    cfgd = true;
  }
  gqa_v8<Br, Bc, D><<<GRID, BLOCK, OCC_SMEM>>>(d_Q, d_O, d_LSE, Ktmap, Vtmap,
                          B, Hq, Hkv, G, S, scale);
}

// V9 — V8 + software-pipelined QK^T (gqa_v9 kernel). Same signature/launch path as V8;
// only the kernel differs, so the K/V tensor maps are built the same way.
template<int Br, int Bc, int D>
void launch_gqa_v9(
  __nv_bfloat16 *d_Q, __nv_bfloat16 *d_K, __nv_bfloat16 *d_V,
  __nv_bfloat16 *d_O, float *d_LSE,
  int B, int Hq, int Hkv, int S, int G, float scale
){
  static_assert(Br % 64 == 0, "Br must be a multiple of 64 for tcgen05 M = 64");
  static_assert(Bc % 8 == 0, "Bc must be a multiple of 8 for tcgen05 N = 8");
  static_assert(D  % 16 == 0, "D  must be a multiple of 16 for tcgen05 dense");

  dim3 GRID(B, Hq, S/Br);
  dim3 BLOCK(128);

  constexpr int OCC_SMEM = 64 * 1024;   // 1 CTA/SM (per-SM TMEM), as V2-V4
  static bool cfgd = false;
  static CUtensorMap Ktmap, Vtmap;
  if(!cfgd){
    const uint64_t kvRows = (uint64_t)B * Hkv * S;   // K/V flattened as [B*Hkv*S, D]
    Ktmap = make_tma_2d(d_K, kvRows, (uint64_t)D, (uint32_t)Bc, (uint32_t)D);
    Vtmap = make_tma_2d(d_V, kvRows, (uint64_t)D, (uint32_t)Bc, (uint32_t)D);
    cudaFuncSetAttribute(gqa_v9<Br, Bc, D>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, OCC_SMEM);
    cfgd = true;
  }
  gqa_v9<Br, Bc, D><<<GRID, BLOCK, OCC_SMEM>>>(d_Q, d_O, d_LSE, Ktmap, Vtmap,
                          B, Hq, Hkv, G, S, scale);
}

// V10 — V8 kernel at Bc = 64 (config variant, like V6 was for Br). Halves the KV-tile
// count 128 -> 64 so each tile's fixed sync/readout/issue overhead amortizes over 2x
// the work — the V7/V8/V9 plateau pointed at exactly this per-tile overhead.
// NOTE: no dummy dynamic smem here, unlike V5-V9. At Bc=64 the kernel's static smem is
// ~177KB (sS/sP/staging all double), so two blocks can't fit on an SM anyway — occupancy
// is already pinned to 1 CTA/SM. Adding the 64KB dynamic reservation would exceed the
// per-block smem limit and the launch would fail.
template<int Br, int Bc, int D>
void launch_gqa_v10(
  __nv_bfloat16 *d_Q, __nv_bfloat16 *d_K, __nv_bfloat16 *d_V,
  __nv_bfloat16 *d_O, float *d_LSE,
  int B, int Hq, int Hkv, int S, int G, float scale
){
  static_assert(Br % 64 == 0, "Br must be a multiple of 64 for tcgen05 M = 64");
  static_assert(Bc % 8 == 0, "Bc must be a multiple of 8 for tcgen05 N = 8");
  static_assert(D  % 16 == 0, "D  must be a multiple of 16 for tcgen05 dense");

  dim3 GRID(B, Hq, S/Br);
  dim3 BLOCK(128);

  static bool cfgd = false;
  static CUtensorMap Ktmap, Vtmap;
  if(!cfgd){
    const uint64_t kvRows = (uint64_t)B * Hkv * S;   // K/V flattened as [B*Hkv*S, D]
    Ktmap = make_tma_2d(d_K, kvRows, (uint64_t)D, (uint32_t)Bc, (uint32_t)D);
    Vtmap = make_tma_2d(d_V, kvRows, (uint64_t)D, (uint32_t)Bc, (uint32_t)D);
    cfgd = true;
  }
  gqa_v8<Br, Bc, D><<<GRID, BLOCK>>>(d_Q, d_O, d_LSE, Ktmap, Vtmap,
                          B, Hq, Hkv, G, S, scale);
}

// V11 — V10 + P@V readout fused into the O accumulation (gqa_v11 kernel, Bc=64).
// Same no-dynamic-smem launch as V10: static smem (~145KB after dropping sPV) still
// pins occupancy to 1 CTA/SM on its own (two blocks would need ~291KB > 228KB/SM).
template<int Br, int Bc, int D>
void launch_gqa_v11(
  __nv_bfloat16 *d_Q, __nv_bfloat16 *d_K, __nv_bfloat16 *d_V,
  __nv_bfloat16 *d_O, float *d_LSE,
  int B, int Hq, int Hkv, int S, int G, float scale
){
  static_assert(Br % 64 == 0, "Br must be a multiple of 64 for tcgen05 M = 64");
  static_assert(Bc % 8 == 0, "Bc must be a multiple of 8 for tcgen05 N = 8");
  static_assert(D  % 16 == 0, "D  must be a multiple of 16 for tcgen05 dense");

  dim3 GRID(B, Hq, S/Br);
  dim3 BLOCK(128);

  static bool cfgd = false;
  static CUtensorMap Ktmap, Vtmap;
  if(!cfgd){
    const uint64_t kvRows = (uint64_t)B * Hkv * S;   // K/V flattened as [B*Hkv*S, D]
    Ktmap = make_tma_2d(d_K, kvRows, (uint64_t)D, (uint32_t)Bc, (uint32_t)D);
    Vtmap = make_tma_2d(d_V, kvRows, (uint64_t)D, (uint32_t)Bc, (uint32_t)D);
    cfgd = true;
  }
  gqa_v11<Br, Bc, D><<<GRID, BLOCK>>>(d_Q, d_O, d_LSE, Ktmap, Vtmap,
                          B, Hq, Hkv, G, S, scale);
}

// V12 — V11 + single-buffered TMA staging (gqa_v12 kernel, Bc=64). Isolates the cost
// of giving up the TMA prefetch overlap before V13 spends the freed smem on Bc=128.
// Static smem ~129KB still pins 1 CTA/SM on its own.
template<int Br, int Bc, int D>
void launch_gqa_v12(
  __nv_bfloat16 *d_Q, __nv_bfloat16 *d_K, __nv_bfloat16 *d_V,
  __nv_bfloat16 *d_O, float *d_LSE,
  int B, int Hq, int Hkv, int S, int G, float scale
){
  static_assert(Br % 64 == 0, "Br must be a multiple of 64 for tcgen05 M = 64");
  static_assert(Bc % 8 == 0, "Bc must be a multiple of 8 for tcgen05 N = 8");
  static_assert(D  % 16 == 0, "D  must be a multiple of 16 for tcgen05 dense");

  dim3 GRID(B, Hq, S/Br);
  dim3 BLOCK(128);

  static bool cfgd = false;
  static CUtensorMap Ktmap, Vtmap;
  if(!cfgd){
    const uint64_t kvRows = (uint64_t)B * Hkv * S;   // K/V flattened as [B*Hkv*S, D]
    Ktmap = make_tma_2d(d_K, kvRows, (uint64_t)D, (uint32_t)Bc, (uint32_t)D);
    Vtmap = make_tma_2d(d_V, kvRows, (uint64_t)D, (uint32_t)Bc, (uint32_t)D);
    cfgd = true;
  }
  gqa_v12<Br, Bc, D><<<GRID, BLOCK>>>(d_Q, d_O, d_LSE, Ktmap, Vtmap,
                          B, Hq, Hkv, G, S, scale);
}

// V13 — V12 kernel at Bc = 128 (the smem-diet payoff): 32 KV tiles, ~209KB static smem
// — only fits because V11 dropped sPV and V12 single-buffered the staging. Separate
// launcher because the template instantiation needs its own tensor maps (box_rows=128).
template<int Br, int Bc, int D>
void launch_gqa_v13(
  __nv_bfloat16 *d_Q, __nv_bfloat16 *d_K, __nv_bfloat16 *d_V,
  __nv_bfloat16 *d_O, float *d_LSE,
  int B, int Hq, int Hkv, int S, int G, float scale
){
  static_assert(Br % 64 == 0, "Br must be a multiple of 64 for tcgen05 M = 64");
  static_assert(Bc % 8 == 0, "Bc must be a multiple of 8 for tcgen05 N = 8");
  static_assert(D  % 16 == 0, "D  must be a multiple of 16 for tcgen05 dense");

  dim3 GRID(B, Hq, S/Br);
  dim3 BLOCK(128);

  static bool cfgd = false;
  static CUtensorMap Ktmap, Vtmap;
  if(!cfgd){
    const uint64_t kvRows = (uint64_t)B * Hkv * S;   // K/V flattened as [B*Hkv*S, D]
    Ktmap = make_tma_2d(d_K, kvRows, (uint64_t)D, (uint32_t)Bc, (uint32_t)D);
    Vtmap = make_tma_2d(d_V, kvRows, (uint64_t)D, (uint32_t)Bc, (uint32_t)D);
    cfgd = true;
  }
  gqa_v12<Br, Bc, D><<<GRID, BLOCK>>>(d_Q, d_O, d_LSE, Ktmap, Vtmap,
                          B, Hq, Hkv, G, S, scale);
}

// V14 — V13 + cluster launch scaffolding (gqa_v14 kernel, Bc=128). __cluster_dims__
// on the kernel makes the plain <<<>>> launch cluster-aware; gridDim.z=32 is even.
template<int Br, int Bc, int D>
void launch_gqa_v14(
  __nv_bfloat16 *d_Q, __nv_bfloat16 *d_K, __nv_bfloat16 *d_V,
  __nv_bfloat16 *d_O, float *d_LSE,
  int B, int Hq, int Hkv, int S, int G, float scale
){
  static_assert(Br % 64 == 0, "Br must be a multiple of 64 for tcgen05 M = 64");
  static_assert(Bc % 8 == 0, "Bc must be a multiple of 8 for tcgen05 N = 8");
  static_assert(D  % 16 == 0, "D  must be a multiple of 16 for tcgen05 dense");

  dim3 GRID(B, Hq, S/Br);   // z must be even (cluster dim 2 along z)
  dim3 BLOCK(128);

  static bool cfgd = false;
  static CUtensorMap Ktmap, Vtmap;
  if(!cfgd){
    const uint64_t kvRows = (uint64_t)B * Hkv * S;   // K/V flattened as [B*Hkv*S, D]
    Ktmap = make_tma_2d(d_K, kvRows, (uint64_t)D, (uint32_t)Bc, (uint32_t)D);
    Vtmap = make_tma_2d(d_V, kvRows, (uint64_t)D, (uint32_t)Bc, (uint32_t)D);
    cfgd = true;
  }
  gqa_v14<Br, Bc, D><<<GRID, BLOCK>>>(d_Q, d_O, d_LSE, Ktmap, Vtmap,
                          B, Hq, Hkv, G, S, scale);
}

// V15 — V14 + multicast K/V TMA (gqa_v15 kernel, Bc=128): each K/V tile fetched from
// HBM once per CLUSTER and multicast to both CTAs' staging.
template<int Br, int Bc, int D>
void launch_gqa_v15(
  __nv_bfloat16 *d_Q, __nv_bfloat16 *d_K, __nv_bfloat16 *d_V,
  __nv_bfloat16 *d_O, float *d_LSE,
  int B, int Hq, int Hkv, int S, int G, float scale
){
  static_assert(Br % 64 == 0, "Br must be a multiple of 64 for tcgen05 M = 64");
  static_assert(Bc % 8 == 0, "Bc must be a multiple of 8 for tcgen05 N = 8");
  static_assert(D  % 16 == 0, "D  must be a multiple of 16 for tcgen05 dense");

  dim3 GRID(B, Hq, S/Br);   // z must be even (cluster dim 2 along z)
  dim3 BLOCK(128);

  static bool cfgd = false;
  static CUtensorMap Ktmap, Vtmap;
  if(!cfgd){
    const uint64_t kvRows = (uint64_t)B * Hkv * S;   // K/V flattened as [B*Hkv*S, D]
    Ktmap = make_tma_2d(d_K, kvRows, (uint64_t)D, (uint32_t)Bc, (uint32_t)D);
    Vtmap = make_tma_2d(d_V, kvRows, (uint64_t)D, (uint32_t)Bc, (uint32_t)D);
    cfgd = true;
  }
  gqa_v15<Br, Bc, D><<<GRID, BLOCK>>>(d_Q, d_O, d_LSE, Ktmap, Vtmap,
                          B, Hq, Hkv, G, S, scale);
}

// V16 — persistent kernel on the V13 core (gqa_v16, Bc=128): grid = #SMs, one
// resident CTA per SM (smem pins that anyway), each looping over work items.
template<int Br, int Bc, int D>
void launch_gqa_v16(
  __nv_bfloat16 *d_Q, __nv_bfloat16 *d_K, __nv_bfloat16 *d_V,
  __nv_bfloat16 *d_O, float *d_LSE,
  int B, int Hq, int Hkv, int S, int G, float scale
){
  static_assert(Br % 64 == 0, "Br must be a multiple of 64 for tcgen05 M = 64");
  static_assert(Bc % 8 == 0, "Bc must be a multiple of 8 for tcgen05 N = 8");
  static_assert(D  % 16 == 0, "D  must be a multiple of 16 for tcgen05 dense");

  static bool cfgd = false;
  static CUtensorMap Ktmap, Vtmap;
  static int nsm = 0;
  if(!cfgd){
    const uint64_t kvRows = (uint64_t)B * Hkv * S;   // K/V flattened as [B*Hkv*S, D]
    Ktmap = make_tma_2d(d_K, kvRows, (uint64_t)D, (uint32_t)Bc, (uint32_t)D);
    Vtmap = make_tma_2d(d_V, kvRows, (uint64_t)D, (uint32_t)Bc, (uint32_t)D);
    int dev; cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
    cfgd = true;
  }
  const int totalWork = B * Hq * (S / Br);
  dim3 GRID(nsm < totalWork ? nsm : totalWork);
  dim3 BLOCK(128);
  gqa_v16<Br, Bc, D><<<GRID, BLOCK>>>(d_Q, d_O, d_LSE, Ktmap, Vtmap,
                          B, Hq, Hkv, G, S, scale);
}

// V17 — V13 core + warp specialization (gqa_v17, Bc=128): a dedicated 5th "producer"
// warp (threads 128..159) issues K/V TMA loads decoupled from the 4 "consumer" warps
// (threads 0..127, unchanged V13 math), handshaking via load_bar/free_bar mbarriers
// instead of a block-wide __syncthreads. Same single-buffered smem footprint as V13 —
// this step is about overlapping TMA latency with compute, not about adding a buffer.
template<int Br, int Bc, int D>
void launch_gqa_v17(
  __nv_bfloat16 *d_Q, __nv_bfloat16 *d_K, __nv_bfloat16 *d_V,
  __nv_bfloat16 *d_O, float *d_LSE,
  int B, int Hq, int Hkv, int S, int G, float scale
){
  static_assert(Br == 128, "V17's consumer group is hardwired to 128 threads");
  static_assert(Bc % 8 == 0, "Bc must be a multiple of 8 for tcgen05 N = 8");
  static_assert(D  % 16 == 0, "D  must be a multiple of 16 for tcgen05 dense");

  dim3 GRID(B, Hq, S/Br);
  dim3 BLOCK(160);   // 128 consumer threads (warps 0-3) + 32 producer threads (warp 4)

  static bool cfgd = false;
  static CUtensorMap Ktmap, Vtmap;
  if(!cfgd){
    const uint64_t kvRows = (uint64_t)B * Hkv * S;   // K/V flattened as [B*Hkv*S, D]
    Ktmap = make_tma_2d(d_K, kvRows, (uint64_t)D, (uint32_t)Bc, (uint32_t)D);
    Vtmap = make_tma_2d(d_V, kvRows, (uint64_t)D, (uint32_t)Bc, (uint32_t)D);
    cfgd = true;
  }
  gqa_v17<Br, Bc, D><<<GRID, BLOCK>>>(d_Q, d_O, d_LSE, Ktmap, Vtmap,
                          B, Hq, Hkv, G, S, scale);
}

// V18 — V17 + double-buffered TMA staging (gqa_v18, Bc=128): ping-pong sKstage/sVstage
// so the producer can run a full iteration ahead instead of just to the reorder point.
// sS moves to bf16 to fit the extra staging buffer under the static-smem ceiling.
template<int Br, int Bc, int D>
void launch_gqa_v18(
  __nv_bfloat16 *d_Q, __nv_bfloat16 *d_K, __nv_bfloat16 *d_V,
  __nv_bfloat16 *d_O, float *d_LSE,
  int B, int Hq, int Hkv, int S, int G, float scale
){
  static_assert(Br == 128, "V18's consumer group is hardwired to 128 threads");
  static_assert(Bc % 8 == 0, "Bc must be a multiple of 8 for tcgen05 N = 8");
  static_assert(D  % 16 == 0, "D  must be a multiple of 16 for tcgen05 dense");

  dim3 GRID(B, Hq, S/Br);
  dim3 BLOCK(160);   // 128 consumer threads (warps 0-3) + 32 producer threads (warp 4)

  static bool cfgd = false;
  static CUtensorMap Ktmap, Vtmap;
  if(!cfgd){
    const uint64_t kvRows = (uint64_t)B * Hkv * S;   // K/V flattened as [B*Hkv*S, D]
    Ktmap = make_tma_2d(d_K, kvRows, (uint64_t)D, (uint32_t)Bc, (uint32_t)D);
    Vtmap = make_tma_2d(d_V, kvRows, (uint64_t)D, (uint32_t)Bc, (uint32_t)D);
    cfgd = true;
  }
  gqa_v18<Br, Bc, D><<<GRID, BLOCK>>>(d_Q, d_O, d_LSE, Ktmap, Vtmap,
                          B, Hq, Hkv, G, S, scale);
}


int main(){
  std::cout << "Benchmarking Grouped-Query Attention kernels — Blackwell SM_103 (B300)\n";

  constexpr int B   = 8;     //! batch - later try 32
  constexpr int Hq  = 12;     // number of query heads
  constexpr int Hkv = 4;      // number of key/value heads
  constexpr int G   = Hq/Hkv; // groups per KV head
  constexpr int S   = 4096;   // context length
  constexpr int D   = 64;     // head dimension
  constexpr int Br  = 16;     // tile size along the query sequence dimension (v1/v2, one warp)
  constexpr int Bc  = 32;     // tile size along the key/value sequence dimension
  constexpr int Br_64 = 64;   // v3 (tcgen05) requires M = 64 → Br = 64, 4 warps/block
  constexpr int Br_128 = 128;   // v6 (tcgen05) requires N % 8 == 0 → Bc = 32
  constexpr int Bc_64 = 64;   // v10: halve the KV-tile count (128 -> 64 tiles)
  constexpr int Bc_128 = 128; // v13: 32 KV tiles (needs the V11+V12 smem diet)

  static_assert(Hq % Hkv == 0, "Hq must be divisible by Hkv");
  static_assert(S  % Br    == 0, "S must be divisible by Br");
  static_assert(S  % Br_64 == 0, "S must be divisible by Br_64");
  static_assert(S  % Br_128 == 0, "S must be divisible by Br_128");
  static_assert(S  % Bc    == 0, "S must be divisible by Bc");
  static_assert(S  % Bc_64 == 0, "S must be divisible by Bc_64");
  static_assert(S  % Bc_128 == 0, "S must be divisible by Bc_128");

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
    launch_gqa_v0<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_O.data(),   d_O,   Nq   * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_LSE.data(), d_LSE, Nlse * sizeof(float),         cudaMemcpyDeviceToHost));

    // widen the bf16 output to fp32 so checkResult / reportPrecision can compare it
    std::vector<float> h_O_f32(Nq);
    for(size_t i = 0; i < Nq; ++i) h_O_f32[i] = __bfloat162float(h_O[i]);

    // bf16 attention: ~2^-8 relative precision, so use bf16-scale tolerances.
    std::cout << "\nCorrectness V0 (WMMA two-pass vs PyTorch bf16 SDPA):\n";
    reportPrecision("  output O ", h_O_ref.data(),   h_O_f32.data(), Nq);
    reportPrecision("  lse      ", h_LSE_ref.data(), h_LSE.data(),   Nlse);
    std::cout << "  O   : "; checkResult(h_O_ref.data(),   h_O_f32.data(), Nq,   2e-2f, 2e-2f);
    std::cout << "  LSE : "; checkResult(h_LSE_ref.data(), h_LSE.data(),   Nlse, 2e-2f, 2e-2f);

    // ── V1 : WMMA online softmax — same bf16 in / bf16 out comparison path ──
    launch_gqa_v1<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_O.data(),   d_O,   Nq   * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_LSE.data(), d_LSE, Nlse * sizeof(float),         cudaMemcpyDeviceToHost));

    // widen the bf16 output to fp32 so checkResult / reportPrecision can compare it
    for(size_t i = 0; i < Nq; ++i) h_O_f32[i] = __bfloat162float(h_O[i]);

    std::cout << "\nCorrectness V1 (WMMA online softmax vs PyTorch bf16 SDPA):\n";
    reportPrecision("  output O ", h_O_ref.data(),   h_O_f32.data(), Nq);
    reportPrecision("  lse      ", h_LSE_ref.data(), h_LSE.data(),   Nlse);
    std::cout << "  O   : "; checkResult(h_O_ref.data(),   h_O_f32.data(), Nq,   2e-2f, 2e-2f);
    std::cout << "  LSE : "; checkResult(h_LSE_ref.data(), h_LSE.data(),   Nlse, 2e-2f, 2e-2f);

    // ── V2 : tcgen05 online softmax (Br = 64, 4 warps/block) ──
    launch_gqa_v2<Br_64, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_O.data(),   d_O,   Nq   * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_LSE.data(), d_LSE, Nlse * sizeof(float),         cudaMemcpyDeviceToHost));

    for(size_t i = 0; i < Nq; ++i) h_O_f32[i] = __bfloat162float(h_O[i]);

    std::cout << "\nCorrectness V2 (tcgen05 online softmax vs PyTorch bf16 SDPA):\n";
    reportPrecision("  output O ", h_O_ref.data(),   h_O_f32.data(), Nq);
    reportPrecision("  lse      ", h_LSE_ref.data(), h_LSE.data(),   Nlse);
    std::cout << "  O   : "; checkResult(h_O_ref.data(),   h_O_f32.data(), Nq,   2e-2f, 2e-2f);
    std::cout << "  LSE : "; checkResult(h_LSE_ref.data(), h_LSE.data(),   Nlse, 2e-2f, 2e-2f);

    // ── V3 : tcgen05 + cp.async double-buffered KV loads ──
    launch_gqa_v3<Br_64, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_O.data(),   d_O,   Nq   * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_LSE.data(), d_LSE, Nlse * sizeof(float),         cudaMemcpyDeviceToHost));

    for(size_t i = 0; i < Nq; ++i) h_O_f32[i] = __bfloat162float(h_O[i]);

    std::cout << "\nCorrectness V3 (tcgen05 + cp.async pipeline vs PyTorch bf16 SDPA):\n";
    reportPrecision("  output O ", h_O_ref.data(),   h_O_f32.data(), Nq);
    reportPrecision("  lse      ", h_LSE_ref.data(), h_LSE.data(),   Nlse);
    std::cout << "  O   : "; checkResult(h_O_ref.data(),   h_O_f32.data(), Nq,   2e-2f, 2e-2f);
    std::cout << "  LSE : "; checkResult(h_LSE_ref.data(), h_LSE.data(),   Nlse, 2e-2f, 2e-2f);

    // ── V4 : V3 + vectorized TMEM readout ──
    launch_gqa_v4<Br_64, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_O.data(),   d_O,   Nq   * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_LSE.data(), d_LSE, Nlse * sizeof(float),         cudaMemcpyDeviceToHost));

    for(size_t i = 0; i < Nq; ++i) h_O_f32[i] = __bfloat162float(h_O[i]);

    std::cout << "\nCorrectness V4 (V3 + vectorized readout vs PyTorch bf16 SDPA):\n";
    reportPrecision("  output O ", h_O_ref.data(),   h_O_f32.data(), Nq);
    reportPrecision("  lse      ", h_LSE_ref.data(), h_LSE.data(),   Nlse);
    std::cout << "  O   : "; checkResult(h_O_ref.data(),   h_O_f32.data(), Nq,   2e-2f, 2e-2f);
    std::cout << "  LSE : "; checkResult(h_LSE_ref.data(), h_LSE.data(),   Nlse, 2e-2f, 2e-2f);

    // ── V5 : V4 + TMA KV loads ──
    launch_gqa_v5<Br_64, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_O.data(),   d_O,   Nq   * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_LSE.data(), d_LSE, Nlse * sizeof(float),         cudaMemcpyDeviceToHost));

    for(size_t i = 0; i < Nq; ++i) h_O_f32[i] = __bfloat162float(h_O[i]);

    std::cout << "\nCorrectness V5 (V4 + TMA KV loads vs PyTorch bf16 SDPA):\n";
    reportPrecision("  output O ", h_O_ref.data(),   h_O_f32.data(), Nq);
    reportPrecision("  lse      ", h_LSE_ref.data(), h_LSE.data(),   Nlse);
    std::cout << "  O   : "; checkResult(h_O_ref.data(),   h_O_f32.data(), Nq,   2e-2f, 2e-2f);
    std::cout << "  LSE : "; checkResult(h_LSE_ref.data(), h_LSE.data(),   Nlse, 2e-2f, 2e-2f);

    // ── V6 : V5 + Br = 128 ──
    launch_gqa_v6<Br_128, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_O.data(),   d_O,   Nq   * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_LSE.data(), d_LSE, Nlse * sizeof(float),         cudaMemcpyDeviceToHost));

    for(size_t i = 0; i < Nq; ++i) h_O_f32[i] = __bfloat162float(h_O[i]);

    std::cout << "\nCorrectness V6 (V5 + Br = 128 vs PyTorch bf16 SDPA):\n";
    reportPrecision("  output O ", h_O_ref.data(),   h_O_f32.data(), Nq);
    reportPrecision("  lse      ", h_LSE_ref.data(), h_LSE.data(),   Nlse);
    std::cout << "  O   : "; checkResult(h_O_ref.data(),   h_O_f32.data(), Nq,   2e-2f, 2e-2f);
    std::cout << "  LSE : "; checkResult(h_LSE_ref.data(), h_LSE.data(),   Nlse, 2e-2f, 2e-2f);

    // ── V7 : V6 + log2-domain softmax ──
    launch_gqa_v7<Br_128, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_O.data(),   d_O,   Nq   * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_LSE.data(), d_LSE, Nlse * sizeof(float),         cudaMemcpyDeviceToHost));

    for(size_t i = 0; i < Nq; ++i) h_O_f32[i] = __bfloat162float(h_O[i]);

    std::cout << "\nCorrectness V7 (V6 + log2-domain softmax vs PyTorch bf16 SDPA):\n";
    reportPrecision("  output O ", h_O_ref.data(),   h_O_f32.data(), Nq);
    reportPrecision("  lse      ", h_LSE_ref.data(), h_LSE.data(),   Nlse);
    std::cout << "  O   : "; checkResult(h_O_ref.data(),   h_O_f32.data(), Nq,   2e-2f, 2e-2f);
    std::cout << "  LSE : "; checkResult(h_LSE_ref.data(), h_LSE.data(),   Nlse, 2e-2f, 2e-2f);

    // ── V8 : V7 + packed bf16 conversion ──
    launch_gqa_v8<Br_128, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_O.data(),   d_O,   Nq   * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_LSE.data(), d_LSE, Nlse * sizeof(float),         cudaMemcpyDeviceToHost));

    for(size_t i = 0; i < Nq; ++i) h_O_f32[i] = __bfloat162float(h_O[i]);

    std::cout << "\nCorrectness V8 (V7 + packed bf16 conversion vs PyTorch bf16 SDPA):\n";
    reportPrecision("  output O ", h_O_ref.data(),   h_O_f32.data(), Nq);
    reportPrecision("  lse      ", h_LSE_ref.data(), h_LSE.data(),   Nlse);
    std::cout << "  O   : "; checkResult(h_O_ref.data(),   h_O_f32.data(), Nq,   2e-2f, 2e-2f);
    std::cout << "  LSE : "; checkResult(h_LSE_ref.data(), h_LSE.data(),   Nlse, 2e-2f, 2e-2f);

    // ── V9 : V8 + software-pipelined QK^T (MMA/softmax overlap) ──
    launch_gqa_v9<Br_128, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_O.data(),   d_O,   Nq   * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_LSE.data(), d_LSE, Nlse * sizeof(float),         cudaMemcpyDeviceToHost));

    for(size_t i = 0; i < Nq; ++i) h_O_f32[i] = __bfloat162float(h_O[i]);

    std::cout << "\nCorrectness V9 (V8 + software-pipelined QK^T vs PyTorch bf16 SDPA):\n";
    reportPrecision("  output O ", h_O_ref.data(),   h_O_f32.data(), Nq);
    reportPrecision("  lse      ", h_LSE_ref.data(), h_LSE.data(),   Nlse);
    std::cout << "  O   : "; checkResult(h_O_ref.data(),   h_O_f32.data(), Nq,   2e-2f, 2e-2f);
    std::cout << "  LSE : "; checkResult(h_LSE_ref.data(), h_LSE.data(),   Nlse, 2e-2f, 2e-2f);

    // ── V10 : V8 kernel at Bc = 64 (half the KV tiles) ──
    launch_gqa_v10<Br_128, Bc_64, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_O.data(),   d_O,   Nq   * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_LSE.data(), d_LSE, Nlse * sizeof(float),         cudaMemcpyDeviceToHost));

    for(size_t i = 0; i < Nq; ++i) h_O_f32[i] = __bfloat162float(h_O[i]);

    std::cout << "\nCorrectness V10 (V8 at Bc = 64 vs PyTorch bf16 SDPA):\n";
    reportPrecision("  output O ", h_O_ref.data(),   h_O_f32.data(), Nq);
    reportPrecision("  lse      ", h_LSE_ref.data(), h_LSE.data(),   Nlse);
    std::cout << "  O   : "; checkResult(h_O_ref.data(),   h_O_f32.data(), Nq,   2e-2f, 2e-2f);
    std::cout << "  LSE : "; checkResult(h_LSE_ref.data(), h_LSE.data(),   Nlse, 2e-2f, 2e-2f);

    // ── V11 : V10 + P@V readout fused into O accumulation ──
    launch_gqa_v11<Br_128, Bc_64, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_O.data(),   d_O,   Nq   * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_LSE.data(), d_LSE, Nlse * sizeof(float),         cudaMemcpyDeviceToHost));

    for(size_t i = 0; i < Nq; ++i) h_O_f32[i] = __bfloat162float(h_O[i]);

    std::cout << "\nCorrectness V11 (V10 + fused P@V accumulate vs PyTorch bf16 SDPA):\n";
    reportPrecision("  output O ", h_O_ref.data(),   h_O_f32.data(), Nq);
    reportPrecision("  lse      ", h_LSE_ref.data(), h_LSE.data(),   Nlse);
    std::cout << "  O   : "; checkResult(h_O_ref.data(),   h_O_f32.data(), Nq,   2e-2f, 2e-2f);
    std::cout << "  LSE : "; checkResult(h_LSE_ref.data(), h_LSE.data(),   Nlse, 2e-2f, 2e-2f);

    // ── V12 : V11 + single-buffered TMA staging ──
    launch_gqa_v12<Br_128, Bc_64, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_O.data(),   d_O,   Nq   * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_LSE.data(), d_LSE, Nlse * sizeof(float),         cudaMemcpyDeviceToHost));

    for(size_t i = 0; i < Nq; ++i) h_O_f32[i] = __bfloat162float(h_O[i]);

    std::cout << "\nCorrectness V12 (V11 + single-buffered staging vs PyTorch bf16 SDPA):\n";
    reportPrecision("  output O ", h_O_ref.data(),   h_O_f32.data(), Nq);
    reportPrecision("  lse      ", h_LSE_ref.data(), h_LSE.data(),   Nlse);
    std::cout << "  O   : "; checkResult(h_O_ref.data(),   h_O_f32.data(), Nq,   2e-2f, 2e-2f);
    std::cout << "  LSE : "; checkResult(h_LSE_ref.data(), h_LSE.data(),   Nlse, 2e-2f, 2e-2f);

    // ── V13 : V12 kernel at Bc = 128 (32 KV tiles) ──
    launch_gqa_v13<Br_128, Bc_128, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_O.data(),   d_O,   Nq   * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_LSE.data(), d_LSE, Nlse * sizeof(float),         cudaMemcpyDeviceToHost));

    for(size_t i = 0; i < Nq; ++i) h_O_f32[i] = __bfloat162float(h_O[i]);

    std::cout << "\nCorrectness V13 (V12 at Bc = 128 vs PyTorch bf16 SDPA):\n";
    reportPrecision("  output O ", h_O_ref.data(),   h_O_f32.data(), Nq);
    reportPrecision("  lse      ", h_LSE_ref.data(), h_LSE.data(),   Nlse);
    std::cout << "  O   : "; checkResult(h_O_ref.data(),   h_O_f32.data(), Nq,   2e-2f, 2e-2f);
    std::cout << "  LSE : "; checkResult(h_LSE_ref.data(), h_LSE.data(),   Nlse, 2e-2f, 2e-2f);

    // ── V14 : V13 + cluster launch scaffolding ──
    launch_gqa_v14<Br_128, Bc_128, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_O.data(),   d_O,   Nq   * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_LSE.data(), d_LSE, Nlse * sizeof(float),         cudaMemcpyDeviceToHost));

    for(size_t i = 0; i < Nq; ++i) h_O_f32[i] = __bfloat162float(h_O[i]);

    std::cout << "\nCorrectness V14 (V13 + cluster launch vs PyTorch bf16 SDPA):\n";
    reportPrecision("  output O ", h_O_ref.data(),   h_O_f32.data(), Nq);
    reportPrecision("  lse      ", h_LSE_ref.data(), h_LSE.data(),   Nlse);
    std::cout << "  O   : "; checkResult(h_O_ref.data(),   h_O_f32.data(), Nq,   2e-2f, 2e-2f);
    std::cout << "  LSE : "; checkResult(h_LSE_ref.data(), h_LSE.data(),   Nlse, 2e-2f, 2e-2f);

    // ── V15 : V14 + multicast K/V TMA ──
    launch_gqa_v15<Br_128, Bc_128, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_O.data(),   d_O,   Nq   * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_LSE.data(), d_LSE, Nlse * sizeof(float),         cudaMemcpyDeviceToHost));

    for(size_t i = 0; i < Nq; ++i) h_O_f32[i] = __bfloat162float(h_O[i]);

    std::cout << "\nCorrectness V15 (V14 + multicast K/V TMA vs PyTorch bf16 SDPA):\n";
    reportPrecision("  output O ", h_O_ref.data(),   h_O_f32.data(), Nq);
    reportPrecision("  lse      ", h_LSE_ref.data(), h_LSE.data(),   Nlse);
    std::cout << "  O   : "; checkResult(h_O_ref.data(),   h_O_f32.data(), Nq,   2e-2f, 2e-2f);
    std::cout << "  LSE : "; checkResult(h_LSE_ref.data(), h_LSE.data(),   Nlse, 2e-2f, 2e-2f);

    // ── V16 : persistent kernel (V13 core, 1 resident CTA/SM) ──
    launch_gqa_v16<Br_128, Bc_128, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_O.data(),   d_O,   Nq   * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_LSE.data(), d_LSE, Nlse * sizeof(float),         cudaMemcpyDeviceToHost));

    for(size_t i = 0; i < Nq; ++i) h_O_f32[i] = __bfloat162float(h_O[i]);

    std::cout << "\nCorrectness V16 (persistent kernel vs PyTorch bf16 SDPA):\n";
    reportPrecision("  output O ", h_O_ref.data(),   h_O_f32.data(), Nq);
    reportPrecision("  lse      ", h_LSE_ref.data(), h_LSE.data(),   Nlse);
    std::cout << "  O   : "; checkResult(h_O_ref.data(),   h_O_f32.data(), Nq,   2e-2f, 2e-2f);
    std::cout << "  LSE : "; checkResult(h_LSE_ref.data(), h_LSE.data(),   Nlse, 2e-2f, 2e-2f);

    // ── V17 : V13 core + warp specialization (dedicated TMA producer warp) ──
    launch_gqa_v17<Br_128, Bc_128, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_O.data(),   d_O,   Nq   * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_LSE.data(), d_LSE, Nlse * sizeof(float),         cudaMemcpyDeviceToHost));

    for(size_t i = 0; i < Nq; ++i) h_O_f32[i] = __bfloat162float(h_O[i]);

    std::cout << "\nCorrectness V17 (V13 + warp-specialized TMA producer vs PyTorch bf16 SDPA):\n";
    reportPrecision("  output O ", h_O_ref.data(),   h_O_f32.data(), Nq);
    reportPrecision("  lse      ", h_LSE_ref.data(), h_LSE.data(),   Nlse);
    std::cout << "  O   : "; checkResult(h_O_ref.data(),   h_O_f32.data(), Nq,   2e-2f, 2e-2f);
    std::cout << "  LSE : "; checkResult(h_LSE_ref.data(), h_LSE.data(),   Nlse, 2e-2f, 2e-2f);

    // ── V18 : V17 + double-buffered TMA staging (bf16 sS to fit smem) ──
    launch_gqa_v18<Br_128, Bc_128, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_O.data(),   d_O,   Nq   * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_LSE.data(), d_LSE, Nlse * sizeof(float),         cudaMemcpyDeviceToHost));

    for(size_t i = 0; i < Nq; ++i) h_O_f32[i] = __bfloat162float(h_O[i]);

    std::cout << "\nCorrectness V18 (V17 + double-buffered TMA staging vs PyTorch bf16 SDPA):\n";
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

  KernelStats stats_v0 = benchmarkKernel(
    [&](){ launch_gqa_v0<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); },
    100, 25, flops, bytes
  );
  displayStats("V0 — WMMA two-pass (stable softmax, bf16)", stats_v0);

  KernelStats stats_v1 = benchmarkKernel(
    [&](){ launch_gqa_v1<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); },
    100, 25, flops, bytes
  );
  displayStats("V1 — WMMA online softmax (single-pass, bf16)", stats_v1);

  KernelStats stats_v2 = benchmarkKernel(
    [&](){ launch_gqa_v2<Br_64, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); },
    100, 25, flops, bytes
  );
  displayStats("V2 — tcgen05 online softmax (single-pass, bf16)", stats_v2);

  KernelStats stats_v3 = benchmarkKernel(
    [&](){ launch_gqa_v3<Br_64, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); },
    100, 25, flops, bytes
  );
  displayStats("V3 — tcgen05 + cp.async KV pipeline (single-pass, bf16)", stats_v3);

  KernelStats stats_v4 = benchmarkKernel(
    [&](){ launch_gqa_v4<Br_64, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); },
    100, 25, flops, bytes
  );
  displayStats("V4 — V3 + vectorized TMEM readout (single-pass, bf16)", stats_v4);

  KernelStats stats_v5 = benchmarkKernel(
    [&](){ launch_gqa_v5<Br_64, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); },
    100, 25, flops, bytes
  );
  displayStats("V5 — V4 + TMA KV loads (single-pass, bf16)", stats_v5);

  KernelStats stats_v6 = benchmarkKernel(
    [&](){ launch_gqa_v6<Br_128, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); },
    100, 25, flops, bytes
  );
  displayStats("V6 — V5 + Br = 128 (single-pass, bf16)", stats_v6);

  KernelStats stats_v7 = benchmarkKernel(
    [&](){ launch_gqa_v7<Br_128, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); },
    100, 25, flops, bytes
  );
  displayStats("V7 — V6 + log2-domain softmax (single-pass, bf16)", stats_v7);

  KernelStats stats_v8 = benchmarkKernel(
    [&](){ launch_gqa_v8<Br_128, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); },
    100, 25, flops, bytes
  );
  displayStats("V8 — V7 + packed bf16 conversion (single-pass, bf16)", stats_v8);

  KernelStats stats_v9 = benchmarkKernel(
    [&](){ launch_gqa_v9<Br_128, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); },
    100, 25, flops, bytes
  );
  displayStats("V9 — V8 + software-pipelined QK^T (MMA/softmax overlap, bf16)", stats_v9);

  KernelStats stats_v10 = benchmarkKernel(
    [&](){ launch_gqa_v10<Br_128, Bc_64, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); },
    100, 25, flops, bytes
  );
  displayStats("V10 — V8 at Bc = 64 (half the KV tiles, bf16)", stats_v10);

  KernelStats stats_v11 = benchmarkKernel(
    [&](){ launch_gqa_v11<Br_128, Bc_64, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); },
    100, 25, flops, bytes
  );
  displayStats("V11 — V10 + fused P@V accumulate (no sPV staging, bf16)", stats_v11);

  KernelStats stats_v12 = benchmarkKernel(
    [&](){ launch_gqa_v12<Br_128, Bc_64, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); },
    100, 25, flops, bytes
  );
  displayStats("V12 — V11 + single-buffered TMA staging (no prefetch, bf16)", stats_v12);

  KernelStats stats_v13 = benchmarkKernel(
    [&](){ launch_gqa_v13<Br_128, Bc_128, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); },
    100, 25, flops, bytes
  );
  displayStats("V13 — V12 at Bc = 128 (32 KV tiles, bf16)", stats_v13);

  KernelStats stats_v14 = benchmarkKernel(
    [&](){ launch_gqa_v14<Br_128, Bc_128, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); },
    100, 25, flops, bytes
  );
  displayStats("V14 — V13 + cluster launch (2-CTA scaffolding, bf16)", stats_v14);

  KernelStats stats_v15 = benchmarkKernel(
    [&](){ launch_gqa_v15<Br_128, Bc_128, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); },
    100, 25, flops, bytes
  );
  displayStats("V15 — V14 + multicast K/V TMA (1 fetch per cluster, bf16)", stats_v15);

  KernelStats stats_v16 = benchmarkKernel(
    [&](){ launch_gqa_v16<Br_128, Bc_128, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); },
    100, 25, flops, bytes
  );
  displayStats("V16 — persistent kernel (V13 core, 1 resident CTA/SM, bf16)", stats_v16);

  KernelStats stats_v17 = benchmarkKernel(
    [&](){ launch_gqa_v17<Br_128, Bc_128, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); },
    100, 25, flops, bytes
  );
  displayStats("V17 — V13 + warp specialization (dedicated TMA producer warp, bf16)", stats_v17);

  KernelStats stats_v18 = benchmarkKernel(
    [&](){ launch_gqa_v18<Br_128, Bc_128, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); },
    100, 25, flops, bytes
  );
  displayStats("V18 — V17 + double-buffered TMA staging (bf16 sS, bf16)", stats_v18);

  CUDA_CHECK(cudaFree(d_Q));
  CUDA_CHECK(cudaFree(d_K));
  CUDA_CHECK(cudaFree(d_V));
  CUDA_CHECK(cudaFree(d_O));
  CUDA_CHECK(cudaFree(d_LSE));

  return 0;
}
