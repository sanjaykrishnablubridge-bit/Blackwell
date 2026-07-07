// =============================================================================
// FUSED GQA Flash Attention FORWARD — QK-norm + RoPE + attention (bf16)
//
// Self-contained benchmark + correctness harness, mirroring GQA_sm103.cu. This
// file builds to its own executable (one main()); add new kernel versions and
// they are correctness-checked and timed automatically.
//
// The fused op, per (token, head) row over head_dim D, applied to Q and K
// BEFORE the QK^T GEMM:
//   1. QK-norm (RMSNorm over D):  rstd = 1/sqrt(mean(x^2)+eps),
//                                 x_hat = x * rstd * gamma.   (skip if gamma==null)
//   2. RoPE rotation of the normalized vector (NeoX half-split or GPT-J interleaved).
// V is untouched. q_rstd / k_rstd are written out (fp32) for the backward, like LSE.
//
// Precision: bf16 storage, fp32 math. The cos/sin cache is fp32
// [cache_seq_len, D] (cos in [0,D/2), sin in [D/2,D)); theta baked in.
//
// CORRECTNESS: this harness computes its OWN ground truth on the GPU — a naive,
// straightforward implementation of the exact fused spec (see reference kernels
// below), precision-matched to the tiled kernels (same bf16 rounding points).
// No external .bin / PyTorch reference is required; it works at full size.
//
// -----------------------------------------------------------------------------
// HOW TO ADD A NEW VERSION (e.g. tcgen05, cp.async, TMA — see GQA_sm103.cu):
//   1. Write   __global__ void gqa_fused_vN(...)      (any internal design)
//   2. Write   template<...> void launch_gqa_fused_vN(...)  with the SAME
//      argument list as launch_gqa_fused_v0 so the harness call is uniform.
//   3. In main(), add ONE line to the `versions` vector:
//        versions.push_back({"VN — my description",
//          [&]{ launch_gqa_fused_vN<Br,Bc,D>( /* same args as V0 */ ); }});
//   It is then checked for correctness AND benchmarked, no other edits needed.
//
// Build + run:  from the repo root,  cmake --build build --target GQA_fused_sm103
//               ./build/bin/GQA_fused_sm103
// (tcgen05 versions require an sm_103 GPU — see kernel_guidelines / the memory.)
// =============================================================================
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cmath>
#include <cstdio>
#include <vector>
#include <string>
#include <utility>
#include <functional>

#include "utils/kernelUtils.cuh"
#include "utils/kernelBench.cuh"

using namespace nvcuda;

// ===========================================================================
//  Shared device helper: QK-norm (RMSNorm) then RoPE on a shared-memory tile.
//  `s` is [rows, D] row-major bf16; one warp lane per row (rows <= 32).
//  gamma == nullptr -> skip QK-norm (RoPE only). rstd_out != nullptr -> write
//  per-row rstd (stride 1). Math in fp32; values stored back as bf16, with a
//  bf16 round between norm and RoPE (the reference mirrors this exactly).
// ===========================================================================
template<bool IS_NEOX, int D>
__device__ __forceinline__ void norm_rope_tile(
    __nv_bfloat16* s, int rows, int lane,
    const float* __restrict__ cos_sin_cache,
    const float* __restrict__ gamma,   // [D] or nullptr
    int base_pos, int cache_seq_len, float eps,
    float* __restrict__ rstd_out)      // per-row stride-1 dest, or nullptr
{
    const int half = D / 2;
    if (lane < rows) {
        __nv_bfloat16* row = s + lane * D;

        // --- QK-norm over D ---
        if (gamma != nullptr) {
            float ss = 0.0f;
            for (int d = 0; d < D; ++d) {
                float v = __bfloat162float(row[d]);
                ss += v * v;
            }
            float rstd = rsqrtf(ss / (float)D + eps);
            if (rstd_out != nullptr) rstd_out[lane] = rstd;
            for (int d = 0; d < D; ++d)
                row[d] = __float2bfloat16(__bfloat162float(row[d]) * rstd * gamma[d]);
        }

        // --- RoPE (in place) ---
        const int pos = base_pos + lane;
        if (pos >= 0 && pos < cache_seq_len) {
            const float* cache_row = cos_sin_cache + (long)pos * D;
            for (int i = 0; i < half; ++i) {
                int a_idx = IS_NEOX ? i        : 2 * i;
                int b_idx = IS_NEOX ? i + half : 2 * i + 1;
                float x = __bfloat162float(row[a_idx]);
                float y = __bfloat162float(row[b_idx]);
                float c  = cache_row[i];
                float sn = cache_row[i + half];
                row[a_idx] = __float2bfloat16(x * c - y * sn);
                row[b_idx] = __float2bfloat16(x * sn + y * c);
            }
        }
    }
}

// ===========================================================================
//  V0 : WMMA online-softmax fused kernel (bf16). Known-good baseline.
//  One block = one (batch, query-head, Br-row tile); one warp per block.
// ===========================================================================
template<int Br, int Bc, int D, bool IS_NEOX>
__global__ void gqa_fused_v0(
  __nv_bfloat16 *d_Q,
  __nv_bfloat16 *d_K,
  __nv_bfloat16 *d_V,
  __nv_bfloat16 *d_O,
  float *d_LSE,
  const float *d_cache,
  const float *d_qg,        // q_gamma [D] or nullptr
  const float *d_kg,        // k_gamma [D] or nullptr
  float *d_qrstd,           // [B, Hq, S]  or nullptr
  float *d_krstd,           // [B, Hkv, S] or nullptr
  int B, int Hq, int Hkv, int G, int S,
  int cache_seq_len, int pos_offset, float eps,
  float scale, bool is_causal
){
  const int b      = blockIdx.x;
  const int hq     = blockIdx.y;
  const int q_tile = blockIdx.z;
  const int hkv    = hq / G;
  const int lane   = threadIdx.x;

  const int q_row0   = q_tile * Br;
  const int nKVTiles = S / Bc;

  const long qBase  = ((long)(b * Hq  + hq)  * S  + q_row0) * D;
  const long kvBase = ((long)(b * Hkv + hkv) * S) * D;
  const long lBase  = ((long)(b * Hq  + hq)  * S) + q_row0;

  __shared__ __align__(16) __nv_bfloat16 sQ[Br * D];
  __shared__ __align__(16) __nv_bfloat16 sK[Bc * D];
  __shared__ __align__(16) __nv_bfloat16 sV[Bc * D];
  __shared__ __align__(16) __nv_bfloat16 sP[Br * Bc];
  __shared__ __align__(16) float         sS[Br * Bc];
  __shared__ __align__(16) float         sO[Br * D];
  __shared__ __align__(16) float         sPV[Br * D];
  __shared__ float sm[Br];
  __shared__ float sl[Br];
  __shared__ float sCorr[Br];

  // Load Q tile, zero running output, init stats.
  for(int i = lane; i < Br * D; i += 32){
    sQ[i] = d_Q[qBase + i];
    sO[i] = 0.0f;
  }
  if(lane < Br){
    sm[lane] = -INFINITY;
    sl[lane] = 0.0f;
  }
  __syncwarp();

  // === FUSED: QK-norm + RoPE on Q (once; Q tile reused across all KV tiles) ===
  {
    float* qrstd_dst = (d_qrstd != nullptr) ? (d_qrstd + lBase) : nullptr;
    norm_rope_tile<IS_NEOX, D>(sQ, Br, lane, d_cache, d_qg,
                               q_row0 + pos_offset, cache_seq_len, eps, qrstd_dst);
  }
  __syncwarp();

  // Causal: KV tiles beginning past this tile's last query row are fully masked.
  const int kc_end = is_causal
      ? (((q_row0 + Br - 1) / Bc + 1) < nKVTiles ? ((q_row0 + Br - 1) / Bc + 1) : nKVTiles)
      : nKVTiles;

  for(int kc = 0; kc < kc_end; ++kc){
    const long kBase = kvBase + (long)kc * Bc * D;
    for(int i = lane; i < Bc * D; i += 32){
      sK[i] = d_K[kBase + i];
      sV[i] = d_V[kBase + i];
    }
    __syncwarp();

    // === FUSED: QK-norm + RoPE on this K tile (before QK^T) ===
    // k_rstd written only by the first query-head of each group (hq % G == 0)
    // to avoid G-fold duplicate writes; every head still norms its own smem copy.
    {
      const long krBase = ((long)(b * Hkv + hkv) * S) + kc * Bc;
      float* krstd_dst =
          (d_krstd != nullptr && (hq % G) == 0) ? (d_krstd + krBase) : nullptr;
      norm_rope_tile<IS_NEOX, D>(sK, Bc, lane, d_cache, d_kg,
                                 kc * Bc + pos_offset, cache_seq_len, eps, krstd_dst);
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

    // Online-softmax stats + unnormalized P, one lane per query row.
    if(lane < Br){
      const float m_old = sm[lane];
      const float l_old = sl[lane];
      const int   q_idx = q_row0 + lane;

      float tile_max = -INFINITY;
      for(int j = 0; j < Bc; ++j){
        float s = sS[lane * Bc + j];
        if(is_causal && (kc * Bc + j) > q_idx) s = -INFINITY;
        tile_max = fmaxf(tile_max, s);
      }

      const float m_new = fmaxf(m_old, tile_max);
      const float corr  = __expf(m_old - m_new);

      float p_sum = 0.0f;
      for(int j = 0; j < Bc; ++j){
        float p;
        if(is_causal && (kc * Bc + j) > q_idx) p = 0.0f;
        else                                   p = __expf(sS[lane * Bc + j] - m_new);
        sP[lane * Bc + j] = __float2bfloat16(p);
        p_sum += p;
      }

      sm[lane]    = m_new;
      sl[lane]    = l_old * corr + p_sum;
      sCorr[lane] = corr;
    }
    __syncwarp();

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

    for(int i = lane; i < Br * D; i += 32){
      sO[i] += sPV[i];
    }
    __syncwarp();
  } // end KV loop

  // Finalize: normalize by running denominator and write LSE.
  for(int i = lane; i < Br * D; i += 32){
    d_O[qBase + i] = __float2bfloat16(sO[i] / sl[i / D]);
  }
  if(lane < Br){
    d_LSE[lBase + lane] = sm[lane] + logf(sl[lane]);
  }
}

// V0 launcher. The `interleaved` bool selects the RoPE pairing:
//   interleaved == false -> NeoX half-split (IS_NEOX = true)
//   interleaved == true  -> GPT-J interleaved (IS_NEOX = false)
template<int Br, int Bc, int D>
void launch_gqa_fused_v0(
  __nv_bfloat16 *d_Q, __nv_bfloat16 *d_K, __nv_bfloat16 *d_V,
  __nv_bfloat16 *d_O, float *d_LSE,
  const float *d_cache, const float *d_qg, const float *d_kg,
  float *d_qrstd, float *d_krstd,
  int B, int Hq, int Hkv, int G, int S,
  int cache_seq_len, int pos_offset, float eps, bool interleaved, float scale,
  bool is_causal
){
  static_assert(Br % 16 == 0, "Br must be a multiple of 16 (WMMA tile)");
  static_assert(Bc % 16 == 0, "Bc must be a multiple of 16 (WMMA tile)");
  static_assert(D  % 16 == 0, "D  must be a multiple of 16 (WMMA tile)");
  static_assert(Br == 16, "gqa_fused_v0 processes exactly 16 query rows per block");
  static_assert(D % 2 == 0, "head_dim must be even for RoPE");

  dim3 GRID(B, Hq, S / Br);
  dim3 BLOCK(32);
  if (interleaved) {
    gqa_fused_v0<Br, Bc, D, false><<<GRID, BLOCK>>>(
        d_Q, d_K, d_V, d_O, d_LSE, d_cache, d_qg, d_kg, d_qrstd, d_krstd,
        B, Hq, Hkv, G, S, cache_seq_len, pos_offset, eps, scale, is_causal);
  } else {
    gqa_fused_v0<Br, Bc, D, true><<<GRID, BLOCK>>>(
        d_Q, d_K, d_V, d_O, d_LSE, d_cache, d_qg, d_kg, d_qrstd, d_krstd,
        B, Hq, Hkv, G, S, cache_seq_len, pos_offset, eps, scale, is_causal);
  }
}

// ===========================================================================
//  [SCAFFOLD] Add your next version here — copy the two functions above,
//  rename to gqa_fused_v1 / launch_gqa_fused_v1, change the internals, then
//  register it in main() (one push_back). See the header comment.
// ===========================================================================


// ===========================================================================
//  GROUND-TRUTH REFERENCE (GPU, self-contained). Naive, straightforward
//  implementation of the exact fused spec, precision-matched to the kernels
//  (same bf16 rounding points), so agreement is limited only by fp32
//  accumulation order — comfortably within the bf16 tolerance used below.
// ===========================================================================

// Stage 1: apply QK-norm (if gamma!=null) then RoPE to every [D] row of `in`,
// writing bf16 `out` and (optionally) per-row rstd. One thread per row.
template<int D>
__global__ void ref_norm_rope(
    const __nv_bfloat16* __restrict__ in,
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ rstd_out,             // or nullptr
    const float* __restrict__ cache,
    const float* __restrict__ gamma,          // [D] or nullptr
    long rows, int S, int cache_seq_len, int pos_offset, float eps, bool is_neox)
{
  long r = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  const int half = D / 2;
  const __nv_bfloat16* row  = in  + r * D;
  __nv_bfloat16*       orow = out + r * D;

  float x[D];
  for (int d = 0; d < D; ++d) x[d] = __bfloat162float(row[d]);

  if (gamma != nullptr) {
    float ss = 0.0f;
    for (int d = 0; d < D; ++d) ss += x[d] * x[d];
    float rstd = rsqrtf(ss / (float)D + eps);
    if (rstd_out != nullptr) rstd_out[r] = rstd;
    // bf16 round between norm and RoPE (matches norm_rope_tile).
    for (int d = 0; d < D; ++d)
      x[d] = __bfloat162float(__float2bfloat16(x[d] * rstd * gamma[d]));
  }

  const int pos = (int)(r % S) + pos_offset;
  if (pos >= 0 && pos < cache_seq_len) {
    const float* cr = cache + (long)pos * D;
    for (int i = 0; i < half; ++i) {
      int a = is_neox ? i        : 2 * i;
      int bb = is_neox ? i + half : 2 * i + 1;
      float xa = x[a], xb = x[bb];
      float c  = cr[i];
      float sn = cr[i + half];
      x[a]  = xa * c  - xb * sn;
      x[bb] = xa * sn + xb * c;
    }
  }
  for (int d = 0; d < D; ++d) orow[d] = __float2bfloat16(x[d]);
}

// Stage 2: naive per-row online-softmax attention over the (already normed +
// RoPE'd) Qn / Kn and raw V. One thread per (b, hq, query-row). P is rounded to
// bf16 before the P·V accumulate, matching the tiled kernel's bf16 P.
template<int D>
__global__ void ref_attention(
    const __nv_bfloat16* __restrict__ Qn,
    const __nv_bfloat16* __restrict__ Kn,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int Hq, int Hkv, int G, int S, float scale, bool is_causal)
{
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;   // (b, hq, s)
  long total = (long)B * Hq * S;
  if (idx >= total) return;

  const int s   = (int)(idx % S);
  const int hq  = (int)((idx / S) % Hq);
  const int b   = (int)(idx / ((long)S * Hq));
  const int hkv = hq / G;

  const __nv_bfloat16* q = Qn + idx * D;
  const long kvBase = ((long)(b * Hkv + hkv) * S) * D;

  float qrow[D], acc[D];
  for (int d = 0; d < D; ++d) { qrow[d] = __bfloat162float(q[d]); acc[d] = 0.0f; }

  float m = -INFINITY, l = 0.0f;
  const int kmax = is_causal ? (s + 1) : S;
  for (int j = 0; j < kmax; ++j) {
    const __nv_bfloat16* k = Kn + kvBase + (long)j * D;
    float dot = 0.0f;
    for (int d = 0; d < D; ++d) dot += qrow[d] * __bfloat162float(k[d]);
    dot *= scale;

    float m_new = fmaxf(m, dot);
    float corr  = __expf(m - m_new);         // 0 on the first key (m = -inf)
    float p     = __expf(dot - m_new);
    float pb    = __bfloat162float(__float2bfloat16(p));   // bf16 P (matches kernel)

    const __nv_bfloat16* v = V + kvBase + (long)j * D;
    for (int d = 0; d < D; ++d) acc[d] = acc[d] * corr + pb * __bfloat162float(v[d]);
    l = l * corr + p;
    m = m_new;
  }

  const long oBase = idx * D;
  for (int d = 0; d < D; ++d) O[oBase + d] = __float2bfloat16(acc[d] / l);
  LSE[idx] = m + logf(l);
}


int main(){
  std::cout << "Benchmarking FUSED GQA (QK-norm + RoPE + attention) kernels\n";

  // ─────────────────── Problem + fused config (edit here) ───────────────────
  constexpr int B   = 16;     // batch
  constexpr int Hq  = 12;     // query heads
  constexpr int Hkv = 4;      // key/value heads
  constexpr int G   = Hq / Hkv;
  constexpr int S   = 4096;   // context length
  constexpr int D   = 64;     // head dim (RoPE needs D even; WMMA needs D % 16 == 0)
  constexpr int Br  = 16;     // query-row tile (V0 = one warp, 16 rows)
  constexpr int Bc  = 32;     // key/value tile

  constexpr bool INTERLEAVED = false;  // false = NeoX half-split, true = GPT-J interleaved
  constexpr bool IS_CAUSAL   = false;  // causal masking on/off
  constexpr bool USE_QKNORM  = true;   // apply RMSNorm(QK-norm) before RoPE
  const     float EPS        = 1e-6f;  // RMSNorm epsilon
  const     float THETA      = 10000.0f;
  constexpr int   POS_OFFSET = 0;
  const     int   CACHE_SEQ_LEN = S + POS_OFFSET;   // must be >= S + pos_offset
  const     bool  IS_NEOX    = !INTERLEAVED;

  static_assert(Hq % Hkv == 0, "Hq must be divisible by Hkv");
  static_assert(S % Br == 0 && S % Bc == 0, "S must be divisible by Br and Bc");
  static_assert(D % 2 == 0 && D % 16 == 0, "D must be even and a multiple of 16");

  const float scale = 1.0f / std::sqrt((float)D);

  const size_t Nq    = (size_t)B * Hq  * S * D;   // Q, O   [B, Hq,  S, D]
  const size_t Nkv   = (size_t)B * Hkv * S * D;   // K, V   [B, Hkv, S, D]
  const size_t Nlse  = (size_t)B * Hq  * S;       // LSE, q_rstd  [B, Hq,  S]
  const size_t Nkr   = (size_t)B * Hkv * S;       // k_rstd       [B, Hkv, S]
  const size_t Ncache = (size_t)CACHE_SEQ_LEN * D;

  // ─────────────────── Host inputs ───────────────────
  std::vector<__nv_bfloat16> h_Q(Nq), h_K(Nkv), h_V(Nkv);
  std::vector<float>         h_qg(D), h_kg(D), h_cache(Ncache);

  initPtr(h_Q.data(), (int)Nq);
  initPtr(h_K.data(), (int)Nkv);
  initPtr(h_V.data(), (int)Nkv);

  // gamma in [0.5, 1.5) so the QK-norm scale is actually exercised.
  { std::mt19937 rng(123); std::uniform_real_distribution<float> u(0.5f, 1.5f);
    for (int d = 0; d < D; ++d) { h_qg[d] = u(rng); h_kg[d] = u(rng); } }

  // cos/sin cache: cos in [0, D/2), sin in [D/2, D); angle = pos * theta^(-2i/D).
  { const int half = D / 2;
    for (int pos = 0; pos < CACHE_SEQ_LEN; ++pos)
      for (int i = 0; i < half; ++i) {
        float inv_freq = std::pow(THETA, -2.0f * (float)i / (float)D);
        float ang = (float)pos * inv_freq;
        h_cache[(size_t)pos * D + i]        = std::cos(ang);
        h_cache[(size_t)pos * D + half + i] = std::sin(ang);
      } }

  // ─────────────────── Device buffers ───────────────────
  __nv_bfloat16 *d_Q, *d_K, *d_V, *d_O;
  float *d_LSE, *d_cache, *d_qg, *d_kg, *d_qrstd, *d_krstd;
  CUDA_CHECK(cudaMalloc(&d_Q,     Nq     * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_K,     Nkv    * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_V,     Nkv    * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_O,     Nq     * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_LSE,   Nlse   * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_cache, Ncache * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_qg,    D      * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_kg,    D      * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_qrstd, Nlse   * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_krstd, Nkr    * sizeof(float)));

  // Reference (golden) buffers.
  __nv_bfloat16 *d_Qn, *d_Kn, *d_Oref;
  float *d_LSEref, *d_qrstd_ref, *d_krstd_ref;
  CUDA_CHECK(cudaMalloc(&d_Qn,        Nq   * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_Kn,        Nkv  * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_Oref,      Nq   * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_LSEref,    Nlse * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_qrstd_ref, Nlse * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_krstd_ref, Nkr  * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_Q,     h_Q.data(),     Nq     * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_K,     h_K.data(),     Nkv    * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_V,     h_V.data(),     Nkv    * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_cache, h_cache.data(), Ncache * sizeof(float),         cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_qg,    h_qg.data(),    D      * sizeof(float),         cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_kg,    h_kg.data(),    D      * sizeof(float),         cudaMemcpyHostToDevice));

  // Kernels see gamma/rstd only when QK-norm is enabled.
  const float* qg     = USE_QKNORM ? d_qg    : nullptr;
  const float* kg     = USE_QKNORM ? d_kg    : nullptr;
  float*       qrstd  = USE_QKNORM ? d_qrstd : nullptr;
  float*       krstd  = USE_QKNORM ? d_krstd : nullptr;

  // ─────────────────── Build the golden reference on GPU ───────────────────
  {
    const int T = 256;
    const long qRows = (long)B * Hq  * S;
    const long kRows = (long)B * Hkv * S;
    ref_norm_rope<D><<<(unsigned)((qRows + T - 1) / T), T>>>(
        d_Q, d_Qn, USE_QKNORM ? d_qrstd_ref : nullptr, d_cache,
        USE_QKNORM ? d_qg : nullptr, qRows, S, CACHE_SEQ_LEN, POS_OFFSET, EPS, IS_NEOX);
    ref_norm_rope<D><<<(unsigned)((kRows + T - 1) / T), T>>>(
        d_K, d_Kn, USE_QKNORM ? d_krstd_ref : nullptr, d_cache,
        USE_QKNORM ? d_kg : nullptr, kRows, S, CACHE_SEQ_LEN, POS_OFFSET, EPS, IS_NEOX);
    const long attnRows = (long)B * Hq * S;
    ref_attention<D><<<(unsigned)((attnRows + T - 1) / T), T>>>(
        d_Qn, d_Kn, d_V, d_Oref, d_LSEref, B, Hq, Hkv, G, S, scale, IS_CAUSAL);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // Reference -> host (widen O to fp32 for the comparators).
  std::vector<float> h_Oref(Nq), h_LSEref(Nlse), h_qrstd_ref(Nlse), h_krstd_ref(Nkr);
  {
    std::vector<__nv_bfloat16> tmpO(Nq);
    CUDA_CHECK(cudaMemcpy(tmpO.data(),        d_Oref,      Nq   * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < Nq; ++i) h_Oref[i] = __bfloat162float(tmpO[i]);
    CUDA_CHECK(cudaMemcpy(h_LSEref.data(),    d_LSEref,    Nlse * sizeof(float),         cudaMemcpyDeviceToHost));
    if (USE_QKNORM) {
      CUDA_CHECK(cudaMemcpy(h_qrstd_ref.data(), d_qrstd_ref, Nlse * sizeof(float),       cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(h_krstd_ref.data(), d_krstd_ref, Nkr  * sizeof(float),       cudaMemcpyDeviceToHost));
    }
  }

  // ─────────────────── Version registry ───────────────────
  // Each entry: (name, launch-lambda). Add a line to register a new version.
  std::vector<std::pair<std::string, std::function<void()>>> versions;
  versions.push_back({"V0 — fused WMMA online softmax (bf16)",
    [&]{ launch_gqa_fused_v0<Br, Bc, D>(
           d_Q, d_K, d_V, d_O, d_LSE, d_cache, qg, kg, qrstd, krstd,
           B, Hq, Hkv, G, S, CACHE_SEQ_LEN, POS_OFFSET, EPS, INTERLEAVED, scale, IS_CAUSAL); }});
  // versions.push_back({"V1 — my next fused version",
  //   [&]{ launch_gqa_fused_v1<Br, Bc, D>( /* same args as V0 */ ); }});

  // ─────────────────── Correctness (vs the GPU golden reference) ───────────────────
  std::cout << "\n============================== CORRECTNESS ==============================\n";
  std::cout << "Config: B=" << B << " Hq=" << Hq << " Hkv=" << Hkv << " S=" << S << " D=" << D
            << " | RoPE=" << (IS_NEOX ? "NeoX" : "interleaved")
            << " causal=" << (IS_CAUSAL ? "on" : "off")
            << " qk_norm=" << (USE_QKNORM ? "on" : "off") << "\n";

  std::vector<__nv_bfloat16> h_O(Nq);
  std::vector<float> h_O_f32(Nq), h_LSE(Nlse), h_qrstd(Nlse), h_krstd(Nkr);
  for (auto& v : versions) {
    CUDA_CHECK(cudaMemset(d_O,   0, Nq   * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemset(d_LSE, 0, Nlse * sizeof(float)));
    v.second();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_O.data(),   d_O,   Nq   * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_LSE.data(), d_LSE, Nlse * sizeof(float),         cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < Nq; ++i) h_O_f32[i] = __bfloat162float(h_O[i]);

    std::cout << "\n--- " << v.first << " ---\n";
    reportPrecision("  output O ", h_Oref.data(),   h_O_f32.data(), Nq);
    reportPrecision("  lse      ", h_LSEref.data(), h_LSE.data(),   Nlse);
    std::cout << "  O   : "; checkResult(h_Oref.data(),   h_O_f32.data(), Nq,   2e-2f, 2e-2f);
    std::cout << "  LSE : "; checkResult(h_LSEref.data(), h_LSE.data(),   Nlse, 2e-2f, 2e-2f);

    if (USE_QKNORM) {
      CUDA_CHECK(cudaMemcpy(h_qrstd.data(), d_qrstd, Nlse * sizeof(float), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(h_krstd.data(), d_krstd, Nkr  * sizeof(float), cudaMemcpyDeviceToHost));
      reportPrecision("  q_rstd   ", h_qrstd_ref.data(), h_qrstd.data(), Nlse);
      reportPrecision("  k_rstd   ", h_krstd_ref.data(), h_krstd.data(), Nkr);
      std::cout << "  qrstd: "; checkResult(h_qrstd_ref.data(), h_qrstd.data(), Nlse, 2e-2f, 2e-2f);
      std::cout << "  krstd: "; checkResult(h_krstd_ref.data(), h_krstd.data(), Nkr,  2e-2f, 2e-2f);
    }
  }

  // ─────────────────── Benchmark ───────────────────
  // Algorithmic attention FLOPs: 4 * B * Hq * S * S * D (QKᵀ + P·V, ×2 for MAC).
  // (Norm + RoPE are negligible; causal ~halves the real count — reported full.)
  long long flops = 4LL * B * Hq * (long long)S * S * D;
  size_t bytes = (2 * Nq + 2 * Nkv) * sizeof(__nv_bfloat16) + Nlse * sizeof(float);

  std::cout << "\n============================== BENCHMARK ==============================\n";
  for (auto& v : versions) {
    KernelStats st = benchmarkKernel(v.second, 100, 25, flops, bytes);
    displayStats(v.first, st);
  }

  CUDA_CHECK(cudaFree(d_Q));   CUDA_CHECK(cudaFree(d_K));   CUDA_CHECK(cudaFree(d_V));
  CUDA_CHECK(cudaFree(d_O));   CUDA_CHECK(cudaFree(d_LSE)); CUDA_CHECK(cudaFree(d_cache));
  CUDA_CHECK(cudaFree(d_qg));  CUDA_CHECK(cudaFree(d_kg));
  CUDA_CHECK(cudaFree(d_qrstd)); CUDA_CHECK(cudaFree(d_krstd));
  CUDA_CHECK(cudaFree(d_Qn));  CUDA_CHECK(cudaFree(d_Kn));  CUDA_CHECK(cudaFree(d_Oref));
  CUDA_CHECK(cudaFree(d_LSEref));
  CUDA_CHECK(cudaFree(d_qrstd_ref)); CUDA_CHECK(cudaFree(d_krstd_ref));
  return 0;
}
