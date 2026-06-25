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
// This variant takes *two* sequence lengths:
//   Sq  : query  sequence length   (Q / O / LSE are [B, Hq,  Sq,  D])
//   Skv : key/value sequence length (K / V       are [B, Hkv, Skv, D])
// This is what you need for cross-attention or KV-cache decode, where the query
// and key/value lengths differ.
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
  int Sq,    // query sequence length
  int Skv,   // key/value sequence length
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
  const int q_tile = blockIdx.z;   // which Br-row tile    (0 .. Sq/Br-1)
  const int hkv    = hq / G;       // GQA: query head hq shares this kv head
  const int lane   = threadIdx.x;  // 0..31 — single warp per block

  const int q_row0    = q_tile * Br;  // first query row this block computes
  const int nKeyTiles = Skv / Bc;     // number of Bc-wide key tiles to stream

  // Flat offsets into the [B,H,S,D] (and [B,H,S] for LSE) row-major arrays.
  // Note Q/O/LSE use Sq for their sequence stride, K/V use Skv.
  const long qBase  = ((long)(b * Hq  + hq ) * Sq  + q_row0) * D; // Q / O tile start
  const long kvBase = ((long)(b * Hkv + hkv) * Skv         ) * D; // K / V head start
  const long lBase  =  (long)(b * Hq  + hq ) * Sq  + q_row0;      // LSE tile start

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

//* ============================
//* Kernel Launcher
//* ============================
//* Tiled mapping: one block per (batch, query-head, query-row-tile).
//*   blockIdx.x = b       (0 .. B-1)
//*   blockIdx.y = hq      (0 .. Hq-1)    -> kv head = hq / G
//*   blockIdx.z = q_tile  (0 .. Sq/Br-1) -> query rows [q_tile*Br : q_tile*Br + Br)
//* One warp per block because WMMA is a warp-collective op. Tiles live in static
//* __shared__ memory (sizes are compile-time via Br/Bc/D), so no dynamic smem.
template<int Br, int Bc, int D>
void launch_gqa_v1(
  __nv_bfloat16 *d_Q, __nv_bfloat16 *d_K, __nv_bfloat16 *d_V,
  __nv_bfloat16 *d_O, float *d_LSE,
  int B, int Hq, int Hkv, int Sq, int Skv, int G, float scale
){
  static_assert(Br % 16 == 0, "Br must be a multiple of 16 (WMMA tile)");
  static_assert(Bc % 16 == 0, "Bc must be a multiple of 16 (WMMA tile)");
  static_assert(D  % 16 == 0, "D  must be a multiple of 16 (WMMA tile)");

  dim3 GRID(B, Hq, Sq / Br);  // one block per (batch, query-head, query-row-tile)
  dim3 BLOCK(32);             // ONE warp per block
  gqa_v1<Br, Bc, D><<<GRID, BLOCK>>>(d_Q, d_K, d_V, d_O, d_LSE,
                                     B, Hq, Hkv, Sq, Skv, G, scale);
}

int main(){
  std::cout << "Benchmarking Grouped-Query Attention (split Q / KV seqlen) — Blackwell SM_103\n";

  constexpr int B   = 16;     //! batch
  constexpr int Hq  = 12;     // number of query heads
  constexpr int Hkv = 4;      // number of key/value heads
  constexpr int G   = Hq/Hkv; // groups per KV head
  constexpr int Sq  = 4096;   // query sequence length
  constexpr int Skv = 2048;   // key/value sequence length (differs from Sq)
  constexpr int D   = 64;     // head dimension
  constexpr int Br  = 16;     // tile size along the query sequence dimension
  constexpr int Bc  = 32;     // tile size along the key/value sequence dimension

  static_assert(Hq % Hkv == 0, "Hq must be divisible by Hkv");
  static_assert(Sq  % Br  == 0, "Sq must be divisible by Br");
  static_assert(Skv % Bc  == 0, "Skv must be divisible by Bc");

  const float scale = 1.0f / sqrtf((float)D);

  const size_t Nq   = (size_t)B * Hq  * Sq  * D;  // Q and O   [B, Hq,  Sq,  D]
  const size_t Nkv  = (size_t)B * Hkv * Skv * D;  // K and V   [B, Hkv, Skv, D]
  const size_t Nlse = (size_t)B * Hq  * Sq;       // LSE       [B, Hq,  Sq]

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

  bool has_ref = fileMatchesSize("data/gqa_split_q.bin", Nq);
  if(has_ref){
    loadBinBF16("data/gqa_split_q.bin", h_Q, Nq);
    loadBinBF16("data/gqa_split_k.bin", h_K, Nkv);
    loadBinBF16("data/gqa_split_v.bin", h_V, Nkv);
    loadBin("data/gqa_split_o.bin",   h_O_ref.data(),   Nq);
    loadBin("data/gqa_split_lse.bin", h_LSE_ref.data(), Nlse);
    std::cout << "\nLoaded PyTorch reference from data/gqa_split_*.bin\n";
  } else {
    initPtr(h_Q.data(), (int)Nq);
    initPtr(h_K.data(), (int)Nkv);
    initPtr(h_V.data(), (int)Nkv);
    std::cout << "\nNo matching reference files found — using random data (benchmarks only)\n";
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
    launch_gqa_v1<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, Sq, Skv, G, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_O.data(),   d_O,   Nq   * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_LSE.data(), d_LSE, Nlse * sizeof(float),         cudaMemcpyDeviceToHost));

    // widen the bf16 output to fp32 so checkResult / reportPrecision can compare it
    std::vector<float> h_O_f32(Nq);
    for(size_t i = 0; i < Nq; ++i) h_O_f32[i] = __bfloat162float(h_O[i]);

    // bf16 attention: ~2^-8 relative precision, so use bf16-scale tolerances.
    std::cout << "\nCorrectness V1 (WMMA two-pass, split Q/KV seqlen vs PyTorch bf16 SDPA):\n";
    reportPrecision("  output O ", h_O_ref.data(),   h_O_f32.data(), Nq);
    reportPrecision("  lse      ", h_LSE_ref.data(), h_LSE.data(),   Nlse);
    std::cout << "  O   : "; checkResult(h_O_ref.data(),   h_O_f32.data(), Nq,   2e-2f, 2e-2f);
    std::cout << "  LSE : "; checkResult(h_LSE_ref.data(), h_LSE.data(),   Nlse, 2e-2f, 2e-2f);
  }

  //* ── Benchmark ──────────────────────────────────────────────────────────
  //* Attention FLOPs (algorithmic): 4 * B * Hq * Sq * Skv * D  (QKᵀ + P·V, ×2 for MAC).
  long long flops = 4LL * B * Hq * (long long)Sq * Skv * D;
  size_t bytes = (2 * Nq + 2 * Nkv) * sizeof(__nv_bfloat16) + Nlse * sizeof(float);

  KernelStats stats_v1 = benchmarkKernel(
    [&](){ launch_gqa_v1<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, Sq, Skv, G, scale); },
    100, 25, flops, bytes
  );
  displayStats("V1 — WMMA two-pass (stable softmax, split Q/KV seqlen, bf16)", stats_v1);

  CUDA_CHECK(cudaFree(d_Q));
  CUDA_CHECK(cudaFree(d_K));
  CUDA_CHECK(cudaFree(d_V));
  CUDA_CHECK(cudaFree(d_O));
  CUDA_CHECK(cudaFree(d_LSE));

  return 0;
}
