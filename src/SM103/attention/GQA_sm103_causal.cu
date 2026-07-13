// GQA_sm103_causal.cu — causal-masked GQA, built on GQA_sm103.cu's V19 (the fastest
// fully-verified kernel: warp-specialized, atom-native K TMA, fp16 sS, cta_group::1).
// Kept in its OWN file rather than folded into GQA_sm103.cu: masking is a different
// axis of work from that file's V0-V23 performance ladder (changes the reference data,
// the correctness target, and the tiling/loop-bound logic), and GQA_sm103.cu is
// already a large, stable, individually-verified history that a mask flag threaded
// through 24 kernel variants would put at real risk for no benefit.
//
// Causal masking here exploits Br == Bc (both 128, same as V19): for a given query
// tile q_tile (rows [q_tile*Br, q_tile*Br+Br)), key tiles align 1:1 with query tiles,
// so:
//   - key tiles kc > q_tile are entirely above the diagonal (fully masked) -> SKIPPED
//     outright by looping only kc in [0, q_tile], not [0, S/Bc). This is the real perf
//     lever: causal attention only visits ~half the (q_tile, kc) pairs a full/non-causal
//     kernel would (exactly nTiles*(nTiles+1)/2 out of nTiles*nTiles tile-pairs).
//   - key tiles kc < q_tile are entirely BELOW the diagonal (fully visible) -> no
//     per-element masking needed at all.
//   - the single tile kc == q_tile is the DIAGONAL tile -> needs a per-element mask:
//     local column j is masked (query cannot see it) iff j > local row (== tid, since
//     Br==128 consumer threads map 1:1 to the Br=128 query rows). This is applied
//     directly to sS (already read out of TMEM, already in base-2 units) right before
//     the online-softmax reads it — no cross-thread synchronization needed since each
//     consumer thread only ever reads/writes its own row of sS.
//
// Everything else (operand layouts, warp specialization, mbarrier protocol, TMEM
// readout, log2-domain online softmax) is unchanged from V19 — see GQA_sm103.cu's
// V2/V7/V17/V18/V19 headers for the full derivation history of each piece.

#include <cuda_runtime.h>
#include <cuda.h>
#include <cooperative_groups.h>
#include <cuda/barrier>
#include <stdio.h>
#include <cassert>
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "utils/kernelUtils.cuh"
#include "utils/kernelBench.cuh"

namespace cg = cooperative_groups;

// ---------------------------------------------------------------------------
// tcgen05 matrix-descriptor helpers (verbatim from GQA_sm103.cu — see that file's
// V2 header for the full derivation of the canonical K-major atom layout).
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint64_t desc_encode(uint64_t x){
  return (x & 0x3FFFFull) >> 4;
}

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

__device__ uint32_t make_idesc_bf16(int M, int N){
  uint32_t idesc = 0;
  idesc |= (1u << 4);                            // D = fp32
  idesc |= (1u << 7);                            // A = bf16
  idesc |= (1u << 10);                           // B = bf16
  idesc |= ((uint32_t)(N >> 3) << 17);           // N in 8-element units
  idesc |= ((uint32_t)(M >> 4) << 24);           // M in 16-element units
  return idesc;
}

__device__ __forceinline__ int canon_idx(int row, int k, int rows){
  return (k / 8) * rows * 8 + row * 8 + (k % 8);
}

__device__ uint64_t advance_desc_katom(uint64_t desc, int katom, int rows){
  uint64_t units    = (uint64_t)katom * (uint64_t)rows * 2ull;
  uint64_t base_addr = desc & 0x3FFFull;
  uint64_t new_addr  = (base_addr + units) & 0x3FFFull;
  return (desc & ~0x3FFFull) | new_addr;
}

__device__ __forceinline__ void mbar_init(uint32_t bar, int count){
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(count));
}
__device__ __forceinline__ void mbar_commit_mma(uint32_t bar){
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
               :: "r"(bar) : "memory");
}
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
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes){
  asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar){
  asm volatile("mbarrier.arrive.shared.b64 _, [%0];" :: "r"(bar) : "memory");
}

__device__ __forceinline__ void tma_load_2d(uint32_t smem_addr, const void* tmap, int c, int r, uint32_t bar){
  asm volatile(
    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
    :: "r"(smem_addr), "l"(tmap), "r"(c), "r"(r), "r"(bar) : "memory");
}
__device__ __forceinline__ void tma_load_3d(uint32_t smem_addr, const void* tmap, int x, int y, int z, uint32_t bar){
  asm volatile(
    "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];"
    :: "r"(smem_addr), "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(bar) : "memory");
}

__device__ __forceinline__ float ex2_approx(float x){
  float y;
  // NOT volatile — see GQA_sm103.cu's V7 header for why marking this volatile
  // serializes the softmax exp loop and regresses perf.
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ void consumer_sync(){
  asm volatile("bar.sync 1, 128;" ::: "memory");
}

// TMEM readout: fp16-packed scores (see GQA_sm103.cu's V18 header for the bf16 vs
// fp16 precision derivation — this is the fp16 variant V19 uses).
__device__ void tmem_readout_to_smem_fp16_vec(
  __half* smem_out,
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
      __half* o = &smem_out[row * smem_stride + col];
      *reinterpret_cast<__half2*>(&o[0]) = __floats2half2_rn(
          reinterpret_cast<float&>(r0) * scale, reinterpret_cast<float&>(r1) * scale);
      *reinterpret_cast<__half2*>(&o[2]) = __floats2half2_rn(
          reinterpret_cast<float&>(r2) * scale, reinterpret_cast<float&>(r3) * scale);
      *reinterpret_cast<__half2*>(&o[4]) = __floats2half2_rn(
          reinterpret_cast<float&>(r4) * scale, reinterpret_cast<float&>(r5) * scale);
      *reinterpret_cast<__half2*>(&o[6]) = __floats2half2_rn(
          reinterpret_cast<float&>(r6) * scale, reinterpret_cast<float&>(r7) * scale);
    }
  }
}

// TMEM readout that accumulates straight into an fp32 smem accumulator (P@V's output).
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
//  gqa_v19_causal : V19 + causal masking (tile-skip + single diagonal-tile mask)
// =================================
template<int Br, int Bc, int D>
__global__ void gqa_v19_causal(
  __nv_bfloat16 *d_Q,
  __nv_bfloat16 *d_O,
  float *d_LSE,
  const __grid_constant__ CUtensorMap Ktmap3d,   // K, atom-native 3-rank map (see make_tma_3d_katom)
  const __grid_constant__ CUtensorMap Vtmap,     // V as flattened [B*Hkv*S, D], plain 2D map
  int B,
  int Hq,
  int Hkv,
  int G,
  int S,
  float scale
){
  static_assert(Br == 128, "consumer group is hardwired to 128 threads (TMEM readout needs warps 0-3)");
  static_assert(Br == Bc, "causal tile-skip + single diagonal-tile mask requires Br == Bc");

  const int b      = blockIdx.x;
  const int hq     = blockIdx.y;
  const int q_tile = blockIdx.z;
  const int hkv    = hq / G;
  const int tid    = threadIdx.x;              // 0..159: 0..127 consumer, 128..159 producer

  const int q_row0   = q_tile * Br;
  const int nKVTiles = q_tile + 1;   // CAUSAL: only key tiles [0, q_tile] can be visible
  const int kvRow0   = (b * Hkv + hkv) * S;   // first K/V row of this head in the flat tensor

  const long qBase  = ((long)(b * Hq + hq) * S + q_row0) * D;
  const long lBase  = ((long)(b * Hq + hq) * S + q_row0);

  // Fold log2(e) into the score scale so QK^T lands in base-2 units (see V7 header).
  const float scale_l2e = scale * 1.4426950408889634f;   // scale * float32 log2(e)

  __shared__ __align__(16)  __nv_bfloat16 sQ[Br * D];
  __shared__ __align__(128) __nv_bfloat16 sK[2][Bc * D];        // atom-native, TMA lands directly — no reorder
  __shared__ __align__(128) __nv_bfloat16 sVstage[2][Bc * D];   // raw landing, row-major (needs transpose reorder)
  __shared__ __align__(16)  __nv_bfloat16 sV[Bc * D];           // canonical [D,Bc] transposed (reorder output)
  __shared__ __align__(16)  __nv_bfloat16 sP[Br * Bc];
  __shared__ __align__(16)  __half        sS[Br * Bc];          // fp16, not fp32 — see V18 header
  __shared__ __align__(16)  float         sO[Br * D];           // P@V readout accumulates here directly
  __shared__ float sm[Br];
  __shared__ float sl[Br];
  __shared__ float sCorr[Br];
  __shared__ __align__(8) uint64_t s_mma_bar;
  __shared__ __align__(8) uint64_t s_load_bar_K[2];   // producer -> consumer: K slot ready
  __shared__ __align__(8) uint64_t s_free_bar_K[2];   // consumer -> producer: K slot free (post QK^T MMA)
  __shared__ __align__(8) uint64_t s_load_bar_V[2];   // producer -> consumer: V slot ready
  __shared__ __align__(8) uint64_t s_free_bar_V[2];   // consumer -> producer: V slot free (post reorder)

  for(int i = tid; i < Br * D; i += blockDim.x){
    const int r = i / D, c = i % D;
    sQ[canon_idx(r, c, Br)] = d_Q[qBase + i];
    sO[i] = 0.0f;
  }
  if(tid < Br){ sm[tid] = -INFINITY; sl[tid] = 0.0f; }

  const uint32_t mma_bar   = (uint32_t)__cvta_generic_to_shared(&s_mma_bar);
  const uint32_t lbarK0    = (uint32_t)__cvta_generic_to_shared(&s_load_bar_K[0]);
  const uint32_t lbarK1    = (uint32_t)__cvta_generic_to_shared(&s_load_bar_K[1]);
  const uint32_t fbarK0    = (uint32_t)__cvta_generic_to_shared(&s_free_bar_K[0]);
  const uint32_t fbarK1    = (uint32_t)__cvta_generic_to_shared(&s_free_bar_K[1]);
  const uint32_t lbarV0    = (uint32_t)__cvta_generic_to_shared(&s_load_bar_V[0]);
  const uint32_t lbarV1    = (uint32_t)__cvta_generic_to_shared(&s_load_bar_V[1]);
  const uint32_t fbarV0    = (uint32_t)__cvta_generic_to_shared(&s_free_bar_V[0]);
  const uint32_t fbarV1    = (uint32_t)__cvta_generic_to_shared(&s_free_bar_V[1]);
  if(tid == 0){
    mbar_init(mma_bar, 1);
    mbar_init(lbarK0, 1); mbar_init(lbarK1, 1);
    mbar_init(fbarK0, 1); mbar_init(fbarK1, 1);
    mbar_init(lbarV0, 1); mbar_init(lbarV1, 1);
    mbar_init(fbarV0, 1); mbar_init(fbarV1, 1);
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

  const uint32_t TX = (uint32_t)Bc * (uint32_t)D * (uint32_t)sizeof(__nv_bfloat16);   // one operand's box bytes
  const uint32_t sK_addr[2] = {
    (uint32_t)__cvta_generic_to_shared(sK[0]),
    (uint32_t)__cvta_generic_to_shared(sK[1])
  };
  const uint32_t sVstage_addr[2] = {
    (uint32_t)__cvta_generic_to_shared(sVstage[0]),
    (uint32_t)__cvta_generic_to_shared(sVstage[1])
  };
  const uint32_t lbarK[2] = {lbarK0, lbarK1};
  const uint32_t fbarK[2] = {fbarK0, fbarK1};
  const uint32_t lbarV[2] = {lbarV0, lbarV1};
  const uint32_t fbarV[2] = {fbarV0, fbarV1};

  if(tid >= 128){
    // ---- Producer warp: K (atom-native) and V (raw stage) are now independent. ----
    // CAUSAL: only prefetches kc in [0, q_tile] — tiles above the diagonal are never
    // even loaded, let alone computed.
    if(tid == 128){
      int free_phase_K[2] = {0, 0};
      int free_phase_V[2] = {0, 0};
      for(int kc = 0; kc < nKVTiles; ++kc){
        const int slot = kc & 1;
        // Slots 0 and 1 start empty, so the first wait for each is at kc == slot+2.
        if(kc >= 2){
          mbar_wait(fbarK[slot], free_phase_K[slot]); free_phase_K[slot] ^= 1;
          mbar_wait(fbarV[slot], free_phase_V[slot]); free_phase_V[slot] ^= 1;
        }
        const int r = kvRow0 + kc * Bc;

        mbar_expect_tx(lbarK[slot], TX);
        tma_load_3d(sK_addr[slot], &Ktmap3d, 0, r, 0, lbarK[slot]);
        mbar_arrive(lbarK[slot]);

        mbar_expect_tx(lbarV[slot], TX);
        tma_load_2d(sVstage_addr[slot], &Vtmap, 0, r, lbarV[slot]);
        mbar_arrive(lbarV[slot]);
      }
    }
  } else {
    // ---- Consumer warps (0-3): MMA-issue (K used directly) + V-reorder + softmax + PV. ----
    int mbar_phase = 0;
    int load_phase_K[2] = {0, 0};
    int load_phase_V[2] = {0, 0};
    const uint64_t descQ_base = make_smem_desc(sQ, Br);

    for(int kc = 0; kc < nKVTiles; ++kc){
      const int slot = kc & 1;
      mbar_wait(lbarK[slot], load_phase_K[slot]); load_phase_K[slot] ^= 1;
      mbar_wait(lbarV[slot], load_phase_V[slot]); load_phase_V[slot] ^= 1;
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      consumer_sync();

      // V only — K needs no reorder, it's already in canon_idx layout in sK[slot].
      for(int i = tid; i < Bc * D; i += Br){
        const int bc = i / D, d = i % D;
        sV[canon_idx(d, bc, D)] = sVstage[slot][i];   // transposed
      }
      consumer_sync();
      // V's slot is fully drained — let the producer reuse it while MMA/softmax/PV run.
      if(tid == 0) mbar_arrive(fbarV[slot]);

      // S2 = (Q @ K^T) * (scale*log2e) -> sS[Br, Bc], fp16 (scores already in base-2 units)
      {
        const uint64_t descK_base = make_smem_desc(sK[slot], Bc);
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
        // K's slot has now been fully read by the QK^T MMA — safe to let the producer reuse it.
        if(tid == 0) mbar_arrive(fbarK[slot]);
        tmem_readout_to_smem_fp16_vec(sS, tmem_addr, Br, Bc, Bc, scale_l2e);   // prescale by log2e here
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");

        // CAUSAL diagonal-tile mask: only the LAST tile (kc == q_tile) straddles the
        // diagonal. Column j (this rank's key j) is invisible to query row `tid` iff
        // j > tid — safe with no extra sync since each consumer thread only ever
        // touches its own row (row == tid exactly, given Br == 128 consumer threads).
        if(kc == q_tile){
          const __half neg_inf = __float2half(-INFINITY);
          for(int j = tid + 1; j < Bc; ++j) sS[tid * Bc + j] = neg_inf;
        }
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
                           fmaxf(__half2float(sS[tid * Bc + j]),
                                 fmaxf(__half2float(sS[tid * Bc + j + 1]),
                                       __half2float(sS[tid * Bc + j + 2]))));
        for(; j < Bc; ++j) tile_max = fmaxf(tile_max, __half2float(sS[tid * Bc + j]));

        const float m_new = fmaxf(m_old, tile_max);
        const float corr  = ex2_approx(m_old - m_new);   // == exp(m_old - m_new)

        // Pack two probabilities per F2FP (F2FP.BF16.F32.PACK_AB) + one 32-bit STS.
        float p_sum = 0.0f;
        for(int j2 = 0; j2 < Bc; j2 += 2){
          const float p0 = ex2_approx(__half2float(sS[tid * Bc + j2])     - m_new);   // == exp(S - m_new)
          const float p1 = ex2_approx(__half2float(sS[tid * Bc + j2 + 1]) - m_new);
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

} // end of gqa_v19_causal

//* ============================
//* TMA descriptor builders (verbatim from GQA_sm103.cu)
//* ============================
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

static CUtensorMap make_tma_3d_katom(__nv_bfloat16* gptr, uint64_t rows, uint64_t cols,
                                     uint32_t box_rows){
  constexpr uint32_t ATOM = 8;
  assert(cols % ATOM == 0);
  CUtensorMap tmap{};
  uint64_t gdim[3]    = { ATOM, rows, cols / ATOM };
  uint64_t gstride[2] = { cols * sizeof(__nv_bfloat16), ATOM * sizeof(__nv_bfloat16) };
  uint32_t bdim[3]    = { ATOM, box_rows, (uint32_t)(cols / ATOM) };
  uint32_t estride[3] = { 1, 1, 1 };
  CUresult res = cuTensorMapEncodeTiled(
    &tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, gptr, gdim, gstride, bdim, estride,
    CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
    CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  assert(res == CUDA_SUCCESS);
  return tmap;
}

template<int Br, int Bc, int D>
void launch_gqa_v19_causal(
  __nv_bfloat16 *d_Q, __nv_bfloat16 *d_K, __nv_bfloat16 *d_V,
  __nv_bfloat16 *d_O, float *d_LSE,
  int B, int Hq, int Hkv, int S, int G, float scale
){
  static_assert(Br == 128, "consumer group is hardwired to 128 threads");
  static_assert(Br == Bc, "causal tile-skip + diagonal-tile mask requires Br == Bc");
  static_assert(Bc % 8 == 0, "Bc must be a multiple of 8 for tcgen05 N = 8");
  static_assert(D  % 16 == 0, "D  must be a multiple of 16 for tcgen05 dense");
  static_assert(D  % 8  == 0, "D must be a multiple of 8 for the atom-native K TMA map");

  dim3 GRID(B, Hq, S/Br);
  dim3 BLOCK(160);   // 128 consumer threads (warps 0-3) + 32 producer threads (warp 4)

  static bool cfgd = false;
  static CUtensorMap Ktmap3d, Vtmap;
  if(!cfgd){
    const uint64_t kvRows = (uint64_t)B * Hkv * S;   // K/V flattened as [B*Hkv*S, D]
    Ktmap3d = make_tma_3d_katom(d_K, kvRows, (uint64_t)D, (uint32_t)Bc);
    Vtmap   = make_tma_2d(d_V, kvRows, (uint64_t)D, (uint32_t)Bc, (uint32_t)D);
    cfgd = true;
  }
  gqa_v19_causal<Br, Bc, D><<<GRID, BLOCK>>>(d_Q, d_O, d_LSE, Ktmap3d, Vtmap,
                          B, Hq, Hkv, G, S, scale);
}


int main(){
  std::cout << "Benchmarking CAUSAL Grouped-Query Attention — Blackwell SM_103 (B300)\n";

  constexpr int B   = 8;
  constexpr int Hq  = 12;
  constexpr int Hkv = 4;
  constexpr int G   = Hq / Hkv;
  constexpr int S   = 4096;
  constexpr int D   = 64;
  constexpr int Br  = 128;
  constexpr int Bc  = 128;

  static_assert(Hq % Hkv == 0, "Hq must be divisible by Hkv");
  static_assert(S  % Br   == 0, "S must be divisible by Br");
  static_assert(Br == Bc, "causal tile-skip + diagonal-tile mask requires Br == Bc");

  const float scale = 1.0f / sqrtf((float)D);

  const size_t Nq   = (size_t)B * Hq  * S * D;
  const size_t Nkv  = (size_t)B * Hkv * S * D;
  const size_t Nlse = (size_t)B * Hq  * S;

  std::vector<__nv_bfloat16> h_Q(Nq), h_K(Nkv), h_V(Nkv), h_O(Nq);
  std::vector<float>         h_LSE(Nlse);
  std::vector<float>         h_O_ref(Nq), h_LSE_ref(Nlse);

  auto fileMatchesSize = [](const std::string &p, size_t n_floats) -> bool {
    FILE *f = fopen(p.c_str(), "rb");
    if(!f) return false;
    fseek(f, 0, SEEK_END);
    size_t bytes = (size_t)ftell(f);
    fclose(f);
    return bytes == n_floats * sizeof(float);
  };
  auto loadBinBF16 = [](const char *path, std::vector<__nv_bfloat16> &dst, size_t n){
    std::vector<float> tmp(n);
    loadBin(path, tmp.data(), n);
    for(size_t i = 0; i < n; ++i) dst[i] = __float2bfloat16(tmp[i]);
  };

  bool has_ref = fileMatchesSize("data/gqa_causal_q.bin", Nq);
  if(has_ref){
    loadBinBF16("data/gqa_causal_q.bin", h_Q, Nq);
    loadBinBF16("data/gqa_causal_k.bin", h_K, Nkv);
    loadBinBF16("data/gqa_causal_v.bin", h_V, Nkv);
    loadBin("data/gqa_causal_o.bin",   h_O_ref.data(),   Nq);
    loadBin("data/gqa_causal_lse.bin", h_LSE_ref.data(), Nlse);
    std::cout << "\nLoaded causal PyTorch reference from data/gqa_causal_*.bin\n";
  } else {
    initPtr(h_Q.data(), (int)Nq);
    initPtr(h_K.data(), (int)Nkv);
    initPtr(h_V.data(), (int)Nkv);
    std::cout << "\nNo causal reference files found (run base/baseline_gqa_causal.py "
                 "first) — using random data (benchmarks only)\n";
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

  if(has_ref){
    launch_gqa_v19_causal<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_O.data(),   d_O,   Nq   * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_LSE.data(), d_LSE, Nlse * sizeof(float),         cudaMemcpyDeviceToHost));

    std::vector<float> h_O_f32(Nq);
    for(size_t i = 0; i < Nq; ++i) h_O_f32[i] = __bfloat162float(h_O[i]);

    std::cout << "\nCorrectness V19-causal (V19 + causal mask vs PyTorch bf16 causal SDPA):\n";
    reportPrecision("  output O ", h_O_ref.data(),   h_O_f32.data(), Nq);
    reportPrecision("  lse      ", h_LSE_ref.data(), h_LSE.data(),   Nlse);
    std::cout << "  O   : "; checkResult(h_O_ref.data(),   h_O_f32.data(), Nq,   2e-2f, 2e-2f);
    std::cout << "  LSE : "; checkResult(h_LSE_ref.data(), h_LSE.data(),   Nlse, 2e-2f, 2e-2f);
  }

  //* ── Benchmark ──────────────────────────────────────────────────────────
  //* Causal attention FLOPs: only nTiles*(nTiles+1)/2 of the nTiles*nTiles (q_tile,kc)
  //* pairs a full/non-causal kernel would visit are ever computed (Br==Bc==nTiles tiles
  //* wide) — roughly HALF the work of the non-causal V19, not the same S*S count.
  constexpr long long nTiles     = S / Br;                          // 32
  constexpr long long tileVisits = nTiles * (nTiles + 1) / 2;        // 528 (vs 1024 full)
  long long flops = 4LL * B * Hq * (long long)Br * Bc * D * tileVisits;
  size_t bytes = (2 * Nq + 2 * Nkv) * sizeof(__nv_bfloat16) + Nlse * sizeof(float);

  KernelStats stats = benchmarkKernel(
    [&](){ launch_gqa_v19_causal<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); },
    100, 25, flops, bytes
  );
  displayStats("V19-causal — V19 + causal tile-skip + diagonal mask", stats);

  CUDA_CHECK(cudaFree(d_Q));
  CUDA_CHECK(cudaFree(d_K));
  CUDA_CHECK(cudaFree(d_V));
  CUDA_CHECK(cudaFree(d_O));
  CUDA_CHECK(cudaFree(d_LSE));

  return 0;
}
