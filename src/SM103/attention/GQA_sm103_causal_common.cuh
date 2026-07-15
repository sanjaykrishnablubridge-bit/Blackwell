// GQA_sm103_causal_common.cuh — shared device-side helpers for the GQA causal
// attention kernel family (GQA_sm103_causal.cu holds V19-V36; new versions from
// V37 onward live in separate .cu files that #include this header instead of
// duplicating it, since GQA_sm103_causal.cu had grown too large to work in
// comfortably). Extracted verbatim from GQA_sm103_causal.cu on 2026-07-15 —
// see that file for the full derivation history of each helper (referenced
// in the comments below, which are preserved as-is).
#pragma once

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

// Remote arrive on the PEER CTA's mbarrier at the same smem offset: maps this CTA's
// barrier address into cluster rank `peer`'s smem window, then does a RELEASE-scoped
// arrive there. Used as a cross-CTA "operand ready" signal for joint cta_group::2
// MMAs — the issuing CTA (rank 0) reads the peer's smem-resident operand halves, so
// the peer's writes must be ordered (and made cluster-visible) before the issue.
__device__ __forceinline__ void mbar_arrive_peer(uint32_t bar, uint32_t peer){
  uint32_t remote;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(bar), "r"(peer));
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];"
               :: "r"(remote) : "memory");
}
// Cluster-scope ACQUIRE counterpart of mbar_wait — pairs with mbar_arrive_peer's
// release so the peer CTA's shared-memory writes are visible to the waiting thread.
__noinline__ __device__ void mbar_wait_cluster(uint32_t bar, int phase){
  uint32_t ticks = 0x989680;
  asm volatile(
    "{\n\t.reg .pred P1;\n\t"
    "LAB_WAIT_CL:\n\t"
    "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], %1, %2;\n\t"
    "@P1 bra.uni DONE_CL;\n\t"
    "bra.uni LAB_WAIT_CL;\n\t"
    "DONE_CL:\n\t}"
    :: "r"(bar), "r"(phase), "r"(ticks)
  );
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

// V25_causal's wider reorder-phase barrier: spans the 128 compute threads (tid 0-127)
// PLUS the 352 reorder-helper threads (tid 160-511) = 480 total. Uses a DIFFERENT
// named barrier id (2) from consumer_sync's (1) so the two scopes never collide —
// the 32 dedicated producer threads (tid 128-159) never call this and are correctly
// excluded from the count.
__device__ __forceinline__ void reorder_sync(){
  asm volatile("bar.sync 2, 480;" ::: "memory");
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

// Generalized version of the above, for callers whose group doesn't start at
// global warp 0 (mirrors why tmem_readout_to_smem_vec_2cta_g exists alongside
// tmem_readout_to_smem_vec_2cta — same rationale, fp16-output case). Used by
// V30_causal's softmax-hi group (tid 128-255, i.e. global warps 4-7).
__device__ void tmem_readout_to_smem_fp16_vec_g(
  __half* smem_out,
  uint32_t tmem_addr,
  int M,
  int N,
  int smem_stride,
  float scale,
  int warp_id_local,
  int lane
){
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

  const int rows_per_warp = M / 4;
  const uint32_t lane_base = (uint32_t)warp_id_local * 32u;
  const int row = warp_id_local * rows_per_warp + lane;

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

// cta_group::2 counterpart of mbar_commit_mma, used by V20-23_causal: a single call
// from the issuing CTA (rank 0) arrives the SAME relative smem offset's mbarrier in
// every CTA named by `mask` (see GQA_sm103.cu's V20 header for the full derivation).
__device__ __forceinline__ void mbar_commit_mma_2cta_multicast(uint32_t bar, uint16_t mask){
  asm volatile(
    "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"
    :: "r"(bar), "h"(mask) : "memory");
}
// Makes this CTA's mbarrier inits visible cluster-wide before any peer CTA arrives/waits.
__device__ __forceinline__ void cluster_fence_mbarrier_init(){
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

// TMEM readout for cta_group::2: same x8-vectorized readout as the ::1 versions, but
// the lane_base also folds in rank_warp_offset (= crank * Br/32) so each CTA of the
// pair reads out only its own row-band of the shared 2*Br-row accumulator.
__device__ void tmem_readout_to_smem_vec_2cta(
  float* smem_out,
  uint32_t tmem_addr,
  int M,
  int N,
  int smem_stride,
  float scale,
  uint32_t rank_warp_offset
){
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

  const int warp_id = threadIdx.x / 32;
  const int lane    = threadIdx.x % 32;
  const int rows_per_warp = M / 4;
  const uint32_t lane_base = (rank_warp_offset + (uint32_t)warp_id) * 32u;
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

// cta_group::2 counterpart of tmem_readout_accum_vec — same rank_warp_offset row-band
// addressing as tmem_readout_to_smem_vec_2cta above.
__device__ void tmem_readout_accum_vec_2cta(
  float* smem_acc,
  uint32_t tmem_addr,
  int M,
  int N,
  int smem_stride,
  uint32_t rank_warp_offset
){
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

  const int warp_id = threadIdx.x / 32;
  const int lane    = threadIdx.x % 32;
  const int rows_per_warp = M / 4;
  const uint32_t lane_base = (rank_warp_offset + (uint32_t)warp_id) * 32u;
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

// Generalized versions of the two readouts above, for V26_causal's ping-pong design:
// the plain versions derive warp_id/lane straight from threadIdx.x, which only gives
// the 0-3 range the TMEM sub-partition math expects when the calling group's threads
// start at global warp 0. V26's half-B compute group lives at tid 128-255 (global
// warps 4-7), so it must pass its OWN group-local warp_id explicitly instead.
__device__ void tmem_readout_to_smem_vec_2cta_g(
  float* smem_out,
  uint32_t tmem_addr,
  int M,
  int N,
  int smem_stride,
  float scale,
  uint32_t rank_warp_offset,
  int warp_id_local,
  int lane
){
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

  const int rows_per_warp = M / 4;
  const uint32_t lane_base = (rank_warp_offset + (uint32_t)warp_id_local) * 32u;
  const int row = warp_id_local * rows_per_warp + lane;

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

__device__ void tmem_readout_accum_vec_2cta_g(
  float* smem_acc,
  uint32_t tmem_addr,
  int M,
  int N,
  int smem_stride,
  uint32_t rank_warp_offset,
  int warp_id_local,
  int lane
){
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

  const int rows_per_warp = M / 4;
  const uint32_t lane_base = (rank_warp_offset + (uint32_t)warp_id_local) * 32u;
  const int row = warp_id_local * rows_per_warp + lane;

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

// Readouts for per-CTA M=64 accumulators — the cta_group::2, idesc-M=128 shape that
// V27's 64-row half-pipelines issue (each CTA of the pair holds 64 rows of D).
// The "M/4 rows per sub-partition, lanes 32w..32w+15" mapping used by every readout
// above is an M=128-per-CTA layout and does NOT extend down to M=64. For M=64 the
// D layout is COLUMN-HALVED across the full lane space instead:
//   lanes 0-63   : rows 0-63, output columns [0, N/2)
//   lanes 64-127 : rows 0-63, output columns [N/2, N)
// and only N/2 TMEM columns are occupied. Assuming the 16-rows-per-sub-partition
// mapping here is exactly what produced V26/V27's deterministic corruption:
// row-scrambled sS for rows 16-63 (LSE first failing at row 18), and leftover raw
// QK^T scores read into O's upper D/2 columns (O first failing at exactly column
// 32). The gqa_tmem_probe kernel + main()'s probe analysis verify this mapping
// empirically on hardware with no layout assumption baked in.
// All 128 threads of the calling group participate: thread t owns
// (row = t & 63, column-half = t >> 6). Because each row's two column-halves land
// from two DIFFERENT threads, the caller MUST barrier after this returns and before
// any per-row consumption of smem_out.
__device__ void tmem_readout_to_smem_vec_2cta_m64(
  float* smem_out,
  uint32_t tmem_addr,
  int N,
  int smem_stride,
  float scale,
  uint32_t rank_lane_offset,
  int ltid
){
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

  const int row  = ltid & 63;
  const int col0 = (ltid >> 6) * (N / 2);
  const uint32_t lane_base = rank_lane_offset + (uint32_t)ltid;

  for(int col = 0; col < N / 2; col += 8){
    uint32_t r0,r1,r2,r3,r4,r5,r6,r7;
    asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
      : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7)
      : "r"(tmem_addr + (lane_base << 16) + (uint32_t)col)
      : "memory"
    );
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    float* o = &smem_out[row * smem_stride + col0 + col];
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

__device__ void tmem_readout_accum_vec_2cta_m64(
  float* smem_acc,
  uint32_t tmem_addr,
  int N,
  int smem_stride,
  uint32_t rank_lane_offset,
  int ltid
){
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

  const int row  = ltid & 63;
  const int col0 = (ltid >> 6) * (N / 2);
  const uint32_t lane_base = rank_lane_offset + (uint32_t)ltid;

  for(int col = 0; col < N / 2; col += 8){
    uint32_t r0,r1,r2,r3,r4,r5,r6,r7;
    asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
      : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7)
      : "r"(tmem_addr + (lane_base << 16) + (uint32_t)col)
      : "memory"
    );
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    float* o = &smem_acc[row * smem_stride + col0 + col];
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

// V26_causal's half-B internal barrier (half-A reuses consumer_sync, bar id 1).
// Different id from V25's reorder_sync (id 2) — irrelevant here since these are
// separate __global__ functions with independent barrier-id namespaces, but kept
// distinct (id 3) to avoid any confusion when reading the two kernels side by side.
__device__ __forceinline__ void sync_half_b(){
  asm volatile("bar.sync 3, 128;" ::: "memory");
}

// V27_causal's shared rescale-warpgroup internal barrier (softmax-A reuses
// consumer_sync id 1, softmax-B reuses sync_half_b id 3) — distinct id (4) since
// this is a THIRD disjoint set of 128 threads within the same kernel.
__device__ __forceinline__ void sync_rescale(){
  asm volatile("bar.sync 4, 128;" ::: "memory");
}

// V30_causal's shared-loader-warpgroup internal barrier — id 5, distinct from
// consumer_sync/sync_half_b/sync_rescale (ids 1/3/4), the FOURTH disjoint set of
// 128 threads. Scopes the cooperative V reorder-copy: ensures all 128 loader
// threads have finished reading sVstage/writing sV before any one of them signals
// "V ready" to the rescale group.
__device__ __forceinline__ void sync_loader(){
  asm volatile("bar.sync 5, 128;" ::: "memory");
}


// --- TMA tensor-map builders (originally defined mid-file in GQA_sm103_causal.cu,
// moved here since every launcher needs them) ---
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
