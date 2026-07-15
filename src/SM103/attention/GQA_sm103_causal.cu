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

// =================================
//  gqa_v20_causal .. gqa_v23_causal : cta_group::2 (2-CTA pair) + causal masking.
//
// Cluster pairing complication (see GQA_sm103.cu's cta_group::2 header comments for
// the base non-causal kernels): __cluster_dims__(2,1,1) pairs blocks (2c, 2c+1) along
// blockIdx.x == q_tile, and only rank 0 issues the joint tcgen05.mma — so BOTH ranks
// must iterate the SAME kc range, even though they have DIFFERENT causal deadlines
// (rank 0's own_q_tile = 2c, rank 1's = 2c+1). The loop bound must therefore cover the
// pair's LATER (higher/odd) q_tile: nKVTiles = (q_tile | 1) + 1 — a communication-free
// bitwise trick (q_tile|1 maps both 2c and 2c+1 to 2c+1).
//
// Per-rank masking then has THREE cases relative to this CTA's own q_tile (blockIdx.x
// is already per-CTA, unlike the shared loop bound):
//   kc <  q_tile : fully visible, no mask (all keys are in the past for this rank)
//   kc == q_tile : diagonal tile, mask column j > tid (same as the ::1 kernels)
//   kc >  q_tile : only possible for rank 0 (the pair's lower q_tile) on the pair's
//                  final shared iteration — that tile is entirely FUTURE for rank 0,
//                  so mask the WHOLE row to -inf. Rank 1 still needs the real (shared)
//                  MMA that iteration for its own diagonal, so the tile-skip can't be
//                  pushed into the loop bound; it has to be a per-rank full-row mask
//                  after readout instead. P@V needs no separate masking: a fully
//                  -inf-masked row softmaxes to all-zero probabilities, contributing
//                  nothing to O.
//
// KNOWN-FAILING: gqa_v20_causal reproducibly fails correctness (O mismatches at
// column D/2, LSE matches exactly) — this is NOT a masking bug. V20's "duplicate
// full K/V on both ranks" trick was never a real fix for cta_group::2's B-operand
// split, just a workaround that happened to pass tolerance in the non-causal
// kernel because the resulting error was small relative to that kernel's (larger)
// output magnitude. Causal masking shrinks many rows' true output magnitude
// (they attend to far fewer keys), so the SAME absolute duplication error now
// exceeds the relative-error tolerance. The only real fix is V21's genuine
// N-half B-split — which is exactly what gqa_v21_causal already is. Kept here
// only as the historical "here's the naive attempt, here's why it's wrong" rung
// of the ladder, mirroring V20's role in GQA_sm103.cu; use V21_causal onward for
// anything correctness-sensitive.
// =================================

template<int Br, int Bc, int D>
__global__ void __cluster_dims__(2, 1, 1) gqa_v20_causal(
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
  static_assert(Br == Bc, "causal tile-skip + diagonal-tile mask requires Br == Bc");

  const int q_tile = blockIdx.x;   // paired dimension MUST be x — see gqa_v20 header
  const int hq     = blockIdx.y;
  const int b      = blockIdx.z;
  const int hkv    = hq / G;
  const int tid    = threadIdx.x;

  cg::cluster_group cluster = cg::this_cluster();
  const unsigned int crank  = cluster.block_rank();   // 0 = issuer/K-V owner, 1 = peer

  const int q_row0   = q_tile * Br;
  const int nKVTiles = (q_tile | 1) + 1;   // CAUSAL: cover the pair's higher/odd q_tile
  const int kvRow0   = (b * Hkv + hkv) * S;   // first K/V row of this head in the flat tensor

  const long qBase  = ((long)(b * Hq + hq) * S + q_row0) * D;
  const long lBase  = ((long)(b * Hq + hq) * S + q_row0);

  const float scale_l2e = scale * 1.4426950408889634f;   // scale * float32 log2(e)

  __shared__ __align__(16)  __nv_bfloat16 sQ[Br * D];
  __shared__ __align__(128) __nv_bfloat16 sKstage[Bc * D];     // TMA target, row-major, SINGLE buffer
  __shared__ __align__(128) __nv_bfloat16 sVstage[Bc * D];
  __shared__ __align__(16)  __nv_bfloat16 sK[Bc * D];          // canonical K-major (reorder output)
  __shared__ __align__(16)  __nv_bfloat16 sV[Bc * D];          // canonical [D,Bc] transposed
  __shared__ __align__(16)  __nv_bfloat16 sP[Br * Bc];
  __shared__ __align__(16)  float         sS[Br * Bc];         // fp32 — no precision trade in this version
  __shared__ __align__(16)  float         sO[Br * D];          // P@V readout accumulates here directly
  __shared__ float sm[Br];
  __shared__ float sl[Br];
  __shared__ float sCorr[Br];
  __shared__ __align__(8) uint64_t s_mma_bar;
  __shared__ __align__(8) uint64_t s_load_bar;                 // single TMA completion barrier (rank 0 only)

  for(int i = tid; i < Br * D; i += blockDim.x){
    const int r = i / D, c = i % D;
    sQ[canon_idx(r, c, Br)] = d_Q[qBase + i];
    sO[i] = 0.0f;
  }
  if(tid < Br){ sm[tid] = -INFINITY; sl[tid] = 0.0f; }

  const uint32_t mma_bar = (uint32_t)__cvta_generic_to_shared(&s_mma_bar);
  const uint32_t lbar    = (uint32_t)__cvta_generic_to_shared(&s_load_bar);
  if(tid == 0){
    mbar_init(mma_bar, 1);
    mbar_init(lbar, 1);
    cluster_fence_mbarrier_init();
  }
  cluster.sync();

  constexpr uint32_t NCOLS = (Bc > D) ? (uint32_t)Bc : (uint32_t)D;
  static_assert(NCOLS >= 32 && (NCOLS & (NCOLS - 1)) == 0,
                "tcgen05 column count must be a power of two >= 32");
  constexpr uint32_t RANK_WARP_SPAN = (uint32_t)Br / 32u;

  uint32_t tmem_addr;
  {
    __shared__ uint32_t s_tmem_addr;
    if(tid < 32){
      uint32_t s_addr = (uint32_t)__cvta_generic_to_shared(&s_tmem_addr);
      asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                   :: "r"(s_addr), "r"(NCOLS) : "memory");
      asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::: "memory");
    }
    __syncthreads();
    tmem_addr = s_tmem_addr;
  }

  int mbar_phase = 0;
  int load_phase = 0;
  const uint32_t TX = 2u * (uint32_t)Bc * (uint32_t)D * (uint32_t)sizeof(__nv_bfloat16);
  const uint32_t rank_warp_offset = crank * RANK_WARP_SPAN;

  const uint64_t descQ_base = make_smem_desc(sQ, Br);

  for(int kc = 0; kc < nKVTiles; ++kc){
    {
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
        sK[canon_idx(bc, d, Bc)] = sKstage[i];
        sV[canon_idx(d, bc, D)]  = sVstage[i];
      }
      __syncthreads();
    }

    {
      const uint32_t idesc = make_idesc_bf16(2 * Br, Bc);
      if(crank == 0 && tid == 0){
        const uint64_t descK_base = make_smem_desc(sK, Bc);
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        for(int kt = 0; kt < D/16; ++kt){
          uint64_t descQ = advance_desc_katom(descQ_base, kt, Br);
          uint64_t descK = advance_desc_katom(descK_base, kt, Bc);
          uint32_t accumulate = (kt > 0) ? 1u : 0u;
          asm volatile(
            "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
            :: "r"(tmem_addr), "l"(descQ), "l"(descK), "r"(idesc), "r"(accumulate) : "memory");
        }
        mbar_commit_mma_2cta_multicast(mma_bar, 0b11);
      }
      mbar_wait(mma_bar, mbar_phase); mbar_phase ^= 1;
      tmem_readout_to_smem_vec_2cta(sS, tmem_addr, Br, Bc, Bc, scale_l2e, rank_warp_offset);
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");

      // CAUSAL per-rank mask — see the header comment above this kernel group.
      if(tid < Br){
        if(kc > q_tile){
          for(int j = 0; j < Bc; ++j) sS[tid * Bc + j] = -INFINITY;
        } else if(kc == q_tile){
          for(int j = tid + 1; j < Bc; ++j) sS[tid * Bc + j] = -INFINITY;
        }
      }
      __syncthreads();
    }

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

    {
      const uint64_t descP_base = make_smem_desc(sP, Br);
      const uint32_t idesc      = make_idesc_bf16(2 * Br, D);
      if(crank == 0 && tid == 0){
        const uint64_t descV_base = make_smem_desc(sV, D);
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        for(int kt = 0; kt < Bc/16; ++kt){
          uint64_t descP = advance_desc_katom(descP_base, kt, Br);
          uint64_t descV = advance_desc_katom(descV_base, kt, D);
          uint32_t accumulate = (kt > 0) ? 1u : 0u;
          asm volatile(
            "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
            :: "r"(tmem_addr), "l"(descP), "l"(descV), "r"(idesc), "r"(accumulate) : "memory");
        }
        mbar_commit_mma_2cta_multicast(mma_bar, 0b11);
      }
      mbar_wait(mma_bar, mbar_phase); mbar_phase ^= 1;
      tmem_readout_accum_vec_2cta(sO, tmem_addr, Br, D, D, rank_warp_offset);
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
      __syncthreads();
    }
  } // end kv loop

  cluster.sync();
  if(tid < 32)
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
                 :: "r"(tmem_addr), "r"(NCOLS) : "memory");
  __syncthreads();

  for(int i = 2 * tid; i < Br * D; i += 2 * blockDim.x){
    const float denom = sl[i / D];
    *reinterpret_cast<__nv_bfloat162*>(&d_O[qBase + i]) =
        __floats2bfloat162_rn(sO[i] / denom, sO[i + 1] / denom);
  }
  if(tid < Br)
    d_LSE[lBase + tid] = 0.6931471805599453f * (sm[tid] + log2f(sl[tid]));

} // end of gqa_v20_causal

// gqa_v21_causal — V20_causal + genuine N-half split B operand (same B-split as
// GQA_sm103.cu's gqa_v21). Causal loop bound / masking identical to V20_causal above.
template<int Br, int Bc, int D>
__global__ void __cluster_dims__(2, 1, 1) gqa_v21_causal(
  __nv_bfloat16 *d_Q,
  __nv_bfloat16 *d_O,
  float *d_LSE,
  const __grid_constant__ CUtensorMap Ktmap_half,  // K, box_rows = Bc/2 (key-range split)
  const __grid_constant__ CUtensorMap Vtmap_half,  // V, box_cols = D/2  (head-dim split)
  int B,
  int Hq,
  int Hkv,
  int G,
  int S,
  float scale
){
  static_assert(Br == Bc, "causal tile-skip + diagonal-tile mask requires Br == Bc");
  static_assert(Bc % 2 == 0, "Bc must be even to split the key range in half");
  static_assert(D  % 2 == 0, "D must be even to split the head dim in half");
  constexpr int Bc_half = Bc / 2;
  constexpr int D_half  = D / 2;

  const int q_tile = blockIdx.x;
  const int hq     = blockIdx.y;
  const int b      = blockIdx.z;
  const int hkv    = hq / G;
  const int tid    = threadIdx.x;

  cg::cluster_group cluster = cg::this_cluster();
  const unsigned int crank  = cluster.block_rank();

  const int q_row0   = q_tile * Br;
  const int nKVTiles = (q_tile | 1) + 1;   // CAUSAL: cover the pair's higher/odd q_tile
  const int kvRow0   = (b * Hkv + hkv) * S;

  const long qBase  = ((long)(b * Hq + hq) * S + q_row0) * D;
  const long lBase  = ((long)(b * Hq + hq) * S + q_row0);

  const float scale_l2e = scale * 1.4426950408889634f;

  __shared__ __align__(16)  __nv_bfloat16 sQ[Br * D];
  __shared__ __align__(128) __nv_bfloat16 sKstage[Bc_half * D];
  __shared__ __align__(128) __nv_bfloat16 sVstage[Bc * D_half];
  __shared__ __align__(16)  __nv_bfloat16 sK[Bc_half * D];
  __shared__ __align__(16)  __nv_bfloat16 sV[D_half * Bc];
  __shared__ __align__(16)  __nv_bfloat16 sP[Br * Bc];
  __shared__ __align__(16)  float         sS[Br * Bc];
  __shared__ __align__(16)  float         sO[Br * D];
  __shared__ float sm[Br];
  __shared__ float sl[Br];
  __shared__ float sCorr[Br];
  __shared__ __align__(8) uint64_t s_mma_bar;
  __shared__ __align__(8) uint64_t s_load_bar;

  for(int i = tid; i < Br * D; i += blockDim.x){
    const int r = i / D, c = i % D;
    sQ[canon_idx(r, c, Br)] = d_Q[qBase + i];
    sO[i] = 0.0f;
  }
  if(tid < Br){ sm[tid] = -INFINITY; sl[tid] = 0.0f; }

  const uint32_t mma_bar = (uint32_t)__cvta_generic_to_shared(&s_mma_bar);
  const uint32_t lbar    = (uint32_t)__cvta_generic_to_shared(&s_load_bar);
  if(tid == 0){
    mbar_init(mma_bar, 1);
    mbar_init(lbar, 1);
    cluster_fence_mbarrier_init();
  }
  cluster.sync();

  constexpr uint32_t NCOLS = (Bc > D) ? (uint32_t)Bc : (uint32_t)D;
  static_assert(NCOLS >= 32 && (NCOLS & (NCOLS - 1)) == 0,
                "tcgen05 column count must be a power of two >= 32");
  constexpr uint32_t RANK_WARP_SPAN = (uint32_t)Br / 32u;

  uint32_t tmem_addr;
  {
    __shared__ uint32_t s_tmem_addr;
    if(tid < 32){
      uint32_t s_addr = (uint32_t)__cvta_generic_to_shared(&s_tmem_addr);
      asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                   :: "r"(s_addr), "r"(NCOLS) : "memory");
      asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::: "memory");
    }
    __syncthreads();
    tmem_addr = s_tmem_addr;
  }

  int mbar_phase = 0;
  int load_phase = 0;
  const uint32_t TX = ((uint32_t)Bc_half * (uint32_t)D + (uint32_t)Bc * (uint32_t)D_half)
                    * (uint32_t)sizeof(__nv_bfloat16);
  const uint32_t rank_warp_offset = crank * RANK_WARP_SPAN;

  const uint64_t descQ_base = make_smem_desc(sQ, Br);

  for(int kc = 0; kc < nKVTiles; ++kc){
    {
      if(tid == 0){
        const int kRow = kvRow0 + kc * Bc + (int)crank * Bc_half;
        const int vCol = (int)crank * D_half;
        mbar_expect_tx(lbar, TX);
        tma_load_2d((uint32_t)__cvta_generic_to_shared(sKstage), &Ktmap_half, 0, kRow, lbar);
        tma_load_2d((uint32_t)__cvta_generic_to_shared(sVstage), &Vtmap_half, vCol, kvRow0 + kc * Bc, lbar);
        mbar_arrive(lbar);
      }
      mbar_wait(lbar, load_phase); load_phase ^= 1;
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      __syncthreads();

      for(int i = tid; i < Bc_half * D; i += blockDim.x){
        const int bc = i / D, d = i % D;
        sK[canon_idx(bc, d, Bc_half)] = sKstage[i];
      }
      for(int i = tid; i < Bc * D_half; i += blockDim.x){
        const int bc = i / D_half, d = i % D_half;
        sV[canon_idx(d, bc, D_half)] = sVstage[i];
      }
      __syncthreads();
    }

    {
      const uint32_t idesc = make_idesc_bf16(2 * Br, Bc);
      if(crank == 0 && tid == 0){
        const uint64_t descK_base = make_smem_desc(sK, Bc_half);
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        for(int kt = 0; kt < D/16; ++kt){
          uint64_t descQ = advance_desc_katom(descQ_base, kt, Br);
          uint64_t descK = advance_desc_katom(descK_base, kt, Bc_half);
          uint32_t accumulate = (kt > 0) ? 1u : 0u;
          asm volatile(
            "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
            :: "r"(tmem_addr), "l"(descQ), "l"(descK), "r"(idesc), "r"(accumulate) : "memory");
        }
        mbar_commit_mma_2cta_multicast(mma_bar, 0b11);
      }
      mbar_wait(mma_bar, mbar_phase); mbar_phase ^= 1;
      tmem_readout_to_smem_vec_2cta(sS, tmem_addr, Br, Bc, Bc, scale_l2e, rank_warp_offset);
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");

      if(tid < Br){
        if(kc > q_tile){
          for(int j = 0; j < Bc; ++j) sS[tid * Bc + j] = -INFINITY;
        } else if(kc == q_tile){
          for(int j = tid + 1; j < Bc; ++j) sS[tid * Bc + j] = -INFINITY;
        }
      }
      __syncthreads();
    }

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

    {
      const uint64_t descP_base = make_smem_desc(sP, Br);
      const uint32_t idesc      = make_idesc_bf16(2 * Br, D);
      if(crank == 0 && tid == 0){
        const uint64_t descV_base = make_smem_desc(sV, D_half);
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        for(int kt = 0; kt < Bc/16; ++kt){
          uint64_t descP = advance_desc_katom(descP_base, kt, Br);
          uint64_t descV = advance_desc_katom(descV_base, kt, D_half);
          uint32_t accumulate = (kt > 0) ? 1u : 0u;
          asm volatile(
            "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
            :: "r"(tmem_addr), "l"(descP), "l"(descV), "r"(idesc), "r"(accumulate) : "memory");
        }
        mbar_commit_mma_2cta_multicast(mma_bar, 0b11);
      }
      mbar_wait(mma_bar, mbar_phase); mbar_phase ^= 1;
      tmem_readout_accum_vec_2cta(sO, tmem_addr, Br, D, D, rank_warp_offset);
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
      __syncthreads();
    }
  } // end kv loop

  cluster.sync();
  if(tid < 32)
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
                 :: "r"(tmem_addr), "r"(NCOLS) : "memory");
  __syncthreads();

  for(int i = 2 * tid; i < Br * D; i += 2 * blockDim.x){
    const float denom = sl[i / D];
    *reinterpret_cast<__nv_bfloat162*>(&d_O[qBase + i]) =
        __floats2bfloat162_rn(sO[i] / denom, sO[i + 1] / denom);
  }
  if(tid < Br)
    d_LSE[lBase + tid] = 0.6931471805599453f * (sm[tid] + log2f(sl[tid]));

} // end of gqa_v21_causal

// gqa_v22_causal — V21_causal + warp specialization (dedicated producer warp, threads
// 128..159, decoupled from the 128 consumer threads via load_bar/free_bar + consumer_sync).
// Same causal loop bound / three-case mask as V20/V21_causal.
template<int Br, int Bc, int D>
__global__ void __cluster_dims__(2, 1, 1) gqa_v22_causal(
  __nv_bfloat16 *d_Q,
  __nv_bfloat16 *d_O,
  float *d_LSE,
  const __grid_constant__ CUtensorMap Ktmap_half,
  const __grid_constant__ CUtensorMap Vtmap_half,
  int B,
  int Hq,
  int Hkv,
  int G,
  int S,
  float scale
){
  static_assert(Bc % 2 == 0, "Bc must be even to split the key range in half");
  static_assert(D  % 2 == 0, "D must be even to split the head dim in half");
  static_assert(Br == 128, "V22_causal's consumer group is hardwired to 128 threads");
  static_assert(Br == Bc, "causal tile-skip + diagonal-tile mask requires Br == Bc");
  constexpr int Bc_half = Bc / 2;
  constexpr int D_half  = D / 2;

  const int q_tile = blockIdx.x;
  const int hq     = blockIdx.y;
  const int b      = blockIdx.z;
  const int hkv    = hq / G;
  const int tid    = threadIdx.x;   // 0..159: 0..127 consumer, 128..159 producer

  cg::cluster_group cluster = cg::this_cluster();
  const unsigned int crank  = cluster.block_rank();

  const int q_row0   = q_tile * Br;
  const int nKVTiles = (q_tile | 1) + 1;   // CAUSAL: cover the pair's higher/odd q_tile
  const int kvRow0   = (b * Hkv + hkv) * S;

  const long qBase  = ((long)(b * Hq + hq) * S + q_row0) * D;
  const long lBase  = ((long)(b * Hq + hq) * S + q_row0);

  const float scale_l2e = scale * 1.4426950408889634f;

  __shared__ __align__(16)  __nv_bfloat16 sQ[Br * D];
  __shared__ __align__(128) __nv_bfloat16 sKstage[Bc_half * D];
  __shared__ __align__(128) __nv_bfloat16 sVstage[Bc * D_half];
  __shared__ __align__(16)  __nv_bfloat16 sK[Bc_half * D];
  __shared__ __align__(16)  __nv_bfloat16 sV[D_half * Bc];
  __shared__ __align__(16)  __nv_bfloat16 sP[Br * Bc];
  __shared__ __align__(16)  float         sS[Br * Bc];
  __shared__ __align__(16)  float         sO[Br * D];
  __shared__ float sm[Br];
  __shared__ float sl[Br];
  __shared__ float sCorr[Br];
  __shared__ __align__(8) uint64_t s_mma_bar;
  __shared__ __align__(8) uint64_t s_load_bar;
  __shared__ __align__(8) uint64_t s_free_bar;

  for(int i = tid; i < Br * D; i += blockDim.x){
    const int r = i / D, c = i % D;
    sQ[canon_idx(r, c, Br)] = d_Q[qBase + i];
    sO[i] = 0.0f;
  }
  if(tid < Br){ sm[tid] = -INFINITY; sl[tid] = 0.0f; }

  const uint32_t mma_bar = (uint32_t)__cvta_generic_to_shared(&s_mma_bar);
  const uint32_t lbar    = (uint32_t)__cvta_generic_to_shared(&s_load_bar);
  const uint32_t fbar    = (uint32_t)__cvta_generic_to_shared(&s_free_bar);
  if(tid == 0){
    mbar_init(mma_bar, 1);
    mbar_init(lbar, 1);
    mbar_init(fbar, 1);
    cluster_fence_mbarrier_init();
  }
  cluster.sync();

  constexpr uint32_t NCOLS = (Bc > D) ? (uint32_t)Bc : (uint32_t)D;
  static_assert(NCOLS >= 32 && (NCOLS & (NCOLS - 1)) == 0,
                "tcgen05 column count must be a power of two >= 32");
  constexpr uint32_t RANK_WARP_SPAN = (uint32_t)Br / 32u;

  uint32_t tmem_addr;
  {
    __shared__ uint32_t s_tmem_addr;
    if(tid < 32){
      uint32_t s_addr = (uint32_t)__cvta_generic_to_shared(&s_tmem_addr);
      asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                   :: "r"(s_addr), "r"(NCOLS) : "memory");
      asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::: "memory");
    }
    __syncthreads();
    tmem_addr = s_tmem_addr;
  }

  const uint32_t TX = ((uint32_t)Bc_half * (uint32_t)D + (uint32_t)Bc * (uint32_t)D_half)
                    * (uint32_t)sizeof(__nv_bfloat16);
  const uint32_t rank_warp_offset = crank * RANK_WARP_SPAN;
  const uint32_t sKstage_addr = (uint32_t)__cvta_generic_to_shared(sKstage);
  const uint32_t sVstage_addr = (uint32_t)__cvta_generic_to_shared(sVstage);

  if(tid >= 128){
    if(tid == 128){
      int free_phase = 0;
      for(int kc = 0; kc < nKVTiles; ++kc){
        if(kc > 0){ mbar_wait(fbar, free_phase); free_phase ^= 1; }
        const int kRow = kvRow0 + kc * Bc + (int)crank * Bc_half;
        const int vCol = (int)crank * D_half;
        mbar_expect_tx(lbar, TX);
        tma_load_2d(sKstage_addr, &Ktmap_half, 0, kRow, lbar);
        tma_load_2d(sVstage_addr, &Vtmap_half, vCol, kvRow0 + kc * Bc, lbar);
        mbar_arrive(lbar);
      }
    }
  } else {
    const uint64_t descQ_base = make_smem_desc(sQ, Br);
    int mbar_phase = 0;
    int load_phase = 0;

    for(int kc = 0; kc < nKVTiles; ++kc){
      mbar_wait(lbar, load_phase); load_phase ^= 1;
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      consumer_sync();

      for(int i = tid; i < Bc_half * D; i += Br){
        const int bc = i / D, d = i % D;
        sK[canon_idx(bc, d, Bc_half)] = sKstage[i];
      }
      for(int i = tid; i < Bc * D_half; i += Br){
        const int bc = i / D_half, d = i % D_half;
        sV[canon_idx(d, bc, D_half)] = sVstage[i];
      }
      consumer_sync();
      if(tid == 0) mbar_arrive(fbar);

      {
        const uint32_t idesc = make_idesc_bf16(2 * Br, Bc);
        if(crank == 0 && tid == 0){
          const uint64_t descK_base = make_smem_desc(sK, Bc_half);
          asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
          for(int kt = 0; kt < D/16; ++kt){
            uint64_t descQ = advance_desc_katom(descQ_base, kt, Br);
            uint64_t descK = advance_desc_katom(descK_base, kt, Bc_half);
            uint32_t accumulate = (kt > 0) ? 1u : 0u;
            asm volatile(
              "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
              "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
              :: "r"(tmem_addr), "l"(descQ), "l"(descK), "r"(idesc), "r"(accumulate) : "memory");
          }
          mbar_commit_mma_2cta_multicast(mma_bar, 0b11);
        }
        mbar_wait(mma_bar, mbar_phase); mbar_phase ^= 1;
        tmem_readout_to_smem_vec_2cta(sS, tmem_addr, Br, Bc, Bc, scale_l2e, rank_warp_offset);
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");

        if(kc > q_tile){
          for(int j = 0; j < Bc; ++j) sS[tid * Bc + j] = -INFINITY;
        } else if(kc == q_tile){
          for(int j = tid + 1; j < Bc; ++j) sS[tid * Bc + j] = -INFINITY;
        }
        consumer_sync();
      }

      {
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
      consumer_sync();

      for(int i = tid; i < Br * D; i += Br) sO[i] *= sCorr[i / D];
      consumer_sync();

      {
        const uint64_t descP_base = make_smem_desc(sP, Br);
        const uint32_t idesc      = make_idesc_bf16(2 * Br, D);
        if(crank == 0 && tid == 0){
          const uint64_t descV_base = make_smem_desc(sV, D_half);
          asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
          for(int kt = 0; kt < Bc/16; ++kt){
            uint64_t descP = advance_desc_katom(descP_base, kt, Br);
            uint64_t descV = advance_desc_katom(descV_base, kt, D_half);
            uint32_t accumulate = (kt > 0) ? 1u : 0u;
            asm volatile(
              "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
              "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
              :: "r"(tmem_addr), "l"(descP), "l"(descV), "r"(idesc), "r"(accumulate) : "memory");
          }
          mbar_commit_mma_2cta_multicast(mma_bar, 0b11);
        }
        mbar_wait(mma_bar, mbar_phase); mbar_phase ^= 1;
        tmem_readout_accum_vec_2cta(sO, tmem_addr, Br, D, D, rank_warp_offset);
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
        consumer_sync();
      }
    } // end kv loop
  }

  __syncthreads();
  cluster.sync();
  if(tid < 32)
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
                 :: "r"(tmem_addr), "r"(NCOLS) : "memory");
  __syncthreads();

  for(int i = 2 * tid; i < Br * D; i += 2 * blockDim.x){
    const float denom = sl[i / D];
    *reinterpret_cast<__nv_bfloat162*>(&d_O[qBase + i]) =
        __floats2bfloat162_rn(sO[i] / denom, sO[i + 1] / denom);
  }
  if(tid < Br)
    d_LSE[lBase + tid] = 0.6931471805599453f * (sm[tid] + log2f(sl[tid]));

} // end of gqa_v22_causal

// gqa_v23_causal — V22_causal + double-buffered TMA staging (ping-pong sKstage/
// sVstage). Same causal loop bound / three-case mask as V20-22_causal.
template<int Br, int Bc, int D>
__global__ void __cluster_dims__(2, 1, 1) gqa_v23_causal(
  __nv_bfloat16 *d_Q,
  __nv_bfloat16 *d_O,
  float *d_LSE,
  const __grid_constant__ CUtensorMap Ktmap_half,
  const __grid_constant__ CUtensorMap Vtmap_half,
  int B,
  int Hq,
  int Hkv,
  int G,
  int S,
  float scale
){
  static_assert(Bc % 2 == 0, "Bc must be even to split the key range in half");
  static_assert(D  % 2 == 0, "D must be even to split the head dim in half");
  static_assert(Br == 128, "V23_causal's consumer group is hardwired to 128 threads");
  static_assert(Br == Bc, "causal tile-skip + diagonal-tile mask requires Br == Bc");
  constexpr int Bc_half = Bc / 2;
  constexpr int D_half  = D / 2;

  const int q_tile = blockIdx.x;
  const int hq     = blockIdx.y;
  const int b      = blockIdx.z;
  const int hkv    = hq / G;
  const int tid    = threadIdx.x;   // 0..159: 0..127 consumer, 128..159 producer

  cg::cluster_group cluster = cg::this_cluster();
  const unsigned int crank  = cluster.block_rank();

  const int q_row0   = q_tile * Br;
  const int nKVTiles = (q_tile | 1) + 1;   // CAUSAL: cover the pair's higher/odd q_tile
  const int kvRow0   = (b * Hkv + hkv) * S;

  const long qBase  = ((long)(b * Hq + hq) * S + q_row0) * D;
  const long lBase  = ((long)(b * Hq + hq) * S + q_row0);

  const float scale_l2e = scale * 1.4426950408889634f;

  __shared__ __align__(16)  __nv_bfloat16 sQ[Br * D];
  __shared__ __align__(128) __nv_bfloat16 sKstage[2][Bc_half * D];
  __shared__ __align__(128) __nv_bfloat16 sVstage[2][Bc * D_half];
  __shared__ __align__(16)  __nv_bfloat16 sK[Bc_half * D];
  __shared__ __align__(16)  __nv_bfloat16 sV[D_half * Bc];
  __shared__ __align__(16)  __nv_bfloat16 sP[Br * Bc];
  __shared__ __align__(16)  float         sS[Br * Bc];
  __shared__ __align__(16)  float         sO[Br * D];
  __shared__ float sm[Br];
  __shared__ float sl[Br];
  __shared__ float sCorr[Br];
  __shared__ __align__(8) uint64_t s_mma_bar;
  __shared__ __align__(8) uint64_t s_load_bar[2];
  __shared__ __align__(8) uint64_t s_free_bar[2];

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
    cluster_fence_mbarrier_init();
  }
  cluster.sync();

  constexpr uint32_t NCOLS = (Bc > D) ? (uint32_t)Bc : (uint32_t)D;
  static_assert(NCOLS >= 32 && (NCOLS & (NCOLS - 1)) == 0,
                "tcgen05 column count must be a power of two >= 32");
  constexpr uint32_t RANK_WARP_SPAN = (uint32_t)Br / 32u;

  uint32_t tmem_addr;
  {
    __shared__ uint32_t s_tmem_addr;
    if(tid < 32){
      uint32_t s_addr = (uint32_t)__cvta_generic_to_shared(&s_tmem_addr);
      asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                   :: "r"(s_addr), "r"(NCOLS) : "memory");
      asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::: "memory");
    }
    __syncthreads();
    tmem_addr = s_tmem_addr;
  }

  const uint32_t TX = ((uint32_t)Bc_half * (uint32_t)D + (uint32_t)Bc * (uint32_t)D_half)
                    * (uint32_t)sizeof(__nv_bfloat16);
  const uint32_t rank_warp_offset = crank * RANK_WARP_SPAN;
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
    if(tid == 128){
      int free_phase[2] = {0, 0};
      for(int kc = 0; kc < nKVTiles; ++kc){
        const int slot = kc & 1;
        if(kc >= 2){ mbar_wait(fbar[slot], free_phase[slot]); free_phase[slot] ^= 1; }
        const int kRow = kvRow0 + kc * Bc + (int)crank * Bc_half;
        const int vCol = (int)crank * D_half;
        mbar_expect_tx(lbar[slot], TX);
        tma_load_2d(sKstage_addr[slot], &Ktmap_half, 0, kRow, lbar[slot]);
        tma_load_2d(sVstage_addr[slot], &Vtmap_half, vCol, kvRow0 + kc * Bc, lbar[slot]);
        mbar_arrive(lbar[slot]);
      }
    }
  } else {
    const uint64_t descQ_base = make_smem_desc(sQ, Br);
    int mbar_phase = 0;
    int load_phase[2] = {0, 0};

    for(int kc = 0; kc < nKVTiles; ++kc){
      const int slot = kc & 1;
      mbar_wait(lbar[slot], load_phase[slot]); load_phase[slot] ^= 1;
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      consumer_sync();

      for(int i = tid; i < Bc_half * D; i += Br){
        const int bc = i / D, d = i % D;
        sK[canon_idx(bc, d, Bc_half)] = sKstage[slot][i];
      }
      for(int i = tid; i < Bc * D_half; i += Br){
        const int bc = i / D_half, d = i % D_half;
        sV[canon_idx(d, bc, D_half)] = sVstage[slot][i];
      }
      consumer_sync();
      if(tid == 0) mbar_arrive(fbar[slot]);

      {
        const uint32_t idesc = make_idesc_bf16(2 * Br, Bc);
        if(crank == 0 && tid == 0){
          const uint64_t descK_base = make_smem_desc(sK, Bc_half);
          asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
          for(int kt = 0; kt < D/16; ++kt){
            uint64_t descQ = advance_desc_katom(descQ_base, kt, Br);
            uint64_t descK = advance_desc_katom(descK_base, kt, Bc_half);
            uint32_t accumulate = (kt > 0) ? 1u : 0u;
            asm volatile(
              "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
              "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
              :: "r"(tmem_addr), "l"(descQ), "l"(descK), "r"(idesc), "r"(accumulate) : "memory");
          }
          mbar_commit_mma_2cta_multicast(mma_bar, 0b11);
        }
        mbar_wait(mma_bar, mbar_phase); mbar_phase ^= 1;
        tmem_readout_to_smem_vec_2cta(sS, tmem_addr, Br, Bc, Bc, scale_l2e, rank_warp_offset);
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");

        if(kc > q_tile){
          for(int j = 0; j < Bc; ++j) sS[tid * Bc + j] = -INFINITY;
        } else if(kc == q_tile){
          for(int j = tid + 1; j < Bc; ++j) sS[tid * Bc + j] = -INFINITY;
        }
        consumer_sync();
      }

      {
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
      consumer_sync();

      for(int i = tid; i < Br * D; i += Br) sO[i] *= sCorr[i / D];
      consumer_sync();

      {
        const uint64_t descP_base = make_smem_desc(sP, Br);
        const uint32_t idesc      = make_idesc_bf16(2 * Br, D);
        if(crank == 0 && tid == 0){
          const uint64_t descV_base = make_smem_desc(sV, D_half);
          asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
          for(int kt = 0; kt < Bc/16; ++kt){
            uint64_t descP = advance_desc_katom(descP_base, kt, Br);
            uint64_t descV = advance_desc_katom(descV_base, kt, D_half);
            uint32_t accumulate = (kt > 0) ? 1u : 0u;
            asm volatile(
              "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
              "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
              :: "r"(tmem_addr), "l"(descP), "l"(descV), "r"(idesc), "r"(accumulate) : "memory");
          }
          mbar_commit_mma_2cta_multicast(mma_bar, 0b11);
        }
        mbar_wait(mma_bar, mbar_phase); mbar_phase ^= 1;
        tmem_readout_accum_vec_2cta(sO, tmem_addr, Br, D, D, rank_warp_offset);
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
        consumer_sync();
      }
    } // end kv loop
  }

  __syncthreads();
  cluster.sync();
  if(tid < 32)
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
                 :: "r"(tmem_addr), "r"(NCOLS) : "memory");
  __syncthreads();

  for(int i = 2 * tid; i < Br * D; i += 2 * blockDim.x){
    const float denom = sl[i / D];
    *reinterpret_cast<__nv_bfloat162*>(&d_O[qBase + i]) =
        __floats2bfloat162_rn(sO[i] / denom, sO[i + 1] / denom);
  }
  if(tid < Br)
    d_LSE[lBase + tid] = 0.6931471805599453f * (sm[tid] + log2f(sl[tid]));

} // end of gqa_v23_causal

// gqa_v24_causal — V23_causal + persistent launch, matching cuDNN's observed launch
// shape for this problem size (kernel name decodes to 128x128x64 tile, cga2x1x1,
// 4x1x1 warpgroups = 512 threads; grid was (16,12,8) vs our (32,12,8)). This version
// isolates JUST the persistent-grid change (grid.x halved, each CTA-pair loops over
// 2 virtual q_tiles) on top of V23_causal's exact threading model (160 threads: 128
// consumer + 32 producer) — deliberately NOT yet adopting the wider 512-thread/
// 4-warpgroup block, so the perf delta from persistence alone can be measured before
// taking on that much riskier rebuild (see V16's persistent-kernel note: it isolated
// persistence the same way before combining with other changes).
//
// TMEM alloc/dealloc is hoisted OUTSIDE the outer loop (paid once per CTA instead of
// once per q_tile, mirroring V16). Mbarriers are DELIBERATELY re-initialized at the
// top of EACH outer iteration rather than also hoisted: carrying the software phase
// counters across iterations without also carrying an exactly-matching hardware
// mbarrier phase risks a subtle wait/phase-parity mismatch whenever a slot's use
// count differs in parity between the two virtual q_tiles (e.g. a short first
// q_tile leaves a pending, never-waited-on arrival that would desync a persisted
// phase counter for the second q_tile) — not worth the risk without hardware access
// to verify it. Re-init cost is a few mbarrier.init instructions + one cluster.sync()
// per outer iteration (2 per CTA lifetime instead of 1 per q_tile-launch), so the
// launch-count reduction (the actual lever being tested here) is still captured.
template<int Br, int Bc, int D>
__global__ void __cluster_dims__(2, 1, 1) gqa_v24_causal(
  __nv_bfloat16 *d_Q,
  __nv_bfloat16 *d_O,
  float *d_LSE,
  const __grid_constant__ CUtensorMap Ktmap_half,
  const __grid_constant__ CUtensorMap Vtmap_half,
  int B,
  int Hq,
  int Hkv,
  int G,
  int S,
  float scale
){
  static_assert(Bc % 2 == 0, "Bc must be even to split the key range in half");
  static_assert(D  % 2 == 0, "D must be even to split the head dim in half");
  static_assert(Br == 128, "V24_causal's consumer group is hardwired to 128 threads");
  static_assert(Br == Bc, "causal tile-skip + diagonal-tile mask requires Br == Bc");
  constexpr int Bc_half = Bc / 2;
  constexpr int D_half  = D / 2;

  const int q_tile_base = blockIdx.x;   // 0..gridDim.x-1 — this CTA's FIRST virtual q_tile
  const int hq  = blockIdx.y;
  const int b   = blockIdx.z;
  const int hkv = hq / G;
  const int tid = threadIdx.x;   // 0..159: 0..127 consumer, 128..159 producer

  cg::cluster_group cluster = cg::this_cluster();
  const unsigned int crank  = cluster.block_rank();

  const int kvRow0 = (b * Hkv + hkv) * S;   // same for every virtual q_tile in this CTA
  const float scale_l2e = scale * 1.4426950408889634f;

  // Persistent: this CTA-pair covers (S/Br)/gridDim.x virtual q_tiles, spaced gridDim.x
  // apart (q_tile_base, q_tile_base+gridDim.x, ...) — assumes exact divisibility, which
  // the launcher guarantees for the configured grid (S/Br=32, gridDim.x=16 -> nOuter=2).
  const int nOuter = (S / Br) / (int)gridDim.x;

  __shared__ __align__(16)  __nv_bfloat16 sQ[Br * D];
  __shared__ __align__(128) __nv_bfloat16 sKstage[2][Bc_half * D];
  __shared__ __align__(128) __nv_bfloat16 sVstage[2][Bc * D_half];
  __shared__ __align__(16)  __nv_bfloat16 sK[Bc_half * D];
  __shared__ __align__(16)  __nv_bfloat16 sV[D_half * Bc];
  __shared__ __align__(16)  __nv_bfloat16 sP[Br * Bc];
  __shared__ __align__(16)  float         sS[Br * Bc];
  __shared__ __align__(16)  float         sO[Br * D];
  __shared__ float sm[Br];
  __shared__ float sl[Br];
  __shared__ float sCorr[Br];
  __shared__ __align__(8) uint64_t s_mma_bar;
  __shared__ __align__(8) uint64_t s_load_bar[2];
  __shared__ __align__(8) uint64_t s_free_bar[2];

  const uint32_t mma_bar = (uint32_t)__cvta_generic_to_shared(&s_mma_bar);
  const uint32_t lbar0   = (uint32_t)__cvta_generic_to_shared(&s_load_bar[0]);
  const uint32_t lbar1   = (uint32_t)__cvta_generic_to_shared(&s_load_bar[1]);
  const uint32_t fbar0   = (uint32_t)__cvta_generic_to_shared(&s_free_bar[0]);
  const uint32_t fbar1   = (uint32_t)__cvta_generic_to_shared(&s_free_bar[1]);
  const uint32_t lbar[2] = {lbar0, lbar1};
  const uint32_t fbar[2] = {fbar0, fbar1};

  constexpr uint32_t NCOLS = (Bc > D) ? (uint32_t)Bc : (uint32_t)D;
  static_assert(NCOLS >= 32 && (NCOLS & (NCOLS - 1)) == 0,
                "tcgen05 column count must be a power of two >= 32");
  constexpr uint32_t RANK_WARP_SPAN = (uint32_t)Br / 32u;
  const uint32_t rank_warp_offset = crank * RANK_WARP_SPAN;

  const uint32_t TX = ((uint32_t)Bc_half * (uint32_t)D + (uint32_t)Bc * (uint32_t)D_half)
                    * (uint32_t)sizeof(__nv_bfloat16);
  const uint32_t sKstage_addr[2] = {
    (uint32_t)__cvta_generic_to_shared(sKstage[0]),
    (uint32_t)__cvta_generic_to_shared(sKstage[1])
  };
  const uint32_t sVstage_addr[2] = {
    (uint32_t)__cvta_generic_to_shared(sVstage[0]),
    (uint32_t)__cvta_generic_to_shared(sVstage[1])
  };

  // ---- TMEM alloc: ONCE for the whole CTA lifetime (hoisted out of the outer loop). ----
  uint32_t tmem_addr;
  {
    __shared__ uint32_t s_tmem_addr;
    if(tid < 32){
      uint32_t s_addr = (uint32_t)__cvta_generic_to_shared(&s_tmem_addr);
      asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                   :: "r"(s_addr), "r"(NCOLS) : "memory");
      asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::: "memory");
    }
    __syncthreads();
    tmem_addr = s_tmem_addr;
  }

  for(int outer = 0; outer < nOuter; ++outer){
    const int q_tile = q_tile_base + outer * (int)gridDim.x;
    const int q_row0 = q_tile * Br;
    const int nKVTiles = (q_tile | 1) + 1;   // CAUSAL: cover the pair's higher/odd q_tile

    const long qBase = ((long)(b * Hq + hq) * S + q_row0) * D;
    const long lBase = ((long)(b * Hq + hq) * S + q_row0);

    for(int i = tid; i < Br * D; i += blockDim.x){
      const int r = i / D, c = i % D;
      sQ[canon_idx(r, c, Br)] = d_Q[qBase + i];
      sO[i] = 0.0f;
    }
    if(tid < Br){ sm[tid] = -INFINITY; sl[tid] = 0.0f; }

    // Re-init mbarriers each outer iteration — see header comment for why this isn't
    // also hoisted. cluster.sync() here does double duty: makes the Q/O reset above
    // visible block-wide AND makes this iteration's mbar inits visible cluster-wide.
    if(tid == 0){
      mbar_init(mma_bar, 1);
      mbar_init(lbar0, 1); mbar_init(lbar1, 1);
      mbar_init(fbar0, 1); mbar_init(fbar1, 1);
      cluster_fence_mbarrier_init();
    }
    cluster.sync();

    if(tid >= 128){
      if(tid == 128){
        int free_phase[2] = {0, 0};
        for(int kc = 0; kc < nKVTiles; ++kc){
          const int slot = kc & 1;
          if(kc >= 2){ mbar_wait(fbar[slot], free_phase[slot]); free_phase[slot] ^= 1; }
          const int kRow = kvRow0 + kc * Bc + (int)crank * Bc_half;
          const int vCol = (int)crank * D_half;
          mbar_expect_tx(lbar[slot], TX);
          tma_load_2d(sKstage_addr[slot], &Ktmap_half, 0, kRow, lbar[slot]);
          tma_load_2d(sVstage_addr[slot], &Vtmap_half, vCol, kvRow0 + kc * Bc, lbar[slot]);
          mbar_arrive(lbar[slot]);
        }
      }
    } else {
      const uint64_t descQ_base = make_smem_desc(sQ, Br);
      int mbar_phase = 0;
      int load_phase[2] = {0, 0};

      for(int kc = 0; kc < nKVTiles; ++kc){
        const int slot = kc & 1;
        mbar_wait(lbar[slot], load_phase[slot]); load_phase[slot] ^= 1;
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        consumer_sync();

        for(int i = tid; i < Bc_half * D; i += Br){
          const int bc = i / D, d = i % D;
          sK[canon_idx(bc, d, Bc_half)] = sKstage[slot][i];
        }
        for(int i = tid; i < Bc * D_half; i += Br){
          const int bc = i / D_half, d = i % D_half;
          sV[canon_idx(d, bc, D_half)] = sVstage[slot][i];
        }
        consumer_sync();
        if(tid == 0) mbar_arrive(fbar[slot]);

        {
          const uint32_t idesc = make_idesc_bf16(2 * Br, Bc);
          if(crank == 0 && tid == 0){
            const uint64_t descK_base = make_smem_desc(sK, Bc_half);
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            for(int kt = 0; kt < D/16; ++kt){
              uint64_t descQ = advance_desc_katom(descQ_base, kt, Br);
              uint64_t descK = advance_desc_katom(descK_base, kt, Bc_half);
              uint32_t accumulate = (kt > 0) ? 1u : 0u;
              asm volatile(
                "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
                "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
                :: "r"(tmem_addr), "l"(descQ), "l"(descK), "r"(idesc), "r"(accumulate) : "memory");
            }
            mbar_commit_mma_2cta_multicast(mma_bar, 0b11);
          }
          mbar_wait(mma_bar, mbar_phase); mbar_phase ^= 1;
          tmem_readout_to_smem_vec_2cta(sS, tmem_addr, Br, Bc, Bc, scale_l2e, rank_warp_offset);
          asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");

          if(kc > q_tile){
            for(int j = 0; j < Bc; ++j) sS[tid * Bc + j] = -INFINITY;
          } else if(kc == q_tile){
            for(int j = tid + 1; j < Bc; ++j) sS[tid * Bc + j] = -INFINITY;
          }
          consumer_sync();
        }

        {
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
        consumer_sync();

        for(int i = tid; i < Br * D; i += Br) sO[i] *= sCorr[i / D];
        consumer_sync();

        {
          const uint64_t descP_base = make_smem_desc(sP, Br);
          const uint32_t idesc      = make_idesc_bf16(2 * Br, D);
          if(crank == 0 && tid == 0){
            const uint64_t descV_base = make_smem_desc(sV, D_half);
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            for(int kt = 0; kt < Bc/16; ++kt){
              uint64_t descP = advance_desc_katom(descP_base, kt, Br);
              uint64_t descV = advance_desc_katom(descV_base, kt, D_half);
              uint32_t accumulate = (kt > 0) ? 1u : 0u;
              asm volatile(
                "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
                "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
                :: "r"(tmem_addr), "l"(descP), "l"(descV), "r"(idesc), "r"(accumulate) : "memory");
            }
            mbar_commit_mma_2cta_multicast(mma_bar, 0b11);
          }
          mbar_wait(mma_bar, mbar_phase); mbar_phase ^= 1;
          tmem_readout_accum_vec_2cta(sO, tmem_addr, Br, D, D, rank_warp_offset);
          asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
          consumer_sync();
        }
      } // end kv loop
    }

    __syncthreads();   // full-block reconvergence: producer is done, consumers are done

    for(int i = 2 * tid; i < Br * D; i += 2 * blockDim.x){
      const float denom = sl[i / D];
      *reinterpret_cast<__nv_bfloat162*>(&d_O[qBase + i]) =
          __floats2bfloat162_rn(sO[i] / denom, sO[i + 1] / denom);
    }
    if(tid < Br)
      d_LSE[lBase + tid] = 0.6931471805599453f * (sm[tid] + log2f(sl[tid]));
  } // end outer (persistent) loop

  // ---- TMEM dealloc: ONCE, after all virtual q_tiles this CTA covers. ----
  cluster.sync();   // both ranks fully done with TMEM before either deallocs
  if(tid < 32)
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
                 :: "r"(tmem_addr), "r"(NCOLS) : "memory");

} // end of gqa_v24_causal

// gqa_v25_causal — V23_causal + widened block (512 threads), staged step 1 towards a
// real FA3-style warpgroup ping-pong (V24's persistent-launch test came back a
// regression, matching V16's finding — persistence alone isn't cuDNN's lever, so
// this isolates the OTHER half of their launch-shape difference: a much wider block).
// Deliberately does NOT touch the MMA/TMEM structure (still ONE M=256 combined
// accumulator, ONE MMA issue per tile, same as V20-24) — that's the higher-risk
// ping-pong rebuild reserved for a later version. What's actually new here:
//   tid   0..127 : compute group — byte-for-byte the same as V23_causal (MMA issue,
//                  TMEM readout, causal mask, online softmax, P@V), using
//                  consumer_sync() (bar id 1, 128 threads) for ITS OWN internal
//                  phases exactly as before.
//   tid 128..159 : producer warp — unchanged from V23_causal (tid==128 only, issues
//                  both K and V TMA sequentially into the ping-pong staging slots).
//   tid 160..511 : NEW reorder-helper group (352 threads) — joins the compute group
//                  ONLY for the K/V reorder-copy step, via reorder_sync() (bar id 2,
//                  480 threads = 128 compute + 352 helper). Widens that copy from
//                  128-way to 480-way parallel. Combined participant index for the
//                  copy is `pidx = tid<128 ? tid : tid-32` (contiguous 0..479, the
//                  32 producer-only tids are deliberately excluded) so the range is
//                  covered exactly once with no gaps/overlaps.
// Helper threads do NOT touch fbar arrival (still tid==0 only, in the compute group)
// and never call consumer_sync() — they sit out the softmax/MMA/PV phases entirely,
// re-entering only at the next kc's reorder_sync(). Epilogue write-out already scales
// with blockDim.x for free, so widening the block also widens that step automatically.
template<int Br, int Bc, int D>
__global__ void __cluster_dims__(2, 1, 1) gqa_v25_causal(
  __nv_bfloat16 *d_Q,
  __nv_bfloat16 *d_O,
  float *d_LSE,
  const __grid_constant__ CUtensorMap Ktmap_half,
  const __grid_constant__ CUtensorMap Vtmap_half,
  int B,
  int Hq,
  int Hkv,
  int G,
  int S,
  float scale
){
  static_assert(Bc % 2 == 0, "Bc must be even to split the key range in half");
  static_assert(D  % 2 == 0, "D must be even to split the head dim in half");
  static_assert(Br == 128, "V25_causal's compute group is hardwired to 128 threads");
  static_assert(Br == Bc, "causal tile-skip + diagonal-tile mask requires Br == Bc");
  constexpr int Bc_half = Bc / 2;
  constexpr int D_half  = D / 2;
  constexpr int NREORDER = 480;   // 128 compute + 352 helper participants in the reorder copy

  const int q_tile = blockIdx.x;
  const int hq     = blockIdx.y;
  const int b      = blockIdx.z;
  const int hkv    = hq / G;
  const int tid    = threadIdx.x;   // 0..511: 0..127 compute, 128..159 producer, 160..511 helper

  cg::cluster_group cluster = cg::this_cluster();
  const unsigned int crank  = cluster.block_rank();

  const int q_row0   = q_tile * Br;
  const int nKVTiles = (q_tile | 1) + 1;   // CAUSAL: cover the pair's higher/odd q_tile
  const int kvRow0   = (b * Hkv + hkv) * S;

  const long qBase  = ((long)(b * Hq + hq) * S + q_row0) * D;
  const long lBase  = ((long)(b * Hq + hq) * S + q_row0);

  const float scale_l2e = scale * 1.4426950408889634f;

  __shared__ __align__(16)  __nv_bfloat16 sQ[Br * D];
  __shared__ __align__(128) __nv_bfloat16 sKstage[2][Bc_half * D];
  __shared__ __align__(128) __nv_bfloat16 sVstage[2][Bc * D_half];
  __shared__ __align__(16)  __nv_bfloat16 sK[Bc_half * D];
  __shared__ __align__(16)  __nv_bfloat16 sV[D_half * Bc];
  __shared__ __align__(16)  __nv_bfloat16 sP[Br * Bc];
  __shared__ __align__(16)  float         sS[Br * Bc];
  __shared__ __align__(16)  float         sO[Br * D];
  __shared__ float sm[Br];
  __shared__ float sl[Br];
  __shared__ float sCorr[Br];
  __shared__ __align__(8) uint64_t s_mma_bar;
  __shared__ __align__(8) uint64_t s_load_bar[2];
  __shared__ __align__(8) uint64_t s_free_bar[2];

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
    cluster_fence_mbarrier_init();
  }
  cluster.sync();   // CTA-collective — all 512 threads

  constexpr uint32_t NCOLS = (Bc > D) ? (uint32_t)Bc : (uint32_t)D;
  static_assert(NCOLS >= 32 && (NCOLS & (NCOLS - 1)) == 0,
                "tcgen05 column count must be a power of two >= 32");
  constexpr uint32_t RANK_WARP_SPAN = (uint32_t)Br / 32u;

  uint32_t tmem_addr;
  {
    __shared__ uint32_t s_tmem_addr;
    if(tid < 32){
      uint32_t s_addr = (uint32_t)__cvta_generic_to_shared(&s_tmem_addr);
      asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                   :: "r"(s_addr), "r"(NCOLS) : "memory");
      asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::: "memory");
    }
    __syncthreads();   // full 512-thread barrier to broadcast tmem_addr
    tmem_addr = s_tmem_addr;
  }

  const uint32_t TX = ((uint32_t)Bc_half * (uint32_t)D + (uint32_t)Bc * (uint32_t)D_half)
                    * (uint32_t)sizeof(__nv_bfloat16);
  const uint32_t rank_warp_offset = crank * RANK_WARP_SPAN;
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

  if(tid >= 128 && tid < 160){
    // ---- Producer warp: unchanged from V23_causal. ----
    if(tid == 128){
      int free_phase[2] = {0, 0};
      for(int kc = 0; kc < nKVTiles; ++kc){
        const int slot = kc & 1;
        if(kc >= 2){ mbar_wait(fbar[slot], free_phase[slot]); free_phase[slot] ^= 1; }
        const int kRow = kvRow0 + kc * Bc + (int)crank * Bc_half;
        const int vCol = (int)crank * D_half;
        mbar_expect_tx(lbar[slot], TX);
        tma_load_2d(sKstage_addr[slot], &Ktmap_half, 0, kRow, lbar[slot]);
        tma_load_2d(sVstage_addr[slot], &Vtmap_half, vCol, kvRow0 + kc * Bc, lbar[slot]);
        mbar_arrive(lbar[slot]);
      }
    }
  } else if(tid >= 160){
    // ---- Reorder-helper group: widens the K/V reorder copy alongside compute. ----
    const int pidx = tid - 32;   // maps tid [160,511] -> contiguous [128,479]
    int load_phase[2] = {0, 0};

    for(int kc = 0; kc < nKVTiles; ++kc){
      const int slot = kc & 1;
      mbar_wait(lbar[slot], load_phase[slot]); load_phase[slot] ^= 1;
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      reorder_sync();

      for(int i = pidx; i < Bc_half * D; i += NREORDER){
        const int bc = i / D, d = i % D;
        sK[canon_idx(bc, d, Bc_half)] = sKstage[slot][i];
      }
      for(int i = pidx; i < Bc * D_half; i += NREORDER){
        const int bc = i / D_half, d = i % D_half;
        sV[canon_idx(d, bc, D_half)] = sVstage[slot][i];
      }
      reorder_sync();
      // Helpers take no further part this iteration — softmax/MMA/PV are compute-only.
    }
  } else {
    // ---- Compute group (tid < 128): same math as V23_causal; only the reorder-copy
    // sync scope widens (reorder_sync instead of consumer_sync) to include helpers.
    const int pidx = tid;   // maps tid [0,127] -> contiguous [0,127]
    const uint64_t descQ_base = make_smem_desc(sQ, Br);
    int mbar_phase = 0;
    int load_phase[2] = {0, 0};

    for(int kc = 0; kc < nKVTiles; ++kc){
      const int slot = kc & 1;
      mbar_wait(lbar[slot], load_phase[slot]); load_phase[slot] ^= 1;
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      reorder_sync();

      for(int i = pidx; i < Bc_half * D; i += NREORDER){
        const int bc = i / D, d = i % D;
        sK[canon_idx(bc, d, Bc_half)] = sKstage[slot][i];
      }
      for(int i = pidx; i < Bc * D_half; i += NREORDER){
        const int bc = i / D_half, d = i % D_half;
        sV[canon_idx(d, bc, D_half)] = sVstage[slot][i];
      }
      reorder_sync();
      if(tid == 0) mbar_arrive(fbar[slot]);

      {
        const uint32_t idesc = make_idesc_bf16(2 * Br, Bc);
        if(crank == 0 && tid == 0){
          const uint64_t descK_base = make_smem_desc(sK, Bc_half);
          asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
          for(int kt = 0; kt < D/16; ++kt){
            uint64_t descQ = advance_desc_katom(descQ_base, kt, Br);
            uint64_t descK = advance_desc_katom(descK_base, kt, Bc_half);
            uint32_t accumulate = (kt > 0) ? 1u : 0u;
            asm volatile(
              "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
              "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
              :: "r"(tmem_addr), "l"(descQ), "l"(descK), "r"(idesc), "r"(accumulate) : "memory");
          }
          mbar_commit_mma_2cta_multicast(mma_bar, 0b11);
        }
        mbar_wait(mma_bar, mbar_phase); mbar_phase ^= 1;
        tmem_readout_to_smem_vec_2cta(sS, tmem_addr, Br, Bc, Bc, scale_l2e, rank_warp_offset);
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");

        if(kc > q_tile){
          for(int j = 0; j < Bc; ++j) sS[tid * Bc + j] = -INFINITY;
        } else if(kc == q_tile){
          for(int j = tid + 1; j < Bc; ++j) sS[tid * Bc + j] = -INFINITY;
        }
        consumer_sync();
      }

      {
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
      consumer_sync();

      for(int i = tid; i < Br * D; i += Br) sO[i] *= sCorr[i / D];
      consumer_sync();

      {
        const uint64_t descP_base = make_smem_desc(sP, Br);
        const uint32_t idesc      = make_idesc_bf16(2 * Br, D);
        if(crank == 0 && tid == 0){
          const uint64_t descV_base = make_smem_desc(sV, D_half);
          asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
          for(int kt = 0; kt < Bc/16; ++kt){
            uint64_t descP = advance_desc_katom(descP_base, kt, Br);
            uint64_t descV = advance_desc_katom(descV_base, kt, D_half);
            uint32_t accumulate = (kt > 0) ? 1u : 0u;
            asm volatile(
              "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
              "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
              :: "r"(tmem_addr), "l"(descP), "l"(descV), "r"(idesc), "r"(accumulate) : "memory");
          }
          mbar_commit_mma_2cta_multicast(mma_bar, 0b11);
        }
        mbar_wait(mma_bar, mbar_phase); mbar_phase ^= 1;
        tmem_readout_accum_vec_2cta(sO, tmem_addr, Br, D, D, rank_warp_offset);
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
        consumer_sync();
      }
    } // end kv loop
  }

  __syncthreads();   // full-block reconvergence: producer, helper, and compute all done
  cluster.sync();
  if(tid < 32)
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
                 :: "r"(tmem_addr), "r"(NCOLS) : "memory");
  __syncthreads();

  for(int i = 2 * tid; i < Br * D; i += 2 * blockDim.x){
    const float denom = sl[i / D];
    *reinterpret_cast<__nv_bfloat162*>(&d_O[qBase + i]) =
        __floats2bfloat162_rn(sO[i] / denom, sO[i + 1] / denom);
  }
  if(tid < Br)
    d_LSE[lBase + tid] = 0.6931471805599453f * (sm[tid] + log2f(sl[tid]));

} // end of gqa_v25_causal

// gqa_v26_causal — the real FA3-style ping-pong: this CTA's own 128 query rows are
// split into two INDEPENDENT 64-row halves (half-A = rows 0-63, half-B = rows
// 64-127), each with its own complete pipeline (own canonical Q, own reordered K/V,
// own TMEM region, own mbarriers, own softmax state) driven by a genuinely separate
// set of 128 threads (4 warps). No explicit stagger between the halves is coded —
// since they are physically independent warps, the SM's own warp scheduler can
// dispatch half-B's instructions while half-A is stalled on its mbar_wait (and vice
// versa), which is the real mechanism for overlap; V9's earlier same-warp
// instruction-reordering attempt COULDN'T get this because it was the same physical
// warps serialized through one instruction stream regardless of code order.
//
// Two things V25's race taught us, addressed here from the start:
//   1. K/V canonical buffers (sK/sV) must be duplicated per half (sK_A/sK_B,
//      sV_A/sV_B) — NOT shared — since each half's MMA consumption pace is now
//      independent, and a shared destination buffer would let one half overwrite
//      data the other half's tensor-core MMA is still asynchronously reading.
//   2. The shared K/V STAGING buffer's "slot is free" signal must be TWO independent
//      single-count mbarriers (fbarA[slot], fbarB[slot]), not one count=2 mbarrier.
//      A raw arrival counter can't distinguish "both halves confirmed" from "one
//      half confirmed twice" — since the two halves' paces can drift arbitrarily
//      (that's the whole point of decoupling them), one half could lap the other by
//      a full 2-slot cycle and satisfy a count=2 threshold alone, letting the
//      producer overwrite a slot the OTHER half hasn't even read yet. Two
//      independent per-half phase counters make each side's confirmation immune to
//      the other's drift. This only couples the (cheap) reorder step across halves,
//      not their (expensive) MMA/softmax pace — that's what keeps the two halves'
//      compute genuinely free to interleave.
// Also: the existing 2cta readout functions derive warp_id from raw threadIdx.x,
// which only yields 0-3 when a group's threads start at global warp 0. Half-B lives
// at tid 128-255 (global warps 4-7), so it uses the generalized _g readout variants
// with an explicit LOCAL warp_id instead. And because rows_per_warp = Br_half/4 = 16
// here (not 32, since Br_half=64 not 128), "row == tid" is no longer a free
// coincidence — only tid where (tid%32) < 16 hold a valid row (row_local =
// warp_id_local*16 + lane), so the softmax/mask/PV block gates on that explicitly
// instead of using tid directly (the reorder-copy and epilogue steps are generic
// full-range strides and need no such gating).
//
// TMEM: ONE combined tcgen05.alloc for 2*NCOLS columns; half-B addressed via
// tmem_addr_A + NCOLS. (A "two independent allocs" variant was tried in between —
// it hard-crashed, because a second tcgen05.alloc while the first is still live is
// documented as illegal; reverted.) The address-offset scheme itself is valid: the
// real TMEM address format is `taddr + (row<<16) + col` (confirmed against public
// tcgen05 documentation), so adding NCOLS — well under the 16-bit column field —
// correctly lands in the next disjoint column-block. The actual corruption bug
// that the offset was originally (wrongly) blamed for was RANK_WARP_SPAN_HALF
// (see above). Still DOUBLES total TMEM column usage vs V20-23 (a scarce per-SM
// resource) — that capacity question remains to be confirmed on hardware.
//
// KNOWN-FAILING (root cause since identified): the readouts below assume the
// per-CTA M=64 accumulator spreads 16 rows per sub-partition (lanes 32w..32w+15) —
// it does NOT; the M=64 D layout is column-halved across the full lane space. See
// the tmem_readout_*_m64 helpers' header and gqa_tmem_probe. The corrected
// architecture lives in gqa_v27_causal; V26 is kept unmodified as the historical
// rung (like V20-causal), not worth re-verifying separately.
template<int Br, int Bc, int D>
__global__ void __cluster_dims__(2, 1, 1) gqa_v26_causal(
  __nv_bfloat16 *d_Q,
  __nv_bfloat16 *d_O,
  float *d_LSE,
  const __grid_constant__ CUtensorMap Ktmap_half,
  const __grid_constant__ CUtensorMap Vtmap_half,
  int B,
  int Hq,
  int Hkv,
  int G,
  int S,
  float scale
){
  static_assert(Bc % 2 == 0, "Bc must be even to split the key range in half");
  static_assert(D  % 2 == 0, "D must be even to split the head dim in half");
  static_assert(Br == 128, "V26_causal's per-half compute group is hardwired to 4 warps");
  static_assert(Br == Bc, "causal tile-skip + diagonal-tile mask requires Br == Bc");
  constexpr int Bc_half   = Bc / 2;        // N-half split (key-range/head-dim), unrelated to the M-half split below
  constexpr int D_half    = D / 2;
  constexpr int Br_half   = Br / 2;        // M-half split: this CTA's own row count PER half (64)
  constexpr int ROWS_PER_WARP_HALF = Br_half / 4;   // 16 — TMEM always has 4 sub-partitions
  // NOT Br_half/32. This is the cta_group::2 cross-CTA TMEM routing threshold: rows
  // 0-127 (4 warp-units of 32) always address THIS CTA's own physical TMEM bank, rows
  // >=128 always route to the peer's bank — a FIXED hardware convention, independent
  // of the current MMA's M. It only looked like "Br/32" in V20-25 because Br was
  // always 128 there; deriving it from Br_half (64) here gave offset 64 instead of
  // 128, landing crank=1's reads/writes on the wrong physical TMEM location entirely
  // (the root cause of V26's first, corruption failure).
  constexpr int RANK_WARP_SPAN_HALF = 4;

  const int q_tile = blockIdx.x;
  const int hq     = blockIdx.y;
  const int b      = blockIdx.z;
  const int hkv    = hq / G;
  const int tid    = threadIdx.x;   // 0..287: 0..127 half-A, 128..255 half-B, 256..287 producer

  cg::cluster_group cluster = cg::this_cluster();
  const unsigned int crank  = cluster.block_rank();

  const int q_row0   = q_tile * Br;
  const int nKVTiles = (q_tile | 1) + 1;   // CAUSAL: cover the pair's higher/odd q_tile
  const int kvRow0   = (b * Hkv + hkv) * S;

  const long qBaseA = ((long)(b * Hq + hq) * S + q_row0) * D;
  const long qBaseB = qBaseA + (long)Br_half * D;
  const long lBaseA = (long)(b * Hq + hq) * S + q_row0;
  const long lBaseB = lBaseA + Br_half;

  const float scale_l2e = scale * 1.4426950408889634f;

  __shared__ __align__(16)  __nv_bfloat16 sQ_A[Br_half * D];
  __shared__ __align__(16)  __nv_bfloat16 sQ_B[Br_half * D];
  __shared__ __align__(128) __nv_bfloat16 sKstage[2][Bc_half * D];      // shared staging (unchanged from V23)
  __shared__ __align__(128) __nv_bfloat16 sVstage[2][Bc * D_half];
  __shared__ __align__(16)  __nv_bfloat16 sK_A[Bc_half * D];            // per-half canonical copies
  __shared__ __align__(16)  __nv_bfloat16 sK_B[Bc_half * D];
  __shared__ __align__(16)  __nv_bfloat16 sV_A[D_half * Bc];
  __shared__ __align__(16)  __nv_bfloat16 sV_B[D_half * Bc];
  __shared__ __align__(16)  __nv_bfloat16 sP_A[Br_half * Bc];
  __shared__ __align__(16)  __nv_bfloat16 sP_B[Br_half * Bc];
  __shared__ __align__(16)  float         sS_A[Br_half * Bc];
  __shared__ __align__(16)  float         sS_B[Br_half * Bc];
  __shared__ __align__(16)  float         sO_A[Br_half * D];
  __shared__ __align__(16)  float         sO_B[Br_half * D];
  __shared__ float sm_A[Br_half], sl_A[Br_half], sCorr_A[Br_half];
  __shared__ float sm_B[Br_half], sl_B[Br_half], sCorr_B[Br_half];
  __shared__ __align__(8) uint64_t s_mma_bar_A;
  __shared__ __align__(8) uint64_t s_mma_bar_B;
  __shared__ __align__(8) uint64_t s_load_bar[2];
  // Two INDEPENDENT free-signal barriers per slot (not one count=2 barrier): each
  // half's own reorder pace can drift arbitrarily from the other's, and a shared
  // counter can't tell "both halves confirmed slot X" from "half-A alone confirmed
  // slot X twice" — only per-half phase tracking can. The producer waits on BOTH
  // independently before reusing a slot.
  __shared__ __align__(8) uint64_t s_free_bar_A[2];
  __shared__ __align__(8) uint64_t s_free_bar_B[2];

  for(int i = tid; i < Br_half * D && tid < 128; i += 128){
    const int r = i / D, c = i % D;
    sQ_A[canon_idx(r, c, Br_half)] = d_Q[qBaseA + i];
    sO_A[i] = 0.0f;
  }
  for(int i = tid - 128; i < Br_half * D && tid >= 128 && tid < 256; i += 128){
    const int r = i / D, c = i % D;
    sQ_B[canon_idx(r, c, Br_half)] = d_Q[qBaseB + i];
    sO_B[i] = 0.0f;
  }
  if(tid < Br_half){ sm_A[tid] = -INFINITY; sl_A[tid] = 0.0f; }
  if(tid >= 128 && tid < 128 + Br_half){ sm_B[tid-128] = -INFINITY; sl_B[tid-128] = 0.0f; }

  const uint32_t mma_bar_A = (uint32_t)__cvta_generic_to_shared(&s_mma_bar_A);
  const uint32_t mma_bar_B = (uint32_t)__cvta_generic_to_shared(&s_mma_bar_B);
  const uint32_t lbar0     = (uint32_t)__cvta_generic_to_shared(&s_load_bar[0]);
  const uint32_t lbar1     = (uint32_t)__cvta_generic_to_shared(&s_load_bar[1]);
  const uint32_t fbarA0    = (uint32_t)__cvta_generic_to_shared(&s_free_bar_A[0]);
  const uint32_t fbarA1    = (uint32_t)__cvta_generic_to_shared(&s_free_bar_A[1]);
  const uint32_t fbarB0    = (uint32_t)__cvta_generic_to_shared(&s_free_bar_B[0]);
  const uint32_t fbarB1    = (uint32_t)__cvta_generic_to_shared(&s_free_bar_B[1]);
  if(tid == 0){
    mbar_init(mma_bar_A, 1);
    mbar_init(mma_bar_B, 1);
    mbar_init(lbar0, 1); mbar_init(lbar1, 1);
    mbar_init(fbarA0, 1); mbar_init(fbarA1, 1);
    mbar_init(fbarB0, 1); mbar_init(fbarB1, 1);
    cluster_fence_mbarrier_init();
  }
  cluster.sync();   // CTA-collective — all 288 threads

  constexpr uint32_t NCOLS = (Bc > D) ? (uint32_t)Bc : (uint32_t)D;   // per-half column width
  static_assert(NCOLS >= 32 && (NCOLS & (NCOLS - 1)) == 0,
                "tcgen05 column count must be a power of two >= 32");

  // ONE combined tcgen05.alloc for 2*NCOLS columns, half-B addressed via a computed
  // column offset (tmem_addr_B = tmem_addr_A + NCOLS) — reverted from the earlier
  // "two independent allocs" attempt. That attempt was based on a wrong diagnosis:
  // the real TMEM address format is `taddr + (row<<16) + col` (confirmed against
  // public documentation), so adding NCOLS (well under the 16-bit column field)
  // to shift into the next column-block was always valid; the actual corruption
  // bug was RANK_WARP_SPAN_HALF (see above). Two independent allocs are furthermore
  // documented as requiring a dealloc between them — issuing a second alloc while
  // the first is still live is illegal, which is exactly what caused the hard
  // "unspecified launch failure" crash from that attempt.
  constexpr uint32_t NCOLS_TOTAL = NCOLS * 2;
  uint32_t tmem_addr_A;
  {
    __shared__ uint32_t s_tmem_addr;
    if(tid < 32){
      uint32_t s_addr = (uint32_t)__cvta_generic_to_shared(&s_tmem_addr);
      asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                   :: "r"(s_addr), "r"(NCOLS_TOTAL) : "memory");
      asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::: "memory");
    }
    __syncthreads();   // full 288-thread barrier to broadcast tmem_addr
    tmem_addr_A = s_tmem_addr;
  }
  const uint32_t tmem_addr_B = tmem_addr_A + NCOLS;

  const uint32_t TX = ((uint32_t)Bc_half * (uint32_t)D + (uint32_t)Bc * (uint32_t)D_half)
                    * (uint32_t)sizeof(__nv_bfloat16);
  const uint32_t sKstage_addr[2] = {
    (uint32_t)__cvta_generic_to_shared(sKstage[0]),
    (uint32_t)__cvta_generic_to_shared(sKstage[1])
  };
  const uint32_t sVstage_addr[2] = {
    (uint32_t)__cvta_generic_to_shared(sVstage[0]),
    (uint32_t)__cvta_generic_to_shared(sVstage[1])
  };
  const uint32_t lbar[2]  = {lbar0, lbar1};
  const uint32_t fbarA[2] = {fbarA0, fbarA1};
  const uint32_t fbarB[2] = {fbarB0, fbarB1};
  const uint32_t rank_warp_offset_half = crank * RANK_WARP_SPAN_HALF;

  if(tid >= 256){
    // ---- Producer: shared K/V staging, same double-buffered pattern as V23, but
    // now waits on BOTH halves' independent free-signals before reusing a slot. ----
    if(tid == 256){
      int free_phaseA[2] = {0, 0};
      int free_phaseB[2] = {0, 0};
      for(int kc = 0; kc < nKVTiles; ++kc){
        const int slot = kc & 1;
        if(kc >= 2){
          mbar_wait(fbarA[slot], free_phaseA[slot]); free_phaseA[slot] ^= 1;
          mbar_wait(fbarB[slot], free_phaseB[slot]); free_phaseB[slot] ^= 1;
        }
        const int kRow = kvRow0 + kc * Bc + (int)crank * Bc_half;
        const int vCol = (int)crank * D_half;
        mbar_expect_tx(lbar[slot], TX);
        tma_load_2d(sKstage_addr[slot], &Ktmap_half, 0, kRow, lbar[slot]);
        tma_load_2d(sVstage_addr[slot], &Vtmap_half, vCol, kvRow0 + kc * Bc, lbar[slot]);
        mbar_arrive(lbar[slot]);
      }
    }
  } else if(tid >= 128){
    // ---- Half-B: rows 64-127 of this CTA's q_tile. ----
    const int ltid          = tid - 128;             // 0..127, local to half-B
    const int warp_id_local = ltid / 32;              // 0..3
    const int lane          = ltid % 32;
    const int row_local     = warp_id_local * ROWS_PER_WARP_HALF + lane;   // valid (0..63) iff lane < 16
    const uint64_t descQ_base = make_smem_desc(sQ_B, Br_half);
    int mbar_phase = 0;
    int load_phase[2] = {0, 0};

    for(int kc = 0; kc < nKVTiles; ++kc){
      const int slot = kc & 1;
      mbar_wait(lbar[slot], load_phase[slot]); load_phase[slot] ^= 1;
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      sync_half_b();

      for(int i = ltid; i < Bc_half * D; i += 128){
        const int bc = i / D, d = i % D;
        sK_B[canon_idx(bc, d, Bc_half)] = sKstage[slot][i];
      }
      for(int i = ltid; i < Bc * D_half; i += 128){
        const int bc = i / D_half, d = i % D_half;
        sV_B[canon_idx(d, bc, D_half)] = sVstage[slot][i];
      }
      sync_half_b();
      if(ltid == 0) mbar_arrive(fbarB[slot]);   // half-B's own independent free-signal

      {
        const uint32_t idesc = make_idesc_bf16(2 * Br_half, Bc);
        if(crank == 0 && ltid == 0){
          const uint64_t descK_base = make_smem_desc(sK_B, Bc_half);
          asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
          for(int kt = 0; kt < D/16; ++kt){
            uint64_t descQ = advance_desc_katom(descQ_base, kt, Br_half);
            uint64_t descK = advance_desc_katom(descK_base, kt, Bc_half);
            uint32_t accumulate = (kt > 0) ? 1u : 0u;
            asm volatile(
              "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
              "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
              :: "r"(tmem_addr_B), "l"(descQ), "l"(descK), "r"(idesc), "r"(accumulate) : "memory");
          }
          mbar_commit_mma_2cta_multicast(mma_bar_B, 0b11);
        }
        mbar_wait(mma_bar_B, mbar_phase); mbar_phase ^= 1;
        tmem_readout_to_smem_vec_2cta_g(sS_B, tmem_addr_B, Br_half, Bc, Bc, scale_l2e,
                                         rank_warp_offset_half, warp_id_local, lane);
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");

        // CAUSAL mask: half-B's local row maps to tile-position row_local+64.
        if(lane < ROWS_PER_WARP_HALF){
          if(kc > q_tile){
            for(int j = 0; j < Bc; ++j) sS_B[row_local * Bc + j] = -INFINITY;
          } else if(kc == q_tile){
            for(int j = row_local + 64 + 1; j < Bc; ++j) sS_B[row_local * Bc + j] = -INFINITY;
          }
        }
        sync_half_b();
      }

      if(lane < ROWS_PER_WARP_HALF){
        const float m_old = sm_B[row_local];
        const float l_old = sl_B[row_local];

        float tile_max = -INFINITY;
        int j = 0;
        for(; j + 2 < Bc; j += 3)
          tile_max = fmaxf(tile_max,
                           fmaxf(sS_B[row_local * Bc + j],
                                 fmaxf(sS_B[row_local * Bc + j + 1], sS_B[row_local * Bc + j + 2])));
        for(; j < Bc; ++j) tile_max = fmaxf(tile_max, sS_B[row_local * Bc + j]);

        const float m_new = fmaxf(m_old, tile_max);
        const float corr  = ex2_approx(m_old - m_new);

        float p_sum = 0.0f;
        for(int j2 = 0; j2 < Bc; j2 += 2){
          const float p0 = ex2_approx(sS_B[row_local * Bc + j2]     - m_new);
          const float p1 = ex2_approx(sS_B[row_local * Bc + j2 + 1] - m_new);
          *reinterpret_cast<__nv_bfloat162*>(&sP_B[canon_idx(row_local, j2, Br_half)]) =
              __floats2bfloat162_rn(p0, p1);
          p_sum += p0 + p1;
        }
        sm_B[row_local] = m_new; sl_B[row_local] = l_old * corr + p_sum; sCorr_B[row_local] = corr;
      }
      sync_half_b();

      for(int i = ltid; i < Br_half * D; i += 128) sO_B[i] *= sCorr_B[i / D];
      sync_half_b();

      {
        const uint64_t descP_base = make_smem_desc(sP_B, Br_half);
        const uint32_t idesc      = make_idesc_bf16(2 * Br_half, D);
        if(crank == 0 && ltid == 0){
          const uint64_t descV_base = make_smem_desc(sV_B, D_half);
          asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
          for(int kt = 0; kt < Bc/16; ++kt){
            uint64_t descP = advance_desc_katom(descP_base, kt, Br_half);
            uint64_t descV = advance_desc_katom(descV_base, kt, D_half);
            uint32_t accumulate = (kt > 0) ? 1u : 0u;
            asm volatile(
              "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
              "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
              :: "r"(tmem_addr_B), "l"(descP), "l"(descV), "r"(idesc), "r"(accumulate) : "memory");
          }
          mbar_commit_mma_2cta_multicast(mma_bar_B, 0b11);
        }
        mbar_wait(mma_bar_B, mbar_phase); mbar_phase ^= 1;
        tmem_readout_accum_vec_2cta_g(sO_B, tmem_addr_B, Br_half, D, D,
                                       rank_warp_offset_half, warp_id_local, lane);
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
        sync_half_b();
      }
    } // end kv loop (half-B)
  } else {
    // ---- Half-A: rows 0-63 of this CTA's q_tile. ----
    const int warp_id_local = tid / 32;               // 0..3
    const int lane          = tid % 32;
    const int row_local     = warp_id_local * ROWS_PER_WARP_HALF + lane;   // valid (0..63) iff lane < 16
    const uint64_t descQ_base = make_smem_desc(sQ_A, Br_half);
    int mbar_phase = 0;
    int load_phase[2] = {0, 0};

    for(int kc = 0; kc < nKVTiles; ++kc){
      const int slot = kc & 1;
      mbar_wait(lbar[slot], load_phase[slot]); load_phase[slot] ^= 1;
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      consumer_sync();

      for(int i = tid; i < Bc_half * D; i += 128){
        const int bc = i / D, d = i % D;
        sK_A[canon_idx(bc, d, Bc_half)] = sKstage[slot][i];
      }
      for(int i = tid; i < Bc * D_half; i += 128){
        const int bc = i / D_half, d = i % D_half;
        sV_A[canon_idx(d, bc, D_half)] = sVstage[slot][i];
      }
      consumer_sync();
      if(tid == 0) mbar_arrive(fbarA[slot]);   // half-A's own independent free-signal

      {
        const uint32_t idesc = make_idesc_bf16(2 * Br_half, Bc);
        if(crank == 0 && tid == 0){
          const uint64_t descK_base = make_smem_desc(sK_A, Bc_half);
          asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
          for(int kt = 0; kt < D/16; ++kt){
            uint64_t descQ = advance_desc_katom(descQ_base, kt, Br_half);
            uint64_t descK = advance_desc_katom(descK_base, kt, Bc_half);
            uint32_t accumulate = (kt > 0) ? 1u : 0u;
            asm volatile(
              "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
              "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
              :: "r"(tmem_addr_A), "l"(descQ), "l"(descK), "r"(idesc), "r"(accumulate) : "memory");
          }
          mbar_commit_mma_2cta_multicast(mma_bar_A, 0b11);
        }
        mbar_wait(mma_bar_A, mbar_phase); mbar_phase ^= 1;
        tmem_readout_to_smem_vec_2cta_g(sS_A, tmem_addr_A, Br_half, Bc, Bc, scale_l2e,
                                         rank_warp_offset_half, warp_id_local, lane);
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");

        // CAUSAL mask: half-A's local row IS the tile-position directly (rows 0-63).
        if(lane < ROWS_PER_WARP_HALF){
          if(kc > q_tile){
            for(int j = 0; j < Bc; ++j) sS_A[row_local * Bc + j] = -INFINITY;
          } else if(kc == q_tile){
            for(int j = row_local + 1; j < Bc; ++j) sS_A[row_local * Bc + j] = -INFINITY;
          }
        }
        consumer_sync();
      }

      if(lane < ROWS_PER_WARP_HALF){
        const float m_old = sm_A[row_local];
        const float l_old = sl_A[row_local];

        float tile_max = -INFINITY;
        int j = 0;
        for(; j + 2 < Bc; j += 3)
          tile_max = fmaxf(tile_max,
                           fmaxf(sS_A[row_local * Bc + j],
                                 fmaxf(sS_A[row_local * Bc + j + 1], sS_A[row_local * Bc + j + 2])));
        for(; j < Bc; ++j) tile_max = fmaxf(tile_max, sS_A[row_local * Bc + j]);

        const float m_new = fmaxf(m_old, tile_max);
        const float corr  = ex2_approx(m_old - m_new);

        float p_sum = 0.0f;
        for(int j2 = 0; j2 < Bc; j2 += 2){
          const float p0 = ex2_approx(sS_A[row_local * Bc + j2]     - m_new);
          const float p1 = ex2_approx(sS_A[row_local * Bc + j2 + 1] - m_new);
          *reinterpret_cast<__nv_bfloat162*>(&sP_A[canon_idx(row_local, j2, Br_half)]) =
              __floats2bfloat162_rn(p0, p1);
          p_sum += p0 + p1;
        }
        sm_A[row_local] = m_new; sl_A[row_local] = l_old * corr + p_sum; sCorr_A[row_local] = corr;
      }
      consumer_sync();

      for(int i = tid; i < Br_half * D; i += 128) sO_A[i] *= sCorr_A[i / D];
      consumer_sync();

      {
        const uint64_t descP_base = make_smem_desc(sP_A, Br_half);
        const uint32_t idesc      = make_idesc_bf16(2 * Br_half, D);
        if(crank == 0 && tid == 0){
          const uint64_t descV_base = make_smem_desc(sV_A, D_half);
          asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
          for(int kt = 0; kt < Bc/16; ++kt){
            uint64_t descP = advance_desc_katom(descP_base, kt, Br_half);
            uint64_t descV = advance_desc_katom(descV_base, kt, D_half);
            uint32_t accumulate = (kt > 0) ? 1u : 0u;
            asm volatile(
              "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
              "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
              :: "r"(tmem_addr_A), "l"(descP), "l"(descV), "r"(idesc), "r"(accumulate) : "memory");
          }
          mbar_commit_mma_2cta_multicast(mma_bar_A, 0b11);
        }
        mbar_wait(mma_bar_A, mbar_phase); mbar_phase ^= 1;
        tmem_readout_accum_vec_2cta_g(sO_A, tmem_addr_A, Br_half, D, D,
                                       rank_warp_offset_half, warp_id_local, lane);
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
        consumer_sync();
      }
    } // end kv loop (half-A)
  }

  __syncthreads();   // full-block reconvergence: producer + both halves all done
  cluster.sync();
  if(tid < 32)
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
                 :: "r"(tmem_addr_A), "r"(NCOLS_TOTAL) : "memory");
  __syncthreads();

  for(int i = 2 * tid; i < Br_half * D && tid < 128; i += 256){
    const float denom = sl_A[i / D];
    *reinterpret_cast<__nv_bfloat162*>(&d_O[qBaseA + i]) =
        __floats2bfloat162_rn(sO_A[i] / denom, sO_A[i + 1] / denom);
  }
  for(int i = 2 * (tid - 128); i < Br_half * D && tid >= 128 && tid < 256; i += 256){
    const float denom = sl_B[i / D];
    *reinterpret_cast<__nv_bfloat162*>(&d_O[qBaseB + i]) =
        __floats2bfloat162_rn(sO_B[i] / denom, sO_B[i + 1] / denom);
  }
  if(tid < Br_half)
    d_LSE[lBaseA + tid] = 0.6931471805599453f * (sm_A[tid] + log2f(sl_A[tid]));
  if(tid >= 128 && tid < 128 + Br_half)
    d_LSE[lBaseB + (tid - 128)] = 0.6931471805599453f * (sm_B[tid-128] + log2f(sl_B[tid-128]));

} // end of gqa_v26_causal

// gqa_v26_diag_causal — DIAGNOSTIC ONLY: byte-for-byte identical to gqa_v26_causal
// except half-B's entire kv-loop body is gutted to a no-op (it still issues its own
// mbar_wait/reorder/fbarB-arrive so the shared producer never deadlocks waiting on
// fbarB, but performs NO tcgen05.mma, NO TMEM readout, and NO softmax — sO_B/sm_B/sl_B
// stay at their initialized zero/-inf/0 values). Purpose: isolate whether V26's
// persistent D/2-column corruption in half-A's OWN rows (crank=0's own data, not a
// cross-CTA peer-routing case) comes from two independent cta_group::2 MMA/commit
// pipelines genuinely coexisting in one CTA-pair (an undocumented hardware
// interaction), or from a bug in the base per-half pipeline that's unrelated to
// half-B's presence. Half-B's own output rows are expected to be garbage/zero here
// by design — only half-A's rows are meaningful for this test.
template<int Br, int Bc, int D>
__global__ void __cluster_dims__(2, 1, 1) gqa_v26_diag_causal(
  __nv_bfloat16 *d_Q,
  __nv_bfloat16 *d_O,
  float *d_LSE,
  const __grid_constant__ CUtensorMap Ktmap_half,
  const __grid_constant__ CUtensorMap Vtmap_half,
  int B,
  int Hq,
  int Hkv,
  int G,
  int S,
  float scale
){
  static_assert(Bc % 2 == 0, "Bc must be even to split the key range in half");
  static_assert(D  % 2 == 0, "D must be even to split the head dim in half");
  static_assert(Br == 128, "V26_causal's per-half compute group is hardwired to 4 warps");
  static_assert(Br == Bc, "causal tile-skip + diagonal-tile mask requires Br == Bc");
  constexpr int Bc_half   = Bc / 2;
  constexpr int D_half    = D / 2;
  constexpr int Br_half   = Br / 2;
  constexpr int ROWS_PER_WARP_HALF = Br_half / 4;
  constexpr int RANK_WARP_SPAN_HALF = 4;

  const int q_tile = blockIdx.x;
  const int hq     = blockIdx.y;
  const int b      = blockIdx.z;
  const int hkv    = hq / G;
  const int tid    = threadIdx.x;

  cg::cluster_group cluster = cg::this_cluster();
  const unsigned int crank  = cluster.block_rank();

  const int q_row0   = q_tile * Br;
  const int nKVTiles = (q_tile | 1) + 1;
  const int kvRow0   = (b * Hkv + hkv) * S;

  const long qBaseA = ((long)(b * Hq + hq) * S + q_row0) * D;
  const long qBaseB = qBaseA + (long)Br_half * D;
  const long lBaseA = (long)(b * Hq + hq) * S + q_row0;
  const long lBaseB = lBaseA + Br_half;

  const float scale_l2e = scale * 1.4426950408889634f;

  __shared__ __align__(16)  __nv_bfloat16 sQ_A[Br_half * D];
  __shared__ __align__(16)  __nv_bfloat16 sQ_B[Br_half * D];
  __shared__ __align__(128) __nv_bfloat16 sKstage[2][Bc_half * D];
  __shared__ __align__(128) __nv_bfloat16 sVstage[2][Bc * D_half];
  __shared__ __align__(16)  __nv_bfloat16 sK_A[Bc_half * D];
  __shared__ __align__(16)  __nv_bfloat16 sK_B[Bc_half * D];
  __shared__ __align__(16)  __nv_bfloat16 sV_A[D_half * Bc];
  __shared__ __align__(16)  __nv_bfloat16 sV_B[D_half * Bc];
  __shared__ __align__(16)  __nv_bfloat16 sP_A[Br_half * Bc];
  __shared__ __align__(16)  __nv_bfloat16 sP_B[Br_half * Bc];
  __shared__ __align__(16)  float         sS_A[Br_half * Bc];
  __shared__ __align__(16)  float         sS_B[Br_half * Bc];
  __shared__ __align__(16)  float         sO_A[Br_half * D];
  __shared__ __align__(16)  float         sO_B[Br_half * D];
  __shared__ float sm_A[Br_half], sl_A[Br_half], sCorr_A[Br_half];
  __shared__ float sm_B[Br_half], sl_B[Br_half], sCorr_B[Br_half];
  __shared__ __align__(8) uint64_t s_mma_bar_A;
  __shared__ __align__(8) uint64_t s_mma_bar_B;
  __shared__ __align__(8) uint64_t s_load_bar[2];
  __shared__ __align__(8) uint64_t s_free_bar_A[2];
  __shared__ __align__(8) uint64_t s_free_bar_B[2];

  for(int i = tid; i < Br_half * D && tid < 128; i += 128){
    const int r = i / D, c = i % D;
    sQ_A[canon_idx(r, c, Br_half)] = d_Q[qBaseA + i];
    sO_A[i] = 0.0f;
  }
  for(int i = tid - 128; i < Br_half * D && tid >= 128 && tid < 256; i += 128){
    const int r = i / D, c = i % D;
    sQ_B[canon_idx(r, c, Br_half)] = d_Q[qBaseB + i];
    sO_B[i] = 0.0f;
  }
  if(tid < Br_half){ sm_A[tid] = -INFINITY; sl_A[tid] = 0.0f; }
  if(tid >= 128 && tid < 128 + Br_half){ sm_B[tid-128] = -INFINITY; sl_B[tid-128] = 0.0f; }

  const uint32_t mma_bar_A = (uint32_t)__cvta_generic_to_shared(&s_mma_bar_A);
  const uint32_t mma_bar_B = (uint32_t)__cvta_generic_to_shared(&s_mma_bar_B);
  const uint32_t lbar0     = (uint32_t)__cvta_generic_to_shared(&s_load_bar[0]);
  const uint32_t lbar1     = (uint32_t)__cvta_generic_to_shared(&s_load_bar[1]);
  const uint32_t fbarA0    = (uint32_t)__cvta_generic_to_shared(&s_free_bar_A[0]);
  const uint32_t fbarA1    = (uint32_t)__cvta_generic_to_shared(&s_free_bar_A[1]);
  const uint32_t fbarB0    = (uint32_t)__cvta_generic_to_shared(&s_free_bar_B[0]);
  const uint32_t fbarB1    = (uint32_t)__cvta_generic_to_shared(&s_free_bar_B[1]);
  if(tid == 0){
    mbar_init(mma_bar_A, 1);
    mbar_init(mma_bar_B, 1);
    mbar_init(lbar0, 1); mbar_init(lbar1, 1);
    mbar_init(fbarA0, 1); mbar_init(fbarA1, 1);
    mbar_init(fbarB0, 1); mbar_init(fbarB1, 1);
    cluster_fence_mbarrier_init();
  }
  cluster.sync();

  constexpr uint32_t NCOLS = (Bc > D) ? (uint32_t)Bc : (uint32_t)D;
  static_assert(NCOLS >= 32 && (NCOLS & (NCOLS - 1)) == 0,
                "tcgen05 column count must be a power of two >= 32");
  constexpr uint32_t NCOLS_TOTAL = NCOLS * 2;
  uint32_t tmem_addr_A;
  {
    __shared__ uint32_t s_tmem_addr;
    if(tid < 32){
      uint32_t s_addr = (uint32_t)__cvta_generic_to_shared(&s_tmem_addr);
      asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                   :: "r"(s_addr), "r"(NCOLS_TOTAL) : "memory");
      asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::: "memory");
    }
    __syncthreads();
    tmem_addr_A = s_tmem_addr;
  }
  const uint32_t tmem_addr_B = tmem_addr_A + NCOLS;

  const uint32_t TX = ((uint32_t)Bc_half * (uint32_t)D + (uint32_t)Bc * (uint32_t)D_half)
                    * (uint32_t)sizeof(__nv_bfloat16);
  const uint32_t sKstage_addr[2] = {
    (uint32_t)__cvta_generic_to_shared(sKstage[0]),
    (uint32_t)__cvta_generic_to_shared(sKstage[1])
  };
  const uint32_t sVstage_addr[2] = {
    (uint32_t)__cvta_generic_to_shared(sVstage[0]),
    (uint32_t)__cvta_generic_to_shared(sVstage[1])
  };
  const uint32_t lbar[2]  = {lbar0, lbar1};
  const uint32_t fbarA[2] = {fbarA0, fbarA1};
  const uint32_t fbarB[2] = {fbarB0, fbarB1};
  const uint32_t rank_warp_offset_half = crank * RANK_WARP_SPAN_HALF;

  if(tid >= 256){
    if(tid == 256){
      int free_phaseA[2] = {0, 0};
      int free_phaseB[2] = {0, 0};
      for(int kc = 0; kc < nKVTiles; ++kc){
        const int slot = kc & 1;
        if(kc >= 2){
          mbar_wait(fbarA[slot], free_phaseA[slot]); free_phaseA[slot] ^= 1;
          mbar_wait(fbarB[slot], free_phaseB[slot]); free_phaseB[slot] ^= 1;
        }
        const int kRow = kvRow0 + kc * Bc + (int)crank * Bc_half;
        const int vCol = (int)crank * D_half;
        mbar_expect_tx(lbar[slot], TX);
        tma_load_2d(sKstage_addr[slot], &Ktmap_half, 0, kRow, lbar[slot]);
        tma_load_2d(sVstage_addr[slot], &Vtmap_half, vCol, kvRow0 + kc * Bc, lbar[slot]);
        mbar_arrive(lbar[slot]);
      }
    }
  } else if(tid >= 128){
    // ---- Half-B: GUTTED for this diagnostic. Still drains lbar and arrives fbarB
    // each kc (so the shared producer never stalls waiting on half-B specifically),
    // but performs no tcgen05.mma, no readout, no softmax. sO_B/sm_B/sl_B are left at
    // their initialized values — half-B's own output rows are garbage by design.
    const int ltid = tid - 128;
    int load_phase[2] = {0, 0};

    for(int kc = 0; kc < nKVTiles; ++kc){
      const int slot = kc & 1;
      mbar_wait(lbar[slot], load_phase[slot]); load_phase[slot] ^= 1;
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      sync_half_b();
      sync_half_b();
      if(ltid == 0) mbar_arrive(fbarB[slot]);
      sync_half_b();
      sync_half_b();
      sync_half_b();
      sync_half_b();
    } // end kv loop (half-B, gutted)
  } else {
    // ---- Half-A: rows 0-63 of this CTA's q_tile. UNCHANGED from gqa_v26_causal. ----
    const int warp_id_local = tid / 32;
    const int lane          = tid % 32;
    const int row_local     = warp_id_local * ROWS_PER_WARP_HALF + lane;
    const uint64_t descQ_base = make_smem_desc(sQ_A, Br_half);
    int mbar_phase = 0;
    int load_phase[2] = {0, 0};

    for(int kc = 0; kc < nKVTiles; ++kc){
      const int slot = kc & 1;
      mbar_wait(lbar[slot], load_phase[slot]); load_phase[slot] ^= 1;
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      consumer_sync();

      for(int i = tid; i < Bc_half * D; i += 128){
        const int bc = i / D, d = i % D;
        sK_A[canon_idx(bc, d, Bc_half)] = sKstage[slot][i];
      }
      for(int i = tid; i < Bc * D_half; i += 128){
        const int bc = i / D_half, d = i % D_half;
        sV_A[canon_idx(d, bc, D_half)] = sVstage[slot][i];
      }
      consumer_sync();
      if(tid == 0) mbar_arrive(fbarA[slot]);

      {
        const uint32_t idesc = make_idesc_bf16(2 * Br_half, Bc);
        if(crank == 0 && tid == 0){
          const uint64_t descK_base = make_smem_desc(sK_A, Bc_half);
          asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
          for(int kt = 0; kt < D/16; ++kt){
            uint64_t descQ = advance_desc_katom(descQ_base, kt, Br_half);
            uint64_t descK = advance_desc_katom(descK_base, kt, Bc_half);
            uint32_t accumulate = (kt > 0) ? 1u : 0u;
            asm volatile(
              "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
              "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
              :: "r"(tmem_addr_A), "l"(descQ), "l"(descK), "r"(idesc), "r"(accumulate) : "memory");
          }
          mbar_commit_mma_2cta_multicast(mma_bar_A, 0b11);
        }
        mbar_wait(mma_bar_A, mbar_phase); mbar_phase ^= 1;
        tmem_readout_to_smem_vec_2cta_g(sS_A, tmem_addr_A, Br_half, Bc, Bc, scale_l2e,
                                         rank_warp_offset_half, warp_id_local, lane);
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");

        if(lane < ROWS_PER_WARP_HALF){
          if(kc > q_tile){
            for(int j = 0; j < Bc; ++j) sS_A[row_local * Bc + j] = -INFINITY;
          } else if(kc == q_tile){
            for(int j = row_local + 1; j < Bc; ++j) sS_A[row_local * Bc + j] = -INFINITY;
          }
        }
        consumer_sync();
      }

      if(lane < ROWS_PER_WARP_HALF){
        const float m_old = sm_A[row_local];
        const float l_old = sl_A[row_local];

        float tile_max = -INFINITY;
        int j = 0;
        for(; j + 2 < Bc; j += 3)
          tile_max = fmaxf(tile_max,
                           fmaxf(sS_A[row_local * Bc + j],
                                 fmaxf(sS_A[row_local * Bc + j + 1], sS_A[row_local * Bc + j + 2])));
        for(; j < Bc; ++j) tile_max = fmaxf(tile_max, sS_A[row_local * Bc + j]);

        const float m_new = fmaxf(m_old, tile_max);
        const float corr  = ex2_approx(m_old - m_new);

        float p_sum = 0.0f;
        for(int j2 = 0; j2 < Bc; j2 += 2){
          const float p0 = ex2_approx(sS_A[row_local * Bc + j2]     - m_new);
          const float p1 = ex2_approx(sS_A[row_local * Bc + j2 + 1] - m_new);
          *reinterpret_cast<__nv_bfloat162*>(&sP_A[canon_idx(row_local, j2, Br_half)]) =
              __floats2bfloat162_rn(p0, p1);
          p_sum += p0 + p1;
        }
        sm_A[row_local] = m_new; sl_A[row_local] = l_old * corr + p_sum; sCorr_A[row_local] = corr;
      }
      consumer_sync();

      for(int i = tid; i < Br_half * D; i += 128) sO_A[i] *= sCorr_A[i / D];
      consumer_sync();

      {
        const uint64_t descP_base = make_smem_desc(sP_A, Br_half);
        const uint32_t idesc      = make_idesc_bf16(2 * Br_half, D);
        if(crank == 0 && tid == 0){
          const uint64_t descV_base = make_smem_desc(sV_A, D_half);
          asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
          for(int kt = 0; kt < Bc/16; ++kt){
            uint64_t descP = advance_desc_katom(descP_base, kt, Br_half);
            uint64_t descV = advance_desc_katom(descV_base, kt, D_half);
            uint32_t accumulate = (kt > 0) ? 1u : 0u;
            asm volatile(
              "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
              "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
              :: "r"(tmem_addr_A), "l"(descP), "l"(descV), "r"(idesc), "r"(accumulate) : "memory");
          }
          mbar_commit_mma_2cta_multicast(mma_bar_A, 0b11);
        }
        mbar_wait(mma_bar_A, mbar_phase); mbar_phase ^= 1;
        tmem_readout_accum_vec_2cta_g(sO_A, tmem_addr_A, Br_half, D, D,
                                       rank_warp_offset_half, warp_id_local, lane);
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
        consumer_sync();
      }
    } // end kv loop (half-A)
  }

  __syncthreads();
  cluster.sync();
  if(tid < 32)
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
                 :: "r"(tmem_addr_A), "r"(NCOLS_TOTAL) : "memory");
  __syncthreads();

  for(int i = 2 * tid; i < Br_half * D && tid < 128; i += 256){
    const float denom = sl_A[i / D];
    *reinterpret_cast<__nv_bfloat162*>(&d_O[qBaseA + i]) =
        __floats2bfloat162_rn(sO_A[i] / denom, sO_A[i + 1] / denom);
  }
  for(int i = 2 * (tid - 128); i < Br_half * D && tid >= 128 && tid < 256; i += 256){
    const float denom = sl_B[i / D];
    *reinterpret_cast<__nv_bfloat162*>(&d_O[qBaseB + i]) =
        __floats2bfloat162_rn(sO_B[i] / denom, sO_B[i + 1] / denom);
  }
  if(tid < Br_half)
    d_LSE[lBaseA + tid] = 0.6931471805599453f * (sm_A[tid] + log2f(sl_A[tid]));
  if(tid >= 128 && tid < 128 + Br_half)
    d_LSE[lBaseB + (tid - 128)] = 0.6931471805599453f * (sm_B[tid-128] + log2f(sl_B[tid-128]));

} // end of gqa_v26_diag_causal

// gqa_v27_causal — FA4/Twill Blackwell architecture (arXiv:2512.18134 §6.2.2,
// confirmed against the actual paper text): 2 independent "softmax" warpgroups
// (softmax-A/softmax-B, exactly V26's half-A/half-B minus their P@V step: own QK^T
// MMA + readout + causal mask + online-softmax) plus ONE shared "rescale" warpgroup
// that owns P@V MMA + TMEM readout + O-rescale for BOTH sub-tiles. Quote: "softmax
// calculations for each sub-tile onto two different groups of warps..., and
// accumulator rescaling operations for both sub-tiles onto a third group of warps...
// The rescaling is moved to a third group of warps because reading accumulators from
// Tensor Memory requires blocking synchronization; placing it on the warp issuing
// exponentials disrupts the pipeline."
//
// This is the fix for the exact failure mode V9 hit (same-warp GEMM/softmax
// reordering regressed): Figure 2 in the paper shows a blocking wait on one op
// interrupts instruction ISSUE for an independent op on the SAME warp regardless of
// source order, since warps issue in-order. Only genuinely separate physical warps
// let the SM scheduler interleave around the stall — which V26's half-A/half-B
// already demonstrated (no explicit stagger needed, the scheduler does it). V27
// applies that same principle one level deeper: the P@V TMEM read (the actual
// blocking op) moves off the softmax warps entirely, onto its own warpgroup.
//
// TMEM: deliberately UNCHANGED from V26 — each half still uses ONE combined region
// shared between QK^T-scores and the P@V accumulator (NCOLS_TOTAL=256), NOT growing
// TMEM footprint further while V26's already-larger-than-V20-23 footprint remains
// unconfirmed on hardware. Consequence: the region is now reused ACROSS warpgroups
// instead of within one, so two new mbarrier pairs per half replace V26's single
// per-half mma_bar:
//   pready_bar_X   : softmax-X -> rescale. Arrives once per kc, after softmax-X's
//                    QK^T readout AND full online-softmax stats are both written
//                    (sP_X/sCorr_X ready, region X already drained of QK^T's data) —
//                    one wait covers both since they're strictly sequential in
//                    softmax-X's own per-thread program order.
//   tmemfree_bar_X : rescale -> softmax-X. Arrives once per kc, after the rescale
//                    group's P@V MMA + readout have fully drained region X —
//                    softmax-X's NEXT kc waits on this before reusing region X for
//                    its own QK^T MMA (the WAR hazard is now cross-warpgroup instead
//                    of intra-warpgroup, but still needs an explicit wait). Skipped
//                    for kc==0 (nothing pending yet).
//   mma_bar_X_qk / mma_bar_X_pv : replace V26's single per-half mma_bar — QK^T and
//                    P@V are now issued+awaited by DIFFERENT warpgroups, so a single
//                    shared mbarrier can no longer tell which side is waiting.
// This sacrifices some cross-iteration overlap (softmax-X's next QK^T still can't
// start until rescale drains region X) in exchange for not re-risking TMEM capacity
// beyond V26. Once hardware-verified correct, an NCU-guided follow-up can decide
// whether double-buffering the P@V region specifically (removing that remaining
// wait) is worth the extra TMEM.
//
// 416 threads/CTA: 128 softmax-A (tid 0-127) + 128 softmax-B (tid 128-255) + 128
// shared rescale (tid 256-383, handles half-A then half-B sequentially each kc) + 32
// producer (tid 384-415, shared K/V TMA, unchanged from V26).
template<int Br, int Bc, int D>
__global__ void __cluster_dims__(2, 1, 1) gqa_v27_causal(
  __nv_bfloat16 *d_Q,
  __nv_bfloat16 *d_O,
  float *d_LSE,
  const __grid_constant__ CUtensorMap Ktmap_half,
  const __grid_constant__ CUtensorMap Vtmap_half,
  int B,
  int Hq,
  int Hkv,
  int G,
  int S,
  float scale
){
  static_assert(Bc % 2 == 0, "Bc must be even to split the key range in half");
  static_assert(D  % 2 == 0, "D must be even to split the head dim in half");
  static_assert(Br == 128, "V27_causal's per-half compute group is hardwired to 4 warps");
  static_assert(Br == Bc, "causal tile-skip + diagonal-tile mask requires Br == Bc");
  constexpr int Bc_half   = Bc / 2;
  constexpr int D_half    = D / 2;
  constexpr int Br_half   = Br / 2;

  const int q_tile = blockIdx.x;
  const int hq     = blockIdx.y;
  const int b      = blockIdx.z;
  const int hkv    = hq / G;
  const int tid    = threadIdx.x;   // 0..415: 0..127 softmax-A, 128..255 softmax-B, 256..383 rescale, 384..415 producer

  cg::cluster_group cluster = cg::this_cluster();
  const unsigned int crank  = cluster.block_rank();

  const int q_row0   = q_tile * Br;
  const int nKVTiles = (q_tile | 1) + 1;   // CAUSAL: cover the pair's higher/odd q_tile
  const int kvRow0   = (b * Hkv + hkv) * S;

  const long qBaseA = ((long)(b * Hq + hq) * S + q_row0) * D;
  const long qBaseB = qBaseA + (long)Br_half * D;
  const long lBaseA = (long)(b * Hq + hq) * S + q_row0;
  const long lBaseB = lBaseA + Br_half;

  const float scale_l2e = scale * 1.4426950408889634f;

  __shared__ __align__(16)  __nv_bfloat16 sQ_A[Br_half * D];
  __shared__ __align__(16)  __nv_bfloat16 sQ_B[Br_half * D];
  __shared__ __align__(128) __nv_bfloat16 sKstage[2][Bc_half * D];
  __shared__ __align__(128) __nv_bfloat16 sVstage[2][Bc * D_half];
  __shared__ __align__(16)  __nv_bfloat16 sK_A[Bc_half * D];
  __shared__ __align__(16)  __nv_bfloat16 sK_B[Bc_half * D];
  __shared__ __align__(16)  __nv_bfloat16 sV_A[D_half * Bc];
  __shared__ __align__(16)  __nv_bfloat16 sV_B[D_half * Bc];
  __shared__ __align__(16)  __nv_bfloat16 sP_A[Br_half * Bc];
  __shared__ __align__(16)  __nv_bfloat16 sP_B[Br_half * Bc];
  __shared__ __align__(16)  float         sS_A[Br_half * Bc];
  __shared__ __align__(16)  float         sS_B[Br_half * Bc];
  __shared__ __align__(16)  float         sO_A[Br_half * D];
  __shared__ __align__(16)  float         sO_B[Br_half * D];
  __shared__ float sm_A[Br_half], sl_A[Br_half], sCorr_A[Br_half];
  __shared__ float sm_B[Br_half], sl_B[Br_half], sCorr_B[Br_half];
  __shared__ __align__(8) uint64_t s_mma_bar_A_qk;
  __shared__ __align__(8) uint64_t s_mma_bar_A_pv;
  __shared__ __align__(8) uint64_t s_mma_bar_B_qk;
  __shared__ __align__(8) uint64_t s_mma_bar_B_pv;
  __shared__ __align__(8) uint64_t s_load_bar[2];
  __shared__ __align__(8) uint64_t s_free_bar_A[2];
  __shared__ __align__(8) uint64_t s_free_bar_B[2];
  __shared__ __align__(8) uint64_t s_pready_bar_A;
  __shared__ __align__(8) uint64_t s_pready_bar_B;
  __shared__ __align__(8) uint64_t s_tmemfree_bar_A;
  __shared__ __align__(8) uint64_t s_tmemfree_bar_B;
  // Cross-CTA operand-ready signals: rank 1 remote-arrives rank 0's copy after its
  // half of a joint MMA's smem operands is written (sK_X for QK^T; sP_X/sV_X for
  // P@V); rank 0's issuing thread waits with a cluster-scope acquire. Rank 1's own
  // copies are initialized but unused. Without these the joint MMA could read the
  // peer's operands mid-write — the V21-inherited race that lockstep timing masked
  // in V21-23 but that fires occasionally under V27's deeper warpgroup skew.
  __shared__ __align__(8) uint64_t s_kready_bar_A;
  __shared__ __align__(8) uint64_t s_kready_bar_B;
  __shared__ __align__(8) uint64_t s_pvready_bar_A;
  __shared__ __align__(8) uint64_t s_pvready_bar_B;

  for(int i = tid; i < Br_half * D && tid < 128; i += 128){
    const int r = i / D, c = i % D;
    sQ_A[canon_idx(r, c, Br_half)] = d_Q[qBaseA + i];
    sO_A[i] = 0.0f;
  }
  for(int i = tid - 128; i < Br_half * D && tid >= 128 && tid < 256; i += 128){
    const int r = i / D, c = i % D;
    sQ_B[canon_idx(r, c, Br_half)] = d_Q[qBaseB + i];
    sO_B[i] = 0.0f;
  }
  if(tid < Br_half){ sm_A[tid] = -INFINITY; sl_A[tid] = 0.0f; }
  if(tid >= 128 && tid < 128 + Br_half){ sm_B[tid-128] = -INFINITY; sl_B[tid-128] = 0.0f; }

  const uint32_t mma_bar_A_qk = (uint32_t)__cvta_generic_to_shared(&s_mma_bar_A_qk);
  const uint32_t mma_bar_A_pv = (uint32_t)__cvta_generic_to_shared(&s_mma_bar_A_pv);
  const uint32_t mma_bar_B_qk = (uint32_t)__cvta_generic_to_shared(&s_mma_bar_B_qk);
  const uint32_t mma_bar_B_pv = (uint32_t)__cvta_generic_to_shared(&s_mma_bar_B_pv);
  const uint32_t lbar0        = (uint32_t)__cvta_generic_to_shared(&s_load_bar[0]);
  const uint32_t lbar1        = (uint32_t)__cvta_generic_to_shared(&s_load_bar[1]);
  const uint32_t fbarA0       = (uint32_t)__cvta_generic_to_shared(&s_free_bar_A[0]);
  const uint32_t fbarA1       = (uint32_t)__cvta_generic_to_shared(&s_free_bar_A[1]);
  const uint32_t fbarB0       = (uint32_t)__cvta_generic_to_shared(&s_free_bar_B[0]);
  const uint32_t fbarB1       = (uint32_t)__cvta_generic_to_shared(&s_free_bar_B[1]);
  const uint32_t pready_A     = (uint32_t)__cvta_generic_to_shared(&s_pready_bar_A);
  const uint32_t pready_B     = (uint32_t)__cvta_generic_to_shared(&s_pready_bar_B);
  const uint32_t tmemfree_A   = (uint32_t)__cvta_generic_to_shared(&s_tmemfree_bar_A);
  const uint32_t tmemfree_B   = (uint32_t)__cvta_generic_to_shared(&s_tmemfree_bar_B);
  const uint32_t kready_A     = (uint32_t)__cvta_generic_to_shared(&s_kready_bar_A);
  const uint32_t kready_B     = (uint32_t)__cvta_generic_to_shared(&s_kready_bar_B);
  const uint32_t pvready_A    = (uint32_t)__cvta_generic_to_shared(&s_pvready_bar_A);
  const uint32_t pvready_B    = (uint32_t)__cvta_generic_to_shared(&s_pvready_bar_B);
  if(tid == 0){
    mbar_init(mma_bar_A_qk, 1); mbar_init(mma_bar_A_pv, 1);
    mbar_init(mma_bar_B_qk, 1); mbar_init(mma_bar_B_pv, 1);
    mbar_init(lbar0, 1); mbar_init(lbar1, 1);
    mbar_init(fbarA0, 1); mbar_init(fbarA1, 1);
    mbar_init(fbarB0, 1); mbar_init(fbarB1, 1);
    mbar_init(pready_A, 1); mbar_init(pready_B, 1);
    mbar_init(tmemfree_A, 1); mbar_init(tmemfree_B, 1);
    mbar_init(kready_A, 1); mbar_init(kready_B, 1);
    mbar_init(pvready_A, 1); mbar_init(pvready_B, 1);
    cluster_fence_mbarrier_init();
  }
  cluster.sync();   // CTA-collective — all 416 threads

  constexpr uint32_t NCOLS = (Bc > D) ? (uint32_t)Bc : (uint32_t)D;   // per-half column width, UNCHANGED from V26
  static_assert(NCOLS >= 32 && (NCOLS & (NCOLS - 1)) == 0,
                "tcgen05 column count must be a power of two >= 32");
  constexpr uint32_t NCOLS_TOTAL = NCOLS * 2;
  uint32_t tmem_addr_A;
  {
    __shared__ uint32_t s_tmem_addr;
    if(tid < 32){
      uint32_t s_addr = (uint32_t)__cvta_generic_to_shared(&s_tmem_addr);
      asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                   :: "r"(s_addr), "r"(NCOLS_TOTAL) : "memory");
      asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::: "memory");
    }
    __syncthreads();   // full 416-thread barrier to broadcast tmem_addr
    tmem_addr_A = s_tmem_addr;
  }
  const uint32_t tmem_addr_B = tmem_addr_A + NCOLS;

  const uint32_t TX = ((uint32_t)Bc_half * (uint32_t)D + (uint32_t)Bc * (uint32_t)D_half)
                    * (uint32_t)sizeof(__nv_bfloat16);
  const uint32_t sKstage_addr[2] = {
    (uint32_t)__cvta_generic_to_shared(sKstage[0]),
    (uint32_t)__cvta_generic_to_shared(sKstage[1])
  };
  const uint32_t sVstage_addr[2] = {
    (uint32_t)__cvta_generic_to_shared(sVstage[0]),
    (uint32_t)__cvta_generic_to_shared(sVstage[1])
  };
  const uint32_t lbar[2]  = {lbar0, lbar1};
  const uint32_t fbarA[2] = {fbarA0, fbarA1};
  const uint32_t fbarB[2] = {fbarB0, fbarB1};
  // Rank 1 addresses its TMEM lanes at +128, matching V21-23's working idiom (there
  // expressed as rank_warp_offset = crank*4 warp-units of 32 lanes each).
  const uint32_t rank_lane_offset = crank * 128u;

  if(tid >= 384){
    // ---- Producer: shared K/V staging, unchanged from V26/V23. ----
    if(tid == 384){
      int free_phaseA[2] = {0, 0};
      int free_phaseB[2] = {0, 0};
      for(int kc = 0; kc < nKVTiles; ++kc){
        const int slot = kc & 1;
        if(kc >= 2){
          mbar_wait(fbarA[slot], free_phaseA[slot]); free_phaseA[slot] ^= 1;
          mbar_wait(fbarB[slot], free_phaseB[slot]); free_phaseB[slot] ^= 1;
        }
        const int kRow = kvRow0 + kc * Bc + (int)crank * Bc_half;
        const int vCol = (int)crank * D_half;
        mbar_expect_tx(lbar[slot], TX);
        tma_load_2d(sKstage_addr[slot], &Ktmap_half, 0, kRow, lbar[slot]);
        tma_load_2d(sVstage_addr[slot], &Vtmap_half, vCol, kvRow0 + kc * Bc, lbar[slot]);
        mbar_arrive(lbar[slot]);
      }
    }
  } else if(tid >= 256){
    // ---- Shared rescale group: owns P@V MMA + TMEM readout + O-rescale for BOTH
    // sub-tiles, decoupled from both softmax groups (the whole point of V27). ----
    const int rtid = tid - 256;
    int pr_phaseA = 0, pv_phaseA = 0, pvr_phaseA = 0;
    int pr_phaseB = 0, pv_phaseB = 0, pvr_phaseB = 0;

    for(int kc = 0; kc < nKVTiles; ++kc){
      // -- half A --
      mbar_wait(pready_A, pr_phaseA); pr_phaseA ^= 1;
      for(int i = rtid; i < Br_half * D; i += 128) sO_A[i] *= sCorr_A[i / D];
      sync_rescale();
      {
        const uint64_t descP_base = make_smem_desc(sP_A, Br_half);
        const uint32_t idesc      = make_idesc_bf16(2 * Br_half, D);
        if(crank == 0 && rtid == 0){
          // Peer's sP_A/sV_A must be fully written before the joint P@V reads them.
          mbar_wait_cluster(pvready_A, pvr_phaseA); pvr_phaseA ^= 1;
          const uint64_t descV_base = make_smem_desc(sV_A, D_half);
          asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
          for(int kt = 0; kt < Bc/16; ++kt){
            uint64_t descP = advance_desc_katom(descP_base, kt, Br_half);
            uint64_t descV = advance_desc_katom(descV_base, kt, D_half);
            uint32_t accumulate = (kt > 0) ? 1u : 0u;
            asm volatile(
              "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
              "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
              :: "r"(tmem_addr_A), "l"(descP), "l"(descV), "r"(idesc), "r"(accumulate) : "memory");
          }
          mbar_commit_mma_2cta_multicast(mma_bar_A_pv, 0b11);
        }
        mbar_wait(mma_bar_A_pv, pv_phaseA); pv_phaseA ^= 1;
        tmem_readout_accum_vec_2cta_m64(sO_A, tmem_addr_A, D, D, rank_lane_offset, rtid);
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
      }
      sync_rescale();
      if(rtid == 0) mbar_arrive(tmemfree_A);

      // -- half B --
      mbar_wait(pready_B, pr_phaseB); pr_phaseB ^= 1;
      for(int i = rtid; i < Br_half * D; i += 128) sO_B[i] *= sCorr_B[i / D];
      sync_rescale();
      {
        const uint64_t descP_base = make_smem_desc(sP_B, Br_half);
        const uint32_t idesc      = make_idesc_bf16(2 * Br_half, D);
        if(crank == 0 && rtid == 0){
          // Peer's sP_B/sV_B must be fully written before the joint P@V reads them.
          mbar_wait_cluster(pvready_B, pvr_phaseB); pvr_phaseB ^= 1;
          const uint64_t descV_base = make_smem_desc(sV_B, D_half);
          asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
          for(int kt = 0; kt < Bc/16; ++kt){
            uint64_t descP = advance_desc_katom(descP_base, kt, Br_half);
            uint64_t descV = advance_desc_katom(descV_base, kt, D_half);
            uint32_t accumulate = (kt > 0) ? 1u : 0u;
            asm volatile(
              "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
              "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
              :: "r"(tmem_addr_B), "l"(descP), "l"(descV), "r"(idesc), "r"(accumulate) : "memory");
          }
          mbar_commit_mma_2cta_multicast(mma_bar_B_pv, 0b11);
        }
        mbar_wait(mma_bar_B_pv, pv_phaseB); pv_phaseB ^= 1;
        tmem_readout_accum_vec_2cta_m64(sO_B, tmem_addr_B, D, D, rank_lane_offset, rtid);
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
      }
      sync_rescale();
      if(rtid == 0) mbar_arrive(tmemfree_B);
    } // end kv loop (rescale group)
  } else if(tid >= 128){
    // ---- Softmax-B: rows 64-127 of this CTA's q_tile. Own QK^T MMA + readout +
    // causal mask + online-softmax ONLY — P@V moved to the shared rescale group. ----
    const int ltid          = tid - 128;
    const uint64_t descQ_base = make_smem_desc(sQ_B, Br_half);
    int mbar_phase = 0;
    int tf_phase = 0;
    int kr_phase = 0;   // used only by crank 0's ltid 0 (the QK^T issuer)
    int load_phase[2] = {0, 0};

    for(int kc = 0; kc < nKVTiles; ++kc){
      const int slot = kc & 1;
      mbar_wait(lbar[slot], load_phase[slot]); load_phase[slot] ^= 1;
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      sync_half_b();

      // The rescale group's P@V MMA reads sP_B/sV_B ASYNCHRONOUSLY, and its readout
      // drains TMEM region B — all of which this iteration is about to overwrite
      // (reorder clobbers sV_B, softmax clobbers sP_B, QK^T clobbers the region).
      // tmemfree_B arrives only after that P@V readout completes, so the wait must
      // sit BEFORE the reorder, not merely before the QK^T issue.
      if(kc >= 1){ mbar_wait(tmemfree_B, tf_phase); tf_phase ^= 1; }

      for(int i = ltid; i < Bc_half * D; i += 128){
        const int bc = i / D, d = i % D;
        sK_B[canon_idx(bc, d, Bc_half)] = sKstage[slot][i];
      }
      for(int i = ltid; i < Bc * D_half; i += 128){
        const int bc = i / D_half, d = i % D_half;
        sV_B[canon_idx(d, bc, D_half)] = sVstage[slot][i];
      }
      sync_half_b();
      if(ltid == 0) mbar_arrive(fbarB[slot]);
      // Rank 0's joint QK^T reads rank 1's sK_B — signal it ready (cross-CTA RAW).
      if(crank == 1 && ltid == 0) mbar_arrive_peer(kready_B, 0);

      {
        const uint32_t idesc = make_idesc_bf16(2 * Br_half, Bc);
        if(crank == 0 && ltid == 0){
          mbar_wait_cluster(kready_B, kr_phase); kr_phase ^= 1;   // peer's sK_B ready
          const uint64_t descK_base = make_smem_desc(sK_B, Bc_half);
          asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
          for(int kt = 0; kt < D/16; ++kt){
            uint64_t descQ = advance_desc_katom(descQ_base, kt, Br_half);
            uint64_t descK = advance_desc_katom(descK_base, kt, Bc_half);
            uint32_t accumulate = (kt > 0) ? 1u : 0u;
            asm volatile(
              "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
              "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
              :: "r"(tmem_addr_B), "l"(descQ), "l"(descK), "r"(idesc), "r"(accumulate) : "memory");
          }
          mbar_commit_mma_2cta_multicast(mma_bar_B_qk, 0b11);
        }
        mbar_wait(mma_bar_B_qk, mbar_phase); mbar_phase ^= 1;
        tmem_readout_to_smem_vec_2cta_m64(sS_B, tmem_addr_B, Bc, Bc, scale_l2e,
                                          rank_lane_offset, ltid);
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
        // m64 readout: each row arrives from TWO threads (one per column-half) —
        // publish all of sS_B before the per-row mask/softmax below.
        sync_half_b();
      }

      if(ltid < Br_half){
        const int row_local = ltid;   // one thread per row for mask + softmax
        if(kc > q_tile){
          for(int j = 0; j < Bc; ++j) sS_B[row_local * Bc + j] = -INFINITY;
        } else if(kc == q_tile){
          for(int j = row_local + 64 + 1; j < Bc; ++j) sS_B[row_local * Bc + j] = -INFINITY;
        }
        const float m_old = sm_B[row_local];
        const float l_old = sl_B[row_local];

        float tile_max = -INFINITY;
        int j = 0;
        for(; j + 2 < Bc; j += 3)
          tile_max = fmaxf(tile_max,
                           fmaxf(sS_B[row_local * Bc + j],
                                 fmaxf(sS_B[row_local * Bc + j + 1], sS_B[row_local * Bc + j + 2])));
        for(; j < Bc; ++j) tile_max = fmaxf(tile_max, sS_B[row_local * Bc + j]);

        const float m_new = fmaxf(m_old, tile_max);
        const float corr  = ex2_approx(m_old - m_new);

        float p_sum = 0.0f;
        for(int j2 = 0; j2 < Bc; j2 += 2){
          const float p0 = ex2_approx(sS_B[row_local * Bc + j2]     - m_new);
          const float p1 = ex2_approx(sS_B[row_local * Bc + j2 + 1] - m_new);
          *reinterpret_cast<__nv_bfloat162*>(&sP_B[canon_idx(row_local, j2, Br_half)]) =
              __floats2bfloat162_rn(p0, p1);
          p_sum += p0 + p1;
        }
        sm_B[row_local] = m_new; sl_B[row_local] = l_old * corr + p_sum; sCorr_B[row_local] = corr;
      }
      sync_half_b();
      if(ltid == 0) mbar_arrive(pready_B);
      // Rank 0's rescale-issued joint P@V reads rank 1's sP_B/sV_B — signal ready.
      if(crank == 1 && ltid == 0) mbar_arrive_peer(pvready_B, 0);
    } // end kv loop (softmax-B)
  } else {
    // ---- Softmax-A: rows 0-63 of this CTA's q_tile. Same role as softmax-B above. ----
    const uint64_t descQ_base = make_smem_desc(sQ_A, Br_half);
    int mbar_phase = 0;
    int tf_phase = 0;
    int kr_phase = 0;   // used only by crank 0's tid 0 (the QK^T issuer)
    int load_phase[2] = {0, 0};

    for(int kc = 0; kc < nKVTiles; ++kc){
      const int slot = kc & 1;
      mbar_wait(lbar[slot], load_phase[slot]); load_phase[slot] ^= 1;
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      consumer_sync();

      // Same WAR-hazard rule as softmax-B: the rescale group's P@V MMA still reads
      // sP_A/sV_A asynchronously until tmemfree_A arrives — wait BEFORE overwriting.
      if(kc >= 1){ mbar_wait(tmemfree_A, tf_phase); tf_phase ^= 1; }

      for(int i = tid; i < Bc_half * D; i += 128){
        const int bc = i / D, d = i % D;
        sK_A[canon_idx(bc, d, Bc_half)] = sKstage[slot][i];
      }
      for(int i = tid; i < Bc * D_half; i += 128){
        const int bc = i / D_half, d = i % D_half;
        sV_A[canon_idx(d, bc, D_half)] = sVstage[slot][i];
      }
      consumer_sync();
      if(tid == 0) mbar_arrive(fbarA[slot]);
      // Rank 0's joint QK^T reads rank 1's sK_A — signal it ready (cross-CTA RAW).
      if(crank == 1 && tid == 0) mbar_arrive_peer(kready_A, 0);

      {
        const uint32_t idesc = make_idesc_bf16(2 * Br_half, Bc);
        if(crank == 0 && tid == 0){
          mbar_wait_cluster(kready_A, kr_phase); kr_phase ^= 1;   // peer's sK_A ready
          const uint64_t descK_base = make_smem_desc(sK_A, Bc_half);
          asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
          for(int kt = 0; kt < D/16; ++kt){
            uint64_t descQ = advance_desc_katom(descQ_base, kt, Br_half);
            uint64_t descK = advance_desc_katom(descK_base, kt, Bc_half);
            uint32_t accumulate = (kt > 0) ? 1u : 0u;
            asm volatile(
              "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
              "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
              :: "r"(tmem_addr_A), "l"(descQ), "l"(descK), "r"(idesc), "r"(accumulate) : "memory");
          }
          mbar_commit_mma_2cta_multicast(mma_bar_A_qk, 0b11);
        }
        mbar_wait(mma_bar_A_qk, mbar_phase); mbar_phase ^= 1;
        tmem_readout_to_smem_vec_2cta_m64(sS_A, tmem_addr_A, Bc, Bc, scale_l2e,
                                          rank_lane_offset, tid);
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
        // m64 readout: each row arrives from TWO threads (one per column-half) —
        // publish all of sS_A before the per-row mask/softmax below.
        consumer_sync();
      }

      if(tid < Br_half){
        const int row_local = tid;   // one thread per row for mask + softmax
        if(kc > q_tile){
          for(int j = 0; j < Bc; ++j) sS_A[row_local * Bc + j] = -INFINITY;
        } else if(kc == q_tile){
          for(int j = row_local + 1; j < Bc; ++j) sS_A[row_local * Bc + j] = -INFINITY;
        }
        const float m_old = sm_A[row_local];
        const float l_old = sl_A[row_local];

        float tile_max = -INFINITY;
        int j = 0;
        for(; j + 2 < Bc; j += 3)
          tile_max = fmaxf(tile_max,
                           fmaxf(sS_A[row_local * Bc + j],
                                 fmaxf(sS_A[row_local * Bc + j + 1], sS_A[row_local * Bc + j + 2])));
        for(; j < Bc; ++j) tile_max = fmaxf(tile_max, sS_A[row_local * Bc + j]);

        const float m_new = fmaxf(m_old, tile_max);
        const float corr  = ex2_approx(m_old - m_new);

        float p_sum = 0.0f;
        for(int j2 = 0; j2 < Bc; j2 += 2){
          const float p0 = ex2_approx(sS_A[row_local * Bc + j2]     - m_new);
          const float p1 = ex2_approx(sS_A[row_local * Bc + j2 + 1] - m_new);
          *reinterpret_cast<__nv_bfloat162*>(&sP_A[canon_idx(row_local, j2, Br_half)]) =
              __floats2bfloat162_rn(p0, p1);
          p_sum += p0 + p1;
        }
        sm_A[row_local] = m_new; sl_A[row_local] = l_old * corr + p_sum; sCorr_A[row_local] = corr;
      }
      consumer_sync();
      if(tid == 0) mbar_arrive(pready_A);
      // Rank 0's rescale-issued joint P@V reads rank 1's sP_A/sV_A — signal ready.
      if(crank == 1 && tid == 0) mbar_arrive_peer(pvready_A, 0);
    } // end kv loop (softmax-A)
  }

  __syncthreads();   // full-block reconvergence: producer + both softmax groups + rescale all done
  cluster.sync();
  if(tid < 32)
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
                 :: "r"(tmem_addr_A), "r"(NCOLS_TOTAL) : "memory");
  __syncthreads();

  for(int i = 2 * tid; i < Br_half * D && tid < 128; i += 256){
    const float denom = sl_A[i / D];
    *reinterpret_cast<__nv_bfloat162*>(&d_O[qBaseA + i]) =
        __floats2bfloat162_rn(sO_A[i] / denom, sO_A[i + 1] / denom);
  }
  for(int i = 2 * (tid - 128); i < Br_half * D && tid >= 128 && tid < 256; i += 256){
    const float denom = sl_B[i / D];
    *reinterpret_cast<__nv_bfloat162*>(&d_O[qBaseB + i]) =
        __floats2bfloat162_rn(sO_B[i] / denom, sO_B[i + 1] / denom);
  }
  if(tid < Br_half)
    d_LSE[lBaseA + tid] = 0.6931471805599453f * (sm_A[tid] + log2f(sl_A[tid]));
  if(tid >= 128 && tid < 128 + Br_half)
    d_LSE[lBaseB + (tid - 128)] = 0.6931471805599453f * (sm_B[tid-128] + log2f(sl_B[tid-128]));

} // end of gqa_v27_causal

// =================================
//  gqa_tmem_probe — one-shot empirical dump of the per-CTA D layout for the
//  cta_group::2 idesc-M=128 (64 rows per CTA) QK^T MMA shape V27's half-pipelines
//  use. Runs ONE such MMA on the real Q/K data (b=0, h=0, q_tiles 0/1 as the CTA
//  pair, key tile 0) and dumps the RAW fp32 contents of TMEM lanes 0-127 — and the
//  same lanes again at address offset +128 lanes, to settle whether that offset
//  (V21-23's crank-1 idiom) aliases back to the local lanes or reads something
//  else — with NO layout assumption baked in. Host-side analysis in main() matches
//  each lane's contents against host-computed raw QK^T rows and prints the true
//  lane -> (row, column-range) mapping.
// =================================
template<int Br, int Bc, int D>
__global__ void __cluster_dims__(2, 1, 1) gqa_tmem_probe(
  __nv_bfloat16 *d_Q,
  float *d_probe,                                  // [2 cranks][2 lane-bases][128 lanes][Bc cols]
  const __grid_constant__ CUtensorMap Ktmap_half
){
  constexpr int Bc_half = Bc / 2;
  constexpr int Br_half = Br / 2;
  const int q_tile = blockIdx.x;   // 0 (crank 0) / 1 (crank 1)
  const int tid    = threadIdx.x;  // 128 threads

  cg::cluster_group cluster = cg::this_cluster();
  const unsigned int crank  = cluster.block_rank();

  __shared__ __align__(16)  __nv_bfloat16 sQ[Br_half * D];
  __shared__ __align__(128) __nv_bfloat16 sKstage[Bc_half * D];
  __shared__ __align__(16)  __nv_bfloat16 sK[Bc_half * D];
  __shared__ __align__(8) uint64_t s_load_bar;
  __shared__ __align__(8) uint64_t s_mma_bar;

  // Q: this CTA's own 64 rows = global q rows [q_tile*Br, q_tile*Br + Br_half), b=0,h=0.
  const long qBase = (long)q_tile * Br * D;
  for(int i = tid; i < Br_half * D; i += blockDim.x){
    const int r = i / D, c = i % D;
    sQ[canon_idx(r, c, Br_half)] = d_Q[qBase + i];
  }

  const uint32_t lbar = (uint32_t)__cvta_generic_to_shared(&s_load_bar);
  const uint32_t mbar = (uint32_t)__cvta_generic_to_shared(&s_mma_bar);
  if(tid == 0){ mbar_init(lbar, 1); mbar_init(mbar, 1); cluster_fence_mbarrier_init(); }
  cluster.sync();

  constexpr uint32_t NCOLS = (uint32_t)Bc;
  uint32_t tmem_addr;
  {
    __shared__ uint32_t s_tmem_addr;
    if(tid < 32){
      uint32_t s_addr = (uint32_t)__cvta_generic_to_shared(&s_tmem_addr);
      asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                   :: "r"(s_addr), "r"(NCOLS) : "memory");
      asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::: "memory");
    }
    __syncthreads();
    tmem_addr = s_tmem_addr;
  }

  // K: key tile 0, this rank's key half (b=0, hkv=0 -> kvRow0 = 0).
  if(tid == 0){
    const uint32_t TX = (uint32_t)Bc_half * (uint32_t)D * (uint32_t)sizeof(__nv_bfloat16);
    mbar_expect_tx(lbar, TX);
    tma_load_2d((uint32_t)__cvta_generic_to_shared(sKstage), &Ktmap_half, 0, (int)crank * Bc_half, lbar);
    mbar_arrive(lbar);
  }
  mbar_wait(lbar, 0);
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  __syncthreads();
  for(int i = tid; i < Bc_half * D; i += blockDim.x){
    const int bc = i / D, d = i % D;
    sK[canon_idx(bc, d, Bc_half)] = sKstage[i];
  }
  __syncthreads();
  cluster.sync();   // BOTH ranks' sQ/sK ready before the joint MMA (airtight, unlike timing-lucky V21)

  {
    const uint64_t descQ_base = make_smem_desc(sQ, Br_half);
    const uint32_t idesc = make_idesc_bf16(2 * Br_half, Bc);   // M=128: exactly V27's QK^T shape
    if(crank == 0 && tid == 0){
      const uint64_t descK_base = make_smem_desc(sK, Bc_half);
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
      for(int kt = 0; kt < D/16; ++kt){
        uint64_t descQ = advance_desc_katom(descQ_base, kt, Br_half);
        uint64_t descK = advance_desc_katom(descK_base, kt, Bc_half);
        uint32_t accumulate = (kt > 0) ? 1u : 0u;
        asm volatile(
          "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
          "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
          :: "r"(tmem_addr), "l"(descQ), "l"(descK), "r"(idesc), "r"(accumulate) : "memory");
      }
      mbar_commit_mma_2cta_multicast(mbar, 0b11);
    }
    mbar_wait(mbar, 0);
  }

  // Raw dump, no layout assumption: 4 warps cover lanes 0-127, then again at +128.
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  const int w = tid / 32, ln = tid % 32;
  for(int base = 0; base < 2; ++base){
    const uint32_t lane_addr = (uint32_t)(base * 128 + w * 32 + ln) << 16;
    for(int col = 0; col < Bc; col += 8){
      uint32_t r0,r1,r2,r3,r4,r5,r6,r7;
      asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
        : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7)
        : "r"(tmem_addr + lane_addr + (uint32_t)col)
        : "memory"
      );
      asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
      float* o = &d_probe[(((long)crank * 2 + base) * 128 + (w * 32 + ln)) * Bc + col];
      o[0] = reinterpret_cast<float&>(r0);
      o[1] = reinterpret_cast<float&>(r1);
      o[2] = reinterpret_cast<float&>(r2);
      o[3] = reinterpret_cast<float&>(r3);
      o[4] = reinterpret_cast<float&>(r4);
      o[5] = reinterpret_cast<float&>(r5);
      o[6] = reinterpret_cast<float&>(r6);
      o[7] = reinterpret_cast<float&>(r7);
    }
  }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  cluster.sync();
  if(tid < 32)
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
                 :: "r"(tmem_addr), "r"(NCOLS) : "memory");

} // end of gqa_tmem_probe

// =================================
//  gqa_v28_causal — matches cuDNN's ACTUAL causal reference kernel launch config,
//  discovered via NCU on the causal (not the earlier non-causal) reference:
//  cudnn_generated_..._128x128x64_4x1x1_cga1x1x1_kernel0_0 — grid (1536,1,1) flat
//  1D, block (512,1,1), 128 registers/thread. Two decisive differences from what
//  V20-27 targeted (the NON-causal reference's cga2x1x1 launch shape):
//    cga1x1x1, not cga2x1x1 -> single-CTA cta_group::1, NO 2-SM cluster pairing at
//      all. Drops crank/rank_lane_offset, the N-half B-operand split, and the M=64
//      column-halved TMEM readout entirely -- back to V19's simple, already-proven
//      M=128 "row = warp_id*32 + lane" mapping (see V19_causal above).
//    grid 1536 = B*Hq*(nTiles/2), HALF the naive B*Hq*nTiles=3072 -> causal LOAD
//      BALANCING: each CTA is assigned a PAIR of query tiles (q_tile_lo=pair,
//      q_tile_hi=nTiles-1-pair), so every CTA does the same total kc-tile-visit
//      count (nTiles+1) regardless of which pair it draws, instead of early-tile
//      CTAs idling while late-tile CTAs do all the real work (the naive grid's
//      triangular imbalance).
//
//  This version deliberately does NOT attempt the full 4-concurrent-warpgroup
//  overlap between a pair's two tiles that cuDNN's 512-thread/4-warpgroup count
//  suggests it uses internally (that would need the scores/probabilities buffers
//  SHARED between the two tiles to fit under the ~230KB smem budget -- a real
//  complexity/risk step up, and a candidate for a later version). Instead it
//  isolates the OTHER variable: does matching the grid/cluster choice alone help?
//  Each CTA runs V19_causal's exact, already-proven single-tile pipeline TWICE,
//  back-to-back (q_tile_lo fully to completion, then q_tile_hi fully to
//  completion) — so each pass needs NO new masking case at all (loop bound is
//  just its OWN q_tile+1; kc never exceeds it; diagonal-only mask exactly like
//  V19). TMEM alloc/dealloc is hoisted OUTSIDE the 2-pass loop (paid once per CTA,
//  mirroring V24); Q/O/mbarriers are reset at the top of each pass (mirroring
//  V24's established re-init pattern). Accepted trade-off: kc range [0,q_tile_lo]
//  gets TMA-loaded TWICE (once per pass, since the two passes share no state) —
//  redundant but cache-friendly; simple/low-risk beats clever for this round.
//  Block stays at 160 threads (128 consumer + 32 producer, matching V19/22/23) —
//  NOT padded to 512, since there's no 4-way concurrent work actually happening
//  here; only the GRID (1536) and cluster choice (none) are being tested against
//  cuDNN's launch shape. Smem footprint matches V19's own (~194KB, comfortably
//  under budget) since passes fully reuse all buffers rather than needing two
//  live copies.
//
//  Deviates from V19 in ONE respect: sS is fp32, not fp16. V19's fp16 sS carries
//  an inherent, already-documented LSE precision cost at Bc=128 (see V18's header
//  in GQA_sm103.cu) — harmless there (passes tolerance easily), but it put V28 on
//  a visibly different precision tier than V27 (LSE max_abs ~6.7e-4 vs V27's
//  ~1.9e-6), which the user wanted matched. fp32 sS costs +32KB smem (~194KB ->
//  ~226KB) but still fits comfortably. The readout reuses
//  tmem_readout_to_smem_vec_2cta with rank_warp_offset=0 rather than a new
//  function — that helper's address math is pure local-TMEM arithmetic with no
//  cta_group-specific behavior, so it's valid for cta_group::1 too.
// =================================
template<int Br, int Bc, int D>
__global__ void gqa_v28_causal(
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
  static_assert(Br == Bc, "causal tile-skip + diagonal-tile mask requires Br == Bc");

  const int tid = threadIdx.x;   // 0..159: 0..127 consumer, 128..159 producer

  // ---- Flat-grid decomposition: causal load-balanced tile pairing. ----
  const int nTiles     = S / Br;
  const int pairsPerBH = nTiles / 2;
  const int idx  = blockIdx.x;
  const int b    = idx / (Hq * pairsPerBH);
  const int rem  = idx - b * (Hq * pairsPerBH);
  const int hq   = rem / pairsPerBH;
  const int pair = rem - hq * pairsPerBH;
  const int hkv  = hq / G;
  const int kvRow0 = (b * Hkv + hkv) * S;

  const float scale_l2e = scale * 1.4426950408889634f;

  __shared__ __align__(16)  __nv_bfloat16 sQ[Br * D];
  __shared__ __align__(128) __nv_bfloat16 sK[2][Bc * D];        // atom-native, TMA lands directly — no reorder
  __shared__ __align__(128) __nv_bfloat16 sVstage[2][Bc * D];   // raw landing, row-major (needs transpose reorder)
  __shared__ __align__(16)  __nv_bfloat16 sV[Bc * D];           // canonical [D,Bc] transposed (reorder output)
  __shared__ __align__(16)  __nv_bfloat16 sP[Br * Bc];
  __shared__ __align__(16)  float         sS[Br * Bc];          // fp32 — see 2026-07-13 session notes:
                                                                 // fp16 sS at Bc=128 has an inherent LSE
                                                                 // precision cost (V18's finding); fp32
                                                                 // trades +32KB smem to match V27's tier.
  __shared__ __align__(16)  float         sO[Br * D];           // P@V readout accumulates here directly
  __shared__ float sm[Br];
  __shared__ float sl[Br];
  __shared__ float sCorr[Br];
  __shared__ __align__(8) uint64_t s_mma_bar;
  __shared__ __align__(8) uint64_t s_load_bar_K[2];
  __shared__ __align__(8) uint64_t s_free_bar_K[2];
  __shared__ __align__(8) uint64_t s_load_bar_V[2];
  __shared__ __align__(8) uint64_t s_free_bar_V[2];

  const uint32_t mma_bar = (uint32_t)__cvta_generic_to_shared(&s_mma_bar);
  const uint32_t lbarK0  = (uint32_t)__cvta_generic_to_shared(&s_load_bar_K[0]);
  const uint32_t lbarK1  = (uint32_t)__cvta_generic_to_shared(&s_load_bar_K[1]);
  const uint32_t fbarK0  = (uint32_t)__cvta_generic_to_shared(&s_free_bar_K[0]);
  const uint32_t fbarK1  = (uint32_t)__cvta_generic_to_shared(&s_free_bar_K[1]);
  const uint32_t lbarV0  = (uint32_t)__cvta_generic_to_shared(&s_load_bar_V[0]);
  const uint32_t lbarV1  = (uint32_t)__cvta_generic_to_shared(&s_load_bar_V[1]);
  const uint32_t fbarV0  = (uint32_t)__cvta_generic_to_shared(&s_free_bar_V[0]);
  const uint32_t fbarV1  = (uint32_t)__cvta_generic_to_shared(&s_free_bar_V[1]);
  const uint32_t lbarK[2] = {lbarK0, lbarK1};
  const uint32_t fbarK[2] = {fbarK0, fbarK1};
  const uint32_t lbarV[2] = {lbarV0, lbarV1};
  const uint32_t fbarV[2] = {fbarV0, fbarV1};

  constexpr uint32_t NCOLS = (Bc > D) ? (uint32_t)Bc : (uint32_t)D;
  static_assert(NCOLS >= 32 && (NCOLS & (NCOLS - 1)) == 0,
                "tcgen05 column count must be a power of two >= 32");

  // ---- TMEM alloc: ONCE for the whole CTA lifetime (paid once, not once per pass). ----
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

  const uint32_t TX = (uint32_t)Bc * (uint32_t)D * (uint32_t)sizeof(__nv_bfloat16);
  const uint32_t sK_addr[2] = {
    (uint32_t)__cvta_generic_to_shared(sK[0]),
    (uint32_t)__cvta_generic_to_shared(sK[1])
  };
  const uint32_t sVstage_addr[2] = {
    (uint32_t)__cvta_generic_to_shared(sVstage[0]),
    (uint32_t)__cvta_generic_to_shared(sVstage[1])
  };

  for(int pass = 0; pass < 2; ++pass){
    const int q_tile   = (pass == 0) ? pair : (nTiles - 1 - pair);
    const int q_row0   = q_tile * Br;
    const int nKVTiles = q_tile + 1;   // own bound only — never exceeds this pass's tile

    const long qBase = ((long)(b * Hq + hq) * S + q_row0) * D;
    const long lBase = ((long)(b * Hq + hq) * S + q_row0);

    for(int i = tid; i < Br * D; i += blockDim.x){
      const int r = i / D, c = i % D;
      sQ[canon_idx(r, c, Br)] = d_Q[qBase + i];
      sO[i] = 0.0f;
    }
    if(tid < Br){ sm[tid] = -INFINITY; sl[tid] = 0.0f; }

    // Re-init mbarriers each pass — same rationale as V24: carrying phase counters
    // across passes without an exactly-matching hardware phase risks a wait/parity
    // mismatch when the two passes' slot-use-count parities differ (they will,
    // since q_tile_lo and q_tile_hi have different nKVTiles).
    if(tid == 0){
      mbar_init(mma_bar, 1);
      mbar_init(lbarK0, 1); mbar_init(lbarK1, 1);
      mbar_init(fbarK0, 1); mbar_init(fbarK1, 1);
      mbar_init(lbarV0, 1); mbar_init(lbarV1, 1);
      mbar_init(fbarV0, 1); mbar_init(fbarV1, 1);
    }
    __syncthreads();

    if(tid >= 128){
      // ---- Producer warp: K (atom-native) and V (raw stage) independent, double-buffered. ----
      if(tid == 128){
        int free_phase_K[2] = {0, 0};
        int free_phase_V[2] = {0, 0};
        for(int kc = 0; kc < nKVTiles; ++kc){
          const int slot = kc & 1;
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

        for(int i = tid; i < Bc * D; i += Br){
          const int bc = i / D, d = i % D;
          sV[canon_idx(d, bc, D)] = sVstage[slot][i];   // transposed
        }
        consumer_sync();
        if(tid == 0) mbar_arrive(fbarV[slot]);

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
          if(tid == 0) mbar_arrive(fbarK[slot]);
          // rank_warp_offset=0: this readout's math is cta_group-agnostic (pure
          // local TMEM address computation), so the cta_group::2 helper reduces
          // exactly to the cta_group::1 case with no peer-routing offset.
          tmem_readout_to_smem_vec_2cta(sS, tmem_addr, Br, Bc, Bc, scale_l2e, 0u);
          asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");

          // CAUSAL diagonal-tile mask only — this pass's loop bound never exceeds
          // its OWN q_tile, so the "kc > q_tile" full-row case never arises here.
          if(kc == q_tile){
            for(int j = tid + 1; j < Bc; ++j) sS[tid * Bc + j] = -INFINITY;
          }
          consumer_sync();
        }

        {
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
        consumer_sync();

        for(int i = tid; i < Br * D; i += Br) sO[i] *= sCorr[i / D];
        consumer_sync();

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
          tmem_readout_accum_vec(sO, tmem_addr, Br, D, D);
          asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
          consumer_sync();
        }
      } // end kv loop
    }

    __syncthreads();   // full-block reconvergence: producer is done, consumers are done

    for(int i = 2 * tid; i < Br * D; i += 2 * blockDim.x){
      const float denom = sl[i / D];
      *reinterpret_cast<__nv_bfloat162*>(&d_O[qBase + i]) =
          __floats2bfloat162_rn(sO[i] / denom, sO[i + 1] / denom);
    }
    if(tid < Br)
      d_LSE[lBase + tid] = 0.6931471805599453f * (sm[tid] + log2f(sl[tid]));

    __syncthreads();   // reconvergence before the NEXT pass reuses all these buffers
  } // end pass loop (q_tile_lo, then q_tile_hi)

  if(tid < 32)
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(tmem_addr), "r"(NCOLS) : "memory");

} // end of gqa_v28_causal

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

// V20_causal — same launch shape as GQA_sm103.cu's launch_gqa_v20: grid x is the
// paired q_tile dimension (must be x for cta_group::2), full (non-split) K/V TMA maps.
template<int Br, int Bc, int D>
void launch_gqa_v20_causal(
  __nv_bfloat16 *d_Q, __nv_bfloat16 *d_K, __nv_bfloat16 *d_V,
  __nv_bfloat16 *d_O, float *d_LSE,
  int B, int Hq, int Hkv, int S, int G, float scale
){
  static_assert(Br == Bc, "causal tile-skip + diagonal-tile mask requires Br == Bc");
  static_assert(Br % 32 == 0, "Br must be a multiple of 32 (RANK_WARP_SPAN)");
  static_assert(Bc % 8 == 0, "Bc must be a multiple of 8 for tcgen05 N = 8");
  static_assert(D  % 16 == 0, "D  must be a multiple of 16 for tcgen05 dense");

  dim3 GRID(S/Br, Hq, B);   // x must be even (cluster dim 2 along x)
  dim3 BLOCK(128);

  static bool cfgd = false;
  static CUtensorMap Ktmap, Vtmap;
  if(!cfgd){
    const uint64_t kvRows = (uint64_t)B * Hkv * S;
    Ktmap = make_tma_2d(d_K, kvRows, (uint64_t)D, (uint32_t)Bc, (uint32_t)D);
    Vtmap = make_tma_2d(d_V, kvRows, (uint64_t)D, (uint32_t)Bc, (uint32_t)D);
    cfgd = true;
  }
  gqa_v20_causal<Br, Bc, D><<<GRID, BLOCK>>>(d_Q, d_O, d_LSE, Ktmap, Vtmap,
                          B, Hq, Hkv, G, S, scale);
}

// V21_causal — genuine N-half split B operand, same TMA-map shapes as GQA_sm103.cu's
// launch_gqa_v21 (Ktmap_half box_rows=Bc/2, Vtmap_half box_cols=D/2).
template<int Br, int Bc, int D>
void launch_gqa_v21_causal(
  __nv_bfloat16 *d_Q, __nv_bfloat16 *d_K, __nv_bfloat16 *d_V,
  __nv_bfloat16 *d_O, float *d_LSE,
  int B, int Hq, int Hkv, int S, int G, float scale
){
  static_assert(Br == Bc, "causal tile-skip + diagonal-tile mask requires Br == Bc");
  static_assert(Br % 32 == 0, "Br must be a multiple of 32 (RANK_WARP_SPAN)");
  static_assert(Bc % 8 == 0, "Bc must be a multiple of 8 for tcgen05 N = 8");
  static_assert(D  % 16 == 0, "D  must be a multiple of 16 for tcgen05 dense");
  static_assert(Bc % 2 == 0, "Bc must be even to split the key range in half");
  static_assert(D  % 2 == 0, "D must be even to split the head dim in half");

  dim3 GRID(S/Br, Hq, B);
  dim3 BLOCK(128);

  static bool cfgd = false;
  static CUtensorMap Ktmap_half, Vtmap_half;
  if(!cfgd){
    const uint64_t kvRows = (uint64_t)B * Hkv * S;
    Ktmap_half = make_tma_2d(d_K, kvRows, (uint64_t)D, (uint32_t)(Bc / 2), (uint32_t)D);
    Vtmap_half = make_tma_2d(d_V, kvRows, (uint64_t)D, (uint32_t)Bc, (uint32_t)(D / 2));
    cfgd = true;
  }
  gqa_v21_causal<Br, Bc, D><<<GRID, BLOCK>>>(d_Q, d_O, d_LSE, Ktmap_half, Vtmap_half,
                          B, Hq, Hkv, G, S, scale);
}

// V22_causal — warp specialization on top of V21_causal's B-split.
template<int Br, int Bc, int D>
void launch_gqa_v22_causal(
  __nv_bfloat16 *d_Q, __nv_bfloat16 *d_K, __nv_bfloat16 *d_V,
  __nv_bfloat16 *d_O, float *d_LSE,
  int B, int Hq, int Hkv, int S, int G, float scale
){
  static_assert(Br == 128, "V22_causal's consumer group is hardwired to 128 threads");
  static_assert(Br == Bc, "causal tile-skip + diagonal-tile mask requires Br == Bc");
  static_assert(Bc % 8 == 0, "Bc must be a multiple of 8 for tcgen05 N = 8");
  static_assert(D  % 16 == 0, "D  must be a multiple of 16 for tcgen05 dense");
  static_assert(Bc % 2 == 0, "Bc must be even to split the key range in half");
  static_assert(D  % 2 == 0, "D must be even to split the head dim in half");

  dim3 GRID(S/Br, Hq, B);
  dim3 BLOCK(160);   // 128 consumer threads (warps 0-3) + 32 producer threads (warp 4)

  static bool cfgd = false;
  static CUtensorMap Ktmap_half, Vtmap_half;
  if(!cfgd){
    const uint64_t kvRows = (uint64_t)B * Hkv * S;
    Ktmap_half = make_tma_2d(d_K, kvRows, (uint64_t)D, (uint32_t)(Bc / 2), (uint32_t)D);
    Vtmap_half = make_tma_2d(d_V, kvRows, (uint64_t)D, (uint32_t)Bc, (uint32_t)(D / 2));
    cfgd = true;
  }
  gqa_v22_causal<Br, Bc, D><<<GRID, BLOCK>>>(d_Q, d_O, d_LSE, Ktmap_half, Vtmap_half,
                          B, Hq, Hkv, G, S, scale);
}

// V23_causal — double-buffered TMA staging on top of V22_causal.
template<int Br, int Bc, int D>
void launch_gqa_v23_causal(
  __nv_bfloat16 *d_Q, __nv_bfloat16 *d_K, __nv_bfloat16 *d_V,
  __nv_bfloat16 *d_O, float *d_LSE,
  int B, int Hq, int Hkv, int S, int G, float scale
){
  static_assert(Br == 128, "V23_causal's consumer group is hardwired to 128 threads");
  static_assert(Br == Bc, "causal tile-skip + diagonal-tile mask requires Br == Bc");
  static_assert(Bc % 8 == 0, "Bc must be a multiple of 8 for tcgen05 N = 8");
  static_assert(D  % 16 == 0, "D  must be a multiple of 16 for tcgen05 dense");
  static_assert(Bc % 2 == 0, "Bc must be even to split the key range in half");
  static_assert(D  % 2 == 0, "D must be even to split the head dim in half");

  dim3 GRID(S/Br, Hq, B);
  dim3 BLOCK(160);

  static bool cfgd = false;
  static CUtensorMap Ktmap_half, Vtmap_half;
  if(!cfgd){
    const uint64_t kvRows = (uint64_t)B * Hkv * S;
    Ktmap_half = make_tma_2d(d_K, kvRows, (uint64_t)D, (uint32_t)(Bc / 2), (uint32_t)D);
    Vtmap_half = make_tma_2d(d_V, kvRows, (uint64_t)D, (uint32_t)Bc, (uint32_t)(D / 2));
    cfgd = true;
  }
  gqa_v23_causal<Br, Bc, D><<<GRID, BLOCK>>>(d_Q, d_O, d_LSE, Ktmap_half, Vtmap_half,
                          B, Hq, Hkv, G, S, scale);
}

// V24_causal — V23_causal + persistent launch (grid.x halved, each CTA-pair loops
// over 2 virtual q_tiles internally). Matches cuDNN's observed grid.x=16 (vs our
// S/Br=32) for this problem size; same 160-thread model as V23_causal otherwise.
template<int Br, int Bc, int D>
void launch_gqa_v24_causal(
  __nv_bfloat16 *d_Q, __nv_bfloat16 *d_K, __nv_bfloat16 *d_V,
  __nv_bfloat16 *d_O, float *d_LSE,
  int B, int Hq, int Hkv, int S, int G, float scale
){
  static_assert(Br == 128, "V24_causal's consumer group is hardwired to 128 threads");
  static_assert(Br == Bc, "causal tile-skip + diagonal-tile mask requires Br == Bc");
  static_assert(Bc % 8 == 0, "Bc must be a multiple of 8 for tcgen05 N = 8");
  static_assert(D  % 16 == 0, "D  must be a multiple of 16 for tcgen05 dense");
  static_assert(Bc % 2 == 0, "Bc must be even to split the key range in half");
  static_assert(D  % 2 == 0, "D must be even to split the head dim in half");

  dim3 GRID(S / Br / 2, Hq, B);   // half of V23_causal's grid.x — each CTA-pair covers 2 q_tiles
  dim3 BLOCK(160);

  static bool cfgd = false;
  static CUtensorMap Ktmap_half, Vtmap_half;
  if(!cfgd){
    const uint64_t kvRows = (uint64_t)B * Hkv * S;
    Ktmap_half = make_tma_2d(d_K, kvRows, (uint64_t)D, (uint32_t)(Bc / 2), (uint32_t)D);
    Vtmap_half = make_tma_2d(d_V, kvRows, (uint64_t)D, (uint32_t)Bc, (uint32_t)(D / 2));
    cfgd = true;
  }
  gqa_v24_causal<Br, Bc, D><<<GRID, BLOCK>>>(d_Q, d_O, d_LSE, Ktmap_half, Vtmap_half,
                          B, Hq, Hkv, G, S, scale);
}

// V25_causal — V23_causal + widened 512-thread block (128 compute + 32 producer +
// 352 reorder-helper). Same grid shape as V22/V23 (non-persistent — this isolates
// the block-width change on its own, separate from V24's persistent-launch test).
template<int Br, int Bc, int D>
void launch_gqa_v25_causal(
  __nv_bfloat16 *d_Q, __nv_bfloat16 *d_K, __nv_bfloat16 *d_V,
  __nv_bfloat16 *d_O, float *d_LSE,
  int B, int Hq, int Hkv, int S, int G, float scale
){
  static_assert(Br == 128, "V25_causal's compute group is hardwired to 128 threads");
  static_assert(Br == Bc, "causal tile-skip + diagonal-tile mask requires Br == Bc");
  static_assert(Bc % 8 == 0, "Bc must be a multiple of 8 for tcgen05 N = 8");
  static_assert(D  % 16 == 0, "D  must be a multiple of 16 for tcgen05 dense");
  static_assert(Bc % 2 == 0, "Bc must be even to split the key range in half");
  static_assert(D  % 2 == 0, "D must be even to split the head dim in half");

  dim3 GRID(S/Br, Hq, B);
  dim3 BLOCK(512);   // 128 compute + 32 producer + 352 reorder-helper

  static bool cfgd = false;
  static CUtensorMap Ktmap_half, Vtmap_half;
  if(!cfgd){
    const uint64_t kvRows = (uint64_t)B * Hkv * S;
    Ktmap_half = make_tma_2d(d_K, kvRows, (uint64_t)D, (uint32_t)(Bc / 2), (uint32_t)D);
    Vtmap_half = make_tma_2d(d_V, kvRows, (uint64_t)D, (uint32_t)Bc, (uint32_t)(D / 2));
    cfgd = true;
  }
  gqa_v25_causal<Br, Bc, D><<<GRID, BLOCK>>>(d_Q, d_O, d_LSE, Ktmap_half, Vtmap_half,
                          B, Hq, Hkv, G, S, scale);
}

// V26_causal — the real ping-pong: this CTA's 128 rows split into two independent
// 64-row halves (own TMEM, own K/V, own mbarriers), each a genuinely separate set of
// 128 threads, plus a 32-thread shared K/V producer. 288 threads total. Same
// non-persistent grid as V22/V23/V25 — isolates the ping-pong architecture on its
// own before considering combining with persistence.
template<int Br, int Bc, int D>
void launch_gqa_v26_causal(
  __nv_bfloat16 *d_Q, __nv_bfloat16 *d_K, __nv_bfloat16 *d_V,
  __nv_bfloat16 *d_O, float *d_LSE,
  int B, int Hq, int Hkv, int S, int G, float scale
){
  static_assert(Br == 128, "V26_causal's per-half compute group is hardwired to 4 warps");
  static_assert(Br == Bc, "causal tile-skip + diagonal-tile mask requires Br == Bc");
  static_assert(Bc % 8 == 0, "Bc must be a multiple of 8 for tcgen05 N = 8");
  static_assert(D  % 16 == 0, "D  must be a multiple of 16 for tcgen05 dense");
  static_assert(Bc % 2 == 0, "Bc must be even to split the key range in half");
  static_assert(D  % 2 == 0, "D must be even to split the head dim in half");

  dim3 GRID(S/Br, Hq, B);
  dim3 BLOCK(288);   // 128 half-A + 128 half-B + 32 producer

  static bool cfgd = false;
  static CUtensorMap Ktmap_half, Vtmap_half;
  if(!cfgd){
    const uint64_t kvRows = (uint64_t)B * Hkv * S;
    Ktmap_half = make_tma_2d(d_K, kvRows, (uint64_t)D, (uint32_t)(Bc / 2), (uint32_t)D);
    Vtmap_half = make_tma_2d(d_V, kvRows, (uint64_t)D, (uint32_t)Bc, (uint32_t)(D / 2));
    cfgd = true;
  }
  gqa_v26_causal<Br, Bc, D><<<GRID, BLOCK>>>(d_Q, d_O, d_LSE, Ktmap_half, Vtmap_half,
                          B, Hq, Hkv, G, S, scale);
}

// V26_diag_causal — DIAGNOSTIC launcher for gqa_v26_diag_causal (half-B gutted to a
// no-op). Same launch shape as V26_causal. Only half-A's output rows are meaningful.
template<int Br, int Bc, int D>
void launch_gqa_v26_diag_causal(
  __nv_bfloat16 *d_Q, __nv_bfloat16 *d_K, __nv_bfloat16 *d_V,
  __nv_bfloat16 *d_O, float *d_LSE,
  int B, int Hq, int Hkv, int S, int G, float scale
){
  static_assert(Br == 128, "V26_causal's per-half compute group is hardwired to 4 warps");
  static_assert(Br == Bc, "causal tile-skip + diagonal-tile mask requires Br == Bc");
  static_assert(Bc % 8 == 0, "Bc must be a multiple of 8 for tcgen05 N = 8");
  static_assert(D  % 16 == 0, "D  must be a multiple of 16 for tcgen05 dense");
  static_assert(Bc % 2 == 0, "Bc must be even to split the key range in half");
  static_assert(D  % 2 == 0, "D must be even to split the head dim in half");

  dim3 GRID(S/Br, Hq, B);
  dim3 BLOCK(288);

  static bool cfgd = false;
  static CUtensorMap Ktmap_half, Vtmap_half;
  if(!cfgd){
    const uint64_t kvRows = (uint64_t)B * Hkv * S;
    Ktmap_half = make_tma_2d(d_K, kvRows, (uint64_t)D, (uint32_t)(Bc / 2), (uint32_t)D);
    Vtmap_half = make_tma_2d(d_V, kvRows, (uint64_t)D, (uint32_t)Bc, (uint32_t)(D / 2));
    cfgd = true;
  }
  gqa_v26_diag_causal<Br, Bc, D><<<GRID, BLOCK>>>(d_Q, d_O, d_LSE, Ktmap_half, Vtmap_half,
                          B, Hq, Hkv, G, S, scale);
}

// V27_causal — FA4/Twill 3-warpgroup architecture: 2 softmax warpgroups + 1 shared
// rescale warpgroup (owns P@V MMA/readout/O-rescale for both sub-tiles) + producer.
// 416 threads/CTA (128+128+128+32). Same non-persistent grid as V22/V23/V25/V26.
template<int Br, int Bc, int D>
void launch_gqa_v27_causal(
  __nv_bfloat16 *d_Q, __nv_bfloat16 *d_K, __nv_bfloat16 *d_V,
  __nv_bfloat16 *d_O, float *d_LSE,
  int B, int Hq, int Hkv, int S, int G, float scale
){
  static_assert(Br == 128, "V27_causal's per-half compute group is hardwired to 4 warps");
  static_assert(Br == Bc, "causal tile-skip + diagonal-tile mask requires Br == Bc");
  static_assert(Bc % 8 == 0, "Bc must be a multiple of 8 for tcgen05 N = 8");
  static_assert(D  % 16 == 0, "D  must be a multiple of 16 for tcgen05 dense");
  static_assert(Bc % 2 == 0, "Bc must be even to split the key range in half");
  static_assert(D  % 2 == 0, "D must be even to split the head dim in half");

  dim3 GRID(S/Br, Hq, B);
  dim3 BLOCK(416);   // 128 softmax-A + 128 softmax-B + 128 rescale + 32 producer

  static bool cfgd = false;
  static CUtensorMap Ktmap_half, Vtmap_half;
  if(!cfgd){
    const uint64_t kvRows = (uint64_t)B * Hkv * S;
    Ktmap_half = make_tma_2d(d_K, kvRows, (uint64_t)D, (uint32_t)(Bc / 2), (uint32_t)D);
    Vtmap_half = make_tma_2d(d_V, kvRows, (uint64_t)D, (uint32_t)Bc, (uint32_t)(D / 2));
    cfgd = true;
  }
  gqa_v27_causal<Br, Bc, D><<<GRID, BLOCK>>>(d_Q, d_O, d_LSE, Ktmap_half, Vtmap_half,
                          B, Hq, Hkv, G, S, scale);
}

// TMEM-layout probe launcher — one CTA pair (q_tiles 0/1 of b=0,h=0), 128 threads.
// d_probe must hold 2*2*128*Bc floats: [crank][lane-base 0/128][lane][col].
template<int Br, int Bc, int D>
void launch_gqa_tmem_probe(
  __nv_bfloat16 *d_Q, __nv_bfloat16 *d_K, float *d_probe,
  int B, int Hkv, int S
){
  dim3 GRID(2, 1, 1);   // exactly one cluster pair
  dim3 BLOCK(128);

  static bool cfgd = false;
  static CUtensorMap Ktmap_half;
  if(!cfgd){
    const uint64_t kvRows = (uint64_t)B * Hkv * S;
    Ktmap_half = make_tma_2d(d_K, kvRows, (uint64_t)D, (uint32_t)(Bc / 2), (uint32_t)D);
    cfgd = true;
  }
  gqa_tmem_probe<Br, Bc, D><<<GRID, BLOCK>>>(d_Q, d_probe, Ktmap_half);
}

// V28_causal — flat 1D load-balanced-pair grid (1536 = B*Hq*(nTiles/2)), matching
// cuDNN's ACTUAL causal reference (cga1x1x1, no cluster). Block stays 160 threads
// (128 consumer + 32 producer) — see the kernel's header for why this deliberately
// does NOT pad to cuDNN's 512/4-warpgroup count. Ktmap3d/Vtmap are FULL (non-split)
// maps, same as V19_causal's launcher, since cta_group::1 needs no B-operand split.
template<int Br, int Bc, int D>
void launch_gqa_v28_causal(
  __nv_bfloat16 *d_Q, __nv_bfloat16 *d_K, __nv_bfloat16 *d_V,
  __nv_bfloat16 *d_O, float *d_LSE,
  int B, int Hq, int Hkv, int S, int G, float scale
){
  static_assert(Br == 128, "consumer group is hardwired to 128 threads");
  static_assert(Br == Bc, "causal tile-skip + diagonal-tile mask requires Br == Bc");
  static_assert(Bc % 8 == 0, "Bc must be a multiple of 8 for tcgen05 N = 8");
  static_assert(D  % 16 == 0, "D  must be a multiple of 16 for tcgen05 dense");
  static_assert(D  % 8  == 0, "D must be a multiple of 8 for the atom-native K TMA map");

  assert((S / Br) % 2 == 0 && "V28_causal's load-balanced pairing needs an even number of causal query tiles");

  dim3 GRID(B * Hq * ((S / Br) / 2), 1, 1);   // flat 1D — matches cuDNN's (1536,1,1)
  dim3 BLOCK(512);   // matches cuDNN's block size — see kernel header: tid 160-511
                      // are NOT yet given real work by the kv-loop's producer/
                      // consumer split (unchanged from the 160-thread version)

  static bool cfgd = false;
  static CUtensorMap Ktmap3d, Vtmap;
  if(!cfgd){
    const uint64_t kvRows = (uint64_t)B * Hkv * S;
    Ktmap3d = make_tma_3d_katom(d_K, kvRows, (uint64_t)D, (uint32_t)Bc);
    Vtmap   = make_tma_2d(d_V, kvRows, (uint64_t)D, (uint32_t)Bc, (uint32_t)D);
    cfgd = true;
  }
  gqa_v28_causal<Br, Bc, D><<<GRID, BLOCK>>>(d_Q, d_O, d_LSE, Ktmap3d, Vtmap,
                          B, Hq, Hkv, G, S, scale);
}

// =================================
//  gqa_v29_causal — V28 + widened V-reorder-copy (V25's trick), now actually using
//  the 512-thread block V28 only nominally launched with. tid 0-127 stay the
//  compute/consumer group; tid 128-159 stay the producer warp (unchanged from
//  V28); tid 160-511 (352 threads) join ONLY for the V reorder-copy step via
//  reorder_sync() (bar id 2, 480 participants = 128 compute + 352 helper) — K
//  needs no widening since it's atom-native (no reorder at all, per V19/28).
//  Everything else (2-pass sequential lo/hi, fp32 sS, atom-native K, causal mask)
//  is unchanged from V28.
//
//  KNOWN RISK, inherited from V25: V25 (which widened BOTH K and V reorder under
//  a similar joint-barrier scheme) showed a NON-DETERMINISTIC correctness failure
//  on hardware (LSE errors varying 2.152e-3/7.2e-4/1.7e-3/1.03e-3 across reruns)
//  whose exact mechanism was never fully pinned down before the team moved on to
//  the ping-pong architecture instead of fixing it. This version's structure
//  differs in some particulars (separate K/V load-bars instead of one combined
//  bar; only V needs reorder here) but is close enough in spirit that the SAME
//  risk applies until proven otherwise. Run correctness 3-5x (matching how the
//  V27 cross-CTA race was actually caught) before trusting a single passing run —
//  do NOT treat one clean pass as proof this is race-free.
// =================================
template<int Br, int Bc, int D>
__global__ void gqa_v29_causal(
  __nv_bfloat16 *d_Q,
  __nv_bfloat16 *d_O,
  float *d_LSE,
  const __grid_constant__ CUtensorMap Ktmap3d,
  const __grid_constant__ CUtensorMap Vtmap,
  int B,
  int Hq,
  int Hkv,
  int G,
  int S,
  float scale
){
  static_assert(Br == 128, "consumer group is hardwired to 128 threads (TMEM readout needs warps 0-3)");
  static_assert(Br == Bc, "causal tile-skip + diagonal-tile mask requires Br == Bc");
  constexpr int NREORDER = 480;   // 128 compute + 352 helper participants in the V reorder copy
  // sS row stride padded past Bc: each thread owns a full row (stride Bc), and Bc=128 is an
  // exact multiple of the 32-bank shared-memory cycle, so every thread in a warp lands on the
  // SAME bank on every sS access (a 32-way conflict) -- NCU-confirmed (~181M excessive
  // wavefronts at this site alone). +1 makes the stride coprime with 32, which zeroes the
  // conflict outright (not just reduces it) at minimal smem cost -- +8 was tried first but
  // pushed total smem 2.6KB over the 232448-byte hard cap. Padding columns are never read
  // (all loops bound on Bc, not Bc_pad).
  constexpr int Bc_pad = Bc + 1;
  // Same fix, same reasoning, applied to sO: tmem_readout_accum_vec's P@V-output write uses
  // smem_stride=D=64 (also a multiple of 32) -- 32 consecutive lanes (rows) collide on one
  // bank. sO is a pure CUDA-core accumulator, never an MMA operand, so unlike sV it's safe
  // to pad freely.
  constexpr int D_pad = D + 1;

  const int tid = threadIdx.x;   // 0..511: 0..127 compute, 128..159 producer, 160..511 helper

  const int nTiles     = S / Br;
  const int pairsPerBH = nTiles / 2;
  const int idx  = blockIdx.x;
  const int b    = idx / (Hq * pairsPerBH);
  const int rem  = idx - b * (Hq * pairsPerBH);
  const int hq   = rem / pairsPerBH;
  const int pair = rem - hq * pairsPerBH;
  const int hkv  = hq / G;
  const int kvRow0 = (b * Hkv + hkv) * S;

  const float scale_l2e = scale * 1.4426950408889634f;

  __shared__ __align__(16)  __nv_bfloat16 sQ[Br * D];
  __shared__ __align__(128) __nv_bfloat16 sK[2][Bc * D];
  __shared__ __align__(128) __nv_bfloat16 sVstage[2][Bc * D];
  __shared__ __align__(16)  __nv_bfloat16 sV[Bc * D];
  __shared__ __align__(16)  __nv_bfloat16 sP[Br * Bc];
  __shared__ __align__(16)  float         sS[Br * Bc_pad];
  __shared__ __align__(16)  float         sO[Br * D_pad];
  __shared__ float sm[Br];
  __shared__ float sl[Br];
  __shared__ float sCorr[Br];
  __shared__ __align__(8) uint64_t s_mma_bar;
  __shared__ __align__(8) uint64_t s_load_bar_K[2];
  __shared__ __align__(8) uint64_t s_free_bar_K[2];
  __shared__ __align__(8) uint64_t s_load_bar_V[2];
  __shared__ __align__(8) uint64_t s_free_bar_V[2];

  const uint32_t mma_bar = (uint32_t)__cvta_generic_to_shared(&s_mma_bar);
  const uint32_t lbarK0  = (uint32_t)__cvta_generic_to_shared(&s_load_bar_K[0]);
  const uint32_t lbarK1  = (uint32_t)__cvta_generic_to_shared(&s_load_bar_K[1]);
  const uint32_t fbarK0  = (uint32_t)__cvta_generic_to_shared(&s_free_bar_K[0]);
  const uint32_t fbarK1  = (uint32_t)__cvta_generic_to_shared(&s_free_bar_K[1]);
  const uint32_t lbarV0  = (uint32_t)__cvta_generic_to_shared(&s_load_bar_V[0]);
  const uint32_t lbarV1  = (uint32_t)__cvta_generic_to_shared(&s_load_bar_V[1]);
  const uint32_t fbarV0  = (uint32_t)__cvta_generic_to_shared(&s_free_bar_V[0]);
  const uint32_t fbarV1  = (uint32_t)__cvta_generic_to_shared(&s_free_bar_V[1]);
  const uint32_t lbarK[2] = {lbarK0, lbarK1};
  const uint32_t fbarK[2] = {fbarK0, fbarK1};
  const uint32_t lbarV[2] = {lbarV0, lbarV1};
  const uint32_t fbarV[2] = {fbarV0, fbarV1};

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

  const uint32_t TX = (uint32_t)Bc * (uint32_t)D * (uint32_t)sizeof(__nv_bfloat16);
  const uint32_t sK_addr[2] = {
    (uint32_t)__cvta_generic_to_shared(sK[0]),
    (uint32_t)__cvta_generic_to_shared(sK[1])
  };
  const uint32_t sVstage_addr[2] = {
    (uint32_t)__cvta_generic_to_shared(sVstage[0]),
    (uint32_t)__cvta_generic_to_shared(sVstage[1])
  };

  for(int pass = 0; pass < 2; ++pass){
    const int q_tile   = (pass == 0) ? pair : (nTiles - 1 - pair);
    const int q_row0   = q_tile * Br;
    const int nKVTiles = q_tile + 1;

    const long qBase = ((long)(b * Hq + hq) * S + q_row0) * D;
    const long lBase = ((long)(b * Hq + hq) * S + q_row0);

    for(int i = tid; i < Br * D; i += blockDim.x){
      const int r = i / D, c = i % D;
      sQ[canon_idx(r, c, Br)] = d_Q[qBase + i];
      sO[r * D_pad + c] = 0.0f;
    }
    if(tid < Br){ sm[tid] = -INFINITY; sl[tid] = 0.0f; }

    if(tid == 0){
      mbar_init(mma_bar, 1);
      mbar_init(lbarK0, 1); mbar_init(lbarK1, 1);
      mbar_init(fbarK0, 1); mbar_init(fbarK1, 1);
      mbar_init(lbarV0, 1); mbar_init(lbarV1, 1);
      mbar_init(fbarV0, 1); mbar_init(fbarV1, 1);
    }
    __syncthreads();

    if(tid >= 128 && tid < 160){
      // ---- Producer warp: unchanged from V28. ----
      if(tid == 128){
        int free_phase_K[2] = {0, 0};
        int free_phase_V[2] = {0, 0};
        for(int kc = 0; kc < nKVTiles; ++kc){
          const int slot = kc & 1;
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
    } else if(tid >= 160){
      // ---- Reorder-helper group: widens the V reorder-copy alongside compute. ----
      const int pidx = tid - 32;   // maps tid [160,511] -> contiguous [128,479]
      int load_phase_V[2] = {0, 0};
      for(int kc = 0; kc < nKVTiles; ++kc){
        const int slot = kc & 1;
        mbar_wait(lbarV[slot], load_phase_V[slot]); load_phase_V[slot] ^= 1;
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        reorder_sync();
        for(int i = pidx; i < Bc * D; i += NREORDER){
          const int bc = i / D, d = i % D;
          sV[canon_idx(d, bc, D)] = sVstage[slot][i];
        }
        reorder_sync();
        // Helpers take no further part this iteration.
      }
    } else {
      // ---- Compute group (tid < 128): same math as V28; only the V reorder-copy
      // sync scope widens (reorder_sync instead of consumer_sync) to include helpers.
      const int pidx = tid;   // maps tid [0,127] -> contiguous [0,127]
      int mbar_phase = 0;
      int load_phase_K[2] = {0, 0};
      int load_phase_V[2] = {0, 0};
      const uint64_t descQ_base = make_smem_desc(sQ, Br);

      for(int kc = 0; kc < nKVTiles; ++kc){
        const int slot = kc & 1;
        mbar_wait(lbarK[slot], load_phase_K[slot]); load_phase_K[slot] ^= 1;
        mbar_wait(lbarV[slot], load_phase_V[slot]); load_phase_V[slot] ^= 1;
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        reorder_sync();

        for(int i = pidx; i < Bc * D; i += NREORDER){
          const int bc = i / D, d = i % D;
          sV[canon_idx(d, bc, D)] = sVstage[slot][i];
        }
        reorder_sync();
        if(tid == 0) mbar_arrive(fbarV[slot]);

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
          if(tid == 0) mbar_arrive(fbarK[slot]);
          tmem_readout_to_smem_vec_2cta(sS, tmem_addr, Br, Bc, Bc_pad, scale_l2e, 0u);
          asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");

          if(kc == q_tile){
            for(int j = tid + 1; j < Bc; ++j) sS[tid * Bc_pad + j] = -INFINITY;
          }
          consumer_sync();
        }

        {
          const float m_old = sm[tid];
          const float l_old = sl[tid];

          float tile_max = -INFINITY;
          int j = 0;
          for(; j + 2 < Bc; j += 3)
            tile_max = fmaxf(tile_max,
                             fmaxf(sS[tid * Bc_pad + j],
                                   fmaxf(sS[tid * Bc_pad + j + 1], sS[tid * Bc_pad + j + 2])));
          for(; j < Bc; ++j) tile_max = fmaxf(tile_max, sS[tid * Bc_pad + j]);

          const float m_new = fmaxf(m_old, tile_max);
          const float corr  = ex2_approx(m_old - m_new);

          float p_sum = 0.0f;
          for(int j2 = 0; j2 < Bc; j2 += 2){
            const float p0 = ex2_approx(sS[tid * Bc_pad + j2]     - m_new);
            const float p1 = ex2_approx(sS[tid * Bc_pad + j2 + 1] - m_new);
            *reinterpret_cast<__nv_bfloat162*>(&sP[canon_idx(tid, j2, Br)]) =
                __floats2bfloat162_rn(p0, p1);
            p_sum += p0 + p1;
          }
          sm[tid] = m_new; sl[tid] = l_old * corr + p_sum; sCorr[tid] = corr;
        }
        consumer_sync();

        for(int i = tid; i < Br * D; i += Br){ const int r = i / D, c = i % D; sO[r * D_pad + c] *= sCorr[r]; }
        consumer_sync();

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
          tmem_readout_accum_vec(sO, tmem_addr, Br, D, D_pad);
          asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
          consumer_sync();
        }
      } // end kv loop
    }

    __syncthreads();

    for(int i = 2 * tid; i < Br * D; i += 2 * blockDim.x){
      const int r = i / D, c = i % D;
      const float denom = sl[r];
      *reinterpret_cast<__nv_bfloat162*>(&d_O[qBase + i]) =
          __floats2bfloat162_rn(sO[r * D_pad + c] / denom, sO[r * D_pad + c + 1] / denom);
    }
    if(tid < Br)
      d_LSE[lBase + tid] = 0.6931471805599453f * (sm[tid] + log2f(sl[tid]));

    __syncthreads();
  } // end pass loop (q_tile_lo, then q_tile_hi)

  if(tid < 32)
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(tmem_addr), "r"(NCOLS) : "memory");

} // end of gqa_v29_causal

// V29_causal launcher — same grid as V28, but a genuinely-populated 512-thread
// block (128 compute + 32 producer + 352 reorder-helper), matching cuDNN's block
// size for real this time (unlike V28's 160-thread block padded to nothing).
template<int Br, int Bc, int D>
void launch_gqa_v29_causal(
  __nv_bfloat16 *d_Q, __nv_bfloat16 *d_K, __nv_bfloat16 *d_V,
  __nv_bfloat16 *d_O, float *d_LSE,
  int B, int Hq, int Hkv, int S, int G, float scale
){
  static_assert(Br == 128, "consumer group is hardwired to 128 threads");
  static_assert(Br == Bc, "causal tile-skip + diagonal-tile mask requires Br == Bc");
  static_assert(Bc % 8 == 0, "Bc must be a multiple of 8 for tcgen05 N = 8");
  static_assert(D  % 16 == 0, "D  must be a multiple of 16 for tcgen05 dense");
  static_assert(D  % 8  == 0, "D must be a multiple of 8 for the atom-native K TMA map");

  assert((S / Br) % 2 == 0 && "V29_causal's load-balanced pairing needs an even number of causal query tiles");

  dim3 GRID(B * Hq * ((S / Br) / 2), 1, 1);
  dim3 BLOCK(512);

  static bool cfgd = false;
  static CUtensorMap Ktmap3d, Vtmap;
  if(!cfgd){
    const uint64_t kvRows = (uint64_t)B * Hkv * S;
    Ktmap3d = make_tma_3d_katom(d_K, kvRows, (uint64_t)D, (uint32_t)Bc);
    Vtmap   = make_tma_2d(d_V, kvRows, (uint64_t)D, (uint32_t)Bc, (uint32_t)D);
    cfgd = true;
  }
  gqa_v29_causal<Br, Bc, D><<<GRID, BLOCK>>>(d_Q, d_O, d_LSE, Ktmap3d, Vtmap,
                          B, Hq, Hkv, G, S, scale);
}

// =================================
//  gqa_v32_causal — V29 (frozen at 2.0807ms, sS+sO bank-conflict fixes only) +
//  a vectorized-store fix for the LAST known bank conflict: the V reorder-copy's
//  sV[canon_idx(d,bc,D)] write. canon_idx(d,bc,64)=(bc/8)*512+d*8+(bc%8); since
//  bf16 is 2 bytes, 2 elements share one 4-byte bank word, so bank=floor(idx/2)%32.
//  The original scheme (32 threads, fixed bc, d=0..31) hits only 8 distinct banks
//  (4-way conflict) -- PROVABLY unfixable via lane relabeling alone, since bank
//  conflict depends only on the address SET touched per wave, and a fully-
//  coalesced-source wave (32 consecutive i, i.e. one bc) always yields that same
//  8-bank set regardless of which lane writes which address.
//  This version instead changes WHICH elements form a wave: canon_idx(d,bc) and
//  canon_idx(d,bc+1) are ADJACENT (differ by 1) whenever bc%8 is even and <7 --
//  verified this pairing gives a TRUE bijection onto all 32 banks (0 conflict),
//  via a bfloat162 vectorized store. Cost: the two source reads per thread
//  (sVstage[bc*D+d], sVstage[(bc+1)*D+d]) are D=64 elements (128 bytes) apart,
//  and consecutive lanes within a d-group jump by 128 elements (256 bytes) --
//  WORSE source locality than the (reverted) first attempt's 64-element jumps.
//  Built as an isolated new version, NOT applied to V29, specifically so V29
//  stays a clean, unmodified reference point regardless of this experiment's
//  outcome -- if this regresses too, the conclusion is that this canon_idx
//  conflict is a genuine hardware/algorithm floor, not a lane-assignment bug.
// =================================
template<int Br, int Bc, int D>
__global__ void gqa_v32_causal(
  __nv_bfloat16 *d_Q,
  __nv_bfloat16 *d_O,
  float *d_LSE,
  const __grid_constant__ CUtensorMap Ktmap3d,
  const __grid_constant__ CUtensorMap Vtmap,
  int B,
  int Hq,
  int Hkv,
  int G,
  int S,
  float scale
){
  static_assert(Br == 128, "consumer group is hardwired to 128 threads (TMEM readout needs warps 0-3)");
  static_assert(Br == Bc, "causal tile-skip + diagonal-tile mask requires Br == Bc");
  constexpr int NREORDER = 480;   // 128 compute + 352 helper participants in the V reorder copy
  // sS row stride padded past Bc: each thread owns a full row (stride Bc), and Bc=128 is an
  // exact multiple of the 32-bank shared-memory cycle, so every thread in a warp lands on the
  // SAME bank on every sS access (a 32-way conflict) -- NCU-confirmed (~181M excessive
  // wavefronts at this site alone). +1 makes the stride coprime with 32, which zeroes the
  // conflict outright (not just reduces it) at minimal smem cost -- +8 was tried first but
  // pushed total smem 2.6KB over the 232448-byte hard cap. Padding columns are never read
  // (all loops bound on Bc, not Bc_pad).
  constexpr int Bc_pad = Bc + 1;
  // Same fix, same reasoning, applied to sO: tmem_readout_accum_vec's P@V-output write uses
  // smem_stride=D=64 (also a multiple of 32) -- 32 consecutive lanes (rows) collide on one
  // bank. sO is a pure CUDA-core accumulator, never an MMA operand, so unlike sV it's safe
  // to pad freely.
  constexpr int D_pad = D + 1;

  const int tid = threadIdx.x;   // 0..511: 0..127 compute, 128..159 producer, 160..511 helper

  const int nTiles     = S / Br;
  const int pairsPerBH = nTiles / 2;
  const int idx  = blockIdx.x;
  const int b    = idx / (Hq * pairsPerBH);
  const int rem  = idx - b * (Hq * pairsPerBH);
  const int hq   = rem / pairsPerBH;
  const int pair = rem - hq * pairsPerBH;
  const int hkv  = hq / G;
  const int kvRow0 = (b * Hkv + hkv) * S;

  const float scale_l2e = scale * 1.4426950408889634f;

  __shared__ __align__(16)  __nv_bfloat16 sQ[Br * D];
  __shared__ __align__(128) __nv_bfloat16 sK[2][Bc * D];
  __shared__ __align__(128) __nv_bfloat16 sVstage[2][Bc * D];
  __shared__ __align__(16)  __nv_bfloat16 sV[Bc * D];
  __shared__ __align__(16)  __nv_bfloat16 sP[Br * Bc];
  __shared__ __align__(16)  float         sS[Br * Bc_pad];
  __shared__ __align__(16)  float         sO[Br * D_pad];
  __shared__ float sm[Br];
  __shared__ float sl[Br];
  __shared__ float sCorr[Br];
  __shared__ __align__(8) uint64_t s_mma_bar;
  __shared__ __align__(8) uint64_t s_load_bar_K[2];
  __shared__ __align__(8) uint64_t s_free_bar_K[2];
  __shared__ __align__(8) uint64_t s_load_bar_V[2];
  __shared__ __align__(8) uint64_t s_free_bar_V[2];

  const uint32_t mma_bar = (uint32_t)__cvta_generic_to_shared(&s_mma_bar);
  const uint32_t lbarK0  = (uint32_t)__cvta_generic_to_shared(&s_load_bar_K[0]);
  const uint32_t lbarK1  = (uint32_t)__cvta_generic_to_shared(&s_load_bar_K[1]);
  const uint32_t fbarK0  = (uint32_t)__cvta_generic_to_shared(&s_free_bar_K[0]);
  const uint32_t fbarK1  = (uint32_t)__cvta_generic_to_shared(&s_free_bar_K[1]);
  const uint32_t lbarV0  = (uint32_t)__cvta_generic_to_shared(&s_load_bar_V[0]);
  const uint32_t lbarV1  = (uint32_t)__cvta_generic_to_shared(&s_load_bar_V[1]);
  const uint32_t fbarV0  = (uint32_t)__cvta_generic_to_shared(&s_free_bar_V[0]);
  const uint32_t fbarV1  = (uint32_t)__cvta_generic_to_shared(&s_free_bar_V[1]);
  const uint32_t lbarK[2] = {lbarK0, lbarK1};
  const uint32_t fbarK[2] = {fbarK0, fbarK1};
  const uint32_t lbarV[2] = {lbarV0, lbarV1};
  const uint32_t fbarV[2] = {fbarV0, fbarV1};

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

  const uint32_t TX = (uint32_t)Bc * (uint32_t)D * (uint32_t)sizeof(__nv_bfloat16);
  const uint32_t sK_addr[2] = {
    (uint32_t)__cvta_generic_to_shared(sK[0]),
    (uint32_t)__cvta_generic_to_shared(sK[1])
  };
  const uint32_t sVstage_addr[2] = {
    (uint32_t)__cvta_generic_to_shared(sVstage[0]),
    (uint32_t)__cvta_generic_to_shared(sVstage[1])
  };

  for(int pass = 0; pass < 2; ++pass){
    const int q_tile   = (pass == 0) ? pair : (nTiles - 1 - pair);
    const int q_row0   = q_tile * Br;
    const int nKVTiles = q_tile + 1;

    const long qBase = ((long)(b * Hq + hq) * S + q_row0) * D;
    const long lBase = ((long)(b * Hq + hq) * S + q_row0);

    for(int i = tid; i < Br * D; i += blockDim.x){
      const int r = i / D, c = i % D;
      sQ[canon_idx(r, c, Br)] = d_Q[qBase + i];
      sO[r * D_pad + c] = 0.0f;
    }
    if(tid < Br){ sm[tid] = -INFINITY; sl[tid] = 0.0f; }

    if(tid == 0){
      mbar_init(mma_bar, 1);
      mbar_init(lbarK0, 1); mbar_init(lbarK1, 1);
      mbar_init(fbarK0, 1); mbar_init(fbarK1, 1);
      mbar_init(lbarV0, 1); mbar_init(lbarV1, 1);
      mbar_init(fbarV0, 1); mbar_init(fbarV1, 1);
    }
    __syncthreads();

    // Bank-conflict-free (verified: true 0-way, not just improved) V reorder-copy.
    // See header comment above for the derivation.
    constexpr int DCoarseGroups  = D / 8;             // 8   (D=64)
    constexpr int BcChunkGroups  = Bc / 8;            // 16  (Bc=128)
    constexpr int NSlicesV2      = DCoarseGroups * BcChunkGroups;  // 128
    constexpr int NWarpsReorder2 = NREORDER / 32;                 // 15
    auto reorderCopyV2 = [&](int pidx, int slot){
      const int warp_local = pidx / 32;
      const int lane        = pidx % 32;
      const int d_frac      = lane / 4;   // 0..7
      const int bc_pair_idx = lane % 4;   // 0..3
      for(int slice = warp_local; slice < NSlicesV2; slice += NWarpsReorder2){
        const int d_coarse = slice / BcChunkGroups;
        const int bc_chunk = slice % BcChunkGroups;
        const int d        = d_coarse * 8 + d_frac;
        const int bc_even  = bc_chunk * 8 + 2 * bc_pair_idx;
        __nv_bfloat162 packed;
        packed.x = sVstage[slot][bc_even * D + d];
        packed.y = sVstage[slot][(bc_even + 1) * D + d];
        *reinterpret_cast<__nv_bfloat162*>(&sV[canon_idx(d, bc_even, D)]) = packed;
      }
    };

    if(tid >= 128 && tid < 160){
      // ---- Producer warp: unchanged from V28. ----
      if(tid == 128){
        int free_phase_K[2] = {0, 0};
        int free_phase_V[2] = {0, 0};
        for(int kc = 0; kc < nKVTiles; ++kc){
          const int slot = kc & 1;
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
    } else if(tid >= 160){
      // ---- Reorder-helper group: widens the V reorder-copy alongside compute. ----
      const int pidx = tid - 32;   // maps tid [160,511] -> contiguous [128,479]
      int load_phase_V[2] = {0, 0};
      for(int kc = 0; kc < nKVTiles; ++kc){
        const int slot = kc & 1;
        mbar_wait(lbarV[slot], load_phase_V[slot]); load_phase_V[slot] ^= 1;
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        reorder_sync();
        reorderCopyV2(pidx, slot);
        reorder_sync();
        // Helpers take no further part this iteration.
      }
    } else {
      // ---- Compute group (tid < 128): same math as V28; only the V reorder-copy
      // sync scope widens (reorder_sync instead of consumer_sync) to include helpers.
      const int pidx = tid;   // maps tid [0,127] -> contiguous [0,127]
      int mbar_phase = 0;
      int load_phase_K[2] = {0, 0};
      int load_phase_V[2] = {0, 0};
      const uint64_t descQ_base = make_smem_desc(sQ, Br);

      for(int kc = 0; kc < nKVTiles; ++kc){
        const int slot = kc & 1;
        mbar_wait(lbarK[slot], load_phase_K[slot]); load_phase_K[slot] ^= 1;
        mbar_wait(lbarV[slot], load_phase_V[slot]); load_phase_V[slot] ^= 1;
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        reorder_sync();

        reorderCopyV2(pidx, slot);
        reorder_sync();
        if(tid == 0) mbar_arrive(fbarV[slot]);

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
          if(tid == 0) mbar_arrive(fbarK[slot]);
          tmem_readout_to_smem_vec_2cta(sS, tmem_addr, Br, Bc, Bc_pad, scale_l2e, 0u);
          asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");

          if(kc == q_tile){
            for(int j = tid + 1; j < Bc; ++j) sS[tid * Bc_pad + j] = -INFINITY;
          }
          consumer_sync();
        }

        {
          const float m_old = sm[tid];
          const float l_old = sl[tid];

          float tile_max = -INFINITY;
          int j = 0;
          for(; j + 2 < Bc; j += 3)
            tile_max = fmaxf(tile_max,
                             fmaxf(sS[tid * Bc_pad + j],
                                   fmaxf(sS[tid * Bc_pad + j + 1], sS[tid * Bc_pad + j + 2])));
          for(; j < Bc; ++j) tile_max = fmaxf(tile_max, sS[tid * Bc_pad + j]);

          const float m_new = fmaxf(m_old, tile_max);
          const float corr  = ex2_approx(m_old - m_new);

          float p_sum = 0.0f;
          for(int j2 = 0; j2 < Bc; j2 += 2){
            const float p0 = ex2_approx(sS[tid * Bc_pad + j2]     - m_new);
            const float p1 = ex2_approx(sS[tid * Bc_pad + j2 + 1] - m_new);
            *reinterpret_cast<__nv_bfloat162*>(&sP[canon_idx(tid, j2, Br)]) =
                __floats2bfloat162_rn(p0, p1);
            p_sum += p0 + p1;
          }
          sm[tid] = m_new; sl[tid] = l_old * corr + p_sum; sCorr[tid] = corr;
        }
        consumer_sync();

        for(int i = tid; i < Br * D; i += Br){ const int r = i / D, c = i % D; sO[r * D_pad + c] *= sCorr[r]; }
        consumer_sync();

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
          tmem_readout_accum_vec(sO, tmem_addr, Br, D, D_pad);
          asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
          consumer_sync();
        }
      } // end kv loop
    }

    __syncthreads();

    for(int i = 2 * tid; i < Br * D; i += 2 * blockDim.x){
      const int r = i / D, c = i % D;
      const float denom = sl[r];
      *reinterpret_cast<__nv_bfloat162*>(&d_O[qBase + i]) =
          __floats2bfloat162_rn(sO[r * D_pad + c] / denom, sO[r * D_pad + c + 1] / denom);
    }
    if(tid < Br)
      d_LSE[lBase + tid] = 0.6931471805599453f * (sm[tid] + log2f(sl[tid]));

    __syncthreads();
  } // end pass loop (q_tile_lo, then q_tile_hi)

  if(tid < 32)
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(tmem_addr), "r"(NCOLS) : "memory");

} // end of gqa_v32_causal

// V29_causal launcher — same grid as V28, but a genuinely-populated 512-thread
// block (128 compute + 32 producer + 352 reorder-helper), matching cuDNN's block
// size for real this time (unlike V28's 160-thread block padded to nothing).
template<int Br, int Bc, int D>
void launch_gqa_v32_causal(
  __nv_bfloat16 *d_Q, __nv_bfloat16 *d_K, __nv_bfloat16 *d_V,
  __nv_bfloat16 *d_O, float *d_LSE,
  int B, int Hq, int Hkv, int S, int G, float scale
){
  static_assert(Br == 128, "consumer group is hardwired to 128 threads");
  static_assert(Br == Bc, "causal tile-skip + diagonal-tile mask requires Br == Bc");
  static_assert(Bc % 8 == 0, "Bc must be a multiple of 8 for tcgen05 N = 8");
  static_assert(D  % 16 == 0, "D  must be a multiple of 16 for tcgen05 dense");
  static_assert(D  % 8  == 0, "D must be a multiple of 8 for the atom-native K TMA map");

  assert((S / Br) % 2 == 0 && "V29_causal's load-balanced pairing needs an even number of causal query tiles");

  dim3 GRID(B * Hq * ((S / Br) / 2), 1, 1);
  dim3 BLOCK(512);

  static bool cfgd = false;
  static CUtensorMap Ktmap3d, Vtmap;
  if(!cfgd){
    const uint64_t kvRows = (uint64_t)B * Hkv * S;
    Ktmap3d = make_tma_3d_katom(d_K, kvRows, (uint64_t)D, (uint32_t)Bc);
    Vtmap   = make_tma_2d(d_V, kvRows, (uint64_t)D, (uint32_t)Bc, (uint32_t)D);
    cfgd = true;
  }
  gqa_v32_causal<Br, Bc, D><<<GRID, BLOCK>>>(d_Q, d_O, d_LSE, Ktmap3d, Vtmap,
                          B, Hq, Hkv, G, S, scale);
}


// =================================
//  gqa_v33_causal — V32 (frozen at 2.0463ms) + two intra-iteration MMA-issue
//  reorderings (see memory Stage 16 for full derivation).
// =================================
template<int Br, int Bc, int D>
__global__ void gqa_v33_causal(
  __nv_bfloat16 *d_Q,
  __nv_bfloat16 *d_O,
  float *d_LSE,
  const __grid_constant__ CUtensorMap Ktmap3d,
  const __grid_constant__ CUtensorMap Vtmap,
  int B,
  int Hq,
  int Hkv,
  int G,
  int S,
  float scale
){
  static_assert(Br == 128, "consumer group is hardwired to 128 threads (TMEM readout needs warps 0-3)");
  static_assert(Br == Bc, "causal tile-skip + diagonal-tile mask requires Br == Bc");
  constexpr int NREORDER = 480;   // 128 compute + 352 helper participants in the V reorder copy
  // sS row stride padded past Bc: each thread owns a full row (stride Bc), and Bc=128 is an
  // exact multiple of the 32-bank shared-memory cycle, so every thread in a warp lands on the
  // SAME bank on every sS access (a 32-way conflict) -- NCU-confirmed (~181M excessive
  // wavefronts at this site alone). +1 makes the stride coprime with 32, which zeroes the
  // conflict outright (not just reduces it) at minimal smem cost -- +8 was tried first but
  // pushed total smem 2.6KB over the 232448-byte hard cap. Padding columns are never read
  // (all loops bound on Bc, not Bc_pad).
  constexpr int Bc_pad = Bc + 1;
  // Same fix, same reasoning, applied to sO: tmem_readout_accum_vec's P@V-output write uses
  // smem_stride=D=64 (also a multiple of 32) -- 32 consecutive lanes (rows) collide on one
  // bank. sO is a pure CUDA-core accumulator, never an MMA operand, so unlike sV it's safe
  // to pad freely.
  constexpr int D_pad = D + 1;

  const int tid = threadIdx.x;   // 0..511: 0..127 compute, 128..159 producer, 160..511 helper

  const int nTiles     = S / Br;
  const int pairsPerBH = nTiles / 2;
  const int idx  = blockIdx.x;
  const int b    = idx / (Hq * pairsPerBH);
  const int rem  = idx - b * (Hq * pairsPerBH);
  const int hq   = rem / pairsPerBH;
  const int pair = rem - hq * pairsPerBH;
  const int hkv  = hq / G;
  const int kvRow0 = (b * Hkv + hkv) * S;

  const float scale_l2e = scale * 1.4426950408889634f;

  __shared__ __align__(16)  __nv_bfloat16 sQ[Br * D];
  __shared__ __align__(128) __nv_bfloat16 sK[2][Bc * D];
  __shared__ __align__(128) __nv_bfloat16 sVstage[2][Bc * D];
  __shared__ __align__(16)  __nv_bfloat16 sV[Bc * D];
  __shared__ __align__(16)  __nv_bfloat16 sP[Br * Bc];
  __shared__ __align__(16)  float         sS[Br * Bc_pad];
  __shared__ __align__(16)  float         sO[Br * D_pad];
  __shared__ float sm[Br];
  __shared__ float sl[Br];
  __shared__ float sCorr[Br];
  __shared__ __align__(8) uint64_t s_mma_bar;
  __shared__ __align__(8) uint64_t s_load_bar_K[2];
  __shared__ __align__(8) uint64_t s_free_bar_K[2];
  __shared__ __align__(8) uint64_t s_load_bar_V[2];
  __shared__ __align__(8) uint64_t s_free_bar_V[2];

  const uint32_t mma_bar = (uint32_t)__cvta_generic_to_shared(&s_mma_bar);
  const uint32_t lbarK0  = (uint32_t)__cvta_generic_to_shared(&s_load_bar_K[0]);
  const uint32_t lbarK1  = (uint32_t)__cvta_generic_to_shared(&s_load_bar_K[1]);
  const uint32_t fbarK0  = (uint32_t)__cvta_generic_to_shared(&s_free_bar_K[0]);
  const uint32_t fbarK1  = (uint32_t)__cvta_generic_to_shared(&s_free_bar_K[1]);
  const uint32_t lbarV0  = (uint32_t)__cvta_generic_to_shared(&s_load_bar_V[0]);
  const uint32_t lbarV1  = (uint32_t)__cvta_generic_to_shared(&s_load_bar_V[1]);
  const uint32_t fbarV0  = (uint32_t)__cvta_generic_to_shared(&s_free_bar_V[0]);
  const uint32_t fbarV1  = (uint32_t)__cvta_generic_to_shared(&s_free_bar_V[1]);
  const uint32_t lbarK[2] = {lbarK0, lbarK1};
  const uint32_t fbarK[2] = {fbarK0, fbarK1};
  const uint32_t lbarV[2] = {lbarV0, lbarV1};
  const uint32_t fbarV[2] = {fbarV0, fbarV1};

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

  const uint32_t TX = (uint32_t)Bc * (uint32_t)D * (uint32_t)sizeof(__nv_bfloat16);
  const uint32_t sK_addr[2] = {
    (uint32_t)__cvta_generic_to_shared(sK[0]),
    (uint32_t)__cvta_generic_to_shared(sK[1])
  };
  const uint32_t sVstage_addr[2] = {
    (uint32_t)__cvta_generic_to_shared(sVstage[0]),
    (uint32_t)__cvta_generic_to_shared(sVstage[1])
  };

  for(int pass = 0; pass < 2; ++pass){
    const int q_tile   = (pass == 0) ? pair : (nTiles - 1 - pair);
    const int q_row0   = q_tile * Br;
    const int nKVTiles = q_tile + 1;

    const long qBase = ((long)(b * Hq + hq) * S + q_row0) * D;
    const long lBase = ((long)(b * Hq + hq) * S + q_row0);

    for(int i = tid; i < Br * D; i += blockDim.x){
      const int r = i / D, c = i % D;
      sQ[canon_idx(r, c, Br)] = d_Q[qBase + i];
      sO[r * D_pad + c] = 0.0f;
    }
    if(tid < Br){ sm[tid] = -INFINITY; sl[tid] = 0.0f; }

    if(tid == 0){
      mbar_init(mma_bar, 1);
      mbar_init(lbarK0, 1); mbar_init(lbarK1, 1);
      mbar_init(fbarK0, 1); mbar_init(fbarK1, 1);
      mbar_init(lbarV0, 1); mbar_init(lbarV1, 1);
      mbar_init(fbarV0, 1); mbar_init(fbarV1, 1);
    }
    __syncthreads();

    // Bank-conflict-free (verified: true 0-way, not just improved) V reorder-copy.
    // See header comment above for the derivation.
    constexpr int DCoarseGroups  = D / 8;             // 8   (D=64)
    constexpr int BcChunkGroups  = Bc / 8;            // 16  (Bc=128)
    constexpr int NSlicesV2      = DCoarseGroups * BcChunkGroups;  // 128
    constexpr int NWarpsReorder2 = NREORDER / 32;                 // 15
    auto reorderCopyV2 = [&](int pidx, int slot){
      const int warp_local = pidx / 32;
      const int lane        = pidx % 32;
      const int d_frac      = lane / 4;   // 0..7
      const int bc_pair_idx = lane % 4;   // 0..3
      for(int slice = warp_local; slice < NSlicesV2; slice += NWarpsReorder2){
        const int d_coarse = slice / BcChunkGroups;
        const int bc_chunk = slice % BcChunkGroups;
        const int d        = d_coarse * 8 + d_frac;
        const int bc_even  = bc_chunk * 8 + 2 * bc_pair_idx;
        __nv_bfloat162 packed;
        packed.x = sVstage[slot][bc_even * D + d];
        packed.y = sVstage[slot][(bc_even + 1) * D + d];
        *reinterpret_cast<__nv_bfloat162*>(&sV[canon_idx(d, bc_even, D)]) = packed;
      }
    };

    if(tid >= 128 && tid < 160){
      // ---- Producer warp: unchanged from V28. ----
      if(tid == 128){
        int free_phase_K[2] = {0, 0};
        int free_phase_V[2] = {0, 0};
        for(int kc = 0; kc < nKVTiles; ++kc){
          const int slot = kc & 1;
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
    } else if(tid >= 160){
      // ---- Reorder-helper group: widens the V reorder-copy alongside compute. ----
      const int pidx = tid - 32;   // maps tid [160,511] -> contiguous [128,479]
      int load_phase_V[2] = {0, 0};
      for(int kc = 0; kc < nKVTiles; ++kc){
        const int slot = kc & 1;
        mbar_wait(lbarV[slot], load_phase_V[slot]); load_phase_V[slot] ^= 1;
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        reorder_sync();
        reorderCopyV2(pidx, slot);
        reorder_sync();
        // Helpers take no further part this iteration.
      }
    } else {
      // ---- Compute group (tid < 128): same math as V28; only the V reorder-copy
      // sync scope widens (reorder_sync instead of consumer_sync) to include helpers.
      const int pidx = tid;   // maps tid [0,127] -> contiguous [0,127]
      int mbar_phase = 0;
      int load_phase_K[2] = {0, 0};
      int load_phase_V[2] = {0, 0};
      const uint64_t descQ_base = make_smem_desc(sQ, Br);

      for(int kc = 0; kc < nKVTiles; ++kc){
        const int slot = kc & 1;
        mbar_wait(lbarK[slot], load_phase_K[slot]); load_phase_K[slot] ^= 1;
        mbar_wait(lbarV[slot], load_phase_V[slot]); load_phase_V[slot] ^= 1;
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

        const uint64_t descK_base = make_smem_desc(sK[slot], Bc);
        const uint32_t idesc_qk   = make_idesc_bf16(Br, Bc);
        if(tid == 0){
          asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
          for(int kt = 0; kt < D/16; ++kt){
            uint64_t descQ = advance_desc_katom(descQ_base, kt, Br);
            uint64_t descK = advance_desc_katom(descK_base, kt, Bc);
            uint32_t accumulate = (kt > 0) ? 1u : 0u;
            asm volatile(
              "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
              "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
              :: "r"(tmem_addr), "l"(descQ), "l"(descK), "r"(idesc_qk), "r"(accumulate) : "memory");
          }
          mbar_commit_mma(mma_bar);
        }

        reorder_sync();
        reorderCopyV2(pidx, slot);
        reorder_sync();
        if(tid == 0) mbar_arrive(fbarV[slot]);

        {
          mbar_wait(mma_bar, mbar_phase); mbar_phase ^= 1;
          if(tid == 0) mbar_arrive(fbarK[slot]);
          tmem_readout_to_smem_vec_2cta(sS, tmem_addr, Br, Bc, Bc_pad, scale_l2e, 0u);
          asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");

          if(kc == q_tile){
            for(int j = tid + 1; j < Bc; ++j) sS[tid * Bc_pad + j] = -INFINITY;
          }
          consumer_sync();
        }

        {
          const float m_old = sm[tid];
          const float l_old = sl[tid];

          float tile_max = -INFINITY;
          int j = 0;
          for(; j + 2 < Bc; j += 3)
            tile_max = fmaxf(tile_max,
                             fmaxf(sS[tid * Bc_pad + j],
                                   fmaxf(sS[tid * Bc_pad + j + 1], sS[tid * Bc_pad + j + 2])));
          for(; j < Bc; ++j) tile_max = fmaxf(tile_max, sS[tid * Bc_pad + j]);

          const float m_new = fmaxf(m_old, tile_max);
          const float corr  = ex2_approx(m_old - m_new);

          float p_sum = 0.0f;
          for(int j2 = 0; j2 < Bc; j2 += 2){
            const float p0 = ex2_approx(sS[tid * Bc_pad + j2]     - m_new);
            const float p1 = ex2_approx(sS[tid * Bc_pad + j2 + 1] - m_new);
            *reinterpret_cast<__nv_bfloat162*>(&sP[canon_idx(tid, j2, Br)]) =
                __floats2bfloat162_rn(p0, p1);
            p_sum += p0 + p1;
          }
          sm[tid] = m_new; sl[tid] = l_old * corr + p_sum; sCorr[tid] = corr;
        }
        consumer_sync();

        const uint64_t descP_base = make_smem_desc(sP, Br);
        const uint64_t descV_base = make_smem_desc(sV, D);
        const uint32_t idesc_pv   = make_idesc_bf16(Br, D);
        if(tid == 0){
          asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
          for(int kt = 0; kt < Bc/16; ++kt){
            uint64_t descP = advance_desc_katom(descP_base, kt, Br);
            uint64_t descV = advance_desc_katom(descV_base, kt, D);
            uint32_t accumulate = (kt > 0) ? 1u : 0u;
            asm volatile(
              "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
              "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
              :: "r"(tmem_addr), "l"(descP), "l"(descV), "r"(idesc_pv), "r"(accumulate) : "memory");
          }
          mbar_commit_mma(mma_bar);
        }

        for(int i = tid; i < Br * D; i += Br){ const int r = i / D, c = i % D; sO[r * D_pad + c] *= sCorr[r]; }
        consumer_sync();

        {
          mbar_wait(mma_bar, mbar_phase); mbar_phase ^= 1;
          tmem_readout_accum_vec(sO, tmem_addr, Br, D, D_pad);
          asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
          consumer_sync();
        }
      } // end kv loop
    }

    __syncthreads();

    for(int i = 2 * tid; i < Br * D; i += 2 * blockDim.x){
      const int r = i / D, c = i % D;
      const float denom = sl[r];
      *reinterpret_cast<__nv_bfloat162*>(&d_O[qBase + i]) =
          __floats2bfloat162_rn(sO[r * D_pad + c] / denom, sO[r * D_pad + c + 1] / denom);
    }
    if(tid < Br)
      d_LSE[lBase + tid] = 0.6931471805599453f * (sm[tid] + log2f(sl[tid]));

    __syncthreads();
  } // end pass loop (q_tile_lo, then q_tile_hi)

  if(tid < 32)
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(tmem_addr), "r"(NCOLS) : "memory");

} // end of gqa_v33_causal

// V29_causal launcher — same grid as V28, but a genuinely-populated 512-thread
// block (128 compute + 32 producer + 352 reorder-helper), matching cuDNN's block
// size for real this time (unlike V28's 160-thread block padded to nothing).
template<int Br, int Bc, int D>
void launch_gqa_v33_causal(
  __nv_bfloat16 *d_Q, __nv_bfloat16 *d_K, __nv_bfloat16 *d_V,
  __nv_bfloat16 *d_O, float *d_LSE,
  int B, int Hq, int Hkv, int S, int G, float scale
){
  static_assert(Br == 128, "consumer group is hardwired to 128 threads");
  static_assert(Br == Bc, "causal tile-skip + diagonal-tile mask requires Br == Bc");
  static_assert(Bc % 8 == 0, "Bc must be a multiple of 8 for tcgen05 N = 8");
  static_assert(D  % 16 == 0, "D  must be a multiple of 16 for tcgen05 dense");
  static_assert(D  % 8  == 0, "D must be a multiple of 8 for the atom-native K TMA map");

  assert((S / Br) % 2 == 0 && "V29_causal's load-balanced pairing needs an even number of causal query tiles");

  dim3 GRID(B * Hq * ((S / Br) / 2), 1, 1);
  dim3 BLOCK(512);

  static bool cfgd = false;
  static CUtensorMap Ktmap3d, Vtmap;
  if(!cfgd){
    const uint64_t kvRows = (uint64_t)B * Hkv * S;
    Ktmap3d = make_tma_3d_katom(d_K, kvRows, (uint64_t)D, (uint32_t)Bc);
    Vtmap   = make_tma_2d(d_V, kvRows, (uint64_t)D, (uint32_t)Bc, (uint32_t)D);
    cfgd = true;
  }
  gqa_v33_causal<Br, Bc, D><<<GRID, BLOCK>>>(d_Q, d_O, d_LSE, Ktmap3d, Vtmap,
                          B, Hq, Hkv, G, S, scale);
}

// =================================
//  gqa_v30_causal — genuine concurrent lo/hi tile processing under V28's grid+
//  cluster shape (flat 1536-CTA load-balanced-pair grid, cta_group::1, no
//  cluster). 4 real warpgroups (512 threads), mirroring V27's FA4 role split but
//  for 2 DIFFERENT q_tiles in one CTA instead of 2 row-halves of the SAME tile:
//    softmax-lo (tid 0-127)   : q_tile_lo's own QK^T MMA + readout + mask + softmax
//    softmax-hi (tid 128-255) : q_tile_hi's own QK^T MMA + readout + mask + softmax
//    rescale    (tid 256-383) : P@V MMA + TMEM readout + O-rescale, lo THEN hi
//    loader     (tid 384-511) : shared K/V TMA + cooperative V reorder-copy, feeds
//                                BOTH tiles
//
//  Unlike V27, this needs NO cluster/crank machinery at all (single CTA,
//  cta_group::1) -- the entire cross-CTA-race class that took real effort to fix
//  in V27 simply doesn't exist here; every sync below is an ordinary local
//  mbarrier or named barrier.
//
//  The SMEM budget forces real sharing between lo and hi that V27 didn't need
//  (V27's two halves were the SAME q_tile's rows, needing the same K/V anyway;
//  here lo and hi are DIFFERENT q_tiles, so this is a genuine new compromise):
//  reusing the shared kv-loop-bound trick (nKVTiles=q_tile_hi+1 for ALL four
//  groups; softmax-lo reuses V20's "kc>own_tile -> mask whole row" case for kc in
//  (q_tile_lo,q_tile_hi]) means lo and hi need the SAME K/V content whenever both
//  still need it, so K/V staging+canonical buffers are SHARED (single copies) --
//  their READS never conflict (only the loader writes). To fit under budget, sS
//  and sP are ALSO shared (single copies) between lo/hi, which serializes the
//  readout->mask->softmax->P-write step between the two tiles each kc (only one
//  tile's worth runs at a time) -- but each tile's OWN QK^T-MMA-issue+wait (using
//  its own TMEM region) still overlaps with the OTHER tile's whole softmax block,
//  recovering FA3's original ping-pong overlap (GEMM(one tile) under EXP(other
//  tile)) even though full FA4-style per-tile-independent-buffer overlap doesn't
//  fit. K/V staging (sK/sVstage) is SINGLE-buffered here (not double-buffered like
//  V28/29's own K/V), paid for by losing load/compute prefetch overlap for K/V
//  specifically (a separate, already-tested axis — V28/29 cover the double-
//  buffered case). That alone got to ~243KB — still over sm_103a's ACTUAL static
//  shared memory ceiling, discovered empirically here: ptxas hard-caps static
//  smem at exactly 0x38c00 bytes = 227KB (below cuDNN's own reported 232.45KB,
//  which is DYNAMIC shared memory with an explicit cudaFuncAttribute opt-in — a
//  higher ceiling than static smem gets by default; this kernel doesn't use that
//  mechanism). The remaining ~16KB came from sS: fp16 instead of fp32 (32KB vs
//  64KB), reusing V19's precision trade-off — needed a new generalized fp16
//  readout helper (tmem_readout_to_smem_fp16_vec_g) for softmax-hi's tid range,
//  since the existing fp16 readout assumed the caller's group starts at warp 0.
//
//  Correctness-critical ordering (lo ALWAYS goes first each kc, hi second, fixed
//  deterministic order -- NOT a race, a designed protocol):
//    pready_lo / pready_hi : softmax-X arrives after WRITING into shared sS/sP —
//      tells rescale "P is ready to read for the PV MMA."
//    pfree_lo / pfree_hi   : rescale arrives after actually CONSUMING shared sP
//      (i.e. after its PV MMA's commit-wait returns, meaning the tensor core has
//      finished reading it) — tells the OTHER softmax group it's now safe to
//      overwrite shared sS/sP with its own tile's data. Note this is deliberately
//      NOT the same signal as pready: pready only proves softmax-X's OWN write is
//      done, not that rescale has finished READING it — gating the other tile's
//      overwrite on pready instead of pfree would reintroduce exactly the kind of
//      WAR hazard V27's cross-CTA race taught us to check for explicitly.
//    fbarK_lo/hi (dual)     : each softmax group confirms it's done reading sK
//      via its own QK^T MMA-wait; the loader needs BOTH before reusing sK (single-
//      buffered, so this gates every kc, not just kc>=2).
//    vfree_lo/hi (dual)     : rescale confirms it's done reading sV via each
//      tile's PV MMA-wait; the loader needs BOTH before reordering into sV again.
//    vready                 : loader arrives once per kc after the cooperative
//      reorder-copy is fully done (via sync_loader(), scoped to just the 128
//      loader threads) — rescale waits for it once per kc, before lo's PV MMA;
//      the same "ready" state covers hi's PV MMA later in the same kc (no new
//      write to sV happens in between, so no second wait is needed).
//
//  Recommended before trusting benchmarks: run correctness repeatedly (V27's
//  cross-CTA race and V25's reorder race were BOTH only caught by rerunning) —
//  this kernel has substantially more new synchronization than either.
// =================================
template<int Br, int Bc, int D>
__global__ void gqa_v30_causal(
  __nv_bfloat16 *d_Q,
  __nv_bfloat16 *d_O,
  float *d_LSE,
  const __grid_constant__ CUtensorMap Ktmap3d,
  const __grid_constant__ CUtensorMap Vtmap,
  int B,
  int Hq,
  int Hkv,
  int G,
  int S,
  float scale
){
  static_assert(Br == 128, "each tile-group is hardwired to 4 warps (128 threads)");
  static_assert(Br == Bc, "causal tile-skip + diagonal-tile mask requires Br == Bc");

  const int tid = threadIdx.x;   // 0..511: 0..127 softmax-lo, 128..255 softmax-hi, 256..383 rescale, 384..511 loader

  const int nTiles     = S / Br;
  const int pairsPerBH = nTiles / 2;
  const int idx  = blockIdx.x;
  const int b    = idx / (Hq * pairsPerBH);
  const int rem  = idx - b * (Hq * pairsPerBH);
  const int hq   = rem / pairsPerBH;
  const int pair = rem - hq * pairsPerBH;
  const int hkv  = hq / G;
  const int kvRow0 = (b * Hkv + hkv) * S;

  const int q_tile_lo = pair;
  const int q_tile_hi = nTiles - 1 - pair;
  const int nKVTiles   = q_tile_hi + 1;   // shared bound for ALL four groups

  const long qBaseLo = ((long)(b * Hq + hq) * S + q_tile_lo * Br) * D;
  const long qBaseHi = ((long)(b * Hq + hq) * S + q_tile_hi * Br) * D;
  const long lBaseLo = (long)(b * Hq + hq) * S + q_tile_lo * Br;
  const long lBaseHi = (long)(b * Hq + hq) * S + q_tile_hi * Br;

  const float scale_l2e = scale * 1.4426950408889634f;

  __shared__ __align__(16)  __nv_bfloat16 sQ_lo[Br * D];
  __shared__ __align__(16)  __nv_bfloat16 sQ_hi[Br * D];
  __shared__ __align__(128) __nv_bfloat16 sK[Bc * D];             // atom-native, SHARED, single-buffered
  __shared__ __align__(128) __nv_bfloat16 sVstage[Bc * D];        // raw landing, SHARED, single-buffered
  __shared__ __align__(16)  __nv_bfloat16 sV[Bc * D];             // canonical, SHARED, single-buffered
  __shared__ __align__(16)  __half        sS[Br * Bc];            // SHARED, single, fp16 — the only way to
                                                                    // fit V30 under sm_103a's 227KB static
                                                                    // smem hard cap (ptxas-enforced, discovered
                                                                    // empirically); carries V19's documented
                                                                    // fp16-at-Bc=128 LSE precision cost.
  __shared__ __align__(16)  __nv_bfloat16 sP[Br * Bc];            // SHARED, single
  __shared__ __align__(16)  float         sO_lo[Br * D];
  __shared__ __align__(16)  float         sO_hi[Br * D];
  __shared__ float sm_lo[Br], sl_lo[Br], sCorr_lo[Br];
  __shared__ float sm_hi[Br], sl_hi[Br], sCorr_hi[Br];

  __shared__ __align__(8) uint64_t s_lbarK, s_lbarV;
  __shared__ __align__(8) uint64_t s_fbarK_lo, s_fbarK_hi;
  __shared__ __align__(8) uint64_t s_vfree_lo, s_vfree_hi, s_vready;
  __shared__ __align__(8) uint64_t s_mma_lo_qk, s_mma_hi_qk, s_mma_lo_pv, s_mma_hi_pv;
  __shared__ __align__(8) uint64_t s_pready_lo, s_pready_hi, s_pfree_lo, s_pfree_hi;

  const uint32_t lbarK    = (uint32_t)__cvta_generic_to_shared(&s_lbarK);
  const uint32_t lbarV    = (uint32_t)__cvta_generic_to_shared(&s_lbarV);
  const uint32_t fbarK_lo = (uint32_t)__cvta_generic_to_shared(&s_fbarK_lo);
  const uint32_t fbarK_hi = (uint32_t)__cvta_generic_to_shared(&s_fbarK_hi);
  const uint32_t vfree_lo  = (uint32_t)__cvta_generic_to_shared(&s_vfree_lo);
  const uint32_t vfree_hi  = (uint32_t)__cvta_generic_to_shared(&s_vfree_hi);
  const uint32_t vready    = (uint32_t)__cvta_generic_to_shared(&s_vready);
  const uint32_t mma_lo_qk = (uint32_t)__cvta_generic_to_shared(&s_mma_lo_qk);
  const uint32_t mma_hi_qk = (uint32_t)__cvta_generic_to_shared(&s_mma_hi_qk);
  const uint32_t mma_lo_pv = (uint32_t)__cvta_generic_to_shared(&s_mma_lo_pv);
  const uint32_t mma_hi_pv = (uint32_t)__cvta_generic_to_shared(&s_mma_hi_pv);
  const uint32_t pready_lo = (uint32_t)__cvta_generic_to_shared(&s_pready_lo);
  const uint32_t pready_hi = (uint32_t)__cvta_generic_to_shared(&s_pready_hi);
  const uint32_t pfree_lo  = (uint32_t)__cvta_generic_to_shared(&s_pfree_lo);
  const uint32_t pfree_hi  = (uint32_t)__cvta_generic_to_shared(&s_pfree_hi);

  for(int i = tid; i < Br * D && tid < 128; i += 128){
    sQ_lo[canon_idx(i / D, i % D, Br)] = d_Q[qBaseLo + i];
    sO_lo[i] = 0.0f;
  }
  for(int i = tid - 128; i < Br * D && tid >= 128 && tid < 256; i += 128){
    sQ_hi[canon_idx(i / D, i % D, Br)] = d_Q[qBaseHi + i];
    sO_hi[i] = 0.0f;
  }
  if(tid < Br){ sm_lo[tid] = -INFINITY; sl_lo[tid] = 0.0f; }
  if(tid >= 128 && tid < 128 + Br){ sm_hi[tid-128] = -INFINITY; sl_hi[tid-128] = 0.0f; }

  if(tid == 0){
    mbar_init(lbarK, 1); mbar_init(lbarV, 1);
    mbar_init(fbarK_lo, 1); mbar_init(fbarK_hi, 1);
    mbar_init(vfree_lo, 1); mbar_init(vfree_hi, 1); mbar_init(vready, 1);
    mbar_init(mma_lo_qk, 1); mbar_init(mma_hi_qk, 1);
    mbar_init(mma_lo_pv, 1); mbar_init(mma_hi_pv, 1);
    mbar_init(pready_lo, 1); mbar_init(pready_hi, 1);
    mbar_init(pfree_lo, 1); mbar_init(pfree_hi, 1);
  }
  __syncthreads();

  constexpr uint32_t NCOLS = (Bc > D) ? (uint32_t)Bc : (uint32_t)D;
  static_assert(NCOLS >= 32 && (NCOLS & (NCOLS - 1)) == 0,
                "tcgen05 column count must be a power of two >= 32");
  constexpr uint32_t NCOLS_TOTAL = NCOLS * 2;
  uint32_t tmem_addr_lo;
  {
    __shared__ uint32_t s_tmem_addr;
    if(tid < 32){
      uint32_t s_addr = (uint32_t)__cvta_generic_to_shared(&s_tmem_addr);
      asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                   :: "r"(s_addr), "r"(NCOLS_TOTAL) : "memory");
      asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
    __syncthreads();
    tmem_addr_lo = s_tmem_addr;
  }
  const uint32_t tmem_addr_hi = tmem_addr_lo + NCOLS;

  const uint32_t TX_K = (uint32_t)Bc * (uint32_t)D * (uint32_t)sizeof(__nv_bfloat16);
  const uint32_t TX_V = (uint32_t)Bc * (uint32_t)D * (uint32_t)sizeof(__nv_bfloat16);
  const uint32_t sK_addr      = (uint32_t)__cvta_generic_to_shared(sK);
  const uint32_t sVstage_addr = (uint32_t)__cvta_generic_to_shared(sVstage);

  if(tid >= 384){
    // ---- Loader: shared K(atom-native)/V(raw) TMA, plus cooperative V reorder.
    // Single-buffered (sK/sVstage each have exactly one slot), so BOTH the K-reuse
    // gate (fbarK_lo/hi) and the V-canonical-reuse gate (vfree_lo/hi) apply every
    // kc, not just kc>=2 as a double-buffered version would use. ----
    const int ltid = tid - 384;
    int fk_lo_phase = 0, fk_hi_phase = 0;
    int lv_phase = 0;
    int vf_lo_phase = 0, vf_hi_phase = 0;

    for(int kc = 0; kc < nKVTiles; ++kc){
      if(ltid == 0){
        if(kc >= 1){
          mbar_wait(fbarK_lo, fk_lo_phase); fk_lo_phase ^= 1;
          mbar_wait(fbarK_hi, fk_hi_phase); fk_hi_phase ^= 1;
        }
        const int r = kvRow0 + kc * Bc;
        mbar_expect_tx(lbarK, TX_K);
        tma_load_3d(sK_addr, &Ktmap3d, 0, r, 0, lbarK);
        mbar_arrive(lbarK);
        mbar_expect_tx(lbarV, TX_V);
        tma_load_2d(sVstage_addr, &Vtmap, 0, r, lbarV);
        mbar_arrive(lbarV);
      }

      mbar_wait(lbarV, lv_phase); lv_phase ^= 1;
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      if(kc >= 1){
        mbar_wait(vfree_lo, vf_lo_phase); vf_lo_phase ^= 1;
        mbar_wait(vfree_hi, vf_hi_phase); vf_hi_phase ^= 1;
      }
      for(int i = ltid; i < Bc * D; i += 128){
        const int bc = i / D, d = i % D;
        sV[canon_idx(d, bc, D)] = sVstage[i];
      }
      sync_loader();
      if(ltid == 0) mbar_arrive(vready);
    } // end kv loop (loader)
  } else if(tid >= 256){
    // ---- Rescale: P@V MMA + TMEM readout + O-rescale for lo THEN hi, each kc. ----
    const int rtid = tid - 256;
    int pr_lo_phase = 0, pr_hi_phase = 0, vrdy_phase = 0;
    int pv_lo_phase = 0, pv_hi_phase = 0;

    for(int kc = 0; kc < nKVTiles; ++kc){
      // -- lo --
      mbar_wait(pready_lo, pr_lo_phase); pr_lo_phase ^= 1;
      for(int i = rtid; i < Br * D; i += 128) sO_lo[i] *= sCorr_lo[i / D];
      sync_rescale();
      {
        const uint64_t descP_base = make_smem_desc(sP, Br);
        const uint64_t descV_base = make_smem_desc(sV, D);
        const uint32_t idesc      = make_idesc_bf16(Br, D);
        if(rtid == 0){
          mbar_wait(vready, vrdy_phase); vrdy_phase ^= 1;
          asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
          for(int kt = 0; kt < Bc/16; ++kt){
            uint64_t descP = advance_desc_katom(descP_base, kt, Br);
            uint64_t descV = advance_desc_katom(descV_base, kt, D);
            uint32_t accumulate = (kt > 0) ? 1u : 0u;
            asm volatile(
              "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
              "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
              :: "r"(tmem_addr_lo), "l"(descP), "l"(descV), "r"(idesc), "r"(accumulate) : "memory");
          }
          mbar_commit_mma(mma_lo_pv);
        }
        mbar_wait(mma_lo_pv, pv_lo_phase); pv_lo_phase ^= 1;
        tmem_readout_accum_vec_2cta_g(sO_lo, tmem_addr_lo, Br, D, D, 0u, rtid / 32, rtid % 32);
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
      }
      sync_rescale();
      if(rtid == 0){ mbar_arrive(vfree_lo); mbar_arrive(pfree_lo); }

      // -- hi --
      mbar_wait(pready_hi, pr_hi_phase); pr_hi_phase ^= 1;
      for(int i = rtid; i < Br * D; i += 128) sO_hi[i] *= sCorr_hi[i / D];
      sync_rescale();
      {
        const uint64_t descP_base = make_smem_desc(sP, Br);
        const uint64_t descV_base = make_smem_desc(sV, D);
        const uint32_t idesc      = make_idesc_bf16(Br, D);
        if(rtid == 0){
          // V is unchanged since lo's use this kc (no write happens between) — no
          // second vready wait needed, same "ready" state still holds.
          asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
          for(int kt = 0; kt < Bc/16; ++kt){
            uint64_t descP = advance_desc_katom(descP_base, kt, Br);
            uint64_t descV = advance_desc_katom(descV_base, kt, D);
            uint32_t accumulate = (kt > 0) ? 1u : 0u;
            asm volatile(
              "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
              "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
              :: "r"(tmem_addr_hi), "l"(descP), "l"(descV), "r"(idesc), "r"(accumulate) : "memory");
          }
          mbar_commit_mma(mma_hi_pv);
        }
        mbar_wait(mma_hi_pv, pv_hi_phase); pv_hi_phase ^= 1;
        tmem_readout_accum_vec_2cta_g(sO_hi, tmem_addr_hi, Br, D, D, 0u, rtid / 32, rtid % 32);
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
      }
      sync_rescale();
      if(rtid == 0){ mbar_arrive(vfree_hi); mbar_arrive(pfree_hi); }
    } // end kv loop (rescale)
  } else if(tid >= 128){
    // ---- softmax-hi: q_tile_hi's own QK^T + readout + mask + online-softmax. ----
    const int ltid = tid - 128;
    const uint64_t descQ_base = make_smem_desc(sQ_hi, Br);
    int mqk_phase = 0;
    int lk_phase = 0;
    int pf_lo_phase = 0;

    for(int kc = 0; kc < nKVTiles; ++kc){
      mbar_wait(lbarK, lk_phase); lk_phase ^= 1;
      mbar_wait(pfree_lo, pf_lo_phase); pf_lo_phase ^= 1;   // lo always goes first, every kc

      const uint64_t descK_base = make_smem_desc(sK, Bc);
      const uint32_t idesc      = make_idesc_bf16(Br, Bc);
      if(ltid == 0){
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        for(int kt = 0; kt < D/16; ++kt){
          uint64_t descQ = advance_desc_katom(descQ_base, kt, Br);
          uint64_t descK = advance_desc_katom(descK_base, kt, Bc);
          uint32_t accumulate = (kt > 0) ? 1u : 0u;
          asm volatile(
            "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
            :: "r"(tmem_addr_hi), "l"(descQ), "l"(descK), "r"(idesc), "r"(accumulate) : "memory");
        }
        mbar_commit_mma(mma_hi_qk);
      }
      mbar_wait(mma_hi_qk, mqk_phase); mqk_phase ^= 1;
      if(ltid == 0) mbar_arrive(fbarK_hi);

      tmem_readout_to_smem_fp16_vec_g(sS, tmem_addr_hi, Br, Bc, Bc, scale_l2e, ltid / 32, ltid % 32);
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");

      if(ltid < Br){
        if(kc == q_tile_hi){
          const __half neg_inf = __float2half(-INFINITY);
          for(int j = ltid + 1; j < Bc; ++j) sS[ltid * Bc + j] = neg_inf;
        }
      }
      sync_half_b();

      if(ltid < Br){
        const float m_old = sm_hi[ltid];
        const float l_old = sl_hi[ltid];
        float tile_max = -INFINITY;
        int j = 0;
        for(; j + 2 < Bc; j += 3)
          tile_max = fmaxf(tile_max,
                           fmaxf(__half2float(sS[ltid * Bc + j]),
                                 fmaxf(__half2float(sS[ltid * Bc + j + 1]), __half2float(sS[ltid * Bc + j + 2]))));
        for(; j < Bc; ++j) tile_max = fmaxf(tile_max, __half2float(sS[ltid * Bc + j]));

        const float m_new = fmaxf(m_old, tile_max);
        const float corr  = ex2_approx(m_old - m_new);

        float p_sum = 0.0f;
        for(int j2 = 0; j2 < Bc; j2 += 2){
          const float p0 = ex2_approx(__half2float(sS[ltid * Bc + j2])     - m_new);
          const float p1 = ex2_approx(__half2float(sS[ltid * Bc + j2 + 1]) - m_new);
          *reinterpret_cast<__nv_bfloat162*>(&sP[canon_idx(ltid, j2, Br)]) =
              __floats2bfloat162_rn(p0, p1);
          p_sum += p0 + p1;
        }
        sm_hi[ltid] = m_new; sl_hi[ltid] = l_old * corr + p_sum; sCorr_hi[ltid] = corr;
      }
      sync_half_b();
      if(ltid == 0) mbar_arrive(pready_hi);
    } // end kv loop (softmax-hi)
  } else {
    // ---- softmax-lo: q_tile_lo's own QK^T + readout + mask + online-softmax. ----
    const uint64_t descQ_base = make_smem_desc(sQ_lo, Br);
    int mqk_phase = 0;
    int lk_phase = 0;
    int pf_hi_phase = 0;

    for(int kc = 0; kc < nKVTiles; ++kc){
      mbar_wait(lbarK, lk_phase); lk_phase ^= 1;
      if(kc >= 1){ mbar_wait(pfree_hi, pf_hi_phase); pf_hi_phase ^= 1; }   // no prior use at kc==0

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
            :: "r"(tmem_addr_lo), "l"(descQ), "l"(descK), "r"(idesc), "r"(accumulate) : "memory");
        }
        mbar_commit_mma(mma_lo_qk);
      }
      mbar_wait(mma_lo_qk, mqk_phase); mqk_phase ^= 1;
      if(tid == 0) mbar_arrive(fbarK_lo);

      tmem_readout_to_smem_fp16_vec(sS, tmem_addr_lo, Br, Bc, Bc, scale_l2e);
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");

      if(kc > q_tile_lo){
        const __half neg_inf = __float2half(-INFINITY);
        for(int j = 0; j < Bc; ++j) sS[tid * Bc + j] = neg_inf;
      } else if(kc == q_tile_lo){
        const __half neg_inf = __float2half(-INFINITY);
        for(int j = tid + 1; j < Bc; ++j) sS[tid * Bc + j] = neg_inf;
      }
      consumer_sync();

      {
        const float m_old = sm_lo[tid];
        const float l_old = sl_lo[tid];
        float tile_max = -INFINITY;
        int j = 0;
        for(; j + 2 < Bc; j += 3)
          tile_max = fmaxf(tile_max,
                           fmaxf(__half2float(sS[tid * Bc + j]),
                                 fmaxf(__half2float(sS[tid * Bc + j + 1]), __half2float(sS[tid * Bc + j + 2]))));
        for(; j < Bc; ++j) tile_max = fmaxf(tile_max, __half2float(sS[tid * Bc + j]));

        const float m_new = fmaxf(m_old, tile_max);
        const float corr  = ex2_approx(m_old - m_new);

        float p_sum = 0.0f;
        for(int j2 = 0; j2 < Bc; j2 += 2){
          const float p0 = ex2_approx(__half2float(sS[tid * Bc + j2])     - m_new);
          const float p1 = ex2_approx(__half2float(sS[tid * Bc + j2 + 1]) - m_new);
          *reinterpret_cast<__nv_bfloat162*>(&sP[canon_idx(tid, j2, Br)]) =
              __floats2bfloat162_rn(p0, p1);
          p_sum += p0 + p1;
        }
        sm_lo[tid] = m_new; sl_lo[tid] = l_old * corr + p_sum; sCorr_lo[tid] = corr;
      }
      consumer_sync();
      if(tid == 0) mbar_arrive(pready_lo);
    } // end kv loop (softmax-lo)
  }

  __syncthreads();
  if(tid < 32)
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(tmem_addr_lo), "r"(NCOLS_TOTAL) : "memory");
  __syncthreads();

  for(int i = 2 * tid; i < Br * D && tid < 128; i += 256){
    const float denom = sl_lo[i / D];
    *reinterpret_cast<__nv_bfloat162*>(&d_O[qBaseLo + i]) =
        __floats2bfloat162_rn(sO_lo[i] / denom, sO_lo[i + 1] / denom);
  }
  for(int i = 2 * (tid - 128); i < Br * D && tid >= 128 && tid < 256; i += 256){
    const float denom = sl_hi[i / D];
    *reinterpret_cast<__nv_bfloat162*>(&d_O[qBaseHi + i]) =
        __floats2bfloat162_rn(sO_hi[i] / denom, sO_hi[i + 1] / denom);
  }
  if(tid < Br)
    d_LSE[lBaseLo + tid] = 0.6931471805599453f * (sm_lo[tid] + log2f(sl_lo[tid]));
  if(tid >= 128 && tid < 128 + Br)
    d_LSE[lBaseHi + (tid - 128)] = 0.6931471805599453f * (sm_hi[tid-128] + log2f(sl_hi[tid-128]));

} // end of gqa_v30_causal

// V30_causal launcher — same flat load-balanced-pair grid as V28/V29, block 512
// (4 genuinely-populated warpgroups this time, unlike V28's padded version).
template<int Br, int Bc, int D>
void launch_gqa_v30_causal(
  __nv_bfloat16 *d_Q, __nv_bfloat16 *d_K, __nv_bfloat16 *d_V,
  __nv_bfloat16 *d_O, float *d_LSE,
  int B, int Hq, int Hkv, int S, int G, float scale
){
  static_assert(Br == 128, "each tile-group is hardwired to 4 warps (128 threads)");
  static_assert(Br == Bc, "causal tile-skip + diagonal-tile mask requires Br == Bc");
  static_assert(Bc % 8 == 0, "Bc must be a multiple of 8 for tcgen05 N = 8");
  static_assert(D  % 16 == 0, "D  must be a multiple of 16 for tcgen05 dense");
  static_assert(D  % 8  == 0, "D must be a multiple of 8 for the atom-native K TMA map");

  assert((S / Br) % 2 == 0 && "V30_causal's load-balanced pairing needs an even number of causal query tiles");

  dim3 GRID(B * Hq * ((S / Br) / 2), 1, 1);
  dim3 BLOCK(512);

  static bool cfgd = false;
  static CUtensorMap Ktmap3d, Vtmap;
  if(!cfgd){
    const uint64_t kvRows = (uint64_t)B * Hkv * S;
    Ktmap3d = make_tma_3d_katom(d_K, kvRows, (uint64_t)D, (uint32_t)Bc);
    Vtmap   = make_tma_2d(d_V, kvRows, (uint64_t)D, (uint32_t)Bc, (uint32_t)D);
    cfgd = true;
  }
  gqa_v30_causal<Br, Bc, D><<<GRID, BLOCK>>>(d_Q, d_O, d_LSE, Ktmap3d, Vtmap,
                          B, Hq, Hkv, G, S, scale);
}

// =================================
//  V31SmemLayout — manual dynamic-shared-memory partitioning for gqa_v31_causal.
//  Needed because restoring double-buffered K/V (below) pushes V30's ~211KB up to
//  ~243KB, past sm_103a's 227KB STATIC shared memory hard cap (0x38c00 bytes,
//  discovered empirically while building V30) — dynamic shared memory has a
//  higher ceiling (cuDNN's own causal reference kernel already proves >=232.45KB
//  works via this path), but requires manually partitioning one big
//  extern __shared__ blob by byte offset instead of separate typed __shared__
//  arrays. Small state (mbarriers, the TMEM-address broadcast temp) stays
//  ordinary static __shared__ — only the large data buffers move into this blob.
//  ~243KB is ABOVE cuDNN's own observed 232.45KB, so this is deliberately also an
//  experiment to find sm_103a's true dynamic-smem ceiling: if cudaFuncSetAttribute
//  rejects this size, that tells us the real limit sits somewhere in
//  [237978, 248832) bytes; if it succeeds, the ceiling is at least 248832.
// =================================
template<int Br, int Bc, int D>
struct V31SmemLayout {
  static constexpr size_t align_up(size_t x, size_t a){ return (x + a - 1) / a * a; }
  static constexpr size_t sQ_lo    = 0;
  static constexpr size_t sQ_hi    = align_up(sQ_lo    + (size_t)Br * D * sizeof(__nv_bfloat16), 16);
  static constexpr size_t sK       = align_up(sQ_hi    + (size_t)Br * D * sizeof(__nv_bfloat16), 128);
  static constexpr size_t sVstage  = align_up(sK       + 2 * (size_t)Bc * D * sizeof(__nv_bfloat16), 128);
  static constexpr size_t sV       = align_up(sVstage  + 1 * (size_t)Bc * D * sizeof(__nv_bfloat16), 16);
  static constexpr size_t sS       = align_up(sV       + (size_t)Bc * D * sizeof(__nv_bfloat16), 16);
  static constexpr size_t sP       = align_up(sS       + (size_t)Br * Bc * sizeof(__half), 16);
  static constexpr size_t sO_lo    = align_up(sP       + (size_t)Br * Bc * sizeof(__nv_bfloat16), 16);
  static constexpr size_t sO_hi    = align_up(sO_lo    + (size_t)Br * D * sizeof(float), 16);
  static constexpr size_t sm_lo    = align_up(sO_hi    + (size_t)Br * D * sizeof(float), 4);
  static constexpr size_t sl_lo    = sm_lo    + (size_t)Br * sizeof(float);
  static constexpr size_t sCorr_lo = sl_lo    + (size_t)Br * sizeof(float);
  static constexpr size_t sm_hi    = sCorr_lo + (size_t)Br * sizeof(float);
  static constexpr size_t sl_hi    = sm_hi    + (size_t)Br * sizeof(float);
  static constexpr size_t sCorr_hi = sl_hi    + (size_t)Br * sizeof(float);
  static constexpr size_t total    = align_up(sCorr_hi + (size_t)Br * sizeof(float), 16);
};

// gqa_v31_causal — V30 + double-buffered K staging restored (sK gains a second
// slot again, matching V26-29's [2]-slot pattern, gated by fbarK_lo/hi at
// kc>=2 like before -- NOT kc>=1, since single-buffering is what's being undone
// here). sVstage stays SINGLE-buffered (its own dynamic-smem ceiling probe: a
// 243KB request failed with cudaErrorInvalidValue, so this trims back to fit --
// see launch_gqa_v31_causal). sS/sP sharing is DELIBERATELY left UNCHANGED from V30
// (still shared/serialized between lo and hi via pfree/pready) -- this isolates
// ONE variable, matching this whole session's methodology: V30 showed BETTER
// warp-scheduling efficiency than V29 on every NCU metric (more eligible warps,
// higher IPC, less idle-scheduling) yet ran SLOWER in wall-clock -- meaning
// something added real critical-path latency the better scheduling couldn't hide.
// The two candidates were losing double-buffered K/V and the sS/sP round-trip
// serialization; this version removes only the first, to see how much of the gap
// that alone closes before touching the second (a follow-up, if needed, would
// un-share sP specifically -- the more heavily-gated one -- while leaving sS
// shared, since full independence for both doesn't fit even with this dynamic
// headroom: that math comes to ~370KB).
template<int Br, int Bc, int D>
__global__ void gqa_v31_causal(
  __nv_bfloat16 *d_Q,
  __nv_bfloat16 *d_O,
  float *d_LSE,
  const __grid_constant__ CUtensorMap Ktmap3d,
  const __grid_constant__ CUtensorMap Vtmap,
  int B,
  int Hq,
  int Hkv,
  int G,
  int S,
  float scale
){
  static_assert(Br == 128, "each tile-group is hardwired to 4 warps (128 threads)");
  static_assert(Br == Bc, "causal tile-skip + diagonal-tile mask requires Br == Bc");

  const int tid = threadIdx.x;   // 0..511: 0..127 softmax-lo, 128..255 softmax-hi, 256..383 rescale, 384..511 loader

  const int nTiles     = S / Br;
  const int pairsPerBH = nTiles / 2;
  const int idx  = blockIdx.x;
  const int b    = idx / (Hq * pairsPerBH);
  const int rem  = idx - b * (Hq * pairsPerBH);
  const int hq   = rem / pairsPerBH;
  const int pair = rem - hq * pairsPerBH;
  const int hkv  = hq / G;
  const int kvRow0 = (b * Hkv + hkv) * S;

  const int q_tile_lo = pair;
  const int q_tile_hi = nTiles - 1 - pair;
  const int nKVTiles   = q_tile_hi + 1;

  const long qBaseLo = ((long)(b * Hq + hq) * S + q_tile_lo * Br) * D;
  const long qBaseHi = ((long)(b * Hq + hq) * S + q_tile_hi * Br) * D;
  const long lBaseLo = (long)(b * Hq + hq) * S + q_tile_lo * Br;
  const long lBaseHi = (long)(b * Hq + hq) * S + q_tile_hi * Br;

  const float scale_l2e = scale * 1.4426950408889634f;

  extern __shared__ char smem_raw_v31[];
  using L = V31SmemLayout<Br, Bc, D>;
  __nv_bfloat16* sQ_lo   = reinterpret_cast<__nv_bfloat16*>(smem_raw_v31 + L::sQ_lo);
  __nv_bfloat16* sQ_hi   = reinterpret_cast<__nv_bfloat16*>(smem_raw_v31 + L::sQ_hi);
  __nv_bfloat16* sK      = reinterpret_cast<__nv_bfloat16*>(smem_raw_v31 + L::sK);       // flat [2][Bc*D]
  __nv_bfloat16* sVstage = reinterpret_cast<__nv_bfloat16*>(smem_raw_v31 + L::sVstage);  // single [Bc*D], not double-buffered (smem budget)
  __nv_bfloat16* sV      = reinterpret_cast<__nv_bfloat16*>(smem_raw_v31 + L::sV);
  __half*        sS      = reinterpret_cast<__half*>(smem_raw_v31 + L::sS);
  __nv_bfloat16* sP      = reinterpret_cast<__nv_bfloat16*>(smem_raw_v31 + L::sP);
  float*         sO_lo   = reinterpret_cast<float*>(smem_raw_v31 + L::sO_lo);
  float*         sO_hi   = reinterpret_cast<float*>(smem_raw_v31 + L::sO_hi);
  float*         sm_lo    = reinterpret_cast<float*>(smem_raw_v31 + L::sm_lo);
  float*         sl_lo    = reinterpret_cast<float*>(smem_raw_v31 + L::sl_lo);
  float*         sCorr_lo = reinterpret_cast<float*>(smem_raw_v31 + L::sCorr_lo);
  float*         sm_hi    = reinterpret_cast<float*>(smem_raw_v31 + L::sm_hi);
  float*         sl_hi    = reinterpret_cast<float*>(smem_raw_v31 + L::sl_hi);
  float*         sCorr_hi = reinterpret_cast<float*>(smem_raw_v31 + L::sCorr_hi);

  __shared__ __align__(8) uint64_t s_lbarK[2], s_lbarV;
  __shared__ __align__(8) uint64_t s_fbarK_lo[2], s_fbarK_hi[2];
  __shared__ __align__(8) uint64_t s_vfree_lo, s_vfree_hi, s_vready;
  __shared__ __align__(8) uint64_t s_mma_lo_qk, s_mma_hi_qk, s_mma_lo_pv, s_mma_hi_pv;
  __shared__ __align__(8) uint64_t s_pready_lo, s_pready_hi, s_pfree_lo, s_pfree_hi;

  const uint32_t lbarK[2]    = { (uint32_t)__cvta_generic_to_shared(&s_lbarK[0]),
                                  (uint32_t)__cvta_generic_to_shared(&s_lbarK[1]) };
  const uint32_t lbarV       = (uint32_t)__cvta_generic_to_shared(&s_lbarV);
  const uint32_t fbarK_lo[2] = { (uint32_t)__cvta_generic_to_shared(&s_fbarK_lo[0]),
                                  (uint32_t)__cvta_generic_to_shared(&s_fbarK_lo[1]) };
  const uint32_t fbarK_hi[2] = { (uint32_t)__cvta_generic_to_shared(&s_fbarK_hi[0]),
                                  (uint32_t)__cvta_generic_to_shared(&s_fbarK_hi[1]) };
  const uint32_t vfree_lo  = (uint32_t)__cvta_generic_to_shared(&s_vfree_lo);
  const uint32_t vfree_hi  = (uint32_t)__cvta_generic_to_shared(&s_vfree_hi);
  const uint32_t vready    = (uint32_t)__cvta_generic_to_shared(&s_vready);
  const uint32_t mma_lo_qk = (uint32_t)__cvta_generic_to_shared(&s_mma_lo_qk);
  const uint32_t mma_hi_qk = (uint32_t)__cvta_generic_to_shared(&s_mma_hi_qk);
  const uint32_t mma_lo_pv = (uint32_t)__cvta_generic_to_shared(&s_mma_lo_pv);
  const uint32_t mma_hi_pv = (uint32_t)__cvta_generic_to_shared(&s_mma_hi_pv);
  const uint32_t pready_lo = (uint32_t)__cvta_generic_to_shared(&s_pready_lo);
  const uint32_t pready_hi = (uint32_t)__cvta_generic_to_shared(&s_pready_hi);
  const uint32_t pfree_lo  = (uint32_t)__cvta_generic_to_shared(&s_pfree_lo);
  const uint32_t pfree_hi  = (uint32_t)__cvta_generic_to_shared(&s_pfree_hi);

  for(int i = tid; i < Br * D && tid < 128; i += 128){
    sQ_lo[canon_idx(i / D, i % D, Br)] = d_Q[qBaseLo + i];
    sO_lo[i] = 0.0f;
  }
  for(int i = tid - 128; i < Br * D && tid >= 128 && tid < 256; i += 128){
    sQ_hi[canon_idx(i / D, i % D, Br)] = d_Q[qBaseHi + i];
    sO_hi[i] = 0.0f;
  }
  if(tid < Br){ sm_lo[tid] = -INFINITY; sl_lo[tid] = 0.0f; }
  if(tid >= 128 && tid < 128 + Br){ sm_hi[tid-128] = -INFINITY; sl_hi[tid-128] = 0.0f; }

  if(tid == 0){
    mbar_init(lbarK[0], 1); mbar_init(lbarK[1], 1);
    mbar_init(lbarV, 1);
    mbar_init(fbarK_lo[0], 1); mbar_init(fbarK_lo[1], 1);
    mbar_init(fbarK_hi[0], 1); mbar_init(fbarK_hi[1], 1);
    mbar_init(vfree_lo, 1); mbar_init(vfree_hi, 1); mbar_init(vready, 1);
    mbar_init(mma_lo_qk, 1); mbar_init(mma_hi_qk, 1);
    mbar_init(mma_lo_pv, 1); mbar_init(mma_hi_pv, 1);
    mbar_init(pready_lo, 1); mbar_init(pready_hi, 1);
    mbar_init(pfree_lo, 1); mbar_init(pfree_hi, 1);
  }
  __syncthreads();

  constexpr uint32_t NCOLS = (Bc > D) ? (uint32_t)Bc : (uint32_t)D;
  static_assert(NCOLS >= 32 && (NCOLS & (NCOLS - 1)) == 0,
                "tcgen05 column count must be a power of two >= 32");
  constexpr uint32_t NCOLS_TOTAL = NCOLS * 2;
  uint32_t tmem_addr_lo;
  {
    __shared__ uint32_t s_tmem_addr;
    if(tid < 32){
      uint32_t s_addr = (uint32_t)__cvta_generic_to_shared(&s_tmem_addr);
      asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                   :: "r"(s_addr), "r"(NCOLS_TOTAL) : "memory");
      asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    }
    __syncthreads();
    tmem_addr_lo = s_tmem_addr;
  }
  const uint32_t tmem_addr_hi = tmem_addr_lo + NCOLS;

  const uint32_t TX_K = (uint32_t)Bc * (uint32_t)D * (uint32_t)sizeof(__nv_bfloat16);
  const uint32_t TX_V = (uint32_t)Bc * (uint32_t)D * (uint32_t)sizeof(__nv_bfloat16);
  const uint32_t sK_addr[2] = {
    (uint32_t)__cvta_generic_to_shared(sK + 0 * Bc * D),
    (uint32_t)__cvta_generic_to_shared(sK + 1 * Bc * D)
  };
  const uint32_t sVstage_addr = (uint32_t)__cvta_generic_to_shared(sVstage);

  if(tid >= 384){
    // ---- Loader: K TMA double-buffered (the isolated variable under test);
    // V staging stays single-buffered (smem budget) exactly as in V30 --
    // its reorder-copy is self-gated by loader program order, no extra wait
    // needed beyond what's already here. ----
    const int ltid = tid - 384;
    int fk_lo_phase[2] = {0,0}, fk_hi_phase[2] = {0,0};
    int lv_phase = 0;
    int vf_lo_phase = 0, vf_hi_phase = 0;

    for(int kc = 0; kc < nKVTiles; ++kc){
      const int slot = kc & 1;
      if(ltid == 0){
        if(kc >= 2){
          mbar_wait(fbarK_lo[slot], fk_lo_phase[slot]); fk_lo_phase[slot] ^= 1;
          mbar_wait(fbarK_hi[slot], fk_hi_phase[slot]); fk_hi_phase[slot] ^= 1;
        }
        const int r = kvRow0 + kc * Bc;
        mbar_expect_tx(lbarK[slot], TX_K);
        tma_load_3d(sK_addr[slot], &Ktmap3d, 0, r, 0, lbarK[slot]);
        mbar_arrive(lbarK[slot]);
        mbar_expect_tx(lbarV, TX_V);
        tma_load_2d(sVstage_addr, &Vtmap, 0, r, lbarV);
        mbar_arrive(lbarV);
      }

      mbar_wait(lbarV, lv_phase); lv_phase ^= 1;
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      if(kc >= 1){
        mbar_wait(vfree_lo, vf_lo_phase); vf_lo_phase ^= 1;
        mbar_wait(vfree_hi, vf_hi_phase); vf_hi_phase ^= 1;
      }
      for(int i = ltid; i < Bc * D; i += 128){
        const int bc = i / D, d = i % D;
        sV[canon_idx(d, bc, D)] = sVstage[i];
      }
      sync_loader();
      if(ltid == 0) mbar_arrive(vready);
    } // end kv loop (loader)
  } else if(tid >= 256){
    // ---- Rescale: unchanged from V30. ----
    const int rtid = tid - 256;
    int pr_lo_phase = 0, pr_hi_phase = 0, vrdy_phase = 0;
    int pv_lo_phase = 0, pv_hi_phase = 0;

    for(int kc = 0; kc < nKVTiles; ++kc){
      // -- lo --
      mbar_wait(pready_lo, pr_lo_phase); pr_lo_phase ^= 1;
      for(int i = rtid; i < Br * D; i += 128) sO_lo[i] *= sCorr_lo[i / D];
      sync_rescale();
      {
        const uint64_t descP_base = make_smem_desc(sP, Br);
        const uint64_t descV_base = make_smem_desc(sV, D);
        const uint32_t idesc      = make_idesc_bf16(Br, D);
        if(rtid == 0){
          mbar_wait(vready, vrdy_phase); vrdy_phase ^= 1;
          asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
          for(int kt = 0; kt < Bc/16; ++kt){
            uint64_t descP = advance_desc_katom(descP_base, kt, Br);
            uint64_t descV = advance_desc_katom(descV_base, kt, D);
            uint32_t accumulate = (kt > 0) ? 1u : 0u;
            asm volatile(
              "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
              "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
              :: "r"(tmem_addr_lo), "l"(descP), "l"(descV), "r"(idesc), "r"(accumulate) : "memory");
          }
          mbar_commit_mma(mma_lo_pv);
        }
        mbar_wait(mma_lo_pv, pv_lo_phase); pv_lo_phase ^= 1;
        tmem_readout_accum_vec_2cta_g(sO_lo, tmem_addr_lo, Br, D, D, 0u, rtid / 32, rtid % 32);
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
      }
      sync_rescale();
      if(rtid == 0){ mbar_arrive(vfree_lo); mbar_arrive(pfree_lo); }

      // -- hi --
      mbar_wait(pready_hi, pr_hi_phase); pr_hi_phase ^= 1;
      for(int i = rtid; i < Br * D; i += 128) sO_hi[i] *= sCorr_hi[i / D];
      sync_rescale();
      {
        const uint64_t descP_base = make_smem_desc(sP, Br);
        const uint64_t descV_base = make_smem_desc(sV, D);
        const uint32_t idesc      = make_idesc_bf16(Br, D);
        if(rtid == 0){
          asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
          for(int kt = 0; kt < Bc/16; ++kt){
            uint64_t descP = advance_desc_katom(descP_base, kt, Br);
            uint64_t descV = advance_desc_katom(descV_base, kt, D);
            uint32_t accumulate = (kt > 0) ? 1u : 0u;
            asm volatile(
              "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
              "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
              :: "r"(tmem_addr_hi), "l"(descP), "l"(descV), "r"(idesc), "r"(accumulate) : "memory");
          }
          mbar_commit_mma(mma_hi_pv);
        }
        mbar_wait(mma_hi_pv, pv_hi_phase); pv_hi_phase ^= 1;
        tmem_readout_accum_vec_2cta_g(sO_hi, tmem_addr_hi, Br, D, D, 0u, rtid / 32, rtid % 32);
        asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
      }
      sync_rescale();
      if(rtid == 0){ mbar_arrive(vfree_hi); mbar_arrive(pfree_hi); }
    } // end kv loop (rescale)
  } else if(tid >= 128){
    // ---- softmax-hi: q_tile_hi's own QK^T + readout + mask + online-softmax. ----
    const int ltid = tid - 128;
    const uint64_t descQ_base = make_smem_desc(sQ_hi, Br);
    int mqk_phase = 0;
    int lk_phase[2] = {0,0};
    int pf_lo_phase = 0;

    for(int kc = 0; kc < nKVTiles; ++kc){
      const int slot = kc & 1;
      mbar_wait(lbarK[slot], lk_phase[slot]); lk_phase[slot] ^= 1;
      mbar_wait(pfree_lo, pf_lo_phase); pf_lo_phase ^= 1;

      const uint64_t descK_base = make_smem_desc(sK + slot * Bc * D, Bc);
      const uint32_t idesc      = make_idesc_bf16(Br, Bc);
      if(ltid == 0){
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        for(int kt = 0; kt < D/16; ++kt){
          uint64_t descQ = advance_desc_katom(descQ_base, kt, Br);
          uint64_t descK = advance_desc_katom(descK_base, kt, Bc);
          uint32_t accumulate = (kt > 0) ? 1u : 0u;
          asm volatile(
            "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
            :: "r"(tmem_addr_hi), "l"(descQ), "l"(descK), "r"(idesc), "r"(accumulate) : "memory");
        }
        mbar_commit_mma(mma_hi_qk);
      }
      mbar_wait(mma_hi_qk, mqk_phase); mqk_phase ^= 1;
      if(ltid == 0) mbar_arrive(fbarK_hi[slot]);

      tmem_readout_to_smem_fp16_vec_g(sS, tmem_addr_hi, Br, Bc, Bc, scale_l2e, ltid / 32, ltid % 32);
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");

      if(ltid < Br){
        if(kc == q_tile_hi){
          const __half neg_inf = __float2half(-INFINITY);
          for(int j = ltid + 1; j < Bc; ++j) sS[ltid * Bc + j] = neg_inf;
        }
      }
      sync_half_b();

      if(ltid < Br){
        const float m_old = sm_hi[ltid];
        const float l_old = sl_hi[ltid];
        float tile_max = -INFINITY;
        int j = 0;
        for(; j + 2 < Bc; j += 3)
          tile_max = fmaxf(tile_max,
                           fmaxf(__half2float(sS[ltid * Bc + j]),
                                 fmaxf(__half2float(sS[ltid * Bc + j + 1]), __half2float(sS[ltid * Bc + j + 2]))));
        for(; j < Bc; ++j) tile_max = fmaxf(tile_max, __half2float(sS[ltid * Bc + j]));

        const float m_new = fmaxf(m_old, tile_max);
        const float corr  = ex2_approx(m_old - m_new);

        float p_sum = 0.0f;
        for(int j2 = 0; j2 < Bc; j2 += 2){
          const float p0 = ex2_approx(__half2float(sS[ltid * Bc + j2])     - m_new);
          const float p1 = ex2_approx(__half2float(sS[ltid * Bc + j2 + 1]) - m_new);
          *reinterpret_cast<__nv_bfloat162*>(&sP[canon_idx(ltid, j2, Br)]) =
              __floats2bfloat162_rn(p0, p1);
          p_sum += p0 + p1;
        }
        sm_hi[ltid] = m_new; sl_hi[ltid] = l_old * corr + p_sum; sCorr_hi[ltid] = corr;
      }
      sync_half_b();
      if(ltid == 0) mbar_arrive(pready_hi);
    } // end kv loop (softmax-hi)
  } else {
    // ---- softmax-lo: q_tile_lo's own QK^T + readout + mask + online-softmax. ----
    const uint64_t descQ_base = make_smem_desc(sQ_lo, Br);
    int mqk_phase = 0;
    int lk_phase[2] = {0,0};
    int pf_hi_phase = 0;

    for(int kc = 0; kc < nKVTiles; ++kc){
      const int slot = kc & 1;
      mbar_wait(lbarK[slot], lk_phase[slot]); lk_phase[slot] ^= 1;
      if(kc >= 1){ mbar_wait(pfree_hi, pf_hi_phase); pf_hi_phase ^= 1; }

      const uint64_t descK_base = make_smem_desc(sK + slot * Bc * D, Bc);
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
            :: "r"(tmem_addr_lo), "l"(descQ), "l"(descK), "r"(idesc), "r"(accumulate) : "memory");
        }
        mbar_commit_mma(mma_lo_qk);
      }
      mbar_wait(mma_lo_qk, mqk_phase); mqk_phase ^= 1;
      if(tid == 0) mbar_arrive(fbarK_lo[slot]);

      tmem_readout_to_smem_fp16_vec(sS, tmem_addr_lo, Br, Bc, Bc, scale_l2e);
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");

      if(kc > q_tile_lo){
        const __half neg_inf = __float2half(-INFINITY);
        for(int j = 0; j < Bc; ++j) sS[tid * Bc + j] = neg_inf;
      } else if(kc == q_tile_lo){
        const __half neg_inf = __float2half(-INFINITY);
        for(int j = tid + 1; j < Bc; ++j) sS[tid * Bc + j] = neg_inf;
      }
      consumer_sync();

      {
        const float m_old = sm_lo[tid];
        const float l_old = sl_lo[tid];
        float tile_max = -INFINITY;
        int j = 0;
        for(; j + 2 < Bc; j += 3)
          tile_max = fmaxf(tile_max,
                           fmaxf(__half2float(sS[tid * Bc + j]),
                                 fmaxf(__half2float(sS[tid * Bc + j + 1]), __half2float(sS[tid * Bc + j + 2]))));
        for(; j < Bc; ++j) tile_max = fmaxf(tile_max, __half2float(sS[tid * Bc + j]));

        const float m_new = fmaxf(m_old, tile_max);
        const float corr  = ex2_approx(m_old - m_new);

        float p_sum = 0.0f;
        for(int j2 = 0; j2 < Bc; j2 += 2){
          const float p0 = ex2_approx(__half2float(sS[tid * Bc + j2])     - m_new);
          const float p1 = ex2_approx(__half2float(sS[tid * Bc + j2 + 1]) - m_new);
          *reinterpret_cast<__nv_bfloat162*>(&sP[canon_idx(tid, j2, Br)]) =
              __floats2bfloat162_rn(p0, p1);
          p_sum += p0 + p1;
        }
        sm_lo[tid] = m_new; sl_lo[tid] = l_old * corr + p_sum; sCorr_lo[tid] = corr;
      }
      consumer_sync();
      if(tid == 0) mbar_arrive(pready_lo);
    } // end kv loop (softmax-lo)
  }

  __syncthreads();
  if(tid < 32)
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(tmem_addr_lo), "r"(NCOLS_TOTAL) : "memory");
  __syncthreads();

  for(int i = 2 * tid; i < Br * D && tid < 128; i += 256){
    const float denom = sl_lo[i / D];
    *reinterpret_cast<__nv_bfloat162*>(&d_O[qBaseLo + i]) =
        __floats2bfloat162_rn(sO_lo[i] / denom, sO_lo[i + 1] / denom);
  }
  for(int i = 2 * (tid - 128); i < Br * D && tid >= 128 && tid < 256; i += 256){
    const float denom = sl_hi[i / D];
    *reinterpret_cast<__nv_bfloat162*>(&d_O[qBaseHi + i]) =
        __floats2bfloat162_rn(sO_hi[i] / denom, sO_hi[i + 1] / denom);
  }
  if(tid < Br)
    d_LSE[lBaseLo + tid] = 0.6931471805599453f * (sm_lo[tid] + log2f(sl_lo[tid]));
  if(tid >= 128 && tid < 128 + Br)
    d_LSE[lBaseHi + (tid - 128)] = 0.6931471805599453f * (sm_hi[tid-128] + log2f(sl_hi[tid-128]));

} // end of gqa_v31_causal

// V31_causal launcher — same flat load-balanced-pair grid as V28/29/30, block 512.
// Opts into dynamic shared memory above the 227KB static cap via
// cudaFuncSetAttribute, sized exactly to V31SmemLayout's computed total.
template<int Br, int Bc, int D>
void launch_gqa_v31_causal(
  __nv_bfloat16 *d_Q, __nv_bfloat16 *d_K, __nv_bfloat16 *d_V,
  __nv_bfloat16 *d_O, float *d_LSE,
  int B, int Hq, int Hkv, int S, int G, float scale
){
  static_assert(Br == 128, "each tile-group is hardwired to 4 warps (128 threads)");
  static_assert(Br == Bc, "causal tile-skip + diagonal-tile mask requires Br == Bc");
  static_assert(Bc % 8 == 0, "Bc must be a multiple of 8 for tcgen05 N = 8");
  static_assert(D  % 16 == 0, "D  must be a multiple of 16 for tcgen05 dense");
  static_assert(D  % 8  == 0, "D must be a multiple of 8 for the atom-native K TMA map");

  assert((S / Br) % 2 == 0 && "V31_causal's load-balanced pairing needs an even number of causal query tiles");

  constexpr size_t SMEM_TOTAL = V31SmemLayout<Br, Bc, D>::total;

  dim3 GRID(B * Hq * ((S / Br) / 2), 1, 1);
  dim3 BLOCK(512);

  static bool cfgd = false;
  static CUtensorMap Ktmap3d, Vtmap;
  if(!cfgd){
    const uint64_t kvRows = (uint64_t)B * Hkv * S;
    Ktmap3d = make_tma_3d_katom(d_K, kvRows, (uint64_t)D, (uint32_t)Bc);
    Vtmap   = make_tma_2d(d_V, kvRows, (uint64_t)D, (uint32_t)Bc, (uint32_t)D);
    CUDA_CHECK(cudaFuncSetAttribute(gqa_v31_causal<Br, Bc, D>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int)SMEM_TOTAL));
    cfgd = true;
  }
  gqa_v31_causal<Br, Bc, D><<<GRID, BLOCK, SMEM_TOTAL>>>(d_Q, d_O, d_LSE, Ktmap3d, Vtmap,
                          B, Hq, Hkv, G, S, scale);
}


int main(){
  std::cout << "Benchmarking CAUSAL Grouped-Query Attention — Blackwell SM_103 (B300)\n";

  {
    int optinMax = 0;
    cudaDeviceGetAttribute(&optinMax, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    std::cout << "  cudaDevAttrMaxSharedMemoryPerBlockOptin = " << optinMax
               << " bytes (" << (optinMax / 1024.0) << " KB)\n";
  }

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

  //* ── Correctness ────────────────────────────────────────────────────────
  auto runCorrectness = [&](const char* label, auto launchFn){
    launchFn();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_O.data(),   d_O,   Nq   * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_LSE.data(), d_LSE, Nlse * sizeof(float),         cudaMemcpyDeviceToHost));

    std::vector<float> h_O_f32(Nq);
    for(size_t i = 0; i < Nq; ++i) h_O_f32[i] = __bfloat162float(h_O[i]);

    std::cout << "\nCorrectness " << label << " (vs PyTorch bf16 causal SDPA):\n";
    reportPrecision("  output O ", h_O_ref.data(),   h_O_f32.data(), Nq);
    reportPrecision("  lse      ", h_LSE_ref.data(), h_LSE.data(),   Nlse);
    std::cout << "  O   : "; checkResult(h_O_ref.data(),   h_O_f32.data(), Nq,   2e-2f, 2e-2f);
    std::cout << "  LSE : "; checkResult(h_LSE_ref.data(), h_LSE.data(),   Nlse, 2e-2f, 2e-2f);
  };

  if(has_ref){
    runCorrectness("V19-causal", [&](){ launch_gqa_v19_causal<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); });
    runCorrectness("V20-causal", [&](){ launch_gqa_v20_causal<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); });
    runCorrectness("V21-causal", [&](){ launch_gqa_v21_causal<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); });
    runCorrectness("V22-causal", [&](){ launch_gqa_v22_causal<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); });
    runCorrectness("V23-causal", [&](){ launch_gqa_v23_causal<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); });
    runCorrectness("V24-causal", [&](){ launch_gqa_v24_causal<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); });
    runCorrectness("V25-causal", [&](){ launch_gqa_v25_causal<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); });
    runCorrectness("V26-causal", [&](){ launch_gqa_v26_causal<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); });

    //* V26_diag_causal: half-B is gutted to a no-op, so only half-A's rows (the
    //* FIRST Br_half=64 rows of every Br=128 q_tile block) are meaningful. Extract
    //* just those rows from both reference and GPU output before checking.
    {
      launch_gqa_v26_diag_causal<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(h_O.data(),   d_O,   Nq   * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(h_LSE.data(), d_LSE, Nlse * sizeof(float),         cudaMemcpyDeviceToHost));

      std::vector<float> h_O_f32(Nq);
      for(size_t i = 0; i < Nq; ++i) h_O_f32[i] = __bfloat162float(h_O[i]);

      constexpr int Br_half_diag = Br / 2;
      const int nQTiles = S / Br;
      std::vector<float> refA_O, gpuA_O, refA_LSE, gpuA_LSE;
      for(int bb = 0; bb < B; ++bb)
        for(int hh = 0; hh < Hq; ++hh)
          for(int qt = 0; qt < nQTiles; ++qt){
            const long rowBase = ((long)(bb * Hq + hh) * S + qt * Br);
            for(int r = 0; r < Br_half_diag; ++r){
              refA_LSE.push_back(h_LSE_ref[rowBase + r]);
              gpuA_LSE.push_back(h_LSE[rowBase + r]);
              for(int c = 0; c < D; ++c){
                const long idx = (rowBase + r) * D + c;
                refA_O.push_back(h_O_ref[idx]);
                gpuA_O.push_back(h_O_f32[idx]);
              }
            }
          }

      std::cout << "\nCorrectness V26-diag-causal (half-A rows ONLY vs PyTorch bf16 causal SDPA "
                   "— half-B is a deliberate no-op for this test):\n";
      reportPrecision("  output O ", refA_O.data(),   gpuA_O.data(),   refA_O.size());
      reportPrecision("  lse      ", refA_LSE.data(), gpuA_LSE.data(), refA_LSE.size());
      std::cout << "  O   : "; checkResult(refA_O.data(),   gpuA_O.data(),   refA_O.size(),   2e-2f, 2e-2f);
      std::cout << "  LSE : "; checkResult(refA_LSE.data(), gpuA_LSE.data(), refA_LSE.size(), 2e-2f, 2e-2f);
    }

    //* TMEM-layout probe: empirically derive the per-CTA D layout of the M=128
    //* cta_group::2 QK^T (V27's MMA shape) instead of trusting any extrapolated
    //* mapping. Each lane's raw dump is matched against host-computed raw QK^T rows
    //* (either crank's Q half x keys 0..Bc-1) at column offsets 0 and 64.
    {
      const size_t nProbe = 4ull * 128 * Bc;   // [crank][lane-base 0/128][lane][col]
      float* d_probe;
      CUDA_CHECK(cudaMalloc(&d_probe, nProbe * sizeof(float)));
      launch_gqa_tmem_probe<Br, Bc, D>(d_Q, d_K, d_probe, B, Hkv, S);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      std::vector<float> h_probe(nProbe);
      CUDA_CHECK(cudaMemcpy(h_probe.data(), d_probe, nProbe * sizeof(float), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaFree(d_probe));

      // Host raw QK^T reference (fp32 dots of the bf16 inputs, NO scaling): crank c's
      // Q rows are global rows [c*Br, c*Br+64) of b=0,h=0; keys 0..Bc-1 of b=0,hkv=0.
      std::vector<float> refS(2 * 64 * Bc);
      for(int c = 0; c < 2; ++c)
        for(int r = 0; r < 64; ++r)
          for(int k = 0; k < Bc; ++k){
            float acc = 0.f;
            const long qr = (long)(c * Br + r) * D, kr = (long)k * D;
            for(int d = 0; d < D; ++d)
              acc += __bfloat162float(h_Q[qr + d]) * __bfloat162float(h_K[kr + d]);
            refS[(c * 64 + r) * Bc + k] = acc;
          }

      std::cout << "\nTMEM-layout probe (M=128 cta_group::2 QK^T, 64 rows per CTA):\n"
                   "  entry = rRRR@cCC: probe lane's first 64 values match ref row RRR\n"
                   "  (0-63 = crank0's rows, 64-127 = crank1's rows), ref cols CC..CC+63.\n"
                   "  Expected if column-halved: lanes 0-63 -> rXXX@c000, lanes 64-127 -> rXXX@c064.\n";
      for(int c = 0; c < 2; ++c){
        for(int base = 0; base < 2; ++base){
          std::cout << "  crank " << c << ", lane-base " << base * 128 << ":\n";
          for(int l = 0; l < 128; ++l){
            const float* pv = &h_probe[(((size_t)c * 2 + base) * 128 + l) * Bc];
            int bestR = -1, bestC0 = 0; float bestErr = 1e30f;
            for(int rr = 0; rr < 128; ++rr)
              for(int c0 = 0; c0 < Bc; c0 += 64){
                float err = 0.f;
                const float* rv = &refS[(size_t)rr * Bc + c0];
                for(int j = 0; j < 64; ++j) err += fabsf(pv[j] - rv[j]);
                if(err < bestErr){ bestErr = err; bestR = rr; bestC0 = c0; }
              }
            char buf[16];
            if(bestErr < 0.5f) snprintf(buf, sizeof buf, "r%03d@c%03d", bestR, bestC0);
            else               snprintf(buf, sizeof buf, "????@????");
            std::cout << (l % 8 == 0 ? "    " : " ") << buf;
            if(l % 8 == 7) std::cout << "\n";
          }
        }
      }
      float aliasErr = 0.f;
      for(size_t i = 0; i < 128 * (size_t)Bc; ++i)
        aliasErr = fmaxf(aliasErr, fabsf(h_probe[i] - h_probe[128 * (size_t)Bc + i]));
      std::cout << "  lane-base +128 vs 0 (crank 0): max_abs_diff = " << aliasErr
                << (aliasErr < 1e-6f ? "  -> +128 aliases to the same local lanes\n"
                                     : "  -> +128 reads DIFFERENT data\n");
    }

    runCorrectness("V27-causal", [&](){ launch_gqa_v27_causal<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); });
    runCorrectness("V28-causal", [&](){ launch_gqa_v28_causal<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); });
    runCorrectness("V29-causal", [&](){ launch_gqa_v29_causal<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); });
    runCorrectness("V30-causal", [&](){ launch_gqa_v30_causal<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); });
    runCorrectness("V32-causal", [&](){ launch_gqa_v32_causal<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); });
    runCorrectness("V33-causal", [&](){ launch_gqa_v33_causal<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); });
    // V31 (restore double-buffered K/V via dynamic smem) abandoned: B300's true dynamic
    // shared memory opt-in ceiling equals its static cap exactly (232448 bytes, confirmed
    // via cudaDevAttrMaxSharedMemoryPerBlockOptin) -- no headroom above what V30 already
    // uses, so the isolated-variable test doesn't fit without a new precision compromise.
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

  //* V20-23_causal FLOPs: the shared loop bound means the pair jointly issues
  //* nPairs*(nPairs+1) MMA-iterations total (nPairs = (S/Br)/2), each a joint
  //* 2*Br-row QK^T + P@V over Bc keys — this INCLUDES rank 0's one wasted
  //* fully-masked iteration per pair (real hardware cost, just no useful
  //* output), so it's the honest FLOP/s figure for what the kernel executes.
  constexpr long long nPairs2cta    = nTiles / 2;                    // 16
  constexpr long long pairIterSum2cta = nPairs2cta * (nPairs2cta + 1); // 272
  long long flops2cta = 4LL * B * Hq * (2LL * Br) * Bc * D * pairIterSum2cta;

  stats = benchmarkKernel(
    [&](){ launch_gqa_v20_causal<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); },
    100, 25, flops2cta, bytes
  );
  displayStats("V20-causal — cta_group::2 + causal mask (full K/V dup per rank)", stats);

  stats = benchmarkKernel(
    [&](){ launch_gqa_v21_causal<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); },
    100, 25, flops2cta, bytes
  );
  displayStats("V21-causal — cta_group::2 + causal mask (genuine N-half B-split)", stats);

  stats = benchmarkKernel(
    [&](){ launch_gqa_v22_causal<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); },
    100, 25, flops2cta, bytes
  );
  displayStats("V22-causal — V21-causal + warp specialization", stats);

  stats = benchmarkKernel(
    [&](){ launch_gqa_v23_causal<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); },
    100, 25, flops2cta, bytes
  );
  displayStats("V23-causal — V22-causal + double-buffered TMA staging", stats);

  stats = benchmarkKernel(
    [&](){ launch_gqa_v24_causal<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); },
    100, 25, flops2cta, bytes
  );
  displayStats("V24-causal — V23-causal + persistent launch (grid.x halved, 2 tiles/CTA)", stats);

  stats = benchmarkKernel(
    [&](){ launch_gqa_v25_causal<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); },
    100, 25, flops2cta, bytes
  );
  displayStats("V25-causal — V23-causal + 512-thread block (widened reorder-copy)", stats);

  stats = benchmarkKernel(
    [&](){ launch_gqa_v26_causal<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); },
    100, 25, flops2cta, bytes
  );
  displayStats("V26-causal — real ping-pong: 2 independent 64-row half-pipelines", stats);

  stats = benchmarkKernel(
    [&](){ launch_gqa_v27_causal<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); },
    100, 25, flops2cta, bytes
  );
  displayStats("V27-causal — FA4/Twill 3-warpgroup: 2 softmax + 1 shared rescale group", stats);

  //* V28-causal's total real compute is IDENTICAL to V19's (same tileVisits count,
  //* just redistributed across CTAs for load balance: 1536 CTAs * 33 visits/CTA =
  //* 50688 = nTiles*(nTiles+1)/2 * B*Hq = 528*96) — reuse V19's `flops`, not the
  //* 2-CTA-pair `flops2cta` accounting used by V20-27.
  stats = benchmarkKernel(
    [&](){ launch_gqa_v28_causal<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); },
    100, 25, flops, bytes
  );
  displayStats("V28-causal — cga1x1x1 + load-balanced flat grid (matches cuDNN's real causal launch config)", stats);

  stats = benchmarkKernel(
    [&](){ launch_gqa_v29_causal<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); },
    100, 25, flops, bytes
  );
  displayStats("V29-causal — V28 + widened V-reorder-copy across the full 512-thread block", stats);

  stats = benchmarkKernel(
    [&](){ launch_gqa_v30_causal<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); },
    100, 25, flops, bytes
  );
  displayStats("V30-causal — genuine concurrent lo/hi: 2 softmax + 1 shared rescale + 1 loader", stats);

  stats = benchmarkKernel(
    [&](){ launch_gqa_v32_causal<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); },
    100, 25, flops, bytes
  );
  displayStats("V32-causal — V29 + vectorized bfloat162 reorder-copy (true 0-way bank conflict)", stats);

  stats = benchmarkKernel(
    [&](){ launch_gqa_v33_causal<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); },
    100, 25, flops, bytes
  );
  displayStats("V33-causal — V32 + QK^T/PV MMA issued before reorder-copy/rescale (intra-iteration overlap)", stats);

  CUDA_CHECK(cudaFree(d_Q));
  CUDA_CHECK(cudaFree(d_K));
  CUDA_CHECK(cudaFree(d_V));
  CUDA_CHECK(cudaFree(d_O));
  CUDA_CHECK(cudaFree(d_LSE));

  return 0;
}