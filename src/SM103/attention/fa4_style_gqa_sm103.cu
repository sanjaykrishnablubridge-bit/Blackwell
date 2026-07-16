// ============================================================================
// fa4_style_gqa_sm103.cu   (rev 2 — synchronization fixes)
//
// Re-implementation (from SASS reverse-engineering) of the architecture of:
//   cudnn_generated_fort_native_sdpa_sm100_flash_fprop_f16_knob_7
//     _128x128x64_4x1x1_cga1x1x1_kernel0_0
//
// Target: NVIDIA Blackwell B300 (sm_103a; also builds for sm_100a).
// Compile: nvcc -arch=sm_103a -O3 -std=c++17 fa4_style_gqa_sm103.cu -lcuda
//
// ---------------------------------------------------------------------------
// rev 2 changelog (vs. the first version):
//  FIX-1  tcgen05.alloc / relinquish / dealloc are warp-collective
//         (.sync.aligned): now executed by the WHOLE MMA warp, and the alloc
//         is hoisted before role dispatch so the init __syncthreads publishes
//         S.tmem_base to every thread. (Was: inside elect_one() -> hang.)
//  FIX-2  Removed nested elect.sync inside an elected-thread region (UB).
//  FIX-3  Removed named barriers 6/7 for tmem_base publication (the
//         correction WG could read tmem_base before the MMA warp wrote it).
//  FIX-4  K/V ring slots are now released with tcgen05.commit (arrives the
//         mbarrier only after the async MMAs actually finished reading smem).
//         (Was: manual mbar_arrive right after issue -> producer could
//         overwrite K/V mid-MMA.)
//  FIX-5  Correction WG additionally waits PV(j-1) completion
//         (bar_p_freed[(j-1)&1]) before rescaling O. (Was: could rescale O
//         concurrently with an in-flight PV accumulate.)
//  FIX-6  (m,l) state handoff starts at j>=1 (tile 1 was running with
//         m=-inf), and the duplicate tail-state store is removed (two WGs
//         could write TMEM_STATE concurrently near the end).
//  FIX-7  tcgen05.commit uses .shared::cluster (ptxas rejects .shared::cta).
//
// Datatypes preserved: bf16 Q/K/V/O, fp32 accumulation (tcgen05 f16-kind MMA
// with fp32 D), fp32 softmax with running-max stabilization, exp2 with
// log2(e)*scale folded into an FFMA.
//
// ---------------------------------------------------------------------------
// SASS evidence -> design mapping
// ---------------------------------------------------------------------------
//  SASS observation                          | This file
//  ------------------------------------------+------------------------------
//  tid splits at 0x100/0x180; SETMAXREG      | 4 role groups:
//    TRY_ALLOC 0xc0 / DEALLOC 0x58 / 0x28    |   warps 0-7  : softmax x2 (192r)
//                                            |   warps 8-11 : correction (88r)
//                                            |   warps 12-15: load/mma/store(40r)
//  ~35 mbarriers @ smem, SYNCS.ARRIVE /      | mbarrier ring buffers; zero
//    PHASECHK.TRYWAIT everywhere, few BAR    |   __syncthreads in main loop
//  UTMALDG.4D w/ expect-tx arrives           | cp.async.bulk.tensor.4d (TMA)
//  UTCHMMA gdesc,gdesc,tmem  x4, !UPT first  | QK^T: A,B smem desc, 4 MMAs/K64
//  two S accumulators tmem[UR62]/[UR58]      | double-buffered S in TMEM
//  UTCHMMA tmem,gdesc,tmem x8, UP0 first     | PV: A = P in TMEM, cond. accum
//  LDTM.x32 x4 per thread, FMNMX3, no SHFL   | 1 thread owns 1 full S row
//  FFMA2(scale, s, -m*scale) -> MUFU.EX2     | exp2 with folded scale
//  F2FP.BF16.PACK_AB + STTM                  | P packed bf16 -> TMEM (no smem)
//  STTM scalar alpha; LDTM/FMUL2/STTM @+0x100| correction WG rescales O in TMEM
//  STTM.x2 / LDTM.x2 (m,l)                   | ping-pong state via TMEM
//  UTCATOMSWS.FIND_AND_SET + NANOSLEEP       | tcgen05.alloc retry
//  STS.128 + UTMASTG.4D + UTMACMDFLUSH       | epilogue: smem stage + TMA store
//
// ---------------------------------------------------------------------------
// REMAINING CAVEATS:
//  * UMMA smem-descriptor / instruction-descriptor bitfields follow CUTLASS
//    (cutlass/arch/mma_sm100_desc.hpp) and PTX ISA 8.7 — VERIFY against your
//    CUTLASS version. This is the #1 place a silent wrong-answer bug hides.
//  * PV smem-descriptor stride math and the TMEM P-operand column stride
//    (MMA_K/2 words per K=16 step) are marked [VERIFY].
//  * Epilogue stages linearly and the O tensor map is non-swizzled; if you
//    switch it to SW128 the staging stores must be swizzled to match.
// ============================================================================

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cudaTypedefs.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include "utils/kernelUtils.cuh"
#include "utils/kernelBench.cuh"

// ------------------------------ configuration ------------------------------
#ifndef HEAD_DIM
#define HEAD_DIM 128           // 64 or 128
#endif
#define STR_(x) #x
#define STR(x) STR_(x)
#ifndef CAUSAL
#define CAUSAL 1
#endif

constexpr int TILE_M     = 128;              // Q rows per CTA
constexpr int TILE_N     = 128;              // KV rows per inner tile
constexpr int D          = HEAD_DIM;
constexpr int D_CHUNKS   = D / 64;           // TMA/MMA operate in 64-wide d-chunks
constexpr int KV_STAGES  = 2;                // K and V ring depth (3 fits for D=64)
constexpr int MMA_K      = 16;               // tcgen05 kind::f16 K per instruction
constexpr int NUM_THREADS= 512;

// TMEM column map (128 lanes x 512 cols of 32-bit words per CTA).
// P (bf16 packed, 64 cols) is written over the S buffer it came from once the
// S values live in softmax registers — same trick as the SASS (+0x40..).
constexpr int TMEM_S0    = 0;      // S ping [128 cols f32]; P0 packed at S0+0
constexpr int TMEM_S1    = 128;    // S pong [128 cols f32]; P1 packed at S1+0
constexpr int TMEM_O     = 256;    // O accumulator [D cols f32]  (SASS: +0x100)
constexpr int TMEM_ALPHA = 256 + D;      // per-row correction scalar [2: ping/pong]
constexpr int TMEM_STATE = 256 + D + 2;  // per-row (m,l) running state [2 cols]
constexpr int TMEM_COLS  = 512;    // allocate the full bank (SASS allocs once)

// ------------------------------- PTX helpers -------------------------------
#define DEVICE __device__ __forceinline__

DEVICE uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

DEVICE bool elect_one() {           // SASS: ELECT P1, URZ, PT
  uint32_t pred;
  asm volatile("{.reg .pred P;\n elect.sync _|P, 0xffffffff;\n selp.u32 %0,1,0,P;}\n"
               : "=r"(pred));
  return pred != 0;
}

// ---- mbarrier ------------------------------------------------------------
DEVICE void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
               :: "r"(smem_u32(bar)), "r"(count));
}
DEVICE void mbar_arrive(uint64_t* bar) {   // SASS: SYNCS.ARRIVE.TRANS64.A1T0
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n"
               :: "r"(smem_u32(bar)));
}
DEVICE void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
               :: "r"(smem_u32(bar)), "r"(bytes));
}
// Spin on parity. SASS pattern: SYNCS.PHASECHK.TRANS64.TRYWAIT + BRA loop.
DEVICE void mbar_wait(uint64_t* bar, uint32_t phase) {
  asm volatile(
    "{.reg .pred P;\n"
    "WAIT_%=:\n"
    " mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
    " @!P bra WAIT_%=;\n}"
    :: "r"(smem_u32(bar)), "r"(phase));
}

// ---- register re-partitioning (SASS: USETMAXREG.*) -------------------------
DEVICE void regs_alloc_192()  { asm volatile("setmaxnreg.inc.sync.aligned.u32 192;\n"); }
DEVICE void regs_dealloc_88() { asm volatile("setmaxnreg.dec.sync.aligned.u32  88;\n"); }
DEVICE void regs_dealloc_40() { asm volatile("setmaxnreg.dec.sync.aligned.u32  40;\n"); }

// ---- async proxy fences (SASS: FENCE.VIEW.ASYNC.S / .T) ---------------------
DEVICE void fence_async_smem()   { asm volatile("fence.proxy.async.shared::cta;\n"); }
DEVICE void tcgen05_fence_before(){ asm volatile("tcgen05.fence::before_thread_sync;\n"); }
DEVICE void tcgen05_fence_after() { asm volatile("tcgen05.fence::after_thread_sync;\n"); }

// ---- TMA (SASS: UTMALDG.4D / UTMASTG.4D + UTMACMDFLUSH) ---------------------
DEVICE void tma_load_4d(const CUtensorMap* map, void* smem_dst, uint64_t* bar,
                        int c0, int c1, int c2, int c3) {
  asm volatile(
    "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
    " [%0], [%1, {%2, %3, %4, %5}], [%6];\n"
    :: "r"(smem_u32(smem_dst)), "l"(map),
       "r"(c0), "r"(c1), "r"(c2), "r"(c3),
       "r"(smem_u32(bar))
    : "memory");
}
DEVICE void tma_store_4d(const CUtensorMap* map, const void* smem_src,
                         int c0, int c1, int c2, int c3) {
  asm volatile(
    "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
    " [%0, {%1, %2, %3, %4}], [%5];\n"
    :: "l"(map), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(smem_u32(smem_src))
    : "memory");
}
DEVICE void tma_store_commit() { asm volatile("cp.async.bulk.commit_group;\n"); }
template <int N> DEVICE void tma_store_wait() {
  asm volatile("cp.async.bulk.wait_group.read %0;\n" :: "n"(N) : "memory");
}

// ---- TMEM alloc (SASS: UTCATOMSWS.FIND_AND_SET + NANOSLEEP retry) -----------
// FIX-1: these are warp-collective (.sync.aligned) — the WHOLE warp must call.
DEVICE void tmem_alloc_warp(uint32_t* smem_result, uint32_t ncols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
               :: "r"(smem_u32(smem_result)), "r"(ncols));
}
DEVICE void tmem_relinquish_warp() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n");
}
DEVICE void tmem_dealloc_warp(uint32_t taddr, uint32_t ncols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n"
               :: "r"(taddr), "r"(ncols));
}

// ---- UMMA shared-memory matrix descriptor -----------------------------------
// VERIFY vs cutlass::arch (UMMA::SmemDescriptor):
//  bits [ 0:14) matrix start address >> 4
//  bits [16:30) leading byte offset  >> 4
//  bits [32:46) stride  byte offset  >> 4
//  bits [49:52) base offset
//  bits [61:64) layout/swizzle: 0=none, 1=128B, 2=64B, 3=32B (SW128 used here)
DEVICE uint64_t make_smem_desc_sw128(uint32_t smem_addr,
                                     uint32_t lbo_bytes, uint32_t sbo_bytes) {
  uint64_t d = 0;
  d |= (uint64_t)((smem_addr >> 4) & 0x3FFF);
  d |= (uint64_t)((lbo_bytes  >> 4) & 0x3FFF) << 16;
  d |= (uint64_t)((sbo_bytes  >> 4) & 0x3FFF) << 32;
  d |= (uint64_t)1 << 62;                       // SWIZZLE_128B
  return d;
}
// Canonical SW128 K-major bf16 atom (8 x 64 elems): LBO=16B, SBO=1024B.
// The 4 per-K-step descs in the SASS advance by +2 (=32B >> 4), i.e. K+16 elems.
constexpr uint32_t SW128_LBO = 16;
constexpr uint32_t SW128_SBO = 1024;

// ---- UMMA instruction descriptor (kind::f16) --------------------------------
// VERIFY vs cutlass/arch/mma_sm100_desc.hpp (UMMA::InstrDescriptor):
//  [0:2) sparse id2   [2] sparse en   [3] saturate
//  [4:6) D format: 0=f16, 1=f32
//  [7:10) A dtype: 0=f16, 1=bf16      [10:13) B dtype
//  [13] negate A  [14] negate B  [15] transpose A  [16] transpose B
//  [17:23) N >> 3    [24:29) M >> 4   [30:32) max-shift (ws only)
DEVICE constexpr uint32_t make_idesc_f16(int M, int N, bool a_bf16, bool b_bf16,
                                         bool trans_a, bool trans_b) {
  uint32_t d = 0;
  d |= 1u << 4;                       // D = f32 accumulate
  d |= (a_bf16 ? 1u : 0u) << 7;
  d |= (b_bf16 ? 1u : 0u) << 10;
  d |= (trans_a ? 1u : 0u) << 15;
  d |= (trans_b ? 1u : 0u) << 16;
  d |= ((uint32_t)(N >> 3) & 0x3F) << 17;
  d |= ((uint32_t)(M >> 4) & 0x1F) << 24;
  return d;
}

// ---- tcgen05 MMA (SASS: UTCHMMA) --------------------------------------------
// QK^T flavor: A and B from smem descriptors, D in TMEM.
DEVICE void umma_ss(uint32_t d_tmem, uint64_t a_desc, uint64_t b_desc,
                    uint32_t idesc, bool accumulate) {
  asm volatile(
    "{.reg .pred P;\n setp.ne.b32 P, %4, 0;\n"
    " tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, P;\n}"
    :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc),
       "r"((int)accumulate));
}
// PV flavor: A from TMEM (packed bf16 P), B from smem, D in TMEM.
DEVICE void umma_ts(uint32_t d_tmem, uint32_t a_tmem, uint64_t b_desc,
                    uint32_t idesc, bool accumulate) {
  asm volatile(
    "{.reg .pred P;\n setp.ne.b32 P, %4, 0;\n"
    " tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, P;\n}"
    :: "r"(d_tmem), "r"(a_tmem), "l"(b_desc), "r"(idesc),
       "r"((int)accumulate));
}
// SASS: UTCBAR. FIX-7: ptxas requires .shared::cluster here; a CTA-local
// shared address is a valid cluster address, so smem_u32() is still correct.
// Arrives the mbarrier only when all previously issued tcgen05 ops complete —
// this is also what makes it the ONLY correct way to release K/V ring slots
// and P buffers (FIX-4).
DEVICE void umma_commit(uint64_t* bar) {
  asm volatile(
    "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n"
    :: "r"(smem_u32(bar)));
}

// ---- TMEM load/store (SASS: LDTM.x32 / STTM.*) -------------------------------
// TMEM address: bits [31:16] lane, [15:0] column (32-bit word units).
// 32x32b shape: a warp covers 32 lanes starting at the lane in the address.
DEVICE uint32_t tmem_addr(uint32_t base, int lane, int col) {
  return base + ((uint32_t)lane << 16) + (uint32_t)col;
}
DEVICE void tmem_ld_x32(uint32_t taddr, float* r) {  // 32 f32 per thread
  asm volatile(
    "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
    "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];\n"
    : "=f"(r[0]),"=f"(r[1]),"=f"(r[2]),"=f"(r[3]),"=f"(r[4]),"=f"(r[5]),
      "=f"(r[6]),"=f"(r[7]),"=f"(r[8]),"=f"(r[9]),"=f"(r[10]),"=f"(r[11]),
      "=f"(r[12]),"=f"(r[13]),"=f"(r[14]),"=f"(r[15]),"=f"(r[16]),"=f"(r[17]),
      "=f"(r[18]),"=f"(r[19]),"=f"(r[20]),"=f"(r[21]),"=f"(r[22]),"=f"(r[23]),
      "=f"(r[24]),"=f"(r[25]),"=f"(r[26]),"=f"(r[27]),"=f"(r[28]),"=f"(r[29]),
      "=f"(r[30]),"=f"(r[31])
    : "r"(taddr));
}
DEVICE void tmem_ld_x1(uint32_t taddr, float& a) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];\n"
               : "=f"(a) : "r"(taddr));
}
DEVICE void tmem_ld_x2(uint32_t taddr, float& a, float& b) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x2.b32 {%0,%1}, [%2];\n"
               : "=f"(a), "=f"(b) : "r"(taddr));
}
DEVICE void tmem_st_x32(uint32_t taddr, const uint32_t* r) {
  asm volatile(
    "tcgen05.st.sync.aligned.32x32b.x32.b32 [%32], "
    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
    "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31};\n"
    :: "r"(r[0]),"r"(r[1]),"r"(r[2]),"r"(r[3]),"r"(r[4]),"r"(r[5]),
       "r"(r[6]),"r"(r[7]),"r"(r[8]),"r"(r[9]),"r"(r[10]),"r"(r[11]),
       "r"(r[12]),"r"(r[13]),"r"(r[14]),"r"(r[15]),"r"(r[16]),"r"(r[17]),
       "r"(r[18]),"r"(r[19]),"r"(r[20]),"r"(r[21]),"r"(r[22]),"r"(r[23]),
       "r"(r[24]),"r"(r[25]),"r"(r[26]),"r"(r[27]),"r"(r[28]),"r"(r[29]),
       "r"(r[30]),"r"(r[31]), "r"(taddr));
}
DEVICE void tmem_st_x1(uint32_t taddr, float v) {   // SASS: STTM scalar (alpha)
  asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%1], {%0};\n"
               :: "f"(v), "r"(taddr));
}
DEVICE void tmem_st_x2(uint32_t taddr, float a, float b) {
  asm volatile("tcgen05.st.sync.aligned.32x32b.x2.b32 [%2], {%0,%1};\n"
               :: "f"(a), "f"(b), "r"(taddr));
}
DEVICE void tmem_wait_ld() { asm volatile("tcgen05.wait::ld.sync.aligned;\n"); }
DEVICE void tmem_wait_st() { asm volatile("tcgen05.wait::st.sync.aligned;\n"); }

// ------------------------------ kernel params -------------------------------
struct FA4Params {
  CUtensorMap tma_q;   // (d, s_q,  h_q,  b)   SW128
  CUtensorMap tma_k;   // (d, s_kv, h_kv, b)   SW128
  CUtensorMap tma_v;   // (d, s_kv, h_kv, b)   SW128
  CUtensorMap tma_o;   // (d, s_q,  h_q,  b)   linear
  int   b, h_q, h_kv, s_q, s_kv;
  int   group;             // h_q / h_kv  (GQA)
  float scale;             // 1/sqrt(d)
  float scale_log2e;       // scale * log2(e)  — folded, SASS c[0x0][0x480]
};

// ------------------------------ shared memory --------------------------------
struct __align__(1024) SharedStorage {
  // 128B-swizzled bf16 tiles. [stage][d_chunk][128 rows][64 cols]
  __align__(1024) __nv_bfloat16 q[D_CHUNKS][TILE_M][64];
  __align__(1024) __nv_bfloat16 k[KV_STAGES][D_CHUNKS][TILE_N][64];
  __align__(1024) __nv_bfloat16 v[KV_STAGES][D_CHUNKS][TILE_N][64];
  __align__(1024) __nv_bfloat16 o_stage[D_CHUNKS][TILE_M][64];  // epilogue stage
  uint32_t tmem_base;                       // written by tcgen05.alloc
  // ---- mbarriers (SASS initializes ~35 at +0x1c000) ----
  uint64_t bar_q_full;
  uint64_t bar_k_full [KV_STAGES];
  uint64_t bar_k_empty[KV_STAGES];
  uint64_t bar_v_full [KV_STAGES];
  uint64_t bar_v_empty[KV_STAGES];
  uint64_t bar_s_full [2];      // MMA -> softmax(b): S ready in TMEM (commit)
  uint64_t bar_p_full [2];      // softmax(b) -> MMA: P packed in TMEM
  uint64_t bar_p_freed[2];      // MMA commit: PV(b) done; S/P buf reusable AND
                                //   (FIX-5) O safe to rescale for next tile
  uint64_t bar_alpha  [2];      // softmax(b) -> correction: alpha in TMEM
  uint64_t bar_state  [2];      // softmax(b) -> softmax(1-b): (m,l) handoff
  uint64_t bar_corr_done[2];    // correction -> MMA: O rescaled, PV may accum
  uint64_t bar_o_done;          // MMA commit: last PV finished
  uint64_t bar_epi;             // correction -> store warp: o_stage ready
};

// ============================================================================
//                                   KERNEL
// ============================================================================
__global__ void __launch_bounds__(NUM_THREADS, 1)
fa4_gqa_kernel(const __grid_constant__ FA4Params p) {
  extern __shared__ char smem_raw[];
  SharedStorage& S = *reinterpret_cast<SharedStorage*>(smem_raw);

  const int tid  = threadIdx.x;
  const int warp = tid >> 5;

  // ---- grid mapping: x = q tile, y = q head, z = batch ---------------------
  const int q_tile = blockIdx.x;
  const int hq     = blockIdx.y;
  const int hkv    = hq / p.group;
  const int bb     = blockIdx.z;
  const int q_row0 = q_tile * TILE_M;

  const int n_kv_tiles_full = (p.s_kv + TILE_N - 1) / TILE_N;
#if CAUSAL
  const int n_kv_tiles = min(n_kv_tiles_full,
                             (q_row0 + TILE_M - 1) / TILE_N + 1);
#else
  const int n_kv_tiles = n_kv_tiles_full;
#endif

  // ---- init: one warp sets up mbarriers; MMA warp allocates TMEM -----------
  if (warp == 0 && elect_one()) {
    mbar_init(&S.bar_q_full, 1);
    for (int s = 0; s < KV_STAGES; ++s) {
      mbar_init(&S.bar_k_full [s], 1);
      mbar_init(&S.bar_k_empty[s], 1);   // arrived via tcgen05.commit
      mbar_init(&S.bar_v_full [s], 1);
      mbar_init(&S.bar_v_empty[s], 1);   // arrived via tcgen05.commit
    }
    for (int b = 0; b < 2; ++b) {
      mbar_init(&S.bar_s_full [b], 1);   // tcgen05.commit (one arrival)
      mbar_init(&S.bar_p_full [b], 128); // whole softmax WG arrives
      mbar_init(&S.bar_p_freed[b], 1);   // tcgen05.commit
      mbar_init(&S.bar_alpha  [b], 128);
      mbar_init(&S.bar_state  [b], 128);
      mbar_init(&S.bar_corr_done[b], 128);
    }
    mbar_init(&S.bar_o_done, 1);         // tcgen05.commit
    mbar_init(&S.bar_epi, 128);
    fence_async_smem();                  // SASS: FENCE.VIEW.ASYNC.S
  }
  // FIX-1 + FIX-3: whole MMA warp allocates TMEM here, before role dispatch,
  // so the __syncthreads below publishes both the barriers and tmem_base.
  if (warp == 13) {
    tmem_alloc_warp(&S.tmem_base, TMEM_COLS);   // UTCATOMSWS.FIND_AND_SET
    tmem_relinquish_warp();
  }
  __syncthreads();                       // SASS: BAR.SYNC 0x0 (init only)
  const uint32_t tmem = S.tmem_base;

  // =========================================================================
  // ROLE DISPATCH  (SASS: ISETP on tid vs 0x100 / 0x180 + SETMAXREG)
  // =========================================================================
  if (tid < 256) {
    // -----------------------------------------------------------------
    // SOFTMAX warpgroups (ping-pong): WG0 = threads 0..127 -> even tiles,
    // WG1 = 128..255 -> odd tiles. Thread t owns TMEM lane (row) t%128.
    // -----------------------------------------------------------------
    regs_alloc_192();
    const int wg    = tid >> 7;              // 0 or 1
    const int row   = tid & 127;             // owned S row == TMEM lane
    const int lane0 = (warp & 3) * 32;       // warp's TMEM lane base
    const int sbuf_col = wg ? TMEM_S1 : TMEM_S0;

    uint32_t phase_s = 0, phase_pf = 0, phase_state = 0;

    for (int j = wg; j < n_kv_tiles; j += 2) {
      // ---- wait S = Q K^T for tile j (tcgen05.commit by MMA warp) ----
      mbar_wait(&S.bar_s_full[wg], phase_s);
      tcgen05_fence_after();

      // ---- load my full row: 4 x LDTM.x32 = 128 f32 ----
      float s_row[TILE_N];
      #pragma unroll
      for (int c = 0; c < 4; ++c)
        tmem_ld_x32(tmem_addr(tmem, lane0, sbuf_col + 32 * c), &s_row[32 * c]);
      tmem_wait_ld();

      // ---- masking (branchless, matches FSEL/-INF patterns) ----
      const int kv0 = j * TILE_N;
      #pragma unroll
      for (int c = 0; c < TILE_N; ++c) {
        int kv = kv0 + c;
        bool valid = (kv < p.s_kv);
#if CAUSAL
        valid = valid && (kv <= q_row0 + row);
#endif
        s_row[c] = valid ? s_row[c] : -INFINITY;
      }

      // ---- running (m,l) state ----
      // FIX-6: every tile except the very first (j>=1) consumes the state
      // produced by the other WG on the previous tile. (Was j>=2: tile 1 ran
      // with m=-inf — numerically wrong.)
      float m_old = -INFINITY, l_old = 0.f;
      if (j >= 1) {
        mbar_wait(&S.bar_state[1 - wg], phase_state);
        tcgen05_fence_after();
        tmem_ld_x2(tmem_addr(tmem, lane0, TMEM_STATE), m_old, l_old);
        tmem_wait_ld();
        phase_state ^= 1;
      }

      // ---- row max (FMNMX3-style tree) ----
      float m_new = m_old;
      #pragma unroll
      for (int c = 0; c < TILE_N; ++c) m_new = fmaxf(m_new, s_row[c]);

      // ---- exp2 with folded scale: e = exp2(s*sl2e - m*sl2e) (FFMA2+EX2)
      const float neg_m_sl = (m_new == -INFINITY) ? 0.f : -m_new * p.scale_log2e;
      float l_add = 0.f;
      #pragma unroll
      for (int c = 0; c < TILE_N; ++c) {
        float e = (s_row[c] == -INFINITY)
                    ? 0.f
                    : exp2f(fmaf(s_row[c], p.scale_log2e, neg_m_sl));
        s_row[c] = e;
        l_add += e;
      }
      // correction for previously accumulated O (guard -inf as SASS does)
      const float alpha = (m_old == -INFINITY)
                            ? 1.f
                            : exp2f(fmaf(m_old, p.scale_log2e, neg_m_sl));
      const float l_new = fmaf(l_old, alpha, l_add);

      // ---- publish alpha for the correction WG (SASS: scalar STTM) ----
      if (j > 0) {
        tmem_st_x1(tmem_addr(tmem, lane0, TMEM_ALPHA + wg), alpha);
        tmem_wait_st();
        tcgen05_fence_before();
        mbar_arrive(&S.bar_alpha[wg]);
      }

      // ---- publish (m,l) for the other WG (and, on the last tile, for the
      //      epilogue: ordered before bar_o_done through the P->PV chain) ----
      tmem_st_x2(tmem_addr(tmem, lane0, TMEM_STATE), m_new, l_new);
      tmem_wait_st();
      tcgen05_fence_before();
      mbar_arrive(&S.bar_state[wg]);
      // FIX-6: the old "tail state stash" block is deleted — it made two WGs
      // write TMEM_STATE concurrently near the end.

      // ---- pack P to bf16 pairs and store over the S buffer ----
      // (F2FP.BF16.PACK_AB + STTM in SASS; 64 words per row)
      uint32_t p_packed[TILE_N / 2];
      #pragma unroll
      for (int c = 0; c < TILE_N / 2; ++c) {
        __nv_bfloat162 h = __float22bfloat162_rn(
            make_float2(s_row[2 * c], s_row[2 * c + 1]));
        p_packed[c] = *reinterpret_cast<uint32_t*>(&h);
      }
      #pragma unroll
      for (int c = 0; c < 2; ++c)
        tmem_st_x32(tmem_addr(tmem, lane0, sbuf_col + 32 * c),
                    &p_packed[32 * c]);
      tmem_wait_st();
      tcgen05_fence_before();
      mbar_arrive(&S.bar_p_full[wg]);     // whole WG arrives (count=128)

      // ---- wait until PV consumed P before touching this buffer again ----
      mbar_wait(&S.bar_p_freed[wg], phase_pf);
      phase_pf ^= 1;
      phase_s  ^= 1;
    }

  } else if (tid < 384) {
    // -----------------------------------------------------------------
    // CORRECTION + EPILOGUE warpgroup (SASS: dealloc->0x58, LDTM/FMUL2/STTM
    // at O(+0x100), then normalize, STS.128, hand off to TMA-store warp)
    // -----------------------------------------------------------------
    regs_dealloc_88();
    const int row   = tid & 127;
    const int lane0 = (warp & 3) * 32;

    uint32_t phase_a[2] = {0, 0}, phase_pf[2] = {0, 0}, phase_od = 0;
    // Per-tile O rescale: O *= alpha(j) before PV(j) accumulates.
    for (int j = 1; j < n_kv_tiles; ++j) {
      const int wg = j & 1;
      // alpha(j) from softmax WG(j)
      mbar_wait(&S.bar_alpha[wg], phase_a[wg]);          phase_a[wg]    ^= 1;
      // FIX-5: PV(j-1) must have finished accumulating into O before we
      // rescale it. bar_p_freed[(j-1)&1] is the tcgen05.commit of PV(j-1).
      mbar_wait(&S.bar_p_freed[1 - wg], phase_pf[1 - wg]); phase_pf[1 - wg] ^= 1;
      tcgen05_fence_after();

      float alpha;
      tmem_ld_x1(tmem_addr(tmem, lane0, TMEM_ALPHA + wg), alpha);
      tmem_wait_ld();

      float o_row[32];
      #pragma unroll
      for (int c = 0; c < D; c += 32) {
        tmem_ld_x32(tmem_addr(tmem, lane0, TMEM_O + c), o_row);
        tmem_wait_ld();
        uint32_t o_bits[32];
        #pragma unroll
        for (int k = 0; k < 32; ++k) {
          float v = o_row[k] * alpha;                    // FMUL2 in SASS
          o_bits[k] = *reinterpret_cast<uint32_t*>(&v);
        }
        tmem_st_x32(tmem_addr(tmem, lane0, TMEM_O + c), o_bits);
      }
      tmem_wait_st();
      tcgen05_fence_before();
      mbar_arrive(&S.bar_corr_done[wg]);   // MMA may now issue PV(j) accumulate
    }

    // ---- epilogue: wait last PV, read final (m,l), normalize, stage, store
    mbar_wait(&S.bar_o_done, phase_od);
    tcgen05_fence_after();
    float m_fin, l_fin;
    tmem_ld_x2(tmem_addr(tmem, lane0, TMEM_STATE), m_fin, l_fin);
    tmem_wait_ld();
    (void)m_fin;
    const float inv_l = (l_fin > 0.f) ? __frcp_rn(l_fin) : 0.f;  // MUFU.RCP

    #pragma unroll
    for (int dc = 0; dc < D_CHUNKS; ++dc) {
      float o_row[64];
      tmem_ld_x32(tmem_addr(tmem, lane0, TMEM_O + dc * 64),      &o_row[0]);
      tmem_ld_x32(tmem_addr(tmem, lane0, TMEM_O + dc * 64 + 32), &o_row[32]);
      tmem_wait_ld();
      // pack + 128-bit stores into the (linear) staging tile
      #pragma unroll
      for (int c = 0; c < 64; c += 8) {
        __nv_bfloat16 h[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) h[k] = __float2bfloat16(o_row[c + k] * inv_l);
        // NOTE: staging is linear; the O TMA map is declared without swizzle,
        // so this matches. If you switch the O map to SW128 you must swizzle
        // this store to match. [VERIFY]
        *reinterpret_cast<uint4*>(&S.o_stage[dc][row][c]) =
            *reinterpret_cast<uint4*>(h);
      }
    }
    fence_async_smem();                    // SASS: FENCE.VIEW.ASYNC before TMA
    mbar_arrive(&S.bar_epi);               // store warp takes over

  } else if (warp == 12) {
    // -----------------------------------------------------------------
    // TMA LOAD producer warp (SASS: elected lane; expect-tx + UTMALDG.4D)
    // -----------------------------------------------------------------
    regs_dealloc_40();
    if (elect_one()) {
      constexpr uint32_t CHUNK_BYTES = TILE_N * 64 * sizeof(__nv_bfloat16);
      // Q once
      mbar_arrive_expect_tx(&S.bar_q_full, D_CHUNKS * CHUNK_BYTES);
      for (int dc = 0; dc < D_CHUNKS; ++dc)
        tma_load_4d(&p.tma_q, &S.q[dc][0][0], &S.bar_q_full,
                    dc * 64, q_row0, hq, bb);
      // KV ring
      uint32_t phase_ke[KV_STAGES] = {}, phase_ve[KV_STAGES] = {};
      for (int j = 0; j < n_kv_tiles; ++j) {
        const int st = j % KV_STAGES;
        if (j >= KV_STAGES) { mbar_wait(&S.bar_k_empty[st], phase_ke[st]);
                              phase_ke[st] ^= 1; }
        mbar_arrive_expect_tx(&S.bar_k_full[st], D_CHUNKS * CHUNK_BYTES);
        for (int dc = 0; dc < D_CHUNKS; ++dc)
          tma_load_4d(&p.tma_k, &S.k[st][dc][0][0], &S.bar_k_full[st],
                      dc * 64, j * TILE_N, hkv, bb);
        if (j >= KV_STAGES) { mbar_wait(&S.bar_v_empty[st], phase_ve[st]);
                              phase_ve[st] ^= 1; }
        mbar_arrive_expect_tx(&S.bar_v_full[st], D_CHUNKS * CHUNK_BYTES);
        for (int dc = 0; dc < D_CHUNKS; ++dc)
          tma_load_4d(&p.tma_v, &S.v[st][dc][0][0], &S.bar_v_full[st],
                      dc * 64, j * TILE_N, hkv, bb);
      }
    }

  } else if (warp == 13) {
    // -----------------------------------------------------------------
    // MMA warp (SASS: UTCHMMA groups, UTCBAR commits). TMEM was allocated
    // by this whole warp before role dispatch (FIX-1/FIX-3).
    // -----------------------------------------------------------------
    regs_dealloc_40();
    if (elect_one()) {
      const uint32_t idesc_qk = make_idesc_f16(TILE_M, TILE_N, true, true,
                                               /*tA=*/false, /*tB=*/true);
      const uint32_t idesc_pv = make_idesc_f16(TILE_M, 64,     true, true,
                                               false, false);
      uint32_t phase_q = 0, phase_kf[KV_STAGES] = {}, phase_vf[KV_STAGES] = {};
      uint32_t phase_p[2] = {}, phase_cd[2] = {};

      mbar_wait(&S.bar_q_full, phase_q);
      fence_async_smem();

      for (int j = 0; j < n_kv_tiles; ++j) {
        const int st = j % KV_STAGES;
        const int wg = j & 1;
        const int scol = wg ? TMEM_S1 : TMEM_S0;

        // ---- S = Q · K^T : 4 UMMAs per 64-d-chunk, first non-accumulating
        mbar_wait(&S.bar_k_full[st], phase_kf[st]); phase_kf[st] ^= 1;
        fence_async_smem();
        bool acc = false;
        for (int dc = 0; dc < D_CHUNKS; ++dc) {
          uint32_t qa = smem_u32(&S.q[dc][0][0]);
          uint32_t kb = smem_u32(&S.k[st][dc][0][0]);
          #pragma unroll
          for (int kk = 0; kk < 64 / MMA_K; ++kk) {       // 4x, desc += 32B
            umma_ss(tmem + scol,
                    make_smem_desc_sw128(qa + kk * 32, SW128_LBO, SW128_SBO),
                    make_smem_desc_sw128(kb + kk * 32, SW128_LBO, SW128_SBO),
                    idesc_qk, acc);
            acc = true;
          }
        }
        umma_commit(&S.bar_s_full[wg]);     // softmax WG wakes on this
        // FIX-4 + FIX-2: release the K slot with a commit (fires only after
        // the async MMAs above finished READING smem), not a manual arrive.
        umma_commit(&S.bar_k_empty[st]);

        // ---- wait P from softmax, wait O rescaled, then O += P · V ----
        mbar_wait(&S.bar_p_full[wg], phase_p[wg]); phase_p[wg] ^= 1;
        if (j > 0) { mbar_wait(&S.bar_corr_done[wg], phase_cd[wg]);
                     phase_cd[wg] ^= 1; }
        mbar_wait(&S.bar_v_full[st], phase_vf[st]); phase_vf[st] ^= 1;
        fence_async_smem();
        tcgen05_fence_after();

        for (int dc = 0; dc < D_CHUNKS; ++dc) {
          uint32_t vb = smem_u32(&S.v[st][dc][0][0]);
          bool acc_pv = (j > 0);
          #pragma unroll
          for (int kk = 0; kk < TILE_N / MMA_K; ++kk) {   // 8x, K=128
            // A = P from TMEM: bf16 pairs, MMA_K/2 = 8 cols per K=16 step
            umma_ts(tmem + TMEM_O + dc * 64,
                    tmem_addr(tmem, 0, scol + kk * (MMA_K / 2)),
                    // V rows advance 16 per K step: +16 rows * 128B [VERIFY]
                    make_smem_desc_sw128(vb + kk * MMA_K * 128,
                                         SW128_LBO, SW128_SBO),
                    idesc_pv, acc_pv);
            acc_pv = true;
          }
        }
        // PV(j) completion: releases the S/P buffer for softmax WG reuse AND
        // tells the correction WG that O is stable (FIX-5 consumer).
        umma_commit(&S.bar_p_freed[wg]);
        umma_commit(&S.bar_v_empty[st]);    // FIX-4 + FIX-2
        if (j == n_kv_tiles - 1) umma_commit(&S.bar_o_done);
      }
    }

  } else if (warp == 14) {
    // -----------------------------------------------------------------
    // TMA STORE warp (SASS: UTMASTG.4D + UTMACMDFLUSH + DEPBAR)
    // -----------------------------------------------------------------
    regs_dealloc_40();
    uint32_t phase = 0;
    mbar_wait(&S.bar_epi, phase);
    if (elect_one()) {
      for (int dc = 0; dc < D_CHUNKS; ++dc)
        tma_store_4d(&p.tma_o, &S.o_stage[dc][0][0],
                     dc * 64, q_row0, hq, bb);
      tma_store_commit();
      tma_store_wait<0>();
    }

  } else {
    // warp 15: idle helper (SASS parks it too)
    regs_dealloc_40();
  }

  // ---- TMEM teardown --------------------------------------------------------
  // FIX-1: dealloc is warp-collective; the whole MMA warp executes it.
  __syncthreads();
  if (warp == 13) tmem_dealloc_warp(tmem, TMEM_COLS);
}

// ============================================================================
//                                    HOST
// ============================================================================
static void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); exit(1); }
}
static void checkcu(CUresult r, const char* what) {
  if (r != CUDA_SUCCESS) { const char* s; cuGetErrorString(r, &s);
    fprintf(stderr, "%s: %s\n", what, s ? s : "?"); exit(1); }
}

// Build a 4D bf16 tensor map (d, seq, head, batch), inner box 64 (=128B, SW128).
static CUtensorMap make_map(void* ptr, int d, int seq, int heads, int batch,
                            int box_seq, bool swizzle128) {
  CUtensorMap m{};
  uint64_t dims[4]    = {(uint64_t)d, (uint64_t)seq, (uint64_t)heads, (uint64_t)batch};
  uint64_t strides[3] = {(uint64_t)d * 2,
                         (uint64_t)d * 2 * seq,
                         (uint64_t)d * 2 * seq * heads};
  uint32_t box[4]     = {64, (uint32_t)box_seq, 1, 1};
  uint32_t elemstr[4] = {1, 1, 1, 1};
  checkcu(cuTensorMapEncodeTiled(
      &m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, ptr, dims, strides, box, elemstr,
      CU_TENSOR_MAP_INTERLEAVE_NONE,
      swizzle128 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE), "cuTensorMapEncodeTiled");
  return m;
}

// -------- CPU reference (fp32) for validation --------------------------------
static void ref_attention(const std::vector<float>& Q, const std::vector<float>& K,
                          const std::vector<float>& V, std::vector<float>& O,
                          int B, int HQ, int HKV, int SQ, int SKV, int d,
                          float scale, bool causal) {
  int g = HQ / HKV;
  for (int b = 0; b < B; ++b)
   for (int h = 0; h < HQ; ++h) {
    int hk = h / g;
    for (int i = 0; i < SQ; ++i) {
      std::vector<float> s(SKV);
      float m = -INFINITY;
      for (int j = 0; j < SKV; ++j) {
        float acc = 0;
        for (int c = 0; c < d; ++c)
          acc += Q[((size_t)(b*HQ+h)*SQ+i)*d+c] * K[((size_t)(b*HKV+hk)*SKV+j)*d+c];
        acc *= scale;
        if (causal && j > i) acc = -INFINITY;
        s[j] = acc; m = std::max(m, acc);
      }
      float l = 0;
      for (int j = 0; j < SKV; ++j) { s[j] = (s[j]==-INFINITY)?0.f:std::exp(s[j]-m); l += s[j]; }
      for (int c = 0; c < d; ++c) {
        float acc = 0;
        for (int j = 0; j < SKV; ++j) acc += s[j] * V[((size_t)(b*HKV+hk)*SKV+j)*d+c];
        O[((size_t)(b*HQ+h)*SQ+i)*d+c] = acc / l;
      }
    }
   }
}

int main() {
  // Small validation shape; scale up once numerics check out.
  const int B = 8, HQ = 12, HKV = 4, SQ = 4096, SKV = 4096, d = 64;

  size_t nq = (size_t)B*HQ*SQ*d, nk = (size_t)B*HKV*SKV*d;
  std::vector<float> Qf(nq), Kf(nk), Vf(nk), Oref(nq);
  std::mt19937 rng(0); std::normal_distribution<float> dist(0.f, 1.f);
  for (auto& x : Qf) x = dist(rng);
  for (auto& x : Kf) x = dist(rng);
  for (auto& x : Vf) x = dist(rng);

  std::vector<__nv_bfloat16> Qh(nq), Kh(nk), Vh(nk);
  for (size_t i = 0; i < nq; ++i) Qh[i] = __float2bfloat16(Qf[i]);
  for (size_t i = 0; i < nk; ++i) { Kh[i] = __float2bfloat16(Kf[i]);
                                    Vh[i] = __float2bfloat16(Vf[i]); }
  // reference uses the bf16-rounded inputs for a fair comparison
  for (size_t i = 0; i < nq; ++i) Qf[i] = __bfloat162float(Qh[i]);
  for (size_t i = 0; i < nk; ++i) { Kf[i] = __bfloat162float(Kh[i]);
                                    Vf[i] = __bfloat162float(Vh[i]); }

  __nv_bfloat16 *dQ, *dK, *dV, *dO;
  check(cudaMalloc(&dQ, nq*2), "malloc Q"); check(cudaMalloc(&dK, nk*2), "malloc K");
  check(cudaMalloc(&dV, nk*2), "malloc V"); check(cudaMalloc(&dO, nq*2), "malloc O");
  check(cudaMemcpy(dQ, Qh.data(), nq*2, cudaMemcpyHostToDevice), "cp Q");
  check(cudaMemcpy(dK, Kh.data(), nk*2, cudaMemcpyHostToDevice), "cp K");
  check(cudaMemcpy(dV, Vh.data(), nk*2, cudaMemcpyHostToDevice), "cp V");

  FA4Params p{};
  p.tma_q = make_map(dQ, d, SQ,  HQ,  B, TILE_M, true);
  p.tma_k = make_map(dK, d, SKV, HKV, B, TILE_N, true);
  p.tma_v = make_map(dV, d, SKV, HKV, B, TILE_N, true);
  p.tma_o = make_map(dO, d, SQ,  HQ,  B, TILE_M, false);  // linear epilogue stage
  p.b = B; p.h_q = HQ; p.h_kv = HKV; p.s_q = SQ; p.s_kv = SKV;
  p.group = HQ / HKV;
  p.scale = 1.f / std::sqrt((float)d);
  p.scale_log2e = p.scale * 1.4426950408889634f;

  const int smem = sizeof(SharedStorage);
  printf("SharedStorage = %d bytes\n", smem);
  check(cudaFuncSetAttribute(fa4_gqa_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem), "smem attr");

  dim3 grid((SQ + TILE_M - 1) / TILE_M, HQ, B);
  fa4_gqa_kernel<<<grid, NUM_THREADS, smem>>>(p);
  check(cudaDeviceSynchronize(), "kernel");

  std::vector<__nv_bfloat16> Oh(nq);
  check(cudaMemcpy(Oh.data(), dO, nq*2, cudaMemcpyDeviceToHost), "cp O");
  ref_attention(Qf, Kf, Vf, Oref, B, HQ, HKV, SQ, SKV, d, p.scale, CAUSAL);

  double max_abs = 0, rms = 0;
  for (size_t i = 0; i < nq; ++i) {
    double e = std::fabs(__bfloat162float(Oh[i]) - Oref[i]);
    max_abs = std::max(max_abs, e); rms += e*e;
  }
  printf("max_abs=%g  rms=%g  (bf16 out: expect ~1e-2 max, ~1e-3 rms)\n",
         max_abs, std::sqrt(rms / nq));

  check(cudaFree(dQ), "free Q"); check(cudaFree(dK), "free K");
  check(cudaFree(dV), "free V"); check(cudaFree(dO), "free O");

  //* ── Correctness + Benchmark (production-scale shape, matches
  //*    GQA_sm103_causal_v2.cu's B/Hq/Hkv/S so results are directly comparable
  //*    to every V-series kernel in this project) ────────────────────────────
  {
    constexpr int Bb = 8, HQb = 12, HKVb = 4, Sb = 4096;
    static_assert(Sb % TILE_M == 0, "benchmark seqlen must tile evenly");

    size_t nqb = (size_t)Bb*HQb*Sb*D, nkb = (size_t)Bb*HKVb*Sb*D;
    std::vector<__nv_bfloat16> Qb(nqb), Kb(nkb), Vb(nkb);
    std::vector<float> Oref(nqb);

    // Same reference bins every other kernel in this project checks against
    // (produced by base/baseline_gqa_causal.py). NOTE: this kernel doesn't
    // compute LSE at all (no d_LSE output anywhere in the kernel), so unlike
    // GQA_sm103_causal_v2.cu's runCorrectness this only checks O, not LSE.
    auto fileMatchesSize = [](const std::string &path, size_t n_floats) -> bool {
      FILE *f = fopen(path.c_str(), "rb");
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

    // fileMatchesSize's own byte-count check already rejects a D!=64 build
    // (nqb/nkb scale with D, so a HEAD_DIM=128 build's expected size won't
    // match these D=64 reference files) -- no separate D guard needed.
    bool has_ref = fileMatchesSize("data/gqa_causal_q.bin", nqb) &&
                   fileMatchesSize("data/gqa_causal_k.bin", nkb) &&
                   fileMatchesSize("data/gqa_causal_v.bin", nkb) &&
                   fileMatchesSize("data/gqa_causal_o.bin", nqb);
    if(has_ref){
      loadBinBF16("data/gqa_causal_q.bin", Qb, nqb);
      loadBinBF16("data/gqa_causal_k.bin", Kb, nkb);
      loadBinBF16("data/gqa_causal_v.bin", Vb, nkb);
      loadBin("data/gqa_causal_o.bin", Oref.data(), nqb);
      std::cout << "\nLoaded causal PyTorch reference from data/gqa_causal_*.bin\n";
    } else {
      std::mt19937 rngb(1); std::normal_distribution<float> distb(0.f, 1.f);
      for (auto& x : Qb) x = __float2bfloat16(distb(rngb));
      for (auto& x : Kb) x = __float2bfloat16(distb(rngb));
      for (auto& x : Vb) x = __float2bfloat16(distb(rngb));
      std::cout << "\nNo causal reference files found (run base/baseline_gqa_causal.py "
                   "first, and build with HEAD_DIM=64) — using random data "
                   "(benchmark only)\n";
    }

    __nv_bfloat16 *dQb, *dKb, *dVb, *dOb;
    check(cudaMalloc(&dQb, nqb*2), "malloc Qb"); check(cudaMalloc(&dKb, nkb*2), "malloc Kb");
    check(cudaMalloc(&dVb, nkb*2), "malloc Vb"); check(cudaMalloc(&dOb, nqb*2), "malloc Ob");
    check(cudaMemcpy(dQb, Qb.data(), nqb*2, cudaMemcpyHostToDevice), "cp Qb");
    check(cudaMemcpy(dKb, Kb.data(), nkb*2, cudaMemcpyHostToDevice), "cp Kb");
    check(cudaMemcpy(dVb, Vb.data(), nkb*2, cudaMemcpyHostToDevice), "cp Vb");

    FA4Params pb{};
    pb.tma_q = make_map(dQb, D, Sb,  HQb,  Bb, TILE_M, true);
    pb.tma_k = make_map(dKb, D, Sb, HKVb, Bb, TILE_N, true);
    pb.tma_v = make_map(dVb, D, Sb, HKVb, Bb, TILE_N, true);
    pb.tma_o = make_map(dOb, D, Sb,  HQb,  Bb, TILE_M, false);
    pb.b = Bb; pb.h_q = HQb; pb.h_kv = HKVb; pb.s_q = Sb; pb.s_kv = Sb;
    pb.group = HQb / HKVb;
    pb.scale = 1.f / std::sqrt((float)D);
    pb.scale_log2e = pb.scale * 1.4426950408889634f;

    dim3 gridb((Sb + TILE_M - 1) / TILE_M, HQb, Bb);

    if(has_ref){
      fa4_gqa_kernel<<<gridb, NUM_THREADS, smem>>>(pb);
      check(cudaGetLastError(), "kernel launch (correctness)");
      check(cudaDeviceSynchronize(), "kernel sync (correctness)");

      std::vector<__nv_bfloat16> Ohb(nqb);
      check(cudaMemcpy(Ohb.data(), dOb, nqb*2, cudaMemcpyDeviceToHost), "cp Ob (correctness)");
      std::vector<float> Ohb_f32(nqb);
      for(size_t i = 0; i < nqb; ++i) Ohb_f32[i] = __bfloat162float(Ohb[i]);

      std::cout << "\nCorrectness fa4_style_gqa_sm103 (vs PyTorch bf16 causal SDPA):\n";
      reportPrecision("  output O ", Oref.data(), Ohb_f32.data(), nqb);
      std::cout << "  O   : "; checkResult(Oref.data(), Ohb_f32.data(), nqb, 2e-2f, 2e-2f);
    }

    // Causal: only nTiles*(nTiles+1)/2 of the nTiles*nTiles (q_tile, kv_tile)
    // pairs are ever visited (see n_kv_tiles above) — half the full-attention work.
    constexpr long long nTiles     = Sb / TILE_M;
    constexpr long long tileVisits = nTiles * (nTiles + 1) / 2;
    long long flops = 4LL * Bb * HQb * (long long)TILE_M * TILE_N * D * tileVisits;
    size_t bytes = (2 * nqb + 2 * nkb) * sizeof(__nv_bfloat16);

    KernelStats stats = benchmarkKernel(
      [&](){ fa4_gqa_kernel<<<gridb, NUM_THREADS, smem>>>(pb); },
      100, 25, flops, bytes
    );
    displayStats("fa4_style_gqa_sm103 causal (warp-specialized tcgen05 pipeline)"
                 " — B8 Hq12 Hkv4 S4096 D" STR(HEAD_DIM), stats);

    check(cudaFree(dQb), "free Qb"); check(cudaFree(dKb), "free Kb");
    check(cudaFree(dVb), "free Vb"); check(cudaFree(dOb), "free Ob");
  }

  return 0;
}