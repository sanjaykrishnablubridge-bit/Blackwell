// GQA_v48_standalone.cu — standalone extraction of gqa_v48_causal (the
// SM103/B300 causal GQA kernel) plus the exact helper functions and launcher
// it needs to compile and run, with nothing else pulled in.
//
// Copied verbatim from GQA_sm103_causal_v3.cu (kernel + launcher) and
// GQA_sm103_causal_common.cuh (the subset of shared device helpers V48
// actually calls). See those two files for the full derivation history and
// the V37-V48 version ladder this kernel is the end product of.
//
// V48 dependency audit (why each helper below is here and nothing more):
//   desc_encode          -- used by make_smem_desc_mnV_sw128
//   make_idesc_bf16       -- used directly (Q@K^T) and by make_idesc_bf16_bMN
//   mbar_init/_expect_tx/_arrive/_commit_mma -- barrier + TMA + MMA-commit plumbing
//   tma_load_2d           -- Q, K, and V are all loaded via 2D SWIZZLE_128B maps in V48
//   ex2_approx            -- softmax exp2 (fused-FFMA scale+subtract-max)
//   consumer_sync         -- 128-thread compute-group barrier (named barrier 1)
// V48 does NOT need (deliberately omitted): make_smem_desc/advance_desc_katom
// (plain K-major non-swizzled -- superseded by the SW128 variants below),
// canon_idx/reorder_sync (V46 deleted the V reorder-copy), mbar_wait (V46
// replaced it with the inlined mbar_wait_spin spin below), make_tma_3d_katom/
// tma_load_3d (V48 replaced the last atom-native 3D load, Q's, with a plain
// 2D swizzled map), and every tmem_readout_* free function (O is TMEM-resident
// across the whole KV loop since V46 and read out inline in the epilogue).

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cassert>
#include <cstdint>
#include <cmath>

// =============================================================================
// Shared device helpers (verbatim subset of GQA_sm103_causal_common.cuh)
// =============================================================================
__device__ __forceinline__ uint64_t desc_encode(uint64_t x){
  return (x & 0x3FFFFull) >> 4;
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

__device__ __forceinline__ void mbar_init(uint32_t bar, int count){
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(bar), "r"(count));
}
__device__ __forceinline__ void mbar_commit_mma(uint32_t bar){
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
               :: "r"(bar) : "memory");
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

// =============================================================================
// V48-specific helpers (verbatim from GQA_sm103_causal_v3.cu)
// =============================================================================

// FIX 2 (from V46's header): mbar_wait replaced by an inlined
// mbarrier.try_wait.parity spin — cuDNN's own SASS shape
// (SYNCS.PHASECHK.TRYWAIT + @!P0 BRA), avoiding the __noinline__ CALL/RET
// and the 10ms NANOSLEEP hint the common header's mbar_wait passes.
__device__ __forceinline__ void mbar_wait_spin(uint32_t bar, int phase){
  asm volatile(
    "{\n\t.reg .pred P1;\n\t"
    "LAB_WAIT_%=:\n\t"
    "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1;\n\t"
    "@!P1 bra.uni LAB_WAIT_%=;\n\t}"
    :: "r"(bar), "r"(phase) : "memory");
}

// FIX 1a (from V46's header): instruction descriptor with B in MN-major.
// a_major is bit 15, b_major is bit 16 (0 = K-major, 1 = MN-major).
__device__ __forceinline__ uint32_t make_idesc_bf16_bMN(int M, int N){
  return make_idesc_bf16(M, N) | (1u << 16);   // b_major = MN-major
}

// FIX 1b (from V46's header): smem matrix descriptor for MN-major operands
// with 128-byte swizzle. LBO encoded = 1, SBO = 1024 B (8 K-rows x 128 B),
// swizzle bits [63:61] = 1 (128B swizzle), valid bit 46 set.
__device__ __forceinline__ uint64_t make_smem_desc_mnV_sw128(void* smem_ptr){
  uint64_t addr = (uint64_t)__cvta_generic_to_shared(smem_ptr);
  uint64_t desc = desc_encode(addr)
                | (1ull << 16)                    // LBO encoded = 1
                | (desc_encode(1024ull) << 32)    // SBO = 1024 B
                | (1ull << 46)                    // valid bit
                | (2ull << 61);                   // swizzle mode 1 = 128B
  return desc;
}

// Advance an MN-major SW128 descriptor by one MMA K-atom (16 K rows): 16
// rows x 128 B = 2048 B = two SW128 8-row groups (used for V, whose
// reduction axis Bc is the physically-strided/outer axis).
__device__ __forceinline__ uint64_t advance_desc_katom_mnV(uint64_t desc, int katom){
  uint64_t units     = (uint64_t)katom * (2048ull >> 4);
  uint64_t base_addr = desc & 0x3FFFull;
  uint64_t new_addr  = (base_addr + units) & 0x3FFFull;
  return (desc & ~0x3FFFull) | new_addr;
}

// Advance a K-major SW128 descriptor by one MMA K-atom (16 elements along D,
// the reduction axis for Q and K — physically the inner/contiguous axis, so
// this is a small intra-row-span advance, unlike advance_desc_katom_mnV's
// inter-row-group advance).
__device__ __forceinline__ uint64_t advance_desc_katom_kmajor_sw128(uint64_t desc, int katom){
  uint64_t units     = (uint64_t)katom * 2ull;   // 32 B = 2 units of 16 B
  uint64_t base_addr = desc & 0x3FFFull;
  uint64_t new_addr  = (base_addr + units) & 0x3FFFull;
  return (desc & ~0x3FFFull) | new_addr;
}

// Host side: SWIZZLE_128B 2D tensor map, shared by Q/K/V in V48. Box inner
// extent must be <= the swizzle span (64 bf16 = 128 B = exactly one span).
static CUtensorMap make_tma_2d_sw128(__nv_bfloat16* gptr, uint64_t rows, uint64_t cols,
                                     uint32_t box_rows, uint32_t box_cols){
  CUtensorMap tmap{};
  uint64_t gdim[2]    = { cols, rows };
  uint64_t gstride[1] = { cols * sizeof(__nv_bfloat16) };
  uint32_t bdim[2]    = { box_cols, box_rows };
  uint32_t estride[2] = { 1, 1 };
  CUresult res = cuTensorMapEncodeTiled(
    &tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, gptr, gdim, gstride, bdim, estride,
    CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
    CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  assert(res == CUDA_SUCCESS);
  return tmap;
}

// =============================================================================
//  gqa_v48_causal — V47 + Q load via a plain SWIZZLE_128B 2D map, replacing
//  make_tma_3d_katom's atom-native 3D box for Q too.
//
//  Motivation: V47's own NCU capture showed the "uncoalesced global access"
//  finding UNCHANGED, bit-for-bit, from V46 (11,010,048 excessive sectors,
//  87%, identical down to the last digit) despite K's atom-native load being
//  replaced. Correctness was also bit-identical, so the K fix demonstrably
//  worked at the compute level -- it just didn't move this metric at all.
//  The one atom-native load V47 left untouched is Q's (`make_tma_3d_katom`,
//  same mechanism K used to have). Since Q is read exactly ONCE per block
//  (no L2 reuse across the grid, unlike K which is re-read across ~33 kc
//  iterations per block AND shared across blocks with overlapping causal
//  ranges, likely absorbing much of ITS OWN inefficiency into L2 hits), Q is
//  the more likely dominant source of this metric -- this version tests that
//  hypothesis directly.
//
//  Change is a direct mirror of V47's K fix, reusing the exact same derived
//  primitives (no new [VERIFY] point): Q's reduction axis for QK^T is ALSO D
//  (physically contiguous, matching K's), so Q is genuinely K-major here too
//  -- make_idesc_bf16(Br, Bc) needs no a_major bit, exactly as it needed no
//  b_major bit for K. make_smem_desc_mnV_sw128 and
//  advance_desc_katom_kmajor_sw128 (both shape-driven, [Br or Bc, 128B span],
//  not direction-driven) are reused as-is for Q's base descriptor and K-atom
//  advance.
// =============================================================================
template<int Br, int Bc, int D>
__global__ void __launch_bounds__(160, 1) gqa_v48_causal(
  __nv_bfloat16 *d_O,
  float *d_LSE,
  const __grid_constant__ CUtensorMap Qtmap,
  const __grid_constant__ CUtensorMap Ktmap,
  const __grid_constant__ CUtensorMap Vtmap,
  int B, int Hq, int Hkv, int G, int S, float scale
){
  static_assert(Br == 128, "consumer group is hardwired to 128 threads");
  static_assert(Br == Bc,  "causal tile-skip + diagonal-tile mask requires Br == Bc");
  static_assert(D == 64,   "SW128 Q/K/V descriptor constants assume D = 64 (one 128B span)");

  const int tid = threadIdx.x;          // 0..127 compute, 128..159 producer.

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

  __shared__ __align__(1024) __nv_bfloat16 sQ[Br * D];         // SW128 now (was atom-native canon_idx)
  __shared__ __align__(1024) __nv_bfloat16 sK[2][Bc * D];
  __shared__ __align__(1024) __nv_bfloat16 sVstage[2][Bc * D];
  __shared__ float sl[Br];
  __shared__ __align__(8) uint64_t s_mma_bar;
  __shared__ __align__(8) uint64_t s_load_bar_K[2], s_free_bar_K[2];
  __shared__ __align__(8) uint64_t s_load_bar_V[2], s_free_bar_V[2];
  __shared__ __align__(8) uint64_t s_load_bar_Q;

  const uint32_t mma_bar = (uint32_t)__cvta_generic_to_shared(&s_mma_bar);
  const uint32_t lbarK0  = (uint32_t)__cvta_generic_to_shared(&s_load_bar_K[0]);
  const uint32_t lbarK1  = (uint32_t)__cvta_generic_to_shared(&s_load_bar_K[1]);
  const uint32_t fbarK0  = (uint32_t)__cvta_generic_to_shared(&s_free_bar_K[0]);
  const uint32_t fbarK1  = (uint32_t)__cvta_generic_to_shared(&s_free_bar_K[1]);
  const uint32_t lbarV0  = (uint32_t)__cvta_generic_to_shared(&s_load_bar_V[0]);
  const uint32_t lbarV1  = (uint32_t)__cvta_generic_to_shared(&s_load_bar_V[1]);
  const uint32_t fbarV0  = (uint32_t)__cvta_generic_to_shared(&s_free_bar_V[0]);
  const uint32_t fbarV1  = (uint32_t)__cvta_generic_to_shared(&s_free_bar_V[1]);
  const uint32_t qbar    = (uint32_t)__cvta_generic_to_shared(&s_load_bar_Q);

  constexpr uint32_t TMEM_S_COL    = 0;
  constexpr uint32_t TMEM_O_COL    = (uint32_t)Bc;             // 128
  constexpr uint32_t TMEM_M_COL    = TMEM_O_COL + (uint32_t)D; // 192
  constexpr uint32_t TMEM_CORR_COL = TMEM_M_COL + 1;           // 193
  constexpr uint32_t NCOLS = 256;

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

  const int warp_id_c = tid / 32;
  const uint32_t lane_base_c = (uint32_t)warp_id_c * 32u;
  auto tmem_ld32 = [&](uint32_t col, float* r){
    asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
      : "=f"(r[0]),"=f"(r[1]),"=f"(r[2]),"=f"(r[3]),"=f"(r[4]),"=f"(r[5]),
        "=f"(r[6]),"=f"(r[7]),"=f"(r[8]),"=f"(r[9]),"=f"(r[10]),"=f"(r[11]),
        "=f"(r[12]),"=f"(r[13]),"=f"(r[14]),"=f"(r[15]),"=f"(r[16]),"=f"(r[17]),
        "=f"(r[18]),"=f"(r[19]),"=f"(r[20]),"=f"(r[21]),"=f"(r[22]),"=f"(r[23]),
        "=f"(r[24]),"=f"(r[25]),"=f"(r[26]),"=f"(r[27]),"=f"(r[28]),"=f"(r[29]),
        "=f"(r[30]),"=f"(r[31])
      : "r"(tmem_addr + (lane_base_c << 16) + col));
  };
  auto tmem_st32 = [&](uint32_t col, const uint32_t* r){
    asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x32.b32 [%32], "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31};"
      :: "r"(r[0]),"r"(r[1]),"r"(r[2]),"r"(r[3]),"r"(r[4]),"r"(r[5]),
         "r"(r[6]),"r"(r[7]),"r"(r[8]),"r"(r[9]),"r"(r[10]),"r"(r[11]),
         "r"(r[12]),"r"(r[13]),"r"(r[14]),"r"(r[15]),"r"(r[16]),"r"(r[17]),
         "r"(r[18]),"r"(r[19]),"r"(r[20]),"r"(r[21]),"r"(r[22]),"r"(r[23]),
         "r"(r[24]),"r"(r[25]),"r"(r[26]),"r"(r[27]),"r"(r[28]),"r"(r[29]),
         "r"(r[30]),"r"(r[31]), "r"(tmem_addr + (lane_base_c << 16) + col));
  };
  auto tmem_ld_scalar = [&](uint32_t col, float &a){
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];"
                 : "=f"(a) : "r"(tmem_addr + (lane_base_c << 16) + col));
  };
  auto tmem_st_scalar = [&](uint32_t col, float v){
    asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%1], {%0};"
                 :: "f"(v), "r"(tmem_addr + (lane_base_c << 16) + col));
  };
  auto tmem_wait_ld = [](){ asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); };
  auto tmem_wait_st = [](){ asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory"); };

  const uint32_t TX = (uint32_t)Bc * (uint32_t)D * (uint32_t)sizeof(__nv_bfloat16);
  const uint32_t sK_addr0      = (uint32_t)__cvta_generic_to_shared(sK[0]);
  const uint32_t sK_addr1      = (uint32_t)__cvta_generic_to_shared(sK[1]);
  const uint32_t sVstage_addr0 = (uint32_t)__cvta_generic_to_shared(sVstage[0]);
  const uint32_t sVstage_addr1 = (uint32_t)__cvta_generic_to_shared(sVstage[1]);
  const uint32_t sQ_addr       = (uint32_t)__cvta_generic_to_shared(sQ);

  for(int pass = 0; pass < 2; ++pass){
    const int q_tile   = (pass == 0) ? pair : (nTiles - 1 - pair);
    const int q_row0   = q_tile * Br;
    const int nKVTiles = q_tile + 1;

    const long qBase = ((long)(b * Hq + hq) * S + q_row0) * D;
    const long lBase = ((long)(b * Hq + hq) * S + q_row0);

    if(tid < Br){
      tmem_st_scalar(TMEM_M_COL, -INFINITY);
      tmem_wait_st();
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
      sl[tid] = 0.0f;
    }

    if(tid == 0){
      mbar_init(mma_bar, 1);
      mbar_init(lbarK0, 1); mbar_init(lbarK1, 1);
      mbar_init(fbarK0, 1); mbar_init(fbarK1, 1);
      mbar_init(lbarV0, 1); mbar_init(lbarV1, 1);
      mbar_init(fbarV0, 1); mbar_init(fbarV1, 1);
      mbar_init(qbar, 1);
      mbar_expect_tx(qbar, TX);
      tma_load_2d(sQ_addr, &Qtmap, 0, (int)((b * Hq + hq) * S + q_row0), qbar);   // was tma_load_3d
      mbar_arrive(qbar);
    }
    __syncthreads();

    if(tid >= 128){
      // ---- Producer warp: unchanged from V47. ----
      if(tid == 128){
        int free_phase_K_bits = 0, free_phase_V_bits = 0;
        for(int kc = 0; kc < nKVTiles; ++kc){
          const int slot = kc & 1;
          const uint32_t cur_lbarK = (slot == 0) ? lbarK0 : lbarK1;
          const uint32_t cur_fbarK = (slot == 0) ? fbarK0 : fbarK1;
          const uint32_t cur_lbarV = (slot == 0) ? lbarV0 : lbarV1;
          const uint32_t cur_fbarV = (slot == 0) ? fbarV0 : fbarV1;
          const uint32_t cur_sK_addr      = (slot == 0) ? sK_addr0      : sK_addr1;
          const uint32_t cur_sVstage_addr = (slot == 0) ? sVstage_addr0 : sVstage_addr1;
          if(kc >= 2){
            mbar_wait_spin(cur_fbarK, (free_phase_K_bits >> slot) & 1); free_phase_K_bits ^= (1 << slot);
            mbar_wait_spin(cur_fbarV, (free_phase_V_bits >> slot) & 1); free_phase_V_bits ^= (1 << slot);
          }
          const int r = kvRow0 + kc * Bc;
          mbar_expect_tx(cur_lbarK, TX);
          tma_load_2d(cur_sK_addr, &Ktmap, 0, r, cur_lbarK);
          mbar_arrive(cur_lbarK);
          mbar_expect_tx(cur_lbarV, TX);
          tma_load_2d(cur_sVstage_addr, &Vtmap, 0, r, cur_lbarV);
          mbar_arrive(cur_lbarV);
        }
      }
    } else {
      // ---- Compute group (tid < 128) ----
      int mbar_phase = 0;
      int load_phase_K_bits = 0, load_phase_V_bits = 0;
      mbar_wait_spin(qbar, 0);
      const uint64_t descQ_base = make_smem_desc_mnV_sw128(sQ);   // was make_smem_desc(sQ, Br)

      for(int kc = 0; kc < nKVTiles; ++kc){
        const int slot = kc & 1;
        const uint32_t cur_lbarK = (slot == 0) ? lbarK0 : lbarK1;
        const uint32_t cur_lbarV = (slot == 0) ? lbarV0 : lbarV1;
        const uint32_t cur_fbarK = (slot == 0) ? fbarK0 : fbarK1;
        const uint32_t cur_fbarV = (slot == 0) ? fbarV0 : fbarV1;
        mbar_wait_spin(cur_lbarK, (load_phase_K_bits >> slot) & 1); load_phase_K_bits ^= (1 << slot);
        mbar_wait_spin(cur_lbarV, (load_phase_V_bits >> slot) & 1); load_phase_V_bits ^= (1 << slot);
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

        float s_row[Bc];

        { // ---- S = Q @ K^T ---- Q descriptor is now K-major SW128 too (was canon_idx) ----
          const uint64_t descK_base = make_smem_desc_mnV_sw128(sK[slot]);
          const uint32_t idesc      = make_idesc_bf16(Br, Bc);   // unchanged: both Q and K genuinely K-major
          if(tid == 0){
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            for(int kt = 0; kt < D/16; ++kt){
              uint64_t descQ = advance_desc_katom_kmajor_sw128(descQ_base, kt);   // was advance_desc_katom(...,Br)
              uint64_t descK = advance_desc_katom_kmajor_sw128(descK_base, kt);
              uint32_t accumulate = (kt > 0) ? 1u : 0u;
              asm volatile(
                "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
                :: "r"(tmem_addr + TMEM_S_COL), "l"(descQ), "l"(descK), "r"(idesc), "r"(accumulate) : "memory");
            }
            mbar_commit_mma(mma_bar);
          }
          mbar_wait_spin(mma_bar, mbar_phase); mbar_phase ^= 1;
          if(tid == 0) mbar_arrive(cur_fbarK);

          asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
          #pragma unroll
          for(int c = 0; c < Bc; c += 32) tmem_ld32(TMEM_S_COL + (uint32_t)c, &s_row[c]);
          tmem_wait_ld();

          if(kc == q_tile){
            #pragma unroll
            for(int j = 0; j < Bc; ++j) if(j > tid) s_row[j] = -INFINITY;
          }
          consumer_sync();
        }

        { // ---- softmax: identical FFMA-fused math to V42/V46/V47 ----
          asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
          float m_old;
          tmem_ld_scalar(TMEM_M_COL, m_old);
          tmem_wait_ld();
          const float l_old = sl[tid];

          float tile_max = -INFINITY;
          #pragma unroll
          for(int c = 0; c < Bc; ++c) tile_max = fmaxf(tile_max, s_row[c]);

          const float m_new = fmaxf(m_old, tile_max);
          const float corr  = ex2_approx((m_old - m_new) * scale_l2e);
          const float neg_max_scaled = -m_new * scale_l2e;

          float p_sum = 0.0f;
          uint32_t p_packed[Bc / 2];
          #pragma unroll
          for(int j2 = 0; j2 < Bc; j2 += 2){
            const float p0 = ex2_approx(fmaf(s_row[j2],     scale_l2e, neg_max_scaled));
            const float p1 = ex2_approx(fmaf(s_row[j2 + 1], scale_l2e, neg_max_scaled));
            __nv_bfloat162 hp = __floats2bfloat162_rn(p0, p1);
            p_packed[j2 / 2] = *reinterpret_cast<uint32_t*>(&hp);
            p_sum += p0 + p1;
          }

          tmem_st_scalar(TMEM_M_COL, m_new);
          tmem_wait_st();
          #pragma unroll
          for(int c = 0; c < Bc / 2; c += 32) tmem_st32(TMEM_S_COL + (uint32_t)c, &p_packed[c]);
          tmem_wait_st();

          if(kc > 0){
            float o_row[D];
            #pragma unroll
            for(int c = 0; c < D; c += 32) tmem_ld32(TMEM_O_COL + (uint32_t)c, &o_row[c]);
            tmem_wait_ld();
            #pragma unroll
            for(int c = 0; c < D; ++c) o_row[c] *= corr;
            #pragma unroll
            for(int c = 0; c < D; c += 32)
              tmem_st32(TMEM_O_COL + (uint32_t)c, reinterpret_cast<const uint32_t*>(&o_row[c]));
            tmem_wait_st();
          }
          asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");

          sl[tid] = l_old * corr + p_sum;
        }
        consumer_sync();

        { // ---- O += P @ V, V read directly from sVstage (unchanged from V46/V47) ----
          const uint64_t descV_base = make_smem_desc_mnV_sw128(sVstage[slot]);
          const uint32_t idesc      = make_idesc_bf16_bMN(Br, D);
          if(tid == 0){
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            for(int kt = 0; kt < Bc/16; ++kt){
              uint64_t descV = advance_desc_katom_mnV(descV_base, kt);
              uint32_t p_operand_addr = tmem_addr + TMEM_S_COL + (uint32_t)(kt * 8);
              uint32_t accumulate = (kc > 0 || kt > 0) ? 1u : 0u;
              asm volatile(
                "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n\t}\n"
                :: "r"(tmem_addr + TMEM_O_COL), "r"(p_operand_addr), "l"(descV), "r"(idesc), "r"(accumulate) : "memory");
            }
            mbar_commit_mma(mma_bar);
          }
          mbar_wait_spin(mma_bar, mbar_phase); mbar_phase ^= 1;
          if(tid == 0) mbar_arrive(cur_fbarV);
          consumer_sync();
        }
      } // end kv loop

      // ---- Epilogue: single TMEM readout -> normalize -> global (unchanged) ----
      {
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        float o_row[D];
        #pragma unroll
        for(int c = 0; c < D; c += 32) tmem_ld32(TMEM_O_COL + (uint32_t)c, &o_row[c]);
        tmem_wait_ld();
        const float denom = sl[tid];
        #pragma unroll
        for(int c = 0; c < D; c += 2)
          *reinterpret_cast<__nv_bfloat162*>(&d_O[qBase + (long)tid * D + c]) =
              __floats2bfloat162_rn(o_row[c] / denom, o_row[c + 1] / denom);

        float m_final;
        tmem_ld_scalar(TMEM_M_COL, m_final);
        tmem_wait_ld();
        d_LSE[lBase + tid] = 0.6931471805599453f * (m_final * scale_l2e + log2f(sl[tid]));
      }
    }

    __syncthreads();
  } // end pass loop

  if(tid < 32)
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(tmem_addr), "r"(NCOLS) : "memory");
}

// =============================================================================
template<int Br, int Bc, int D>
void launch_gqa_v48_causal(
  __nv_bfloat16 *d_Q, __nv_bfloat16 *d_K, __nv_bfloat16 *d_V,
  __nv_bfloat16 *d_O, float *d_LSE,
  int B, int Hq, int Hkv, int S, int G, float scale
){
  static_assert(Br == 128 && Br == Bc && D == 64, "see kernel static_asserts");
  assert((S / Br) % 2 == 0);

  dim3 GRID(B * Hq * ((S / Br) / 2), 1, 1);
  dim3 BLOCK(160);

  static bool cfgd = false;
  static CUtensorMap Qtmap, Ktmap, Vtmap;
  if(!cfgd){
    const uint64_t kvRows = (uint64_t)B * Hkv * S;
    const uint64_t qRows  = (uint64_t)B * Hq  * S;
    Qtmap = make_tma_2d_sw128(d_Q, qRows,  (uint64_t)D, (uint32_t)Br, (uint32_t)D);  // was make_tma_3d_katom
    Ktmap = make_tma_2d_sw128(d_K, kvRows, (uint64_t)D, (uint32_t)Bc, (uint32_t)D);
    Vtmap = make_tma_2d_sw128(d_V, kvRows, (uint64_t)D, (uint32_t)Bc, (uint32_t)D);
    cfgd = true;
  }
  gqa_v48_causal<Br, Bc, D><<<GRID, BLOCK>>>(d_O, d_LSE, Qtmap, Ktmap, Vtmap,
                                               B, Hq, Hkv, G, S, scale);
}