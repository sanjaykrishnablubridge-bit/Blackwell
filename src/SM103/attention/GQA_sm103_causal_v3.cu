// GQA_sm103_causal_v3.cu — continuation of GQA_sm103_causal_v2.cu's version
// ladder (V37-V45), starting fresh with gqa_v46_causal. Renamed out of
// new_version.cu (which had drifted to an independent "v43nr" label) to keep
// both the file-per-continuation convention this project has followed since
// the V19-V36 -> V37-V45 split, AND a single continuous version count across
// files — V46 is next after V45, not a restart.
//
// V42 (session best as of the split: 1.4091ms, 150.9 TFLOPS, ~6.68x from
// cuDNN's 0.2110ms reference) is copied in below, verbatim, as the baseline
// this file's new versions are compared against — see GQA_sm103_causal_v2.cu
// for V37-V45's full derivation history (V43/V44/V45 were all negative
// results: dedicated MMA-issue thread, mbarrier-based reorder-copy handoff,
// and shuffle-based bank-conflict fix, respectively, each adding more
// synchronization/register overhead than it saved).
//
// gqa_v46_causal below is V42 + a direct SASS-level comparison against
// cuDNN's real kernel: deletes the V reorder-copy entirely via a swizzled
// MN-major V descriptor, inlines mbar_wait, and keeps O TMEM-resident across
// the whole KV loop — hardware-confirmed 0.5863ms/362.6 TFLOPS, correctness
// clean on O and LSE (~2.78x gap to cuDNN, down from V42's 6.68x). See its
// own header comment just below the V42 baseline for the full derivation.
#include "GQA_sm103_causal_common.cuh"

// =================================
//  gqa_v42_causal - V40 (frozen at 1.5372ms, 138.3 TFLOPS) + FFMA-fused softmax
//  scale/subtract-max, directly from reviewing ThunderKittens' production
//  Blackwell MHA kernel (tunderKittens.txt): `__ffma2_rn(scores_reg[si], scale_2,
//  neg_max_scaled_2)` folds "scale then subtract scaled-max" into ONE fused
//  multiply-add per element, computed from a row-max tracked in RAW (unscaled)
//  units throughout.
//
//  Directly motivated by the full_v40.pdf NCU report's FP32 Non-Fused
//  Instructions finding (Est. Speedup: 5.82%), which pinpointed three exact
//  hotspot lines in V40's softmax block: the eager `s_row[c] *= scale_l2e`
//  pass (a standalone FMUL, 25.9M executions) and the `s_row[j2]-m_new` /
//  `p_sum += p0+p1` FADDs in the exp/sum loop. Earlier this session, V34's
//  SASS was checked and nvcc's default -fmad contraction ALREADY fused the
//  scalar scale+subtract there -- but V40 restructured this into two
//  separate blocks with a TMEM round-trip between them (scale happens before
//  m_new is even known), which defeats that auto-fusion. This version
//  restores the fusion explicitly by deferring scale application entirely.
//
//  Mathematically identical to V40, verified by hand before writing any code:
//  V40 computed corr = exp2(m_old_scaled - m_new_scaled) and
//  p_i = exp2(s_row_scaled[i] - m_new_scaled), where "_scaled" means
//  pre-multiplied by scale_l2e. Since scale_l2e is a POSITIVE constant,
//  scale_l2e*(m_old_raw - m_new_raw) == m_old_scaled - m_new_scaled exactly
//  (same for the per-column term), so tracking m_old/m_new in RAW units and
//  applying scale_l2e once via FFMA at the point of use produces the same
//  value (mod ~1 ULP from FMA vs separate-op rounding, immaterial at bf16
//  output precision) while cutting one full pass over s_row (the old eager
//  scale loop) and turning each subtract into a fused multiply-add. Also
//  verified the -INFINITY causal-mask sentinel survives this change
//  unaffected: masking is scale-invariant (scale_l2e > 0 preserves both
//  ordering and the -INFINITY sentinel under multiplication), and m_new is
//  never -INFINITY in practice since causal masking always leaves at least
//  the diagonal element of a valid row unmasked -- the SAME pre-existing
//  invariant V40's own code already relied on, not a new risk.
// =================================
template<int Br, int Bc, int D>
__global__ void __launch_bounds__(512, 1) gqa_v42_causal(
  __nv_bfloat16 *d_O,
  float *d_LSE,
  const __grid_constant__ CUtensorMap Qtmap3d,
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
  // sS is gone in V40 (S is register-resident, see below) -- its bank-conflict-padding
  // rationale from V38 no longer applies.
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
  __shared__ __align__(16)  float         sO[Br * D_pad];
  __shared__ float sl[Br];   // kept in smem -- see header (cross-512-thread read in
                             // the final O-normalize loop, unsafe to TMEM-ify here)
  __shared__ __align__(8) uint64_t s_mma_bar;
  __shared__ __align__(8) uint64_t s_load_bar_K[2];
  __shared__ __align__(8) uint64_t s_free_bar_K[2];
  __shared__ __align__(8) uint64_t s_load_bar_V[2];
  __shared__ __align__(8) uint64_t s_free_bar_V[2];
  __shared__ __align__(8) uint64_t s_load_bar_Q;   // NEW: Q's TMA-load completion (see header)

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

  // V40: S/P share TMEM_S_COL (P bf16-packed reuses S's fp32 columns once S is
  // consumed into registers -- 128 f32 cols hold 128 P values packed as 64 bf16
  // pairs, so P fits in half of TMEM_S_COL's own span). TMEM_O_COL MUST be a
  // SEPARATE region: the PV MMA reads P from TMEM_S_COL as operand A while
  // writing O to TMEM_O_COL in the SAME instruction -- input and output cannot
  // alias. TMEM_M_COL/TMEM_CORR_COL hold the running max and rescale factor
  // (sl stays in smem, see its declaration above).
  constexpr uint32_t TMEM_S_COL    = 0;
  constexpr uint32_t TMEM_O_COL    = (uint32_t)Bc;             // 128
  constexpr uint32_t TMEM_M_COL    = TMEM_O_COL + (uint32_t)D; // 192
  constexpr uint32_t TMEM_CORR_COL = TMEM_M_COL + 1;           // 193
  constexpr uint32_t NCOLS = 256;
  static_assert(NCOLS > TMEM_CORR_COL && (NCOLS & (NCOLS - 1)) == 0,
                "tcgen05 column count must be a power of two >= 32 and cover every region");

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

  // V40: local TMEM-direct helpers. warp_id_c/lane_base_c match
  // tmem_readout_to_smem_vec_2cta's own addressing exactly (rows_per_warp =
  // Br/4 = 32 = warp size for Br=128, so row == tid always -- every thread here
  // owns exactly one TMEM lane/row already, matching S's existing convention).
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
  auto tmem_wait_ld_v40 = [](){ asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); };
  auto tmem_wait_st_v40 = [](){ asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory"); };

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

    for(int i = tid; i < Br * D; i += blockDim.x){
      const int r = i / D, c = i % D;
      sO[r * D_pad + c] = 0.0f;
    }
    if(tid < Br){
      tmem_st_scalar(TMEM_M_COL, -INFINITY);
      tmem_wait_st_v40();
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
      mbar_expect_tx(qbar, TX);   // TX == Br*D*sizeof(bf16) == Bc*D*sizeof(bf16), Br==Bc
      tma_load_3d(sQ_addr, &Qtmap3d, 0, (int)((b * Hq + hq) * S + q_row0), 0, qbar);
      mbar_arrive(qbar);
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
        int free_phase_K_bits = 0;
        int free_phase_V_bits = 0;
        for(int kc = 0; kc < nKVTiles; ++kc){
          const int slot = kc & 1;
          const uint32_t cur_lbarK      = (slot == 0) ? lbarK0      : lbarK1;
          const uint32_t cur_fbarK      = (slot == 0) ? fbarK0      : fbarK1;
          const uint32_t cur_lbarV      = (slot == 0) ? lbarV0      : lbarV1;
          const uint32_t cur_fbarV      = (slot == 0) ? fbarV0      : fbarV1;
          const uint32_t cur_sK_addr      = (slot == 0) ? sK_addr0      : sK_addr1;
          const uint32_t cur_sVstage_addr = (slot == 0) ? sVstage_addr0 : sVstage_addr1;
          if(kc >= 2){
            mbar_wait(cur_fbarK, (free_phase_K_bits >> slot) & 1); free_phase_K_bits ^= (1 << slot);
            mbar_wait(cur_fbarV, (free_phase_V_bits >> slot) & 1); free_phase_V_bits ^= (1 << slot);
          }
          const int r = kvRow0 + kc * Bc;
          mbar_expect_tx(cur_lbarK, TX);
          tma_load_3d(cur_sK_addr, &Ktmap3d, 0, r, 0, cur_lbarK);
          mbar_arrive(cur_lbarK);
          mbar_expect_tx(cur_lbarV, TX);
          tma_load_2d(cur_sVstage_addr, &Vtmap, 0, r, cur_lbarV);
          mbar_arrive(cur_lbarV);
        }
      }
    } else if(tid >= 160){
      // ---- Reorder-helper group: widens the V reorder-copy alongside compute. ----
      const int pidx = tid - 32;   // maps tid [160,511] -> contiguous [128,479]
      int load_phase_V_bits = 0;
      for(int kc = 0; kc < nKVTiles; ++kc){
        const int slot = kc & 1;
        const uint32_t cur_lbarV = (slot == 0) ? lbarV0 : lbarV1;
        mbar_wait(cur_lbarV, (load_phase_V_bits >> slot) & 1); load_phase_V_bits ^= (1 << slot);
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
      int load_phase_K_bits = 0;
      int load_phase_V_bits = 0;
      mbar_wait(qbar, 0);   // Q's TMA completion -- re-init'd (and thus phase-reset) every pass
      const uint64_t descQ_base = make_smem_desc(sQ, Br);

      for(int kc = 0; kc < nKVTiles; ++kc){
        const int slot = kc & 1;
        const uint32_t cur_lbarK = (slot == 0) ? lbarK0 : lbarK1;
        const uint32_t cur_lbarV = (slot == 0) ? lbarV0 : lbarV1;
        const uint32_t cur_fbarK = (slot == 0) ? fbarK0 : fbarK1;
        const uint32_t cur_fbarV = (slot == 0) ? fbarV0 : fbarV1;
        mbar_wait(cur_lbarK, (load_phase_K_bits >> slot) & 1); load_phase_K_bits ^= (1 << slot);
        mbar_wait(cur_lbarV, (load_phase_V_bits >> slot) & 1); load_phase_V_bits ^= (1 << slot);
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        reorder_sync();

        reorderCopyV2(pidx, slot);
        reorder_sync();
        if(tid == 0) mbar_arrive(cur_fbarV);

        // V40: declared at kc-loop scope (not inside the block below) -- s_row must
        // still be alive in the softmax block that follows, unlike V38's sS which
        // was a kernel-scope smem array visible everywhere.
        float s_row[Bc];

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
                :: "r"(tmem_addr + TMEM_S_COL), "l"(descQ), "l"(descK), "r"(idesc), "r"(accumulate) : "memory");
            }
            mbar_commit_mma(mma_bar);
          }
          mbar_wait(mma_bar, mbar_phase); mbar_phase ^= 1;
          if(tid == 0) mbar_arrive(cur_fbarK);

          // V40: S read straight into registers -- no sS smem buffer, no second
          // re-read for the exp pass below (s_row is reused for both). Matches
          // fa4_style_gqa_sm103.cu's technique; see header for the register-
          // pressure caveat this carries (no setmaxnreg re-partitioning yet).
          asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
          #pragma unroll
          for(int c = 0; c < Bc; c += 32) tmem_ld32(TMEM_S_COL + (uint32_t)c, &s_row[c]);
          tmem_wait_ld_v40();
          // V42: NO separate scale pass here -- s_row stays RAW (pre-scale). See
          // header: scale is fused into the exp2 argument via FFMA below, matching
          // ThunderKittens' technique. The running max (TMEM_M_COL) is tracked in
          // RAW units too, so max-ordering is unaffected (scale_l2e > 0 preserves
          // order) and masking to -INFINITY is scale-invariant either way.

          if(kc == q_tile){
            #pragma unroll
            for(int j = 0; j < Bc; ++j) if(j > tid) s_row[j] = -INFINITY;
          }
          consumer_sync();
        }

        {
          asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
          float m_old;
          tmem_ld_scalar(TMEM_M_COL, m_old);
          tmem_wait_ld_v40();
          const float l_old = sl[tid];

          float tile_max = -INFINITY;
          #pragma unroll
          for(int c = 0; c < Bc; ++c) tile_max = fmaxf(tile_max, s_row[c]);

          const float m_new = fmaxf(m_old, tile_max);
          // V42: m_old/m_new are RAW (unscaled) here -- corr and the exp argument
          // both apply scale_l2e explicitly instead of relying on s_row already
          // being pre-scaled. corr is a single per-row op (not per-column), so
          // fusing it further isn't NCU-material; the per-column exp argument
          // below is where the fusion actually matters (Bc/2 times per row).
          const float corr  = ex2_approx((m_old - m_new) * scale_l2e);
          const float neg_max_scaled = -m_new * scale_l2e;   // once per row, reused below

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
          tmem_wait_st_v40();
          tmem_st_scalar(TMEM_CORR_COL, corr);
          tmem_wait_st_v40();
          // P packed sequentially (col c holds j={2c,2c+1}) -- NOT canon_idx-swizzled:
          // that swizzle exists only to avoid smem bank conflicts for a smem-descriptor
          // MMA operand; a TMEM-resident operand (read via the [addr]-bracket MMA form
          // below) has its own fixed hardware addressing and needs no such swizzle.
          #pragma unroll
          for(int c = 0; c < Bc / 2; c += 32) tmem_st32(TMEM_S_COL + (uint32_t)c, &p_packed[c]);
          tmem_wait_st_v40();
          asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");

          sl[tid] = l_old * corr + p_sum;
        }
        consumer_sync();

        {
          asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
          float corr_r;
          tmem_ld_scalar(TMEM_CORR_COL, corr_r);
          tmem_wait_ld_v40();
          #pragma unroll
          for(int c = 0; c < D; ++c) sO[tid * D_pad + c] *= corr_r;
        }
        consumer_sync();

        {
          const uint64_t descV_base = make_smem_desc(sV, D);
          const uint32_t idesc      = make_idesc_bf16(Br, D);
          if(tid == 0){
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            for(int kt = 0; kt < Bc/16; ++kt){
              uint64_t descV = advance_desc_katom(descV_base, kt, D);
              // A operand (P) read from TMEM instead of a smem descriptor -- the
              // [addr]-bracket form, same PTX pattern fa4_style_gqa_sm103.cu compiled
              // clean with. MMA_K=16 per step, 2 bf16/word -> 8 TMEM cols per step.
              uint32_t p_operand_addr = tmem_addr + TMEM_S_COL + (uint32_t)(kt * 8);
              uint32_t accumulate = (kt > 0) ? 1u : 0u;
              asm volatile(
                "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n\t}\n"
                :: "r"(tmem_addr + TMEM_O_COL), "r"(p_operand_addr), "l"(descV), "r"(idesc), "r"(accumulate) : "memory");
            }
            mbar_commit_mma(mma_bar);
          }
          mbar_wait(mma_bar, mbar_phase); mbar_phase ^= 1;
          tmem_readout_accum_vec(sO, tmem_addr + TMEM_O_COL, Br, D, D_pad);
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
    if(tid < Br){
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
      float m_final;
      tmem_ld_scalar(TMEM_M_COL, m_final);
      tmem_wait_ld_v40();
      // V42: m_final is RAW (unscaled) here -- unlike V40/V41 where TMEM_M_COL held
      // an already-scaled max (s_row was pre-scaled before ever reaching TMEM). Must
      // multiply by scale_l2e to match the scaled-max convention every kc iteration's
      // corr/exp-argument math already uses, or LSE comes out wrong by a scale-dependent
      // offset (caught by hardware correctness: LSE mismatched while O still matched,
      // since O's normalization only depends on ratios between p values, not m's absolute
      // scale, but LSE is m's absolute value).
      d_LSE[lBase + tid] = 0.6931471805599453f * (m_final * scale_l2e + log2f(sl[tid]));
    }

    __syncthreads();
  } // end pass loop (q_tile_lo, then q_tile_hi)

  if(tid < 32)
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(tmem_addr), "r"(NCOLS) : "memory");

} // end of gqa_v42_causal

// V29_causal launcher — same grid as V28, but a genuinely-populated 512-thread
// block (128 compute + 32 producer + 352 reorder-helper), matching cuDNN's block
// size for real this time (unlike V28's 160-thread block padded to nothing).
template<int Br, int Bc, int D>
void launch_gqa_v42_causal(
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
  static CUtensorMap Qtmap3d, Ktmap3d, Vtmap;
  if(!cfgd){
    const uint64_t kvRows = (uint64_t)B * Hkv * S;
    const uint64_t qRows  = (uint64_t)B * Hq  * S;
    Qtmap3d = make_tma_3d_katom(d_Q, qRows,  (uint64_t)D, (uint32_t)Br);
    Ktmap3d = make_tma_3d_katom(d_K, kvRows, (uint64_t)D, (uint32_t)Bc);
    Vtmap   = make_tma_2d(d_V, kvRows, (uint64_t)D, (uint32_t)Bc, (uint32_t)D);
    cfgd = true;
  }
  gqa_v42_causal<Br, Bc, D><<<GRID, BLOCK>>>(d_O, d_LSE, Qtmap3d, Ktmap3d, Vtmap,
                          B, Hq, Hkv, G, S, scale);
}

// =============================================================================
//  gqa_v46_causal  ("nr" = no-reorder)
//
//  V42 + the three structural changes the cuDNN SASS comparison demands, in
//  order of measured impact on the V42 profile:
//
//  FIX 1 (46.2% of PC samples + ~7% issue + 35% smem-load peak): DELETE the
//    V reorder-copy entirely. cuDNN's kernel contains 0 PRMT and 18 LDS total
//    -- it never materializes V^T. For O = P@V, V is the B operand with
//    K = Bc, N = D; V as TMA-loaded (row-major Bc x D) is *MN-major* for B.
//    tcgen05's instruction descriptor has a b_major bit for exactly this, and
//    the smem matrix descriptor's LBO/SBO encode the MN-major strides. V42
//    only transposed V because make_idesc_bf16 hardcoded K-major B (which is
//    correct for K in S = Q@K^T, where row-major K *is* K-major).
//    Consequences that cascade out with it:
//      - sV (16 KB) gone, sVstage feeds the MMA directly
//      - the 352-thread helper group gone -> 160-thread block
//      - reorder_sync (480-thread named barrier, 2x per KV tile) gone
//      - the per-tile fence.proxy.async / MEMBAR.ALL.CTA pair shrinks to the
//        single fence the compute group already needed for the TMA->MMA handoff
//
//  FIX 2 (4.4% samples on NANOSLEEP 0x989680 = 10,000,000 ns, inside a
//    non-inlined CALL): mbar_wait is replaced by an inlined
//    mbarrier.try_wait.parity spin, matching cuDNN's 106 inlined
//    SYNCS.PHASECHK.TRYWAIT sites. A 10 ms sleep request in a 1.4 ms kernel
//    means one missed phase check can cost more than the whole kernel
//    (hardware clamps NANOSLEEP, but the clamp is still enormous at this
//    scale, and the CALL/RET + WARPSYNC around it isn't free either).
//
//  FIX 3 (the remaining LDS/STS census: 167/146 vs cuDNN's 18/19): O stays
//    RESIDENT IN TMEM across the whole KV loop. The PV MMA accumulates into
//    TMEM_O_COL with accumulate=1 (except the first tile), and the per-tile
//    rescale O *= corr becomes LDTM -> FMUL -> STTM (cuDNN's 32 LDTM /
//    25 STTM), instead of V42's per-tile 64x LDS/FMUL/STS rescale pass PLUS
//    tmem_readout_accum_vec's 64x LDTM/LDS/FADD/STS accumulate pass. sO
//    (33 KB with padding) disappears; smem is only touched again in the
//    epilogue, straight from a final TMEM readout.
//
//  NOT done here (deliberately -- next version, per "one change per version"):
//    S double-buffering in TMEM to overlap QK^T(n+1) with softmax(n)/PV(n),
//    which is the last structural gap vs cuDNN (48 UTCHMMA / 16 UTCBAR sites
//    vs our 12/2, and its stall profile of cheap waits on legit MMA/TMA
//    barriers). The TMEM budget already allows it: after this version the
//    layout is S:[0,64) packed-P-compatible, O:[128,192), M/CORR at 192/193,
//    leaving [64,128) free for S1.
//
//  !!! TWO ENCODING POINTS TO VERIFY (runCorrectness() before benchmarks) !!!
//  Both encode layout, so a wrong bit is a wrong-answer bug, not a perf bug:
//    [VERIFY-1] b_major (tnspB) bit in the instruction descriptor: bit 16
//               (a_major is 15). Corroborated by the fact that your
//               make_idesc_bf16's field offsets (c@4, a@7, b@10, N@17, M@24)
//               match the CUTLASS UMMA::InstrDescriptor layout exactly.
//    [VERIFY-2] The SW128 MN-major descriptor constants: LBO encoded = 1,
//               SBO = 1024 B, swizzle field [63:61] = 1. See the derivation
//               above make_smem_desc_mnV_sw128 -- and note WHY no-swizzle is
//               impossible for V (this is the reason reorderCopyV2 existed).
// =============================================================================

// ---------------------------------------------------------------------------
// FIX 2: the common header's mbar_wait is ALREADY a try_wait.parity spin --
// the two pathologies visible in the SASS are (a) __noinline__ (the CALL +
// WARPSYNC around every wait) and (b) ticks = 0x989680 passed as the
// suspendTimeHint operand, which is precisely the `NANOSLEEP.SYNCS 0x989680`
// (10,000,000 ns requested) at 4.4% of samples. This variant is the same
// spin, __forceinline__, using the NO-HINT form -- which compiles to exactly
// cuDNN's shape:
//     .Lspin: SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [addr], phase
//             @!P0 BRA .Lspin
// ---------------------------------------------------------------------------
__device__ __forceinline__ void mbar_wait_spin(uint32_t bar, int phase){
  asm volatile(
    "{\n\t.reg .pred P1;\n\t"
    "LAB_WAIT_%=:\n\t"
    "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1;\n\t"
    "@!P1 bra.uni LAB_WAIT_%=;\n\t}"
    :: "r"(bar), "r"(phase) : "memory");
}

// ---------------------------------------------------------------------------
// FIX 1a: instruction descriptor with B in MN-major.
// The common header's make_idesc_bf16 sets c_format@4, a_format@7, b_format@10,
// N@17, M@24 -- which matches the CUTLASS UMMA::InstrDescriptor field layout
// exactly. In that same layout, a_major (tnspA) is bit 15 and b_major (tnspB)
// is bit 16 (0 = K-major, 1 = MN-major). Your header never sets them because
// every operand so far has been K-major; V as-loaded is MN-major, so this is
// make_idesc_bf16 + bit 16.                                        [VERIFY-1]
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t make_idesc_bf16_bMN(int M, int N){
  return make_idesc_bf16(M, N) | (1u << 16);   // b_major = MN-major
}

// ---------------------------------------------------------------------------
// FIX 1b: smem matrix descriptor for V read MN-major with 128-BYTE SWIZZLE.
//
// Why swizzle is REQUIRED (not optional) here: the header's canonical
// no-swizzle layouts (both K-major katom and MN-major) demand that the 8 rows
// of each 8x8 core matrix sit at +16 B from each other (packed 128 B core).
// Row-major V has consecutive K rows 128 B apart -- not encodable in LBO/SBO,
// and TMA can't fix it either, because TMA's innermost dimension must be
// globally contiguous (this is exactly why reorderCopyV2 existed). The
// MN-major SWIZZLE_128B canonical layout, however, is defined as:
//     one 128 B span of MN-contiguous elements  x  8 consecutive K rows,
//     XOR-128B-swizzled
// and with D = 64 bf16, one V row IS exactly one 128 B span -- so a TMA load
// with CU_TENSOR_MAP_SWIZZLE_128B deposits row-major V in this layout
// verbatim. The MMA descriptor then declares swizzle mode 1 (128B) and the
// hardware un-XORs on read. Zero shuffle by construction; this is the cuDNN
// scheme (its V path has no reorder and its kernel name is a swizzled-TMA
// fprop variant).
//
// Field values for MN-major / SW128 (Hopper GMMA == tcgen05 encoding,
// cf. CUTLASS make_gmma_desc):                                     [VERIFY-2]
//   LBO encoded = 1   (leading-dim offset is fixed/ignored for swizzled
//                      canonical layouts; encoded value 1 == 16 B)
//   SBO         = 1024 B = 8 K-rows x 128 B  (encoded 1024 >> 4 = 64)
//   swizzle bits [63:61] = 1  (128B swizzle)
//   valid bit 46 set, matching make_smem_desc.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint64_t make_smem_desc_mnV_sw128(void* smem_ptr){
  uint64_t addr = (uint64_t)__cvta_generic_to_shared(smem_ptr);
  uint64_t desc = desc_encode(addr)
                | (1ull << 16)                    // LBO encoded = 1  [VERIFY-2]
                | (desc_encode(1024ull) << 32)    // SBO = 1024 B
                | (1ull << 46)                    // valid bit (same as make_smem_desc)
                | (2ull << 61);                   // swizzle mode 1 = 128B [VERIFY-2]
  return desc;
}

// Advance the V descriptor by one MMA K-atom (16 K rows): 16 rows x 128 B =
// 2048 B = two SW128 8-row groups. Same low-14-bit address-field arithmetic
// as the header's advance_desc_katom.
__device__ __forceinline__ uint64_t advance_desc_katom_mnV(uint64_t desc, int katom){
  uint64_t units     = (uint64_t)katom * (2048ull >> 4);
  uint64_t base_addr = desc & 0x3FFFull;
  uint64_t new_addr  = (base_addr + units) & 0x3FFFull;
  return (desc & ~0x3FFFull) | new_addr;
}

// Host side: V's tensor map, identical to make_tma_2d except SWIZZLE_128B.
// Box inner extent is 64 bf16 = 128 B = exactly one swizzle span (required:
// inner box bytes must be <= the swizzle span).
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
template<int Br, int Bc, int D>
__global__ void __launch_bounds__(160, 1) gqa_v46_causal(
  __nv_bfloat16 *d_O,
  float *d_LSE,
  const __grid_constant__ CUtensorMap Qtmap3d,
  const __grid_constant__ CUtensorMap Ktmap3d,
  const __grid_constant__ CUtensorMap Vtmap,
  int B, int Hq, int Hkv, int G, int S, float scale
){
  // static_assert(Br == 128, "consumer group is hardwired to 128 threads");
  // static_assert(Br == Bc,  "causal tile-skip + diagonal-tile mask requires Br == Bc");
  // static_assert(D == 64,   "MN-major V descriptor constants below assume D = 64 (see SBO derivation)");

  const int tid = threadIdx.x;          // 0..127 compute, 128..159 producer. Helpers are GONE (FIX 1).

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

  // FIX 1 + FIX 3 shrink smem from ~115 KB to ~66 KB: sV and sO are gone.
  __shared__ __align__(16)  __nv_bfloat16 sQ[Br * D];
  __shared__ __align__(128) __nv_bfloat16 sK[2][Bc * D];
  __shared__ __align__(1024) __nv_bfloat16 sVstage[2][Bc * D];  // MMA reads this DIRECTLY now;
                                                                // 1024B-aligned = one full SW128 atom (8 rows x 128B)
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

  // TMEM layout unchanged from V42 (S/P share [0,128); O in [128,192);
  // M/CORR at 192/193). [64,128) is now provably idle every iteration --
  // reserved for the S double-buffer in the NEXT version.
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
    // FIX 3: no sO zero-fill -- O's first PV MMA writes TMEM with accumulate=0.

    if(tid == 0){
      mbar_init(mma_bar, 1);
      mbar_init(lbarK0, 1); mbar_init(lbarK1, 1);
      mbar_init(fbarK0, 1); mbar_init(fbarK1, 1);
      mbar_init(lbarV0, 1); mbar_init(lbarV1, 1);
      mbar_init(fbarV0, 1); mbar_init(fbarV1, 1);
      mbar_init(qbar, 1);
      mbar_expect_tx(qbar, TX);
      tma_load_3d(sQ_addr, &Qtmap3d, 0, (int)((b * Hq + hq) * S + q_row0), 0, qbar);
      mbar_arrive(qbar);
    }
    __syncthreads();

    if(tid >= 128){
      // ---- Producer warp: unchanged, except waits are the inlined spin. ----
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
          tma_load_3d(cur_sK_addr, &Ktmap3d, 0, r, 0, cur_lbarK);
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
      const uint64_t descQ_base = make_smem_desc(sQ, Br);

      for(int kc = 0; kc < nKVTiles; ++kc){
        const int slot = kc & 1;
        const uint32_t cur_lbarK = (slot == 0) ? lbarK0 : lbarK1;
        const uint32_t cur_lbarV = (slot == 0) ? lbarV0 : lbarV1;
        const uint32_t cur_fbarK = (slot == 0) ? fbarK0 : fbarK1;
        const uint32_t cur_fbarV = (slot == 0) ? fbarV0 : fbarV1;
        mbar_wait_spin(cur_lbarK, (load_phase_K_bits >> slot) & 1); load_phase_K_bits ^= (1 << slot);
        mbar_wait_spin(cur_lbarV, (load_phase_V_bits >> slot) & 1); load_phase_V_bits ^= (1 << slot);
        // One async-proxy fence for the TMA->MMA handoff of BOTH K and V.
        // (V42 needed this in TWO groups plus reorder_sync barriers; FIX 1
        // collapses it to the fence the compute group issues anyway.)
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

        float s_row[Bc];

        { // ---- S = Q @ K^T ---- (unchanged from V42)
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

        { // ---- softmax: identical FFMA-fused math to V42 ----
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

          // FIX 3: O rescale moves from smem to a TMEM<->register round-trip,
          // exactly cuDNN's LDTM -> FMUL -> STTM shape. Skipped for kc==0
          // (O not yet written; PV below uses accumulate=0 then).
          if(kc > 0){
            float o_row[D];   // D = 64: two x32 loads/stores
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

        { // ---- O += P @ V, V read DIRECTLY from sVstage (FIX 1) ----
          const uint64_t descV_base = make_smem_desc_mnV_sw128(sVstage[slot]);
          const uint32_t idesc      = make_idesc_bf16_bMN(Br, D);
          if(tid == 0){
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            for(int kt = 0; kt < Bc/16; ++kt){
              uint64_t descV = advance_desc_katom_mnV(descV_base, kt);
              uint32_t p_operand_addr = tmem_addr + TMEM_S_COL + (uint32_t)(kt * 8);
              // FIX 3: accumulate across the WHOLE KV loop -- only the very
              // first atom of the very first tile clears.
              uint32_t accumulate = (kc > 0 || kt > 0) ? 1u : 0u;
              asm volatile(
                "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n\t}\n"
                :: "r"(tmem_addr + TMEM_O_COL), "r"(p_operand_addr), "l"(descV), "r"(idesc), "r"(accumulate) : "memory");
            }
            mbar_commit_mma(mma_bar);
          }
          mbar_wait_spin(mma_bar, mbar_phase); mbar_phase ^= 1;
          if(tid == 0) mbar_arrive(cur_fbarV);   // V slot free the moment the MMA is done
          // FIX 3: NO tmem_readout_accum_vec here. O stays in TMEM.
          consumer_sync();
        }
      } // end kv loop

      // ---- Epilogue: single TMEM readout -> normalize -> global (FIX 3) ----
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
void launch_gqa_v46_causal(
  __nv_bfloat16 *d_Q, __nv_bfloat16 *d_K, __nv_bfloat16 *d_V,
  __nv_bfloat16 *d_O, float *d_LSE,
  int B, int Hq, int Hkv, int S, int G, float scale
){
  static_assert(Br == 128 && Br == Bc && D == 64, "see kernel static_asserts");
  assert((S / Br) % 2 == 0);

  dim3 GRID(B * Hq * ((S / Br) / 2), 1, 1);
  dim3 BLOCK(160);   // 128 compute + 32 producer. Helper group deleted (FIX 1).

  static bool cfgd = false;
  static CUtensorMap Qtmap3d, Ktmap3d, Vtmap;
  if(!cfgd){
    const uint64_t kvRows = (uint64_t)B * Hkv * S;
    const uint64_t qRows  = (uint64_t)B * Hq  * S;
    Qtmap3d = make_tma_3d_katom(d_Q, qRows,  (uint64_t)D, (uint32_t)Br);
    Ktmap3d = make_tma_3d_katom(d_K, kvRows, (uint64_t)D, (uint32_t)Bc);
    Vtmap   = make_tma_2d_sw128(d_V, kvRows, (uint64_t)D, (uint32_t)Bc, (uint32_t)D);  // FIX 1: 128B-swizzled V
    cfgd = true;
  }
  gqa_v46_causal<Br, Bc, D><<<GRID, BLOCK>>>(d_O, d_LSE, Qtmap3d, Ktmap3d, Vtmap,
                                               B, Hq, Hkv, G, S, scale);
}

// =============================================================================
//  gqa_v47_causal — V46 + K load via a plain SWIZZLE_128B 2D map, replacing
//  make_tma_3d_katom's atom-native 3D box.
//
//  Motivation (from a real NCU capture on V46, not a SASS guess): Source
//  Counters flagged 87% excessive global sectors (Est. Speedup 84.45%), with
//  no equivalent flag on the real cuDNN kernel's own profile. K's atom-native
//  TMA (ATOM=8 innermost dim) lands data directly into the canon_idx atom
//  layout, which is a genuine gather relative to K's row-major HBM storage —
//  exactly why it was fast to ISSUE (no reorder-copy) but slow to actually
//  TRANSFER (many small strided sectors). V hit the same class of problem
//  and V46 already fixed it for V via a plain contiguous 2D SWIZZLE_128B map;
//  this applies the identical mechanism to K.
//
//  Why K's derivation ISN'T just a copy of V's fix, and where the two
//  genuinely differ:
//    - V's reduction axis for O=P@V is Bc (V's own row axis) — physically the
//      OUTER/strided axis, so V's K-atom advance (advance_desc_katom_mnV)
//      steps BETWEEN 8-row swizzle groups (2048B = 16 rows x 128B per step).
//    - K's reduction axis for S=Q@Kᵀ is D — physically the INNER/contiguous
//      axis (one row's whole 128B span), and D=64 means the ENTIRE reduction
//      fits inside ONE row's own swizzle span. Per the SW128 swizzle's own
//      definition (XORs WHICH physical 128B slot each of the 8 logical rows
//      lands in; does not reorder bytes WITHIN a row's own span), advancing
//      16 elements along D never crosses a row/group boundary — it's a
//      simple +32B offset from whatever (possibly-permuted) base address the
//      row's own descriptor bits already select. Hence a NEW, much smaller
//      advance function below (advance_desc_katom_kmajor_sw128), not a reuse
//      of advance_desc_katom_mnV.
//    - K is genuinely K-major here (D, the physically-contiguous axis, IS
//      the reduction axis) — unlike V, which was physically-contiguous-but-
//      logically-MN-major. So the instruction descriptor needs NO b_major
//      bit for K (make_idesc_bf16(Br, Bc) is reused unchanged); only the
//      smem descriptor (which encodes WHERE the data is and how it's
//      swizzled, not which axis is "major") needs the SW128 variant. That
//      smem descriptor's bit values (LBO encoded=1, SBO=1024B, swizzle=1)
//      turn out to be shape-driven, not direction-driven — identical for K
//      and V here since both are laid out as [Bc rows, 128B span] — so
//      make_smem_desc_mnV_sw128 is reused as-is for K's base descriptor.
//
//  !!! NEW DERIVATION, NOT YET HARDWARE-VERIFIED (runCorrectness() first) !!!
//  [VERIFY-3] advance_desc_katom_kmajor_sw128's "no cross-row-group
//             correction needed within a single 128B span" assumption. If
//             wrong, expect a wrong-answer bug (not a perf regression) —
//             same failure signature as V45's shuffle bug: check for exact
//             mismatches, not just tolerance drift.
// =============================================================================

// Advance a K-major SW128 descriptor by one MMA K-atom (16 elements along D,
// the reduction axis) -- see header above for why this is a SMALL, purely
// intra-row-span advance (32B = 16 bf16 elements), unlike V's inter-row-group
// advance_desc_katom_mnV.                                          [VERIFY-3]
__device__ __forceinline__ uint64_t advance_desc_katom_kmajor_sw128(uint64_t desc, int katom){
  uint64_t units     = (uint64_t)katom * 2ull;   // 32 B = 2 units of 16 B
  uint64_t base_addr = desc & 0x3FFFull;
  uint64_t new_addr  = (base_addr + units) & 0x3FFFull;
  return (desc & ~0x3FFFull) | new_addr;
}

template<int Br, int Bc, int D>
__global__ void __launch_bounds__(160, 1) gqa_v47_causal(
  __nv_bfloat16 *d_O,
  float *d_LSE,
  const __grid_constant__ CUtensorMap Qtmap3d,
  const __grid_constant__ CUtensorMap Ktmap,
  const __grid_constant__ CUtensorMap Vtmap,
  int B, int Hq, int Hkv, int G, int S, float scale
){
  static_assert(Br == 128, "consumer group is hardwired to 128 threads");
  static_assert(Br == Bc,  "causal tile-skip + diagonal-tile mask requires Br == Bc");
  static_assert(D == 64,   "SW128 K/V descriptor constants assume D = 64 (one 128B span)");

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

  __shared__ __align__(16)  __nv_bfloat16 sQ[Br * D];
  __shared__ __align__(1024) __nv_bfloat16 sK[2][Bc * D];      // SW128 now (was atom-native canon_idx)
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
      tma_load_3d(sQ_addr, &Qtmap3d, 0, (int)((b * Hq + hq) * S + q_row0), 0, qbar);
      mbar_arrive(qbar);
    }
    __syncthreads();

    if(tid >= 128){
      // ---- Producer warp: K load is now a plain 2D TMA (was 3D atom-native). ----
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
      const uint64_t descQ_base = make_smem_desc(sQ, Br);

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

        { // ---- S = Q @ K^T ---- K descriptor is now K-major SW128 (was canon_idx) ----
          const uint64_t descK_base = make_smem_desc_mnV_sw128(sK[slot]);   // [VERIFY-3], see header
          const uint32_t idesc      = make_idesc_bf16(Br, Bc);              // unchanged: K genuinely K-major
          if(tid == 0){
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            for(int kt = 0; kt < D/16; ++kt){
              uint64_t descQ = advance_desc_katom(descQ_base, kt, Br);
              uint64_t descK = advance_desc_katom_kmajor_sw128(descK_base, kt);   // [VERIFY-3]
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

        { // ---- softmax: identical FFMA-fused math to V42/V46 ----
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

        { // ---- O += P @ V, V read directly from sVstage (unchanged from V46) ----
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

      // ---- Epilogue: single TMEM readout -> normalize -> global (unchanged from V46) ----
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
void launch_gqa_v47_causal(
  __nv_bfloat16 *d_Q, __nv_bfloat16 *d_K, __nv_bfloat16 *d_V,
  __nv_bfloat16 *d_O, float *d_LSE,
  int B, int Hq, int Hkv, int S, int G, float scale
){
  static_assert(Br == 128 && Br == Bc && D == 64, "see kernel static_asserts");
  assert((S / Br) % 2 == 0);

  dim3 GRID(B * Hq * ((S / Br) / 2), 1, 1);
  dim3 BLOCK(160);

  static bool cfgd = false;
  static CUtensorMap Qtmap3d, Ktmap, Vtmap;
  if(!cfgd){
    const uint64_t kvRows = (uint64_t)B * Hkv * S;
    const uint64_t qRows  = (uint64_t)B * Hq  * S;
    Qtmap3d = make_tma_3d_katom(d_Q, qRows,  (uint64_t)D, (uint32_t)Br);
    Ktmap   = make_tma_2d_sw128(d_K, kvRows, (uint64_t)D, (uint32_t)Bc, (uint32_t)D);  // was make_tma_3d_katom
    Vtmap   = make_tma_2d_sw128(d_V, kvRows, (uint64_t)D, (uint32_t)Bc, (uint32_t)D);
    cfgd = true;
  }
  gqa_v47_causal<Br, Bc, D><<<GRID, BLOCK>>>(d_O, d_LSE, Qtmap3d, Ktmap, Vtmap,
                                               B, Hq, Hkv, G, S, scale);
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


// =============================================================================
//  gqa_v49_causal -- V48 + double-buffered S in TMEM + early QK^T issue.
//  See file header for the full delta description.
// =============================================================================
template<int Br, int Bc, int D>
__global__ void __launch_bounds__(160, 1) gqa_v49_causal(
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
  __shared__ __align__(8) uint64_t s_qk_bar[2];   // per-S-slot QK^T completion (was s_mma_bar)
  __shared__ __align__(8) uint64_t s_pv_bar;      // PV completion (single, reused)
  __shared__ __align__(8) uint64_t s_load_bar_K[2], s_free_bar_K[2];
  __shared__ __align__(8) uint64_t s_load_bar_V[2], s_free_bar_V[2];
  __shared__ __align__(8) uint64_t s_load_bar_Q;

  const uint32_t qkbar0 = (uint32_t)__cvta_generic_to_shared(&s_qk_bar[0]);
  const uint32_t qkbar1 = (uint32_t)__cvta_generic_to_shared(&s_qk_bar[1]);
  const uint32_t pv_bar = (uint32_t)__cvta_generic_to_shared(&s_pv_bar);
  const uint32_t lbarK0  = (uint32_t)__cvta_generic_to_shared(&s_load_bar_K[0]);
  const uint32_t lbarK1  = (uint32_t)__cvta_generic_to_shared(&s_load_bar_K[1]);
  const uint32_t fbarK0  = (uint32_t)__cvta_generic_to_shared(&s_free_bar_K[0]);
  const uint32_t fbarK1  = (uint32_t)__cvta_generic_to_shared(&s_free_bar_K[1]);
  const uint32_t lbarV0  = (uint32_t)__cvta_generic_to_shared(&s_load_bar_V[0]);
  const uint32_t lbarV1  = (uint32_t)__cvta_generic_to_shared(&s_load_bar_V[1]);
  const uint32_t fbarV0  = (uint32_t)__cvta_generic_to_shared(&s_free_bar_V[0]);
  const uint32_t fbarV1  = (uint32_t)__cvta_generic_to_shared(&s_free_bar_V[1]);
  const uint32_t qbar    = (uint32_t)__cvta_generic_to_shared(&s_load_bar_Q);

  constexpr uint32_t TMEM_S0_COL   = 0;                          // S slot 0: [0,128)
  constexpr uint32_t TMEM_S1_COL   = (uint32_t)Bc;               // S slot 1: [128,256)
  constexpr uint32_t TMEM_O_COL    = 2u * (uint32_t)Bc;          // O: [256,320)
  constexpr uint32_t TMEM_P_COL    = TMEM_O_COL + (uint32_t)D;   // packed P: [320,384)
  constexpr uint32_t TMEM_M_COL    = TMEM_P_COL + (uint32_t)Bc/2;// 384
  constexpr uint32_t TMEM_CORR_COL = TMEM_M_COL + 1;             // 385
  constexpr uint32_t NCOLS = 512;   // was 256 -- first hardware use of a full
                                    // 512-col allocation in this codebase

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
      mbar_init(qkbar0, 1); mbar_init(qkbar1, 1);
      mbar_init(pv_bar, 1);
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
      int qk_phase_bits = 0;                 // per-slot phase for qk_bar[2]
      int pv_phase = 0;
      int load_phase_K_bits = 0, load_phase_V_bits = 0;   // used by tid==0 only now
      mbar_wait_spin(qbar, 0);
      const uint64_t descQ_base = make_smem_desc_mnV_sw128(sQ);
      const uint32_t idescQK = make_idesc_bf16(Br, Bc);
      const uint32_t idescPV = make_idesc_bf16_bMN(Br, D);

      // QK^T(tile) -> S[tile&1]. tid==0 only. Caller has already waited
      // lbarK[tile&1] and issued the async-proxy fence.
      auto issue_qk = [&](int tile){
        const int s = tile & 1;
        const uint32_t dst = tmem_addr + ((s == 0) ? TMEM_S0_COL : TMEM_S1_COL);
        const uint64_t descK_base = make_smem_desc_mnV_sw128(sK[s]);
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        for(int kt = 0; kt < D/16; ++kt){
          uint64_t descQ = advance_desc_katom_kmajor_sw128(descQ_base, kt);
          uint64_t descK = advance_desc_katom_kmajor_sw128(descK_base, kt);
          uint32_t accumulate = (kt > 0) ? 1u : 0u;
          asm volatile(
            "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}\n"
            :: "r"(dst), "l"(descQ), "l"(descK), "r"(idescQK), "r"(accumulate) : "memory");
        }
        // Commit BEFORE PV(tile-1) is issued: snapshots only this QK^T
        // (everything issued earlier has already been waited on).
        mbar_commit_mma((s == 0) ? qkbar0 : qkbar1);
      };

      // Prologue: get QK^T(0) in flight before entering the loop.
      if(tid == 0){
        mbar_wait_spin(lbarK0, (load_phase_K_bits >> 0) & 1);
        load_phase_K_bits ^= 1;
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        issue_qk(0);
      }

      for(int kc = 0; kc < nKVTiles; ++kc){
        const int slot = kc & 1;
        const uint32_t cur_qkbar = (slot == 0) ? qkbar0 : qkbar1;
        const uint32_t cur_fbarK = (slot == 0) ? fbarK0 : fbarK1;
        const uint32_t cur_fbarV = (slot == 0) ? fbarV0 : fbarV1;
        const uint32_t S_col     = (slot == 0) ? TMEM_S0_COL : TMEM_S1_COL;

        float s_row[Bc];

        { // ---- consume S = Q @ K^T (issued one iteration -- or prologue -- ago) ----
          mbar_wait_spin(cur_qkbar, (qk_phase_bits >> slot) & 1); qk_phase_bits ^= (1 << slot);
          if(tid == 0) mbar_arrive(cur_fbarK);           // K[slot] consumed

          asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
          #pragma unroll
          for(int c = 0; c < Bc; c += 32) tmem_ld32(S_col + (uint32_t)c, &s_row[c]);
          tmem_wait_ld();

          // Early-issue QK^T(kc+1) into the other S slot: from here on the
          // tensor cores compute tile kc+1's scores UNDER softmax(kc).
          // Writing S[1-slot] is safe: its previous contents (tile kc-1's
          // scores) were fully read by all warps before iteration kc-1's
          // first consumer_sync.
          if(tid == 0 && kc + 1 < nKVTiles){
            const int ns = 1 - slot;
            mbar_wait_spin((ns == 0) ? lbarK0 : lbarK1, (load_phase_K_bits >> ns) & 1);
            load_phase_K_bits ^= (1 << ns);
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            issue_qk(kc + 1);
          }

          if(kc == q_tile){
            #pragma unroll
            for(int j = 0; j < Bc; ++j) if(j > tid) s_row[j] = -INFINITY;
          }
          consumer_sync();
        }

        { // ---- softmax: identical math to V48; packed P now goes to its own
          //      fixed TMEM region (P last read by PV(kc-1), already waited) ----
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
          for(int c = 0; c < Bc / 2; c += 32) tmem_st32(TMEM_P_COL + (uint32_t)c, &p_packed[c]);
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

        { // ---- O += P @ V, P read from its fixed region ----
          if(tid == 0){
            mbar_wait_spin((slot == 0) ? lbarV0 : lbarV1, (load_phase_V_bits >> slot) & 1);
            load_phase_V_bits ^= (1 << slot);
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            const uint64_t descV_base = make_smem_desc_mnV_sw128(sVstage[slot]);
            asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
            for(int kt = 0; kt < Bc/16; ++kt){
              uint64_t descV = advance_desc_katom_mnV(descV_base, kt);
              uint32_t p_operand_addr = tmem_addr + TMEM_P_COL + (uint32_t)(kt * 8);
              uint32_t accumulate = (kc > 0 || kt > 0) ? 1u : 0u;
              asm volatile(
                "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %4, 0;\n\t"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n\t}\n"
                :: "r"(tmem_addr + TMEM_O_COL), "r"(p_operand_addr), "l"(descV), "r"(idescPV), "r"(accumulate) : "memory");
            }
            // This commit also sweeps in the possibly-still-running
            // QK^T(kc+1) -- see file header; conservative but correct.
            mbar_commit_mma(pv_bar);
          }
          mbar_wait_spin(pv_bar, pv_phase); pv_phase ^= 1;
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
void launch_gqa_v49_causal(
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
  gqa_v49_causal<Br, Bc, D><<<GRID, BLOCK>>>(d_O, d_LSE, Qtmap, Ktmap, Vtmap,
                                               B, Hq, Hkv, G, S, scale);
}


int main(){
  std::cout << "Benchmarking CAUSAL Grouped-Query Attention — Blackwell SM_103 (B300), v3 file\n";

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
    runCorrectness("V42-causal", [&](){ launch_gqa_v42_causal<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); });
    runCorrectness("V46-causal", [&](){ launch_gqa_v46_causal<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); });
    runCorrectness("V47-causal", [&](){ launch_gqa_v47_causal<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); });
    runCorrectness("V48-causal", [&](){ launch_gqa_v48_causal<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); });
    runCorrectness("V49-causal", [&](){ launch_gqa_v49_causal<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); });
  }

  //* ── Benchmark ──────────────────────────────────────────────────────────
  //* Causal attention FLOPs: only nTiles*(nTiles+1)/2 of the nTiles*nTiles (q_tile,kc)
  //* pairs a full/non-causal kernel would visit are ever computed (Br==Bc==nTiles tiles
  //* wide) — roughly HALF the work of the non-causal V19, not the same S*S count.
  constexpr long long nTiles     = S / Br;                          // 32
  constexpr long long tileVisits = nTiles * (nTiles + 1) / 2;        // 528
  long long flops = 4LL * B * Hq * (long long)Br * Bc * D * tileVisits;
  size_t bytes = (2 * Nq + 2 * Nkv) * sizeof(__nv_bfloat16) + Nlse * sizeof(float);

  KernelStats stats = benchmarkKernel(
    [&](){ launch_gqa_v42_causal<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); },
    100, 25, flops, bytes
  );
  displayStats("V42-causal — baseline copied in from GQA_sm103_causal_v2.cu for this file's comparisons", stats);

  stats = benchmarkKernel(
    [&](){ launch_gqa_v46_causal<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); },
    100, 25, flops, bytes
  );
  displayStats("V46-causal — V42 + delete V reorder-copy (MN-major swizzled V descriptor) "
               "+ inlined mbar_wait spin + O resident in TMEM across the whole KV loop", stats);

  stats = benchmarkKernel(
    [&](){ launch_gqa_v47_causal<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); },
    100, 25, flops, bytes
  );
  displayStats("V47-causal — V46 + K load via plain SWIZZLE_128B 2D map (K-major SW128 descriptor), "
               "replacing make_tma_3d_katom's atom-native 3D box (fixes 87% excessive global sectors)", stats);

  stats = benchmarkKernel(
    [&](){ launch_gqa_v48_causal<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); },
    100, 25, flops, bytes
  );
  displayStats("V48-causal — V47 + Q load via plain SWIZZLE_128B 2D map too (replacing Q's own "
               "atom-native 3D box) -- tests whether Q, not K, was the dominant uncoalesced-access source", stats);

  stats = benchmarkKernel(
    [&](){ launch_gqa_v49_causal<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); },
    100, 25, flops, bytes
  );
  displayStats("V48 + double-buffered S in TMEM + early QK^T issue."
               "atom-native 3D box) -- tests whether Q, not K, was the dominant uncoalesced-access source", stats);

  // Add new versions here: runCorrectness + benchmarkKernel/displayStats,
  // matching the pattern above.

  CUDA_CHECK(cudaFree(d_Q));
  CUDA_CHECK(cudaFree(d_K));
  CUDA_CHECK(cudaFree(d_V));
  CUDA_CHECK(cudaFree(d_O));
  CUDA_CHECK(cudaFree(d_LSE));

  return 0;
}
