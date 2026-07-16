// GQA_sm103_causal_v2.cu — continuation of GQA_sm103_causal.cu's version ladder
// (V19-V36), starting fresh from V37 onward. The original file grew too large
// to work in comfortably; shared device-side helpers now live in
// GQA_sm103_causal_common.cuh (#included below) instead of being duplicated.
//
// V34 (the fastest verified kernel as of the split: 1.9619ms, 108.4 TFLOPS,
// ~9.3x from cuDNN's 0.2110ms reference) is copied in below as the baseline
// every new version in this file should be compared against — see
// GQA_sm103_causal.cu for its full derivation history (sS/sO bank-conflict
// padding, vectorized bank-conflict-free reorder-copy, local-memory
// elimination) and for V19-V36's complete history including V36's negative
// result (genuine 4-warpgroup concurrency, fully optimized, still 44% slower
// than V34 — see that file's Stage 20 for why).
#include "GQA_sm103_causal_common.cuh"

// =================================
//  gqa_v34_causal - V32 (frozen at 2.0463ms, sS+sO+vectorized-reorder-copy fixes)
//  + eliminating LOCAL MEMORY traffic from small runtime-indexed arrays.
//  NCU's "L1TEX Local Store Access Pattern" rule (est. speedup 40.25%) flagged
//  substantial local load/store traffic (~2.03M local loads, ~1.12M local
//  stores, ~250MB total) despite ptxas reporting ZERO register spills --
//  confirmed via SASS disassembly showing an STL.64 instruction right at the
//  initialization of the free_phase_K / free_phase_V 2-element arrays. Root
//  cause: a small array indexed by a RUNTIME variable (slot = kc & 1, not a
//  compile-time constant) can't live in registers -- the compiler is forced
//  to place it in local memory (backed by L1TEX/L2), even though this isn't a
//  register-pressure "spill" in ptxas's usual sense. Every slot-indexed local
//  array in the kernel was affected this way: the barrier-address pairs
//  (read-only after init), the TMA-destination-address pairs (also read-only),
//  and the mbarrier phase-counter pairs (read-write, toggled every iteration).
//  Fix, mechanical and behavior-preserving (not a redesign):
//   - Read-only address pairs: replaced a stale array index with a select
//     between the two scalar variables the array was originally built from --
//     compiles to a register-resident select/cmov instead of a memory access.
//   - Read-write phase counters: packed both slots' phase bits into a single
//     scalar int, replacing array indexing with shift/mask/xor (slot as a
//     shift amount, an ALU op, instead of an address) -- read via shift+mask,
//     toggle via xor. Provably identical toggle semantics to the array
//     version, just a different storage representation.
// =================================
template<int Br, int Bc, int D>
__global__ void gqa_v34_causal(
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
  const uint32_t sK_addr0      = (uint32_t)__cvta_generic_to_shared(sK[0]);
  const uint32_t sK_addr1      = (uint32_t)__cvta_generic_to_shared(sK[1]);
  const uint32_t sVstage_addr0 = (uint32_t)__cvta_generic_to_shared(sVstage[0]);
  const uint32_t sVstage_addr1 = (uint32_t)__cvta_generic_to_shared(sVstage[1]);

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
          if(tid == 0) mbar_arrive(cur_fbarK);
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

} // end of gqa_v34_causal

// V29_causal launcher — same grid as V28, but a genuinely-populated 512-thread
// block (128 compute + 32 producer + 352 reorder-helper), matching cuDNN's block
// size for real this time (unlike V28's 160-thread block padded to nothing).
template<int Br, int Bc, int D>
void launch_gqa_v34_causal(
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
  gqa_v34_causal<Br, Bc, D><<<GRID, BLOCK>>>(d_Q, d_O, d_LSE, Ktmap3d, Vtmap,
                          B, Hq, Hkv, G, S, scale);
}

// =================================
//  gqa_v37_causal - V34 (frozen at 1.9619ms) with the two reorder_sync() named-
//  barrier calls (id 2, 480 threads) REPLACED by mbarrier-based signaling,
//  eliminating the rigid 480-thread rendezvous entirely rather than just
//  reducing its count. Motivated directly by NCU data: Warp Stall Sampling
//  showed 51% of ALL stall samples landing on the helper group's reorderCopyV2
//  call (gated by these two barriers), and Warp State showed Stall Barrier at
//  ~7.6 of ~14 cycles/instruction (57%) -- the dominant cost, and the one
//  cuDNN barely has.
//
//  Verified (before writing any code) that the two barriers each prevent a
//  DIFFERENT, genuine hazard, so neither is simply redundant:
//   - Barrier A (before copy): prevents idle helpers from overwriting the
//     single-buffered sV before compute finishes READING it during the
//     PREVIOUS iteration's PV step (a WAR hazard).
//   - Barrier B (after copy): guarantees helpers' writes to sV are actually
//     VISIBLE to compute's LATER PV read (a real CUDA memory-model
//     requirement -- not just "enough time passing").
//  Reusing a later/earlier barrier to cover both roles doesn't work: compute's
//  PV read happens BEFORE it reaches the next iteration's entry point in its
//  own program order, so no barrier positioned there can retroactively order
//  a read that already happened.
//
//  Fix: replace both with two mbarriers, modeled directly on V30's own proven
//  vfree_lo/hi + vready pattern (the same technique, adapted to V34's
//  sequential -- not concurrent -- architecture):
//   - `vfree` (NEW, count=1): compute's tid==0 arrives it immediately after
//     the PV MMA's mbar_wait(mma_bar,...) succeeds -- the exact point sV's
//     read is provably done (the MMA operand read is complete; the
//     SUBSEQUENT TMEM readout doesn't touch sV at all). Both compute and
//     helpers wait on it before starting their OWN copy contribution, guarded
//     by kc>=1 (nothing to wait for on the first tile, matching this file's
//     existing guard-pattern style elsewhere).
//   - `fbarV0`/`fbarV1` (EXISTING, repurposed): count changed from 1 to
//     NREORDER=480 -- every one of the 480 copy participants (compute's 128
//     lanes + helpers' 352) now arrives it individually right after their OWN
//     reorderCopyV2 slice finishes, instead of only tid==0 arriving it after
//     a full barrier-consolidated rendezvous. The producer's existing
//     mbar_wait(cur_fbarV,...) (gated kc>=2 for double-buffering) is UNCHANGED
//     in meaning -- it now genuinely reflects "every reader of sVstage is
//     done" rather than being gated behind a blocking rendezvous. Compute ALSO
//     waits on this same signal, but only right before it actually needs sV --
//     i.e. right before issuing PV, not immediately after the copy.
//  Verified this reuses an ALREADY-established release/acquire pairing in
//  this codebase: mbar_wait's try_wait already uses explicit .acquire, paired
//  with a plain mbar_arrive (implicit release per PTX's default arrive
//  semantics) -- the exact pairing every TMA-load handoff in this file
//  already relies on; only the ARRIVAL COUNT changes (1 -> 480), which is
//  standard, well-defined mbarrier usage (this is literally what the
//  "expected count" parameter to mbarrier.init is for), not a novel
//  primitive.
//
//  Net effect: compute does its own copy slice, fires two non-blocking
//  arrives (vready via cur_fbarV, no separate mbarrier needed for that), and
//  proceeds straight to QK^T/softmax/rescale -- the SAME real work it always
//  did -- checking back in for the group's full completion only at the last
//  possible moment (right before PV). Helpers do their slice and loop back
//  immediately, gated only by vfree. This avoids V33's failure mode (compute
//  isn't given MORE work before a rigid rendezvous -- there's no rendezvous
//  left to delay) and V35's failure mode (helpers are untouched, still doing
//  the same distributed copy work).
//
//  Correctness note: this touches cross-thread-group memory ordering
//  directly -- the same class of hazard that caused V25/V27's races. Run
//  correctness 5+ times before trusting a single passing result.
// =================================
template<int Br, int Bc, int D>
__global__ void gqa_v37_causal(
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
  __shared__ __align__(8) uint64_t s_vfree;   // NEW: replaces the two reorder_sync() calls (see header)

  const uint32_t mma_bar = (uint32_t)__cvta_generic_to_shared(&s_mma_bar);
  const uint32_t lbarK0  = (uint32_t)__cvta_generic_to_shared(&s_load_bar_K[0]);
  const uint32_t lbarK1  = (uint32_t)__cvta_generic_to_shared(&s_load_bar_K[1]);
  const uint32_t fbarK0  = (uint32_t)__cvta_generic_to_shared(&s_free_bar_K[0]);
  const uint32_t fbarK1  = (uint32_t)__cvta_generic_to_shared(&s_free_bar_K[1]);
  const uint32_t lbarV0  = (uint32_t)__cvta_generic_to_shared(&s_load_bar_V[0]);
  const uint32_t lbarV1  = (uint32_t)__cvta_generic_to_shared(&s_load_bar_V[1]);
  const uint32_t fbarV0  = (uint32_t)__cvta_generic_to_shared(&s_free_bar_V[0]);
  const uint32_t fbarV1  = (uint32_t)__cvta_generic_to_shared(&s_free_bar_V[1]);
  const uint32_t vfree   = (uint32_t)__cvta_generic_to_shared(&s_vfree);

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
  const uint32_t sK_addr0      = (uint32_t)__cvta_generic_to_shared(sK[0]);
  const uint32_t sK_addr1      = (uint32_t)__cvta_generic_to_shared(sK[1]);
  const uint32_t sVstage_addr0 = (uint32_t)__cvta_generic_to_shared(sVstage[0]);
  const uint32_t sVstage_addr1 = (uint32_t)__cvta_generic_to_shared(sVstage[1]);

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
      mbar_init(fbarV0, NREORDER); mbar_init(fbarV1, NREORDER);   // was count=1; see header
      mbar_init(vfree, 1);
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
      // ---- Reorder-helper group: mbarrier-based handoff (vfree + fbarV)
      // instead of the two reorder_sync() calls -- see header. Waits on vfree
      // before overwriting sV, arrives its OWN slice's completion on
      // cur_fbarV, then loops back immediately -- no rendezvous with
      // compute needed. ----
      const int pidx = tid - 32;   // maps tid [160,511] -> contiguous [128,479]
      int load_phase_V_bits = 0;
      int vfree_phase = 0;
      for(int kc = 0; kc < nKVTiles; ++kc){
        const int slot = kc & 1;
        const uint32_t cur_lbarV = (slot == 0) ? lbarV0 : lbarV1;
        const uint32_t cur_fbarV = (slot == 0) ? fbarV0 : fbarV1;
        mbar_wait(cur_lbarV, (load_phase_V_bits >> slot) & 1); load_phase_V_bits ^= (1 << slot);
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        if(kc >= 1){ mbar_wait(vfree, vfree_phase); vfree_phase ^= 1; }
        reorderCopyV2(pidx, slot);
        mbar_arrive(cur_fbarV);
        // Helpers take no further part this iteration.
      }
    } else {
      // ---- Compute group (tid < 128): same math as V28/V34; the reorder-copy
      // handoff with helpers now uses mbarriers (vfree + fbarV) instead of the
      // two reorder_sync() calls -- see header. Compute does its own copy
      // slice, fires two non-blocking arrives, and proceeds straight to QK^T
      // without waiting for helpers -- it only checks back in (fbarV_wait_bits
      // wait, below) right before PV actually needs sV. ----
      const int pidx = tid;   // maps tid [0,127] -> contiguous [0,127]
      int mbar_phase = 0;
      int load_phase_K_bits = 0;
      int load_phase_V_bits = 0;
      int vfree_phase = 0;
      int fbarV_wait_bits = 0;
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
        if(kc >= 1){ mbar_wait(vfree, vfree_phase); vfree_phase ^= 1; }

        reorderCopyV2(pidx, slot);
        mbar_arrive(cur_fbarV);

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
          if(tid == 0) mbar_arrive(cur_fbarK);
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

        // NEW: wait for ALL 480 copy participants (compute+helpers) here, right
        // before PV actually needs sV -- not immediately after the copy. By now
        // helpers have had the whole QK^T+softmax+rescale window to finish.
        mbar_wait(cur_fbarV, (fbarV_wait_bits >> slot) & 1); fbarV_wait_bits ^= (1 << slot);

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
          if(tid == 0) mbar_arrive(vfree);   // NEW: sV's read is done -- safe for the NEXT kc's copy to overwrite it
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

} // end of gqa_v37_causal

// V29_causal launcher — same grid as V28, but a genuinely-populated 512-thread
// block (128 compute + 32 producer + 352 reorder-helper), matching cuDNN's block
// size for real this time (unlike V28's 160-thread block padded to nothing).
template<int Br, int Bc, int D>
void launch_gqa_v37_causal(
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
  gqa_v37_causal<Br, Bc, D><<<GRID, BLOCK>>>(d_Q, d_O, d_LSE, Ktmap3d, Vtmap,
                          B, Hq, Hkv, G, S, scale);
}

// =================================
//  gqa_v38_causal - V34 (frozen at 1.9619ms) + TMA-based Q load, replacing the
//  512-thread manual copy loop (each thread doing Br*D/512=16 plain global
//  loads + swizzled shared-memory stores) with a single hardware TMA transfer,
//  directly from optimization_hints.txt's Experiment 6 ("kill the residual
//  non-TMA global traffic" -- 786K plain global loads / 405K global stores
//  cuDNN's own kernel doesn't have).
//
//  Verified BEFORE writing any code that this is safe, not just fast: K's
//  existing TMA map (make_tma_3d_katom) already achieves the exact canon_idx
//  swizzle purely through its 3D box decomposition -- no hardware swizzle mode
//  is used (CU_TENSOR_MAP_SWIZZLE_NONE), the swizzle comes from splitting the
//  column dimension into (chunk = k/8, atom = k%8) and making both separate
//  TMA box dimensions. Working through the address arithmetic by hand:
//  smem_offset = d2*(ATOM*box_rows) + d1*ATOM + d0 (TMA's box-dim flattening,
//  dim0 fastest) with d0=atom, d1=row, d2=chunk -- exactly equals
//  canon_idx(row=d1, k=d2*8+d0, rows=box_rows) = (k/8)*rows*8 + row*8 + (k%8).
//  Since Q needs the IDENTICAL canon_idx layout for its own MMA A-operand
//  descriptor (make_smem_desc(sQ, Br), same function/shape as K's
//  make_smem_desc(sK[slot], Bc)), the same 3D-atom tensor map technique
//  applies to Q with zero changes to the technique itself -- only the shape
//  parameters differ (rows = B*Hq*S instead of B*Hkv*S, box_rows = Br).
//
//  O's writeback is NOT touched here (deliberately out of scope for this
//  version): O's smem staging (sO) is a plain fp32 accumulator padded to
//  D_pad=D+1 for bank-conflict avoidance during tmem_readout_accum_vec's
//  writes -- NOT canon_idx-swizzled, and NOT contiguous per row (TMA requires
//  a contiguous, unpadded box source). Doing the same trick for O would need
//  a second, unpadded compact staging buffer (Br*D*2 bytes even in bf16 =
//  16384 bytes), and this kernel has only 436 bytes of smem headroom left
//  (232012 / 232448) -- nowhere near enough. Revisit after a redesign frees
//  smem (see optimization_hints.txt Experiment 2 discussion).
//
//  Q needs no double-buffering (unlike K/V): each pass loads exactly ONE Q
//  tile used for the pass's entire KV loop, so a single mbarrier (qbar,
//  count=1, re-init'd every pass) is enough -- the compute group waits on it
//  once, right before building descQ_base, instead of once per kc iteration.
//  This replaces a 512-thread cooperative copy (with its accompanying
//  __syncthreads() already present anyway for the mbar_init below) with a
//  single-thread TMA issue plus a single mbar_wait -- strictly less work on
//  every thread, not a tradeoff.
// =================================
template<int Br, int Bc, int D>
__global__ void gqa_v38_causal(
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
    if(tid < Br){ sm[tid] = -INFINITY; sl[tid] = 0.0f; }

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
          if(tid == 0) mbar_arrive(cur_fbarK);
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

} // end of gqa_v38_causal

// V29_causal launcher — same grid as V28, but a genuinely-populated 512-thread
// block (128 compute + 32 producer + 352 reorder-helper), matching cuDNN's block
// size for real this time (unlike V28's 160-thread block padded to nothing).
template<int Br, int Bc, int D>
void launch_gqa_v38_causal(
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
  gqa_v38_causal<Br, Bc, D><<<GRID, BLOCK>>>(d_O, d_LSE, Qtmap3d, Ktmap3d, Vtmap,
                          B, Hq, Hkv, G, S, scale);
}

// =================================
//  gqa_v39_causal - V34 (frozen at 1.9619ms) + register-resident softmax
//  running-state (m, l), directly from optimization_hints.txt's Experiment 7d
//  ("keep the softmax state in registers -- you're paying smem traffic to
//  compensate for registers you aren't using"), narrowed to exactly the part
//  verified safe by a cross-thread-read audit of every sm[]/sl[]/sCorr[]
//  access in the kernel (done BEFORE writing any code, since sCorr[r] is
//  read cross-thread by the rescale loop every kc iteration -- NOT safe to
//  register-ify -- and sl[r] is read cross-thread once, in the final
//  O-normalize loop, AFTER the kc loop -- safe to defer, not safe to drop):
//   - sm[tid]: NEVER read cross-thread anywhere in the kernel (every access,
//     including the final LSE write, is same-tid) -- eliminated entirely,
//     replaced by a plain per-thread register `m_reg`. Shared array removed
//     (saves Br*4=512 bytes of smem).
//   - sl[tid]: read/written same-tid every kc iteration, but READ
//     cross-thread (denom=sl[r]) only in the final O-normalize loop, which
//     runs AFTER the kc loop ends. Kept fully register-resident (`l_reg`)
//     through the kc loop itself, with a SINGLE commit to the shared sl[]
//     array right after the loop ends -- eliminates nKVTiles-1 redundant
//     shared load+store round-trips per pass (up to 31 for the largest
//     causal tile), down to exactly one store. The array itself must still
//     exist (cross-thread + across-branch consumption), so no smem savings
//     here, only traffic reduction.
//   - sCorr[tid]: NOT touched -- read cross-thread (sCorr[r]) by the rescale
//     loop within the SAME kc iteration it's written, so it cannot be
//     deferred or register-ified without a larger restructure.
// =================================
template<int Br, int Bc, int D>
__global__ void gqa_v39_causal(
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
  const uint32_t sK_addr0      = (uint32_t)__cvta_generic_to_shared(sK[0]);
  const uint32_t sK_addr1      = (uint32_t)__cvta_generic_to_shared(sK[1]);
  const uint32_t sVstage_addr0 = (uint32_t)__cvta_generic_to_shared(sVstage[0]);
  const uint32_t sVstage_addr1 = (uint32_t)__cvta_generic_to_shared(sVstage[1]);

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
    float m_reg = -INFINITY;   // NEW: the old running-max array is eliminated -- never read
                               // cross-thread anywhere in this kernel (verified: every
                               // access to it was same-tid), so it lives in a register.

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
      float l_reg = 0.0f;   // NEW: sl[tid] mid-loop round-trips eliminated -- l is only ever
                            // read/written same-tid WITHIN the kc loop (verified); the single
                            // cross-thread read (final O-normalize, denom=sl[r]) and the
                            // same-tid LSE read both happen AFTER the loop, so one commit to
                            // sl[tid] at the loop's end (below) is sufficient and correct.
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
          if(tid == 0) mbar_arrive(cur_fbarK);
          tmem_readout_to_smem_vec_2cta(sS, tmem_addr, Br, Bc, Bc_pad, scale_l2e, 0u);
          asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");

          if(kc == q_tile){
            for(int j = tid + 1; j < Bc; ++j) sS[tid * Bc_pad + j] = -INFINITY;
          }
          consumer_sync();
        }

        {
          const float m_old = m_reg;
          const float l_old = l_reg;

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
          m_reg = m_new; l_reg = l_old * corr + p_sum; sCorr[tid] = corr;
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
      sl[tid] = l_reg;   // NEW: single commit -- see l_reg declaration above
    }

    __syncthreads();

    for(int i = 2 * tid; i < Br * D; i += 2 * blockDim.x){
      const int r = i / D, c = i % D;
      const float denom = sl[r];
      *reinterpret_cast<__nv_bfloat162*>(&d_O[qBase + i]) =
          __floats2bfloat162_rn(sO[r * D_pad + c] / denom, sO[r * D_pad + c + 1] / denom);
    }
    if(tid < Br)
      d_LSE[lBase + tid] = 0.6931471805599453f * (m_reg + log2f(sl[tid]));

    __syncthreads();
  } // end pass loop (q_tile_lo, then q_tile_hi)

  if(tid < 32)
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(tmem_addr), "r"(NCOLS) : "memory");

} // end of gqa_v39_causal

// V29_causal launcher — same grid as V28, but a genuinely-populated 512-thread
// block (128 compute + 32 producer + 352 reorder-helper), matching cuDNN's block
// size for real this time (unlike V28's 160-thread block padded to nothing).
template<int Br, int Bc, int D>
void launch_gqa_v39_causal(
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
  gqa_v39_causal<Br, Bc, D><<<GRID, BLOCK>>>(d_Q, d_O, d_LSE, Ktmap3d, Vtmap,
                          B, Hq, Hkv, G, S, scale);
}

// =================================
//  gqa_v40_causal - V38 (frozen at 1.9251ms) + TMEM-resident S/P/(m,corr),
//  directly from reviewing fa4_style_gqa_sm103.cu (a from-SASS reconstruction of
//  cuDNN's actual kernel): read S straight from TMEM into registers (no sS smem
//  buffer, no second re-read for the exp pass -- one register copy serves both
//  the max and exp passes), pack P to bf16 pairs and store it in TMEM instead of
//  smem (no sP buffer), and keep the running row-max (m) and rescale factor
//  (corr) TMEM-resident too, mirroring fa4_style's own (m,l,alpha) design.
//
//  NOT ported from fa4_style, on purpose (kept for a later, separate step):
//   - The ping-pong warpgroup / full warp-specialization architecture. This
//     version keeps V38's sequential single-softmax-stream structure and its
//     causal-pair load balancing untouched -- only WHERE S/P/m/corr live
//     changes, not the surrounding control flow.
//   - `sl` (the running softmax denominator) stays in SHARED MEMORY, not TMEM.
//     Audited every sl[] access first: same-tid read/write every kc iteration
//     (safe either way), but the FINAL O-normalize loop reads sl[r] across ALL
//     512 threads (compute+producer+helper), not just the 128 compute threads
//     that own TMEM lanes 0-127 -- tcgen05.ld/st are warp-collective ops tied
//     to the ISSUING warp's own 32 physical lanes, so an arbitrary thread from
//     warp 5+ cannot read row r's TMEM lane without restructuring that whole
//     512-thread loop's thread-to-row mapping. `sm` and `sCorr` don't have this
//     problem -- every access to them (verified) is confined to the 128
//     compute threads, which already own TMEM lanes 0-127 1:1 (row == tid,
//     since rows_per_warp = Br/4 = 32 = warp size) -- so moving THOSE two was
//     safe without touching any loop's thread participation.
//   - The rescale loop (`sO[r*D_pad+c] *= corr`) WAS restructured, from V38's
//     striped "thread t covers flat range [t, t+Br, t+2*Br, ...]" (touching
//     many different rows per thread) to "thread t owns row t, loops over all
//     D columns" -- required to make corr's TMEM read match the same 1
//     thread : 1 row convention as sm/S. Same total iteration count per thread
//     (D=64 either way), just confined to one contiguous row instead of a
//     flat-index stripe across all 128 rows.
//   - `setmaxnreg` register re-partitioning. fa4_style needs this to afford its
//     softmax warpgroup's s_row[128]+p_packed[64] register arrays (192 regs,
//     borrowed from other warpgroups). V38's compute group has no "other
//     warpgroups" to borrow from in this sequential design, so this version's
//     s_row[128]/p_packed[64] competes for registers with everything else V38
//     already needs (64 registers, no s_row/p_packed) -- flagged here as a
//     REAL risk (possible heavy spilling) to check explicitly against ptxas's
//     actual reported numbers once compiled, not assumed away.
//   - Reused the SAME 3 consumer_sync() calls at the SAME points as V38,
//     unchanged, rather than trying to also prove any of them removable in
//     this same step -- keeping this an isolated test of "does TMEM-residency
//     for S/P/m/corr help", not conflated with a synchronization-reduction
//     experiment too (V33/V35/V39's shared lesson: isolate one variable).
//
//  TMEM layout (NCOLS=256, up from V38's 128): TMEM_S_COL=0 (128 cols, S then
//  P bf16-packed reuse of the same columns) | TMEM_O_COL=128 (D=64 cols, MUST
//  be separate from TMEM_S_COL since the PV MMA reads P from TMEM_S_COL as
//  operand A while writing O to TMEM_O_COL in the same instruction -- input
//  and output cannot alias) | TMEM_M_COL=192 | TMEM_CORR_COL=193.
// =================================
template<int Br, int Bc, int D>
__global__ void __launch_bounds__(512, 1) gqa_v40_causal(
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
          #pragma unroll
          for(int c = 0; c < Bc; ++c) s_row[c] *= scale_l2e;

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
          const float corr  = ex2_approx(m_old - m_new);

          float p_sum = 0.0f;
          uint32_t p_packed[Bc / 2];
          #pragma unroll
          for(int j2 = 0; j2 < Bc; j2 += 2){
            const float p0 = ex2_approx(s_row[j2]     - m_new);
            const float p1 = ex2_approx(s_row[j2 + 1] - m_new);
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
      d_LSE[lBase + tid] = 0.6931471805599453f * (m_final + log2f(sl[tid]));
    }

    __syncthreads();
  } // end pass loop (q_tile_lo, then q_tile_hi)

  if(tid < 32)
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(tmem_addr), "r"(NCOLS) : "memory");

} // end of gqa_v40_causal

// V29_causal launcher — same grid as V28, but a genuinely-populated 512-thread
// block (128 compute + 32 producer + 352 reorder-helper), matching cuDNN's block
// size for real this time (unlike V28's 160-thread block padded to nothing).
template<int Br, int Bc, int D>
void launch_gqa_v40_causal(
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
  gqa_v40_causal<Br, Bc, D><<<GRID, BLOCK>>>(d_O, d_LSE, Qtmap3d, Ktmap3d, Vtmap,
                          B, Hq, Hkv, G, S, scale);
}

int main(){
  std::cout << "Benchmarking CAUSAL Grouped-Query Attention — Blackwell SM_103 (B300), v2 file\n";

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
    runCorrectness("V34-causal", [&](){ launch_gqa_v34_causal<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); });
    runCorrectness("V37-causal", [&](){ launch_gqa_v37_causal<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); });
    runCorrectness("V38-causal", [&](){ launch_gqa_v38_causal<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); });
    runCorrectness("V39-causal", [&](){ launch_gqa_v39_causal<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); });
    runCorrectness("V40-causal", [&](){ launch_gqa_v40_causal<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); });
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
    [&](){ launch_gqa_v34_causal<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); },
    100, 25, flops, bytes
  );
  displayStats("V34-causal — baseline copied in from GQA_sm103_causal.cu for this file's comparisons", stats);

  stats = benchmarkKernel(
    [&](){ launch_gqa_v37_causal<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); },
    100, 25, flops, bytes
  );
  displayStats("V37-causal — V34 + mbarrier-based reorder-copy handoff (replaces both reorder_sync() calls)", stats);

  stats = benchmarkKernel(
    [&](){ launch_gqa_v38_causal<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); },
    100, 25, flops, bytes
  );
  displayStats("V38-causal — V34 + TMA-based Q load (Experiment 6)", stats);

  stats = benchmarkKernel(
    [&](){ launch_gqa_v39_causal<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); },
    100, 25, flops, bytes
  );
  displayStats("V39-causal — V34 + register-resident softmax state m/l (Experiment 7d)", stats);

  stats = benchmarkKernel(
    [&](){ launch_gqa_v40_causal<Br, Bc, D>(d_Q, d_K, d_V, d_O, d_LSE, B, Hq, Hkv, S, G, scale); },
    100, 25, flops, bytes
  );
  displayStats("V40-causal — V38 + TMEM-resident S/P/(m,corr) (from fa4_style_gqa_sm103.cu)", stats);

  // Add new versions (V38 onward) here: runCorrectness + benchmarkKernel/displayStats,
  // matching the pattern above.

  CUDA_CHECK(cudaFree(d_Q));
  CUDA_CHECK(cudaFree(d_K));
  CUDA_CHECK(cudaFree(d_V));
  CUDA_CHECK(cudaFree(d_O));
  CUDA_CHECK(cudaFree(d_LSE));

  return 0;
}
