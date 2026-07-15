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

  // Add new versions (V37 onward) here: runCorrectness + benchmarkKernel/displayStats,
  // matching the pattern above.

  CUDA_CHECK(cudaFree(d_Q));
  CUDA_CHECK(cudaFree(d_K));
  CUDA_CHECK(cudaFree(d_V));
  CUDA_CHECK(cudaFree(d_O));
  CUDA_CHECK(cudaFree(d_LSE));

  return 0;
}
