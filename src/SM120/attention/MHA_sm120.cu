#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <random> 
#include "utils/kernelUtils.cuh"
#include "utils/kernelBench.cuh"

using namespace nvcuda;

static constexpr float kLog2e = 1.4426950216293334961f;

// ============================================================
// Device helpers
// ============================================================

__device__ __forceinline__ float exp2_safe(float x) {
    if (x >= -126.0f) return exp2f(x);
    float h = exp2f(x * 0.5f);
    return h * h;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, off);
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int off = 16; off > 0; off >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, off));
    return val;
}

// Split a float into hi (TF32 truncated) and lo (rounded TF32 remainder)
__device__ __forceinline__ void split_tf32(float x, float& hi, float& lo) {
    uint32_t xb   = __float_as_uint(x);
    uint32_t hi_b = xb & 0xffffe000u;
    hi            = __uint_as_float(hi_b);
    float    lr   = x - hi;
    uint32_t lb   = __float_as_uint(lr);
    if ((lb & 0x7f800000u) != 0x7f800000u) lb += 0x1000u;
    lo            = __uint_as_float(lb);
}

// ============================================================
// MHA V3 — TF32 tensor cores + 3-way split-TF32 (full FP32 emulation)
//
// Per K-block (8 K-dim wide), 3 mma calls recover ~22 mantissa bits:
//     acc += A_hi * B_hi    (TF32 × TF32)
//     acc += A_hi * B_lo    (cross term 1)
//     acc += A_lo * B_hi    (cross term 2)
// The A_lo*B_lo term is of order 2^-20 and dropped — same trick CUTLASS
// uses for emulated-FP32 GEMM, gives essentially FP32 accuracy.
//
// Tiles : Br=16, Bc=32, D=64
// Grid  : (S/Br, B*H)
// Block : 32 threads (one warp)
// ============================================================
template<int Br, int Bc, int D>
__global__ void mha_v3_tc(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float*       __restrict__ O,
    float softmax_scale,
    int   S
) {
    static_assert(Br == 16,        "Br must be 16");
    static_assert(D  % 16 == 0,    "D must be a multiple of 16");
    static_assert(Bc % 16 == 0,    "Bc must be a multiple of 16");
    static_assert(D  % 8  == 0,    "D must be a multiple of 8");
    static_assert(Bc % 8  == 0,    "Bc must be a multiple of 8");
    static_assert(Bc % 32 == 0,    "Bc must be a multiple of warp size");
    static_assert(D  % 32 == 0,    "D must be a multiple of warp size");

    constexpr int kColsPerLane = Bc / 32;
    constexpr int kDPerLane    = D  / 32;

    const int lane  = threadIdx.x;
    const int bh    = blockIdx.y;
    const int qtile = blockIdx.x;

    // Dynamic shmem so Bc can exceed the 48KB static cap.
    // Layout (offsets in floats):
    //   [0,         Br*D)      Qs_hi
    //   [Br*D,    2*Br*D)      Qs_lo
    //   [2*Br*D,  2*Br*D+Bc*D) Ks_hi
    //   ... etc
    extern __shared__ float dsmem[];
    float (*Qs_hi)[D]  = (float (*)[D])(dsmem + 0);
    float (*Qs_lo)[D]  = (float (*)[D])(dsmem + Br*D);
    float (*Ks_hi)[D]  = (float (*)[D])(dsmem + 2*Br*D);
    float (*Ks_lo)[D]  = (float (*)[D])(dsmem + 2*Br*D + Bc*D);
    float (*Vs_hi)[D]  = (float (*)[D])(dsmem + 2*Br*D + 2*Bc*D);
    float (*Vs_lo)[D]  = (float (*)[D])(dsmem + 2*Br*D + 3*Bc*D);
    float (*Ss_hi)[Bc] = (float (*)[Bc])(dsmem + 2*Br*D + 4*Bc*D);
    float (*Ss_lo)[Bc] = (float (*)[Bc])(dsmem + 2*Br*D + 4*Bc*D + Br*Bc);
    float (*Os)[D]     = (float (*)[D])(dsmem + 2*Br*D + 4*Bc*D + 2*Br*Bc);

    const float* Qp = Q + bh * S * D + qtile * Br * D;
    const float* Kp = K + bh * S * D;
    const float* Vp = V + bh * S * D;
    float*       Op = O + bh * S * D + qtile * Br * D;

    // Per-lane register state: lane r (0..Br-1) owns row r's running m and l.
    // Other lanes (Br..31) don't carry state; they fetch via __shfl_sync.
    // Saves the 128-byte m_state/l_state arrays that pushed shmem over 48KB.
    float m_local = -INFINITY;
    float l_local = 0.0f;
    #pragma unroll
    for (int i = lane; i < Br * D; i += 32) {
        reinterpret_cast<float*>(Os)[i] = 0.0f;
    }

    // ---- load Q with split-TF32 ----
    for (int i = lane; i < Br * D; i += 32) {
        float q_hi, q_lo;
        split_tf32(Qp[i], q_hi, q_lo);
        reinterpret_cast<float*>(Qs_hi)[i] = q_hi;
        reinterpret_cast<float*>(Qs_lo)[i] = q_lo;
    }
    __syncwarp();

    const int num_ktiles = S / Bc;

    for (int kt = 0; kt < num_ktiles; kt++) {
        const float* Kbase = Kp + kt * Bc * D;
        const float* Vbase = Vp + kt * Bc * D;

        // ---- load K, V with split-TF32 ----
        for (int i = lane; i < Bc * D; i += 32) {
            float k_hi, k_lo, v_hi, v_lo;
            split_tf32(Kbase[i], k_hi, k_lo);
            split_tf32(Vbase[i], v_hi, v_lo);
            reinterpret_cast<float*>(Ks_hi)[i] = k_hi;
            reinterpret_cast<float*>(Ks_lo)[i] = k_lo;
            reinterpret_cast<float*>(Vs_hi)[i] = v_hi;
            reinterpret_cast<float*>(Vs_lo)[i] = v_lo;
        }
        __syncwarp();

        // ============================================================
        // S = Q @ K^T   (3-way split-TF32: Q_hi*K_hi + Q_hi*K_lo + Q_lo*K_hi)
        // ============================================================
        wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a_hi, a_lo;
        wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> b_hi, b_lo;
        wmma::fragment<wmma::accumulator, 16, 16, 8, float> s_frag[Bc / 16];

        #pragma unroll
        for (int n = 0; n < Bc / 16; n++)
            wmma::fill_fragment(s_frag[n], 0.0f);

        #pragma unroll
        for (int kk = 0; kk < D; kk += 8) {
            wmma::load_matrix_sync(a_hi, &Qs_hi[0][kk], D);
            wmma::load_matrix_sync(a_lo, &Qs_lo[0][kk], D);
            #pragma unroll
            for (int n = 0; n < Bc / 16; n++) {
                wmma::load_matrix_sync(b_hi, &Ks_hi[n * 16][kk], D);
                wmma::load_matrix_sync(b_lo, &Ks_lo[n * 16][kk], D);
                // 3 mmas in CUTLASS k3xTF32 order: small terms FIRST, big term LAST.
                // Critical for precision: adding small contributions before the
                // dominant a_hi*b_hi term keeps their bits in the accumulator's
                // mantissa; reversing this order rounds them off (Kahan-style).
                wmma::mma_sync(s_frag[n], a_lo, b_hi, s_frag[n]);
                wmma::mma_sync(s_frag[n], a_hi, b_lo, s_frag[n]);
                wmma::mma_sync(s_frag[n], a_hi, b_hi, s_frag[n]);
            }
        }

        #pragma unroll
        for (int n = 0; n < Bc / 16; n++) {
            wmma::store_matrix_sync(&Ss_hi[0][n * 16], s_frag[n], Bc, wmma::mem_row_major);
        }
        __syncwarp();

        // ============================================================
        // Online softmax: row max → exp2 → row sum → rescale state.
        // Store P split as P_hi / P_lo for the next 3-mma GEMM.
        // ============================================================
        // CUTLASS fuses (softmax_scale * log2e) into a single multiply before
        // softmax — running state (m_local, l_local) lives in log2 space, so
        // exp2 calls don't need a second log2e multiply.  Saves ~1 ULP per
        // exponent vs doing the two multiplies separately.
        const float scale_log2e = softmax_scale * kLog2e;

        for (int r = 0; r < Br; r++) {
            float vv[kColsPerLane];
            #pragma unroll
            for (int c = 0; c < kColsPerLane; c++)
                vv[c] = Ss_hi[r][lane * kColsPerLane + c] * scale_log2e;   // fused scale * log2e

            float local_max = -INFINITY;
            #pragma unroll
            for (int c = 0; c < kColsPerLane; c++) local_max = fmaxf(local_max, vv[c]);
            local_max = warp_reduce_max(local_max);

            // State is in log2 space — no separate kLog2e multiply needed
            float m_old     = __shfl_sync(0xffffffff, m_local, r);
            float l_old     = __shfl_sync(0xffffffff, l_local, r);
            float m_new     = fmaxf(m_old, local_max);
            float scale_old = exp2_safe(m_old - m_new);

            float pp[kColsPerLane];
            #pragma unroll
            for (int c = 0; c < kColsPerLane; c++)
                pp[c] = exp2_safe(vv[c] - m_new);

            float local_sum = 0.0f;
            #pragma unroll
            for (int c = 0; c < kColsPerLane; c++) local_sum += pp[c];
            local_sum = warp_reduce_sum(local_sum);

            // Update row r's state on its owning lane
            if (lane == r) {
                m_local = m_new;
                l_local = l_old * scale_old + local_sum;
            }

            // Split P_hi / P_lo for next 3-mma
            #pragma unroll
            for (int c = 0; c < kColsPerLane; c++) {
                float p_hi, p_lo;
                split_tf32(pp[c], p_hi, p_lo);
                Ss_hi[r][lane * kColsPerLane + c] = p_hi;
                Ss_lo[r][lane * kColsPerLane + c] = p_lo;
            }

            // Rescale running output
            #pragma unroll
            for (int c = 0; c < kDPerLane; c++)
                Os[r][lane * kDPerLane + c] *= scale_old;
        }
        __syncwarp();

        // ============================================================
        // O += P @ V   (3-way split-TF32: P_hi*V_hi + P_hi*V_lo + P_lo*V_hi)
        // ============================================================
        wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> p_hi, p_lo;
        wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> v_hi, v_lo;
        wmma::fragment<wmma::accumulator, 16, 16, 8, float> o_frag[D / 16];

        #pragma unroll
        for (int n = 0; n < D / 16; n++) {
            wmma::load_matrix_sync(o_frag[n], &Os[0][n * 16], D, wmma::mem_row_major);
        }

        #pragma unroll
        for (int kk = 0; kk < Bc; kk += 8) {
            wmma::load_matrix_sync(p_hi, &Ss_hi[0][kk], Bc);
            wmma::load_matrix_sync(p_lo, &Ss_lo[0][kk], Bc);
            #pragma unroll
            for (int n = 0; n < D / 16; n++) {
                wmma::load_matrix_sync(v_hi, &Vs_hi[kk][n * 16], D);
                wmma::load_matrix_sync(v_lo, &Vs_lo[kk][n * 16], D);
                // 3 mmas in CUTLASS k3xTF32 order: small first, big last
                wmma::mma_sync(o_frag[n], p_lo, v_hi, o_frag[n]);
                wmma::mma_sync(o_frag[n], p_hi, v_lo, o_frag[n]);
                wmma::mma_sync(o_frag[n], p_hi, v_hi, o_frag[n]);
            }
        }

        #pragma unroll
        for (int n = 0; n < D / 16; n++) {
            wmma::store_matrix_sync(&Os[0][n * 16], o_frag[n], D, wmma::mem_row_major);
        }
        __syncwarp();
    }

    // ---- final normalize (double-precision reciprocal, matches DFMA in SASS) ----
    // Each lane r in [0..Br) computes its own row's reciprocal; broadcast to all lanes.
    float inv_l_local = (lane < Br) ? (float)(1.0 / (double)l_local) : 0.0f;
    for (int r = 0; r < Br; r++) {
        float inv_l = __shfl_sync(0xffffffff, inv_l_local, r);
        #pragma unroll
        for (int c = 0; c < kDPerLane; c++) {
            Op[r * D + lane * kDPerLane + c] = Os[r][lane * kDPerLane + c] * inv_l;
        }
    }
}

// ============================================================
// Main
// ============================================================
int main() {
    constexpr int B  = 16;
    constexpr int H  = 12;
    constexpr int S  = 1024;
    constexpr int D  = 64;
    constexpr int Br = 16;
    constexpr int Bc = 32;   // tested Bc=64: identical precision, 60% slower (tile size doesn't drive the gap)

    const float scale = 1.0f / sqrtf((float)D);

    const size_t N      = (size_t)B * H * S * D;
    const size_t NBytes = N * sizeof(float);

    std::vector<float> hQ(N), hK(N), hV(N);
    std::vector<float> hO_sdpa(N, 0.0f), hO_fp64(N, 0.0f), hO_gpu(N, 0.0f);

    // Resolve data dir relative to this source file's location at compile time
    // so the binary runs from any CWD.
    #define STRINGIFY2(x) #x
    #define STRINGIFY(x)  STRINGIFY2(x)
    #ifndef MHA_DATA_DIR
        #define MHA_DATA_DIR /home/blubridge028/Sanjay/CUDA/BlackWell/data
    #endif
    const char* data_dir = STRINGIFY(MHA_DATA_DIR);
    char path[512];
    snprintf(path, sizeof(path), "%s/mha_q.bin",        data_dir); loadBin(path, hQ.data(),      N);
    snprintf(path, sizeof(path), "%s/mha_k.bin",        data_dir); loadBin(path, hK.data(),      N);
    snprintf(path, sizeof(path), "%s/mha_v.bin",        data_dir); loadBin(path, hV.data(),      N);
    snprintf(path, sizeof(path), "%s/mha_out.bin",      data_dir); loadBin(path, hO_sdpa.data(), N);
    snprintf(path, sizeof(path), "%s/mha_out_fp64.bin", data_dir); loadBin(path, hO_fp64.data(), N);

    printf("B=%d  H=%d  S=%d  D=%d  Br=%d  Bc=%d  scale=%.6f\n", B, H, S, D, Br, Bc, scale);
    printf("Loaded Q/K/V + references from %s/\n", data_dir);

    float *dQ, *dK, *dV, *dO;
    CUDA_CHECK(cudaMalloc(&dQ, NBytes));
    CUDA_CHECK(cudaMalloc(&dK, NBytes));
    CUDA_CHECK(cudaMalloc(&dV, NBytes));
    CUDA_CHECK(cudaMalloc(&dO, NBytes));

    CUDA_CHECK(cudaMemcpy(dQ, hQ.data(), NBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hK.data(), NBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV.data(), NBytes, cudaMemcpyHostToDevice));

    dim3 grid(S / Br, B * H);
    dim3 block(32);

    // Dynamic shmem layout (floats):
    //   2*Br*D (Qs_hi/lo) + 4*Bc*D (Ks_hi/lo, Vs_hi/lo) + 2*Br*Bc (Ss_hi/lo) + Br*D (Os)
    size_t smem_floats = 2*Br*D + 4*Bc*D + 2*Br*Bc + Br*D;
    size_t smem_bytes  = smem_floats * sizeof(float);
    CUDA_CHECK(cudaFuncSetAttribute(mha_v3_tc<Br, Bc, D>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes));

    printf("Launching mha_v3_tc  grid=(%d,%d)  block=(32)  smem=%zuKB\n",
           S / Br, B * H, smem_bytes / 1024);
    mha_v3_tc<Br, Bc, D><<<grid, block, smem_bytes>>>(dQ, dK, dV, dO, scale, S);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hO_gpu.data(), dO, NBytes, cudaMemcpyDeviceToHost));

    printf("\n=== Precision triangulation ===\n");
    reportPrecision("torch.SDPA   vs  fp64 ", hO_fp64.data(), hO_sdpa.data(), N);
    reportPrecision("mha_v3_tc    vs  fp64 ", hO_fp64.data(), hO_gpu.data(),  N);
    reportPrecision("mha_v3_tc    vs  SDPA ", hO_sdpa.data(), hO_gpu.data(),  N);

    // ---- benchmark ----
    // Attention FLOPs: 2 * (QK^T) + 2 * (PV) = 4 * B*H*S*S*D
    // (3-mma TF32 emulation does 3x this work, but FLOP count is conventionally
    //  reported per the math operations, not the mma calls)
    long long flops = 4LL * B * H * (long long)S * S * D;
    size_t bytes   = 4 * NBytes;   // Q,K,V read + O write
    auto stats = benchmarkKernel(
        [&](){ mha_v3_tc<Br, Bc, D><<<grid, block, smem_bytes>>>(dQ, dK, dV, dO, scale, S); },
        /*iters=*/100, /*warmup=*/25, flops, bytes);
    displayStats("mha_v3_tc", stats);

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    return 0;
}
