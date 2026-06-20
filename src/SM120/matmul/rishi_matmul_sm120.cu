/*
 * tmem_matmul.cu — Tensor-core matmul for SM_120 (RTX 5080, consumer Blackwell)
 *
 * Note: consumer Blackwell (SM_120) does NOT expose TMEM or wgmma — those are
 * datacenter-only (SM_100, B100/B200). This kernel uses the SM_120 path:
 *   mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
 *   ldmatrix.sync.aligned.m8n8.x4 / .x2.trans
 *
 * Computes C[M,N] = A[M,K] * B[K,N] in fp32 from fp16 inputs (row-major).
 *
 * Layout:
 *   1 warp per block (32 threads, __launch_bounds__(32))
 *   Block tile  BM=16, BN=64, BK=16 → 8 m16n8 fragments per K-step.
 *   Grid (N/BN, M/BM).
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>

constexpr int BM = 16;
constexpr int BN = 64;
constexpr int BK = 16;
constexpr int N_FRAGS = BN / 8;   // 8 m16n8 fragments along N

__device__ __forceinline__ uint32_t cvta_smem(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__
void ldmatrix_x4(uint32_t (&r)[4], uint32_t smem_addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(smem_addr));
}

__device__ __forceinline__
void ldmatrix_x2_trans(uint32_t (&r)[2], uint32_t smem_addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1])
        : "r"(smem_addr));
}

__device__ __forceinline__
void mma_m16n8k16(float (&d)[4],
                  const uint32_t (&a)[4], const uint32_t (&b)[2],
                  const float (&c)[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

__global__ __launch_bounds__(32)
void tmem_matmul_kernel(const __half* __restrict__ A,
                        const __half* __restrict__ B,
                        float* __restrict__ C,
                        int M, int K, int N) {
    const int tile_row = blockIdx.y * BM;
    const int tile_col = blockIdx.x * BN;
    const int lane     = threadIdx.x;

    __shared__ __align__(16) __half smA[BM * BK];     // 16 × 16
    __shared__ __align__(16) __half smB[BK * BN];     // 16 × 64

    float acc[N_FRAGS][4];
    #pragma unroll
    for (int n = 0; n < N_FRAGS; ++n)
        acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;

    const int num_k_tiles = (K + BK - 1) / BK;

    for (int kt = 0; kt < num_k_tiles; ++kt) {
        const int k0 = kt * BK;

        //* ── Cooperative load: A tile (256 halves, 8 per lane) ──────────
        #pragma unroll
        for (int i = 0; i < BM * BK / 32; ++i) {
            int idx = i * 32 + lane;
            int r = idx / BK, c = idx % BK;
            int gr = tile_row + r, gc = k0 + c;
            smA[idx] = (gr < M && gc < K) ? A[gr * K + gc] : __float2half(0.f);
        }
        //* B tile (1024 halves, 32 per lane)
        #pragma unroll
        for (int i = 0; i < BK * BN / 32; ++i) {
            int idx = i * 32 + lane;
            int r = idx / BN, c = idx % BN;
            int gr = k0 + r, gc = tile_col + c;
            smB[idx] = (gr < K && gc < N) ? B[gr * N + gc] : __float2half(0.f);
        }
        __syncwarp();

        //* ── Load A frag (m16k16) via ldmatrix.x4 ────────────────────────
        //* The 4 sub-matrices map to mma A-frag positions:
        //*   sub 0 → a[0]: rows 0–7,  cols 0–7
        //*   sub 1 → a[1]: rows 8–15, cols 0–7
        //*   sub 2 → a[2]: rows 0–7,  cols 8–15
        //*   sub 3 → a[3]: rows 8–15, cols 8–15
        int sub  = lane >> 3;          // 0..3
        int srow = lane & 7;           // 0..7
        int a_r_off = (sub & 1) * 8;   // 0 or 8
        int a_c_off = (sub >> 1) * 8;  // 0 or 8
        uint32_t a_addr = cvta_smem(&smA[(a_r_off + srow) * BK + a_c_off]);
        uint32_t a_frag[4];
        ldmatrix_x4(a_frag, a_addr);

        //* ── Per n-fragment: load B frag + mma ──────────────────────────
        #pragma unroll
        for (int nf = 0; nf < N_FRAGS; ++nf) {
            //* B m16n8: 2 sub-matrices via ldmatrix.x2.trans
            //*   sub 0: K-rows 0–7,  N-cols nf*8..nf*8+7
            //*   sub 1: K-rows 8–15, N-cols nf*8..nf*8+7
            int b_sub  = (lane >> 3) & 1;     // wrap to 0/1 for lanes 16-31
            int b_srow = lane & 7;
            int b_r_off = b_sub * 8;
            int b_c_off = nf * 8;
            uint32_t b_addr =
                cvta_smem(&smB[(b_r_off + b_srow) * BN + b_c_off]);
            uint32_t b_frag[2];
            ldmatrix_x2_trans(b_frag, b_addr);

            float d[4];
            mma_m16n8k16(d, a_frag, b_frag, acc[nf]);
            acc[nf][0] = d[0]; acc[nf][1] = d[1];
            acc[nf][2] = d[2]; acc[nf][3] = d[3];
        }
        __syncwarp();
    }

    //* ── Store C  (m16n8 output layout) ─────────────────────────────────
    //*   group = lane / 4,  tg = lane % 4
    //*   c[0] @ (group,   tg*2)
    //*   c[1] @ (group,   tg*2+1)
    //*   c[2] @ (group+8, tg*2)
    //*   c[3] @ (group+8, tg*2+1)
    int group = lane >> 2;
    int tg    = lane & 3;
    #pragma unroll
    for (int nf = 0; nf < N_FRAGS; ++nf) {
        int gm0 = tile_row + group;
        int gm1 = tile_row + group + 8;
        int gn0 = tile_col + nf * 8 + tg * 2;
        int gn1 = gn0 + 1;
        if (gm0 < M) {
            if (gn0 < N) C[gm0 * N + gn0] = acc[nf][0];
            if (gn1 < N) C[gm0 * N + gn1] = acc[nf][1];
        }
        if (gm1 < M) {
            if (gn0 < N) C[gm1 * N + gn0] = acc[nf][2];
            if (gn1 < N) C[gm1 * N + gn1] = acc[nf][3];
        }
    }
}

void launch_tmem_matmul(const __half* dA, const __half* dB, float* dC,
                        int M, int K, int N) {
    dim3 block(32);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    tmem_matmul_kernel<<<grid, block>>>(dA, dB, dC, M, K, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
}

//* ─── Simple host-side test ────────────────────────────────────────────────
int main() {
    const int M = 1024, K = 1024, N = 1024;

    size_t sA = (size_t)M * K * sizeof(__half);
    size_t sB = (size_t)K * N * sizeof(__half);
    size_t sC = (size_t)M * N * sizeof(float);

    __half* hA = (__half*)malloc(sA);
    __half* hB = (__half*)malloc(sB);
    float*  hC = (float* )malloc(sC);

    for (int i = 0; i < M * K; i++) hA[i] = __float2half(1.f / K);
    for (int i = 0; i < K * N; i++) hB[i] = __float2half(1.f);

    __half *dA, *dB; float *dC;
    cudaMalloc(&dA, sA); cudaMalloc(&dB, sB); cudaMalloc(&dC, sC);
    cudaMemcpy(dA, hA, sA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sB, cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, sC);

    launch_tmem_matmul(dA, dB, dC, M, K, N);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++)
        launch_tmem_matmul(dA, dB, dC, M, K, N);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    double tflops = 2.0 * M * K * N * 10 / (ms * 1e-3) / 1e12;
    printf("mma_matmul [%d×%d×%d]: %.3f ms/iter, %.2f TFLOPS\n",
           M, K, N, ms / 10.f, tflops);

    cudaMemcpy(hC, dC, sC, cudaMemcpyDeviceToHost);
    float maxerr = 0;
    for (int i = 0; i < M * N; i++)
        maxerr = fmaxf(maxerr, fabsf(hC[i] - 1.f));
    printf("Max error vs expected=1.0: %.4e %s\n",
           maxerr, maxerr < 1e-2f ? "[PASS]" : "[FAIL]");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}