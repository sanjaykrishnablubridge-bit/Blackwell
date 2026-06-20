#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "utils/kernelUtils.cuh"
#include "utils/kernelBench.cuh"

template<int BLOCK_SIZE, int VEC_PER_THREAD>
__global__ void __launch_bounds__(BLOCK_SIZE)
RMSNorm_vec(const float* __restrict__ X, float* __restrict__ Y,
            const float* __restrict__ gamma, float* mean, float* rstd, size_t n) {

    int tid = threadIdx.x;
    int row = blockIdx.x;

    const float4* x4 = reinterpret_cast<const float4*>(X + row * n);
    float4*       y4 = reinterpret_cast<float4*>(Y + row * n);
    const float4* g4 = reinterpret_cast<const float4*>(gamma);

    int vec = n / 4;

    float4 buf[VEC_PER_THREAD];
    float  sum = 0.f;

    #pragma unroll
    for (int s = 0; s < VEC_PER_THREAD; ++s) {
        int i = tid + s * BLOCK_SIZE;
        if (i < vec) {
            float4 v = __ldg(&x4[i]);
            buf[s] = v;
            sum += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, off);

    constexpr int N_WARPS = BLOCK_SIZE / 32;
    __shared__ float warp_sums[N_WARPS];
    int warp = tid >> 5;
    int lane = tid & 31;
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        sum = (lane < N_WARPS) ? warp_sums[lane] : 0.f;
        #pragma unroll
        for (int off = N_WARPS / 2; off > 0; off >>= 1)
            sum += __shfl_xor_sync(0xffffffff, sum, off);
        if (lane == 0) warp_sums[0] = sum;
    }
    __syncthreads();

    float m   = warp_sums[0] / n;
    float inv = rsqrtf(m + 1e-5f);

    if (tid == 0) {
        mean[row] = m;
        rstd[row] = inv;
    }

    #pragma unroll
    for (int s = 0; s < VEC_PER_THREAD; ++s) {
        int i = tid + s * BLOCK_SIZE;
        if (i < vec) {
            float4 g = __ldg(&g4[i]);
            float4 v = buf[s];
            v.x = v.x * inv * g.x;
            v.y = v.y * inv * g.y;
            v.z = v.z * inv * g.z;
            v.w = v.w * inv * g.w;
            __stwt(reinterpret_cast<float4*>(&y4[i]), v);   // PTX st.global.cs
        }
    }
}

__global__ void __launch_bounds__(128)
RMSNorm_scalar(const float* __restrict__ X, float* __restrict__ Y,
               const float* __restrict__ gamma, float* mean, float* rstd, size_t n) {

    constexpr int BLOCK_SIZE = 128;
    constexpr int N_WARPS    = BLOCK_SIZE / 32;

    int tid    = threadIdx.x;
    int row    = blockIdx.x;

    const float* x_row = X + row * n;
    float*       y_row = Y + row * n;

    float sum = 0.f;
    for (int i = tid; i < (int)n; i += BLOCK_SIZE) {
        float v = x_row[i];
        sum += v * v;
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, off);

    __shared__ float warp_sums[N_WARPS];
    int warp = tid >> 5;
    int lane = tid & 31;
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        sum = (lane < N_WARPS) ? warp_sums[lane] : 0.f;
        #pragma unroll
        for (int off = N_WARPS / 2; off > 0; off >>= 1)
            sum += __shfl_xor_sync(0xffffffff, sum, off);
        if (lane == 0) warp_sums[0] = sum;
    }
    __syncthreads();

    float m   = warp_sums[0] / n;
    float inv = rsqrtf(m + 1e-5f);

    if (tid == 0) {
        mean[row] = m;
        rstd[row] = inv;
    }

    for (int i = tid; i < (int)n; i += BLOCK_SIZE) {
        y_row[i] = x_row[i] * inv * gamma[i];
    }
}

#define LAUNCH_VEC(BLOCK, VEC) \
    RMSNorm_vec<(BLOCK), (VEC)><<<grid, (BLOCK)>>>(X, Y, gamma, mean, rstd, n)

extern "C" void launcher(const float* X, float* Y, const float* gamma,
                         float* mean, float* rstd, size_t n, size_t m) {

    dim3 grid(m);

    if (n % 4 == 0) {
        size_t vec = n / 4;
        if      (vec <=  128) LAUNCH_VEC(128, 1);
        else if (vec <=  256) LAUNCH_VEC(128, 2);
        else if (vec <=  512) LAUNCH_VEC(128, 4);
        else if (vec <= 1024) LAUNCH_VEC(128, 8);
        else if (vec <= 2048) LAUNCH_VEC(128, 16);
        else                  LAUNCH_VEC(128, 32);   // covers n up to 16384
    } else {
        RMSNorm_scalar<<<grid, 128>>>(X, Y, gamma, mean, rstd, n);
    }
}

int main(int argc, char *argv[]) {
    std::cout << "Benchmarking rishi_rmsnorm — Blackwell SM_120\n";

    const char *data_dir = (argc > 3) ? argv[3] : "data";
    int N = (argc > 1) ? std::atoi(argv[1]) : 8192;
    int C = (argc > 2) ? std::atoi(argv[2]) : 1024;
    size_t SIZE   = (size_t)N * C;
    size_t NBytes = SIZE * sizeof(float);

    std::vector<float> h_inp(SIZE), h_gamma(C);
    std::vector<float> h_out(SIZE), h_out_ref(SIZE);
    std::vector<float> h_mean(N), h_rstd(N), h_mean_ref(N), h_rstd_ref(N);

    //* ── Load PyTorch reference data (falls back to random if files absent) ──
    std::string pfx = std::string(data_dir) + "/rmsnorm_";
    auto fileMatchesSize = [](const std::string &p, size_t n_floats) -> bool {
        FILE *f = fopen(p.c_str(), "rb");
        if (!f) return false;
        fseek(f, 0, SEEK_END);
        size_t bytes = (size_t)ftell(f);
        fclose(f);
        return bytes == n_floats * sizeof(float);
    };
    bool has_ref = fileMatchesSize(pfx + "inp.bin", SIZE);
    if (has_ref) {
        loadBin((pfx + "inp.bin").c_str(),   h_inp.data(),      SIZE);
        loadBin((pfx + "gamma.bin").c_str(), h_gamma.data(),    C);
        loadBin((pfx + "out.bin").c_str(),   h_out_ref.data(),  SIZE);
        loadBin((pfx + "mean.bin").c_str(),  h_mean_ref.data(), N);
        loadBin((pfx + "rstd.bin").c_str(),  h_rstd_ref.data(), N);
        std::cout << "\nLoaded PyTorch reference from " << pfx << "*.bin\n";
    } else {
        initVec(h_inp);
        initVec(h_gamma);
        std::cout << "\nNo reference files found — using random data (benchmarks only)\n";
    }

    //* ── Device allocations ──────────────────────────────────────────────────
    float *d_inp, *d_out, *d_gamma, *d_mean, *d_rstd;
    CUDA_CHECK(cudaMalloc(&d_inp,   NBytes));
    CUDA_CHECK(cudaMalloc(&d_out,   NBytes));
    CUDA_CHECK(cudaMalloc(&d_gamma, C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mean,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rstd,  N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_inp,   h_inp.data(),   NBytes,            cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), C * sizeof(float), cudaMemcpyHostToDevice));

    //* ── Correctness check ───────────────────────────────────────────────────
    launcher(d_inp, d_out, d_gamma, d_mean, d_rstd, (size_t)C, (size_t)N);
    CUDA_CHECK(cudaMemcpy(h_out.data(),  d_out,  NBytes,            cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_mean.data(), d_mean, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_rstd.data(), d_rstd, N * sizeof(float), cudaMemcpyDeviceToHost));

    if (has_ref) {
        std::cout << "\nCorrectness (rishi vs PyTorch) — out:   ";
        checkResult(h_out_ref.data(), h_out.data(), SIZE);
        std::cout << "Correctness (rishi vs PyTorch) — mean:  ";
        checkResult(h_mean_ref.data(), h_mean.data(), N);
        std::cout << "Correctness (rishi vs PyTorch) — rstd:  ";
        checkResult(h_rstd_ref.data(), h_rstd.data(), N);

        reportPrecision("rishi out  precision", h_out_ref.data(),  h_out.data(),  SIZE);
        reportPrecision("rishi mean precision", h_mean_ref.data(), h_mean.data(), N);
        reportPrecision("rishi rstd precision", h_rstd_ref.data(), h_rstd.data(), N);
    }

    //* ── Benchmark ───────────────────────────────────────────────────────────
    //* FLOPs per row ≈ 4C  →  4*N*C total
    //* Bandwidth: inp read once into registers + out write = 2*NBytes
    long long flops = 4LL * N * C;
    size_t    bytes = 2 * NBytes;

    KernelStats stats = benchmarkKernel(
        [&](){ launcher(d_inp, d_out, d_gamma, d_mean, d_rstd, (size_t)C, (size_t)N); },
        100, 25, flops, bytes
    );
    displayStats("rishi — vec4 + register cache + xor butterfly (128 threads per row)", stats);

    CUDA_CHECK(cudaFree(d_inp));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_mean));
    CUDA_CHECK(cudaFree(d_rstd));

    return 0;
}

