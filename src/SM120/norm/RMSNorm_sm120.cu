#include <cuda_runtime.h>
#include "utils/kernelBench.cuh"
#include <chrono>
#include <random>

//* =====================
//* Kernels
//* =====================

// V1 - naive one thread per row
__global__ void RMSNorm_v1_serial(
    const float *inp,
    const float *gamma,
    float *mean,
    float *rstd,
    float *out,
    const int N,
    const int C,
    float eps
){
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    const float *x = inp + idx * C;

    if(idx < N){
        float m = 0.0f;
        float sq_sum = 0.0f;
        for(int i = 0; i < C; ++i){
            float sq = x[i] * x[i];
            sq_sum += sq;
        }
        m = sq_sum / C;

        float r = rsqrtf(m + eps);

        float *y = out + idx * C;
        for(int i = 0; i < C; ++i){
           y[i] = (x[i] * r * gamma[i]); 
        }
        mean[idx] = m;
        rstd[idx] = r;
    }
}

// V2 - one block per row
__global__ void RMSNorm_v2_block(
    const float *inp,
    const float *gamma,
    float *mean,
    float *rstd,
    float *out,
    const int N,
    const int C,
    float eps
){
    uint idx    = threadIdx.x;
    uint row    = blockIdx.x;
    uint stride = blockDim.x;
    uint nwarps = blockDim.x / 32;

    if(row >= (uint)N) return;

    __shared__ float warpSqsum[8];
    __shared__ float s_m;
    __shared__ float s_r;

    float sq_sum = 0.0f;
    for(uint i = idx; i < (uint)C; i += stride){
        float x = inp[row * C + i];
        sq_sum += x * x;
    }

    for(int offset = 16; offset > 0; offset >>= 1)
        sq_sum += __shfl_down_sync(0xffffffff, sq_sum, offset);

    if(idx % 32 == 0)
        warpSqsum[idx / 32] = sq_sum;       
    __syncthreads();                     

    if(idx == 0){
        float blk_sqsum = 0.0f;
        for(uint i = 0; i < nwarps; ++i)
            blk_sqsum += warpSqsum[i];
        s_m = blk_sqsum / C;
        s_r = rsqrtf(s_m + eps);
    }
    __syncthreads();

    float r = s_r;
    float m = s_m;

    for(uint i = idx; i < (uint)C; i += stride)
        out[row * C + i] = inp[row * C + i] * r * gamma[i];

    if(idx == 0){
        mean[row] = m;
        rstd[row] = r;
    }
}

// V3 - one block per row 128 threads
__global__ void RMSNorm_v3_block(
    const float *inp,
    const float *gamma,
    float *mean,
    float *rstd,
    float *out,
    const int N,
    const int C,
    float eps
){
    uint idx    = threadIdx.x;
    uint row    = blockIdx.x;
    uint stride = blockDim.x;
    uint nwarps = blockDim.x / 32;

    if(row >= (uint)N) return;

    __shared__ float warpSqsum[4];
    __shared__ float s_m;
    __shared__ float s_r;

    float sq_sum = 0.0f;
    for(uint i = idx; i < (uint)C; i += stride){
        float x = inp[row * C + i];
        sq_sum += x * x;
    }

    for(int offset = 16; offset > 0; offset >>= 1)
        sq_sum += __shfl_down_sync(0xffffffff, sq_sum, offset);

    if(idx % 32 == 0)
        warpSqsum[idx / 32] = sq_sum;       
    __syncthreads();                     

    if(idx == 0){
        float blk_sqsum = 0.0f;
        for(uint i = 0; i < nwarps; ++i)
            blk_sqsum += warpSqsum[i];
        s_m = blk_sqsum / C;
        s_r = rsqrtf(s_m + eps);
    }
    __syncthreads();

    float r = s_r;
    float m = s_m;

    for(uint i = idx; i < (uint)C; i += stride)
        out[row * C + i] = inp[row * C + i] * r * gamma[i];

    if(idx == 0){
        mean[row] = m;
        rstd[row] = r;
    }
}

// V4 - vectorized loads
__global__ void RMSNorm_v4_vec(
    const float* __restrict__ inp,
    const float* __restrict__ gamma,
    float* mean,
    float* rstd,
    float* out,
    const int N,
    const int C,
    float eps
) {
    // With C=1024 and 128 threads, each thread handles 8 elements (2 float4s)
    const int threads_per_block = blockDim.x;
    const int idx = threadIdx.x;
    const int row = blockIdx.x;

    if (row >= N) return;

    // Use shared memory for the final stage of block reduction
    __shared__ float s_rel[32]; 

    float sq_sum = 0.0f;
    const float4* inp_ptr = reinterpret_cast<const float4*>(inp + row * C);

    // Vectorized Load: Each thread pulls 4 floats at once
    // For C=1024 and blockDim=128, i goes from 0 to 1
    #pragma unroll
    for (int i = idx; i < C / 4; i += threads_per_block) {
        float4 val = inp_ptr[i];
        sq_sum += val.x * val.x;
        sq_sum += val.y * val.y;
        sq_sum += val.z * val.z;
        sq_sum += val.w * val.w;
    }

    // --- High Performance Block Reduction ---
    for (int offset = 16; offset > 0; offset >>= 1)
        sq_sum += __shfl_down_sync(0xffffffff, sq_sum, offset);

    if ((idx & 31) == 0) s_rel[idx >> 5] = sq_sum;
    __syncthreads();

    // Final reduction in the first warp
    float blk_sqsum = (idx < (threads_per_block >> 5)) ? s_rel[idx] : 0.0f;
    if (idx < 32) {
        for (int offset = 16; offset > 0; offset >>= 1)
            blk_sqsum += __shfl_down_sync(0xffffffff, blk_sqsum, offset);
        if (idx == 0) {
            float m = blk_sqsum / C;
            s_rel[0] = m;
            s_rel[1] = rsqrtf(m + eps);
        }
    }
    __syncthreads();

    float m = s_rel[0];
    float r = s_rel[1];

    // Vectorized Write
    const float4* gamma_ptr = reinterpret_cast<const float4*>(gamma);
    float4* out_ptr = reinterpret_cast<float4*>(out + row * C);

    #pragma unroll
    for (int i = idx; i < C / 4; i += threads_per_block) {
        float4 x = inp_ptr[i];
        float4 g = gamma_ptr[i];
        float4 result;
        result.x = x.x * r * g.x;
        result.y = x.y * r * g.y;
        result.z = x.z * r * g.z;
        result.w = x.w * r * g.w;
        out_ptr[i] = result;
    }

    if (idx == 0) {
        mean[row] = m;
        rstd[row] = r;
    }
}

// V5 - xor butterfly reduction: all warp lanes hold the sum after reduction
//      so m and r stay in registers — no s_m/s_r smem, no second __syncthreads()
__global__ void RMSNorm_v5_xor(
    const float* __restrict__ inp,
    const float* __restrict__ gamma,
    float *mean,
    float *rstd,
    float *out,
    const int N,
    const int C,
    float eps
){
    uint idx    = threadIdx.x;
    uint row    = blockIdx.x;
    uint stride = blockDim.x;
    uint nwarps = blockDim.x / 32;

    if(row >= (uint)N) return;

    __shared__ float warpSqsum[4];  // 128 threads = 4 warps

    float sq_sum = 0.0f;
    for(uint i = idx; i < (uint)C; i += stride){
        float x = inp[row * C + i];
        sq_sum += x * x;
    }

    // butterfly: after this, every lane in the warp holds the warp sum
    for(int offset = 16; offset > 0; offset >>= 1)
        sq_sum += __shfl_xor_sync(0xffffffff, sq_sum, offset);

    if(idx % 32 == 0)
        warpSqsum[idx / 32] = sq_sum;
    __syncthreads();  // one barrier — wait for all warp sums to land

    // every thread reads warpSqsum and computes m, r in registers
    // no s_m/s_r broadcast, no second __syncthreads()
    float blk_sqsum = 0.0f;
    for(uint i = 0; i < nwarps; ++i)
        blk_sqsum += warpSqsum[i];
    float m = blk_sqsum / C;
    float r = rsqrtf(m + eps);

    for(uint i = idx; i < (uint)C; i += stride)
        out[row * C + i] = inp[row * C + i] * r * gamma[i];

    if(idx == 0){
        mean[row] = m;
        rstd[row] = r;
    }
}

//* =====================
//* CPU Reference
//* =====================

void rmsnorm_cpu(
    const float *inp,
    float *out,
    const float *gamma,
    float *mean,
    float *rstd,
    int N,
    int C
){
    const float eps = 1.0e-5f;
    for(int n = 0; n < N; ++n){
        const float *x = inp + n * C;

        float rms = 0.0f;
        for(int i = 0; i < C; ++i) rms += x[i] * x[i];
        rms /= C;

        float r = 1.0f / sqrtf(rms + eps);

        float *y = out + n * C;
        for(int i = 0; i < C; ++i)
            y[i] = x[i] * r * gamma[i];

        mean[n] = rms;
        rstd[n] = r;
    }
}

//* =====================
//* Launchers
//* =====================

void launch_RMSNorm_v1(
    const float *inp, const float *gamma,
    float *mean, float *rstd, float *out,
    int N, int C
){
    const float eps = 1.0e-5f;
    dim3 BLOCK{256};
    dim3 GRID{((uint32_t)N + BLOCK.x - 1) / BLOCK.x};
    RMSNorm_v1_serial<<<GRID, BLOCK>>>(inp, gamma, mean, rstd, out, N, C, eps);
}

void launch_RMSNorm_v2(
    const float *inp, const float *gamma,
    float *mean, float *rstd, float *out,
    int N, int C
){
    const float eps = 1.0e-5f;
    dim3 BLOCK{256};
    dim3 GRID{(uint32_t)N};
    RMSNorm_v2_block<<<GRID, BLOCK>>>(inp, gamma, mean, rstd, out, N, C, eps);
}

void launch_RMSNorm_v3(
    const float *inp, const float *gamma,
    float *mean, float *rstd, float *out,
    int N, int C
){
    const float eps = 1.0e-5f;
    dim3 BLOCK{128};
    dim3 GRID{(uint32_t)N};
    RMSNorm_v3_block<<<GRID, BLOCK>>>(inp, gamma, mean, rstd, out, N, C, eps);
}

void launch_RMSNorm_v4(
    const float *inp, const float *gamma,
    float *mean, float *rstd, float *out,
    int N, int C
){
    const float eps = 1.0e-5f;
    dim3 BLOCK{128};
    dim3 GRID{(uint32_t)N};
    RMSNorm_v4_vec<<<GRID, BLOCK>>>(inp, gamma, mean, rstd, out, N, C, eps);
}

void launch_RMSNorm_v5(
    const float *inp, const float *gamma,
    float *mean, float *rstd, float *out,
    int N, int C
){
    const float eps = 1.0e-5f;
    dim3 BLOCK{128};
    dim3 GRID{(uint32_t)N};
    RMSNorm_v5_xor<<<GRID, BLOCK>>>(inp, gamma, mean, rstd, out, N, C, eps);
}



int main(int argc, char *argv[]){
    std::cout << "Benchmarking RMSNorm kernels — Blackwell SM_120\n";

    const char *data_dir = (argc > 3) ? argv[3] : "data";
    int N = (argc > 1) ? std::atoi(argv[1]) : 8192;
    int C = (argc > 2) ? std::atoi(argv[2]) : 1024;
    size_t SIZE   = (size_t)N * C;
    size_t NBytes = SIZE * sizeof(float);

    std::vector<float> h_inp(SIZE), h_gamma(C);
    std::vector<float> h_out(SIZE);
    std::vector<float> h_out_ref(SIZE), h_mean_ref(N), h_rstd_ref(N);
    std::vector<float> h_mean(N), h_rstd(N);

    //* ── Load PyTorch reference data (falls back to random if files absent) ──
    std::string pfx = std::string(data_dir) + "/rmsnorm_";
    auto fileMatchesSize = [](const std::string &p, size_t n_floats) -> bool {
        FILE *f = fopen(p.c_str(), "rb");
        if(!f) return false;
        fseek(f, 0, SEEK_END);
        size_t bytes = (size_t)ftell(f);
        fclose(f);
        return bytes == n_floats * sizeof(float);
    };
    bool has_ref = fileMatchesSize(pfx + "inp.bin", SIZE);
    if(has_ref){
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

    //* ── Device allocations ───────────────────────────────────────────────
    float *d_inp, *d_out, *d_gamma, *d_mean, *d_rstd;
    CUDA_CHECK(cudaMalloc(&d_inp,   NBytes));
    CUDA_CHECK(cudaMalloc(&d_out,   NBytes));
    CUDA_CHECK(cudaMalloc(&d_gamma, C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mean,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rstd,  N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_inp,   h_inp.data(),   NBytes,            cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), C * sizeof(float), cudaMemcpyHostToDevice));

    //* ── Correctness checks ───────────────────────────────────────────────
    if(has_ref){
        launch_RMSNorm_v1(d_inp, d_gamma, d_mean, d_rstd, d_out, N, C);
        CUDA_CHECK(cudaMemcpy(h_out.data(),  d_out,  NBytes,            cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_mean.data(), d_mean, N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_rstd.data(), d_rstd, N * sizeof(float), cudaMemcpyDeviceToHost));

        std::cout << "\nCorrectness V1 (naive vs PyTorch) — out:   ";
        checkResult(h_out_ref.data(), h_out.data(), SIZE);
        std::cout << "Correctness V1 (naive vs PyTorch) — mean:  ";
        checkResult(h_mean_ref.data(), h_mean.data(), N);
        std::cout << "Correctness V1 (naive vs PyTorch) — rstd:  ";
        checkResult(h_rstd_ref.data(), h_rstd.data(), N);

        reportPrecision("V1 out  precision", h_out_ref.data(),  h_out.data(),  SIZE);
        reportPrecision("V1 mean precision", h_mean_ref.data(), h_mean.data(), N);
        reportPrecision("V1 rstd precision", h_rstd_ref.data(), h_rstd.data(), N);

        launch_RMSNorm_v2(d_inp, d_gamma, d_mean, d_rstd, d_out, N, C);
        CUDA_CHECK(cudaMemcpy(h_out.data(),  d_out,  NBytes,            cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_mean.data(), d_mean, N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_rstd.data(), d_rstd, N * sizeof(float), cudaMemcpyDeviceToHost));

        std::cout << "\nCorrectness V2 (block vs PyTorch) — out:   ";
        checkResult(h_out_ref.data(), h_out.data(), SIZE);
        std::cout << "Correctness V2 (block vs PyTorch) — mean:  ";
        checkResult(h_mean_ref.data(), h_mean.data(), N);
        std::cout << "Correctness V2 (block vs PyTorch) — rstd:  ";
        checkResult(h_rstd_ref.data(), h_rstd.data(), N);

        reportPrecision("V2 out  precision", h_out_ref.data(),  h_out.data(),  SIZE);
        reportPrecision("V2 mean precision", h_mean_ref.data(), h_mean.data(), N);
        reportPrecision("V2 rstd precision", h_rstd_ref.data(), h_rstd.data(), N);
    }

    if(has_ref){
        launch_RMSNorm_v3(d_inp, d_gamma, d_mean, d_rstd, d_out, N, C);
        CUDA_CHECK(cudaMemcpy(h_out.data(),  d_out,  NBytes,            cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_mean.data(), d_mean, N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_rstd.data(), d_rstd, N * sizeof(float), cudaMemcpyDeviceToHost));

        std::cout << "\nCorrectness V3 (block vs PyTorch) — out:   ";
        checkResult(h_out_ref.data(), h_out.data(), SIZE);
        std::cout << "Correctness V3 (block vs PyTorch) — mean:  ";
        checkResult(h_mean_ref.data(), h_mean.data(), N);
        std::cout << "Correctness V3 (block vs PyTorch) — rstd:  ";
        checkResult(h_rstd_ref.data(), h_rstd.data(), N);

        reportPrecision("V3 out  precision", h_out_ref.data(),  h_out.data(),  SIZE);
        reportPrecision("V3 mean precision", h_mean_ref.data(), h_mean.data(), N);
        reportPrecision("V3 rstd precision", h_rstd_ref.data(), h_rstd.data(), N);

        launch_RMSNorm_v4(d_inp, d_gamma, d_mean, d_rstd, d_out, N, C);
        CUDA_CHECK(cudaMemcpy(h_out.data(),  d_out,  NBytes,            cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_mean.data(), d_mean, N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_rstd.data(), d_rstd, N * sizeof(float), cudaMemcpyDeviceToHost));

        std::cout << "\nCorrectness V4 (vec4 vs PyTorch) — out:   ";
        checkResult(h_out_ref.data(), h_out.data(), SIZE);
        std::cout << "Correctness V4 (vec4 vs PyTorch) — mean:  ";
        checkResult(h_mean_ref.data(), h_mean.data(), N);
        std::cout << "Correctness V4 (vec4 vs PyTorch) — rstd:  ";
        checkResult(h_rstd_ref.data(), h_rstd.data(), N);

        reportPrecision("V4 out  precision", h_out_ref.data(),  h_out.data(),  SIZE);
        reportPrecision("V4 mean precision", h_mean_ref.data(), h_mean.data(), N);
        reportPrecision("V4 rstd precision", h_rstd_ref.data(), h_rstd.data(), N);

        launch_RMSNorm_v5(d_inp, d_gamma, d_mean, d_rstd, d_out, N, C);
        CUDA_CHECK(cudaMemcpy(h_out.data(),  d_out,  NBytes,            cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_mean.data(), d_mean, N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_rstd.data(), d_rstd, N * sizeof(float), cudaMemcpyDeviceToHost));

        std::cout << "\nCorrectness V5 (xor vs PyTorch) — out:   ";
        checkResult(h_out_ref.data(), h_out.data(), SIZE);
        std::cout << "Correctness V5 (xor vs PyTorch) — mean:  ";
        checkResult(h_mean_ref.data(), h_mean.data(), N);
        std::cout << "Correctness V5 (xor vs PyTorch) — rstd:  ";
        checkResult(h_rstd_ref.data(), h_rstd.data(), N);

        reportPrecision("V5 out  precision", h_out_ref.data(),  h_out.data(),  SIZE);
        reportPrecision("V5 mean precision", h_mean_ref.data(), h_mean.data(), N);
        reportPrecision("V5 rstd precision", h_rstd_ref.data(), h_rstd.data(), N);
    }

    //* ── Benchmarks ───────────────────────────────────────────────────────
    //* FLOPs per row ≈ 4C  →  4*N*C total
    //* Bandwidth: V1 reads inp 2x (rms pass, output pass) + 1 write = 3*NBytes
    //* Bandwidth: V2 same — 2 reads + 1 write = 3*NBytes
    long long flops = 4LL * N * C;
    size_t    bytes = 3 * NBytes;

    KernelStats stats_v1 = benchmarkKernel(
        [&](){ launch_RMSNorm_v1(d_inp, d_gamma, d_mean, d_rstd, d_out, N, C); },
        100, 25, flops, bytes
    );
    displayStats("V1 — Naive (one thread per row)", stats_v1);

    KernelStats stats_v2 = benchmarkKernel(
        [&](){ launch_RMSNorm_v2(d_inp, d_gamma, d_mean, d_rstd, d_out, N, C); },
        100, 25, flops, bytes
    );
    displayStats("V2 — Block cooperative (256 threads per row)", stats_v2);

    KernelStats stats_v3 = benchmarkKernel(
        [&](){ launch_RMSNorm_v3(d_inp, d_gamma, d_mean, d_rstd, d_out, N, C); },
        100, 25, flops, bytes
    );
    displayStats("V3 — Block cooperative (128 threads per row)", stats_v3);

    KernelStats stats_v4 = benchmarkKernel(
        [&](){ launch_RMSNorm_v4(d_inp, d_gamma, d_mean, d_rstd, d_out, N, C); },
        100, 25, flops, bytes
    );
    displayStats("V4 — Vec4 float4 loads (128 threads per row)", stats_v4);

    KernelStats stats_v5 = benchmarkKernel(
        [&](){ launch_RMSNorm_v5(d_inp, d_gamma, d_mean, d_rstd, d_out, N, C); },
        100, 25, flops, bytes
    );
    displayStats("V5 — XOR butterfly, no smem broadcast (128 threads per row)", stats_v5);

    CUDA_CHECK(cudaFree(d_inp));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_mean));
    CUDA_CHECK(cudaFree(d_rstd));

    return 0;
}
