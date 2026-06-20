#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "utils/kernelUtils.cuh"
#include "utils/kernelBench.cuh"

__device__ float sigmoid(const float X) {
    return 1.f/(1 + expf(-X));
}

__device__ float swish(const float X) {
    return X * sigmoid(X);
}

__global__ void __launch_bounds__(256, 8)
swiglu(const float4* __restrict__ X, float4* __restrict__ Y, size_t m, size_t n, int cols4) {

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if(row >= m || col >= cols4) return;

    int row_in4  = row * (2 * cols4);
    int row_out4 = row * cols4;

    float4 a = __ldg(&X[row_in4 + col]);
    float4 b = __ldg(&X[row_in4 + cols4 + col]);

    float4 y;
    y.x = swish(a.x) * b.x;
    y.y = swish(a.y) * b.y;
    y.z = swish(a.z) * b.z;
    y.w = swish(a.w) * b.w;

    Y[row_out4 + col] = y;
}

__global__ void swiglu_tail(const float* X, float* Y, size_t m, size_t n, int col_start) {

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = col_start + blockDim.x * blockIdx.x + threadIdx.x;

    if(row >= m || col >= n) return;

    float a = X[row * 2 * n + col];
    float b = X[row * 2 * n + n + col];

    Y[row * n + col] = swish(a) * b;
}

extern "C" void launcher(const float* X, float* Y, size_t m, size_t n) {

    int cols4     = n / 4;
    int cols_rem  = n & 3;
    int col_start = cols4 * 4;

    dim3 block(32, 4);

    if(cols4 > 0) {
        dim3 grid((cols4 + block.x - 1) / block.x,
                  (m     + block.y - 1) / block.y);

        swiglu<<<grid, block>>>(reinterpret_cast<const float4*>(X),
                                reinterpret_cast<float4*>(Y),
                                m, n, cols4);
    }

    if(cols_rem > 0) {
        dim3 grid((cols_rem + block.x - 1) / block.x,
                  (m        + block.y - 1) / block.y);

        swiglu_tail<<<grid, block>>>(X, Y, m, n, col_start);
    }
}

int main(int argc, char *argv[]){
    std::cout << "Benchmarking SwiGLU kernel — Blackwell SM_120\n";

    const char *data_dir = (argc > 3) ? argv[3] : "data";
    size_t M = (argc > 1) ? std::atoi(argv[1]) : 4096;
    size_t N = (argc > 2) ? std::atoi(argv[2]) : 4096;

    size_t IN_SIZE  = M * 2 * N;
    size_t OUT_SIZE = M * N;
    size_t InBytes  = IN_SIZE  * sizeof(float);
    size_t OutBytes = OUT_SIZE * sizeof(float);

    std::vector<float> h_inp(IN_SIZE);
    std::vector<float> h_out(OUT_SIZE), h_out_ref(OUT_SIZE);

    //* ── Load PyTorch reference data ─────────────────────────────────────
    std::string pfx = std::string(data_dir) + "/swiglu_";
    loadBin((pfx + "inp.bin").c_str(), h_inp.data(),     IN_SIZE);
    loadBin((pfx + "out.bin").c_str(), h_out_ref.data(), OUT_SIZE);
    std::cout << "\nLoaded PyTorch reference from " << pfx << "*.bin\n";

    //* ── Device allocations ───────────────────────────────────────────────
    float *d_inp, *d_out;
    CUDA_CHECK(cudaMalloc(&d_inp, InBytes));
    CUDA_CHECK(cudaMalloc(&d_out, OutBytes));
    CUDA_CHECK(cudaMemcpy(d_inp, h_inp.data(), InBytes, cudaMemcpyHostToDevice));

    //* ── Correctness + precision ──────────────────────────────────────────
    launcher(d_inp, d_out, M, N);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, OutBytes, cudaMemcpyDeviceToHost));

    std::cout << "\nCorrectness (vs PyTorch) — out:  ";
    checkResult(h_out_ref.data(), h_out.data(), OUT_SIZE);
    reportPrecision("SwiGLU precision", h_out_ref.data(), h_out.data(), OUT_SIZE);

    //* ── Benchmark ───────────────────────────────────────────────────────
    //* FLOPs: ~6 per output element (exp, add, div, 2x mul for swish, 1x mul with b)
    //* Bandwidth: read 2N floats + write N floats per row = 3*M*N*sizeof(float)
    long long flops = 6LL * M * N;
    size_t    bytes = 3 * OutBytes;

    KernelStats stats = benchmarkKernel(
        [&](){ launcher(d_inp, d_out, M, N); },
        100, 25, flops, bytes
    );
    displayStats("SwiGLU — float4 vectorised", stats);

    CUDA_CHECK(cudaFree(d_inp));
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}