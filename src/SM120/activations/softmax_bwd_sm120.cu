#include <cuda_runtime.h>
#include <cfloat>
#include <stdint.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

#include "utils/kernelUtils.cuh"
#include "utils/kernelBench.cuh"

//* ============================
//* Math behind Softmax Backward
//* ============================
//*
//* Forward:   y_i = exp(x_i) / sum_j exp(x_j)
//*
//* Given:
//*   grad_out[i] = dL/dy_i   (upstream gradient flowing in)
//*   softmax_out[i] = y_i    (saved from the forward pass)
//*
//* We want:
//*   grad_in[i] = dL/dx_i
//*
//* Derivation using the Jacobian of softmax:
//*   dy_j/dx_i = y_j * (delta_ij - y_i)
//*
//*   dL/dx_i = sum_j ( dL/dy_j * dy_j/dx_i )
//*           = sum_j ( grad_out[j] * y_j * (delta_ij - y_i) )
//*           = grad_out[i] * y_i  -  y_i * sum_j( grad_out[j] * y_j )
//*           = y_i * ( grad_out[i] - dot(grad_out, y) )
//*
//* So for every row:
//*   step 1: dot = sum_j( grad_out[j] * softmax_out[j] )   <- scalar per row
//*   step 2: grad_in[i] = softmax_out[i] * (grad_out[i] - dot)
//*
//* Precision note:
//*   All arithmetic is float32, matching PyTorch's production kernel dtype.
//*   IEEE-compliant math (no --use_fast_math) ensures FMA behaviour is the
//*   same as standard float32.  Expected max_rel vs PyTorch float32 reference:
//*   ~1e-5 for V3 (reduction-order rounding), ~1e-2 for V1 (serial O(n) error).

//* ============================
//* Kernel Implementations
//* ============================

//* V1 Backward — one thread per row, fully serial
//* Each thread owns all COLS columns of its row: reads grad_out+softmax_out
//* twice (dot pass, write pass) with stride-1 access.
__global__ void softmax_backward_v1_serial(
    const float *grad_out,
    const float *softmax_out,
    float       *grad_in,
    int64_t rows,
    int64_t cols
){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row >= rows) return;

    float dot = 0.0f;
    for(int c = 0; c < cols; c++)
        dot += grad_out[row * cols + c] * softmax_out[row * cols + c];

    for(int c = 0; c < cols; c++)
        grad_in[row * cols + c] = softmax_out[row * cols + c]
                                * (grad_out[row * cols + c] - dot);
}


//* V2 Backward — one block per row, partial dot products via atomicAdd
//* 256 threads split columns in stride pattern, merge with atomicAdd,
//* then each thread writes back the same columns it read.
__global__ void softmax_backward_v2_atomic(
    const float *grad_out,
    const float *softmax_out,
    float       *grad_in,
    int64_t rows,
    int64_t cols
){
    uint idx = threadIdx.x;
    uint row = blockIdx.x;

    __shared__ float dot;
    if(idx == 0) dot = 0.0f;
    __syncthreads();

    float local_dot = 0.0f;
    uint stride = blockDim.x;
    for(uint i = idx; i < (uint)cols; i += stride)
        local_dot += grad_out[row * cols + i] * softmax_out[row * cols + i];

    atomicAdd(&dot, local_dot);
    __syncthreads();

    for(uint i = idx; i < (uint)cols; i += stride)
        grad_in[row * cols + i] = softmax_out[row * cols + i]
                                * (grad_out[row * cols + i] - dot);
}


//* V3 Backward — warp shuffle reduction, no atomicAdd
//* 5 __shfl_down_sync rounds reduce each warp to lane 0.
//* Lane 0 of each of the 8 warps writes to shared mem.
//* Thread 0 sums the 8 warp totals into g_dot.
__global__ void softmax_backward_v3_warp_reduce(
    const float *grad_out,
    const float *softmax_out,
    float       *grad_in,
    int64_t rows,
    int64_t cols
){
    uint idx = threadIdx.x;
    uint row = blockIdx.x;

    if(row >= (uint)rows) return;

    __shared__ float g_dot;
    __shared__ float warp_dots[8];          //* 256 threads / 32 = 8 warps
    if(idx == 0) g_dot = 0.0f;
    __syncthreads();

    float local_dot = 0.0f;
    uint stride = blockDim.x;
    for(uint i = idx; i < (uint)cols; i += stride)
        local_dot += grad_out[row * cols + i] * softmax_out[row * cols + i];

    //* warp-level tree reduction — 5 rounds halve the active lanes each time
    #pragma unroll
    for(int offset = 16; offset > 0; offset >>= 1)
        local_dot += __shfl_down_sync(0xffffffff, local_dot, offset);

    if(idx % 32 == 0) warp_dots[idx / 32] = local_dot;
    __syncthreads();

    if(idx == 0){
        float block_dot = warp_dots[0];
        for(int i = 1; i < blockDim.x / 32; i++)
            block_dot += warp_dots[i];
        g_dot = block_dot;
    }
    __syncthreads();

    for(uint i = idx; i < (uint)cols; i += stride)
        grad_in[row * cols + i] = softmax_out[row * cols + i]
                                * (grad_out[row * cols + i] - g_dot);
}


//* ============================
//* CPU Reference
//* ============================

void softmax_backward_cpu(
    const float *grad_out,
    const float *softmax_out,
    float       *grad_in,
    uint64_t rows,
    uint64_t cols
){
    for(uint64_t r = 0; r < rows; r++){
        float dot = 0.0f;
        for(uint64_t c = 0; c < cols; c++)
            dot += grad_out[r * cols + c] * softmax_out[r * cols + c];
        for(uint64_t c = 0; c < cols; c++)
            grad_in[r * cols + c] = softmax_out[r * cols + c]
                                  * (grad_out[r * cols + c] - dot);
    }
}


//* ============================
//* Launch Configurations
//* ============================

void launch_softmax_backward_v1(
    const float *grad_out, const float *softmax_out, float *grad_in,
    uint64_t rows, uint64_t cols
){
    dim3 BLOCK{256};
    dim3 GRID{((uint32_t)rows + BLOCK.x - 1) / BLOCK.x};
    softmax_backward_v1_serial<<<GRID, BLOCK>>>(grad_out, softmax_out, grad_in,
                                                (int64_t)rows, (int64_t)cols);
}

void launch_softmax_backward_v2(
    const float *grad_out, const float *softmax_out, float *grad_in,
    uint64_t rows, uint64_t cols
){
    dim3 BLOCK{256};
    dim3 GRID{(uint32_t)rows};
    softmax_backward_v2_atomic<<<GRID, BLOCK>>>(grad_out, softmax_out, grad_in,
                                                (int64_t)rows, (int64_t)cols);
}

void launch_softmax_backward_v3(
    const float *grad_out, const float *softmax_out, float *grad_in,
    uint64_t rows, uint64_t cols
){
    dim3 BLOCK{256};
    dim3 GRID{(uint32_t)rows};
    softmax_backward_v3_warp_reduce<<<GRID, BLOCK>>>(grad_out, softmax_out, grad_in,
                                                     (int64_t)rows, (int64_t)cols);
}


//* ============================
//* Main
//* ============================

int main(){
    std::cout << "=== Softmax Backward V1, V2 & V3 — Blackwell SM_120 ===\n";

    constexpr uint64_t ROWS   = 8192;
    constexpr uint64_t COLS   = 1024;
    constexpr uint64_t SIZE   = ROWS * COLS;
    constexpr uint64_t NBytes = SIZE * sizeof(float);

    std::vector<float> h_raw(SIZE);
    std::vector<float> h_grad_out(SIZE);
    std::vector<float> h_softmax_out(SIZE);
    std::vector<float> h_grad_in(SIZE);
    std::vector<float> h_ref(SIZE);

    //* ── Load PyTorch reference data (falls back to random + CPU ref if absent) ──
    auto fileMatchesSize = [](const std::string &p, size_t n_floats) -> bool {
        FILE *f = fopen(p.c_str(), "rb");
        if(!f) return false;
        fseek(f, 0, SEEK_END);
        size_t bytes = (size_t)ftell(f);
        fclose(f);
        return bytes == n_floats * sizeof(float);
    };
    bool has_ref = fileMatchesSize("data/softmax_bwd_grad_out.bin", SIZE);
    if(has_ref){
        loadBin("data/softmax_bwd_grad_out.bin",    h_grad_out.data(),    SIZE);
        loadBin("data/softmax_bwd_softmax_out.bin", h_softmax_out.data(), SIZE);
        loadBin("data/softmax_bwd_ref.bin",         h_ref.data(),         SIZE);
        std::cout << "Reference: PyTorch float32 autograd\n";
        std::cout << "Loaded inputs + PyTorch reference from data/softmax_bwd_*.bin\n\n";
    } else {
        initVec(h_grad_out);
        initVec(h_raw);
        for(uint64_t r = 0; r < ROWS; r++){
            float sum = 0.0f;
            for(uint64_t c = 0; c < COLS; c++) sum += expf(h_raw[r * COLS + c]);
            for(uint64_t c = 0; c < COLS; c++)
                h_softmax_out[r * COLS + c] = expf(h_raw[r * COLS + c]) / sum;
        }
        auto t0 = std::chrono::high_resolution_clock::now();
        softmax_backward_cpu(h_grad_out.data(), h_softmax_out.data(), h_ref.data(), ROWS, COLS);
        auto t1 = std::chrono::high_resolution_clock::now();
        float cpu_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        std::cout << "Reference: CPU float32 serial\n";
        std::cout << "No reference files found — falling back to CPU reference: " << cpu_ms << " ms\n\n";
    }

    float *d_grad_out, *d_softmax_out, *d_grad_in;
    CUDA_CHECK(cudaMalloc(&d_grad_out,    NBytes));
    CUDA_CHECK(cudaMalloc(&d_softmax_out, NBytes));
    CUDA_CHECK(cudaMalloc(&d_grad_in,     NBytes));
    CUDA_CHECK(cudaMemcpy(d_grad_out,    h_grad_out.data(),    NBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_softmax_out, h_softmax_out.data(), NBytes, cudaMemcpyHostToDevice));

    //* ── Correctness ─────────────────────────────────────────────────────
    const char *ref_label = has_ref ? "PyTorch" : "CPU";
    std::cout << "--- Correctness ---\n";

    launch_softmax_backward_v1(d_grad_out, d_softmax_out, d_grad_in, ROWS, COLS);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_grad_in.data(), d_grad_in, NBytes, cudaMemcpyDeviceToHost));
    std::cout << "V1 (serial backward)   vs " << ref_label << ": ";
    checkResult(h_ref.data(), h_grad_in.data(), SIZE);
    reportPrecision("V1 precision", h_ref.data(), h_grad_in.data(), SIZE);

    launch_softmax_backward_v2(d_grad_out, d_softmax_out, d_grad_in, ROWS, COLS);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_grad_in.data(), d_grad_in, NBytes, cudaMemcpyDeviceToHost));
    std::cout << "V2 (atomic backward)   vs " << ref_label << ": ";
    checkResult(h_ref.data(), h_grad_in.data(), SIZE);
    reportPrecision("V2 precision", h_ref.data(), h_grad_in.data(), SIZE);

    launch_softmax_backward_v3(d_grad_out, d_softmax_out, d_grad_in, ROWS, COLS);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_grad_in.data(), d_grad_in, NBytes, cudaMemcpyDeviceToHost));
    std::cout << "V3 (warp reduce)       vs " << ref_label << ": ";
    checkResult(h_ref.data(), h_grad_in.data(), SIZE);
    reportPrecision("V3 precision", h_ref.data(), h_grad_in.data(), SIZE);

    //* ── Benchmarks ───────────────────────────────────────────────────────
    //* FLOPs: ~5 per element (2 dot pass + 2 output pass + 1 approx)
    //* Bandwidth: grad_out (read) + softmax_out (read x2) + grad_in (write) = 4× NBytes
    std::cout << "\n--- Benchmarks (ROWS=" << ROWS << ", COLS=" << COLS << ") ---\n";
    long long flops = 5LL * (long long)SIZE;
    size_t    bytes = 4 * NBytes;

    KernelStats stats_v1 = benchmarkKernel(
        [&](){ launch_softmax_backward_v1(d_grad_out, d_softmax_out, d_grad_in, ROWS, COLS); },
        100, 10, flops, bytes);
    displayStats("V1 — Serial Backward (one thread per row)", stats_v1);

    KernelStats stats_v2 = benchmarkKernel(
        [&](){ launch_softmax_backward_v2(d_grad_out, d_softmax_out, d_grad_in, ROWS, COLS); },
        100, 10, flops, bytes);
    displayStats("V2 — AtomicAdd Backward (one block per row)", stats_v2);

    KernelStats stats_v3 = benchmarkKernel(
        [&](){ launch_softmax_backward_v3(d_grad_out, d_softmax_out, d_grad_in, ROWS, COLS); },
        100, 10, flops, bytes);
    displayStats("V3 — Warp Reduce Backward (no atomicAdd)", stats_v3);

    CUDA_CHECK(cudaFree(d_grad_out));
    CUDA_CHECK(cudaFree(d_softmax_out));
    CUDA_CHECK(cudaFree(d_grad_in));

    return 0;
}