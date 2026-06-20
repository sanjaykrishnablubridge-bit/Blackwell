#include <cuda_runtime.h>
#include <chrono>
#include <cfloat>
#include <stdint.h>
#include <assert.h>
#include "utils/kernelBench.cuh"

//* ============================
//* Helpers
//* ============================
#define CHUNKS_PER_THREAD 8
//* warp level reduction for max
__device__ float warpReduceMax(float val){
    for(int offset = 16; offset > 0; offset >>=1){
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val,  offset));
    }
    return val;
}

__device__ __forceinline__ void online_merge(
    float& max_a, float& sum_a,
    float max_b, float sum_b
){
    float new_max = fmaxf(max_a, max_b);
    sum_a = sum_a * __expf(max_a - new_max)
          + sum_b * __expf(max_b - new_max);
    max_a = new_max;
}

__device__ __forceinline__ void warp_reduce(float& thread_max, float& thread_sum){
    #pragma unroll
    for(int offset = 16; offset > 0; offset >>= 1){
        float other_max = __shfl_down_sync(0xffffffff, thread_max, offset);
        float other_sum = __shfl_down_sync(0xffffffff, thread_sum, offset);
        online_merge(thread_max, thread_sum, other_max, other_sum);
    }
}

//* ============================
//* Kernel Implementations
//* ============================

//* V1 — one thread per row, fully serial within the row
//* Bottleneck: under-utilizes the GPU on wide matrices
__global__ void softmax_v1_serial(
    const float *in,
    float *out,
    int64_t rows,
    int64_t cols
){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row >= rows) return;

    float sum = 0.0f;
    for(int c = 0; c < cols; c++)
        sum += expf(in[row * cols + c]);
    for(int c = 0; c < cols; c++)
        out[row * cols + c] = expf(in[row * cols + c]) / sum;
}


//* V2 — one block per row, threads cooperate via atomicAdd
//* Bottleneck: atomicAdd serializes the block-level reduction
__global__ void softmax_v2_atomic(
    const float *in,
    float *out,
    int64_t rows,
    int64_t cols
){
    uint idx = threadIdx.x;
    uint row = blockIdx.x;

    if(idx >= (uint)cols) return;

    __shared__ float sum;
    if(threadIdx.x == 0) sum = 0.0f;
    __syncthreads();

    float local_thread_sum = 0.0f;
    uint stride = blockDim.x;
    for(uint i = idx; i < cols; i += stride)
        local_thread_sum += expf(in[row * cols + i]);

    atomicAdd(&sum, local_thread_sum);
    __syncthreads();

    for(uint i = idx; i < cols; i += stride)
        out[row * cols + i] = expf(in[row * cols + i]) / sum;
}

//* V3 — one block per row, threads cooperate via atomicAdd
//* but the softmax is numerically stable
__global__ void softmax_v3_stable(
    const float *in,
    float *out,
    int64_t rows,
    int64_t cols
){
    uint idx = threadIdx.x;
    uint row = blockIdx.x;

    if(row >= rows) return;

    __shared__ float g_max;
    __shared__ float g_sum;
    __shared__ float warp_maxes[8];              //* one slot per warp (256/32)
    if(idx == 0){
        g_sum = 0.0f;
        g_max = -FLT_MAX;
    }
    __syncthreads();

    float local_thread_max = -FLT_MAX;
    uint stride = blockDim.x;
    for(uint i = idx; i < cols; i+=stride){
        local_thread_max = max(local_thread_max, in[(row * cols) + i]);
    }

    float warp_max = warpReduceMax(local_thread_max);
    if(idx % 32 == 0) warp_maxes[idx / 32] = warp_max;   //* lane 0 of each warp writes
    __syncthreads();

    if(idx == 0){                                          //* thread 0 reduces the 8 warp maxes
        float block_max = warp_maxes[0];
        for(int i = 1; i < blockDim.x / 32; i++)
            block_max = fmaxf(block_max, warp_maxes[i]);
        g_max = block_max;
    }
    __syncthreads();

    float local_thread_sum = 0;
    for(uint i = idx; i < cols; i+=stride){
        local_thread_sum += expf(in[(row * cols) + i] - g_max);
    }

    atomicAdd(&g_sum, local_thread_sum);
    __syncthreads();

    for(uint i = idx; i < cols; i += stride){
        out[row * cols + i] = (expf(in[row * cols + i] - g_max)) / g_sum;
    }
}

//* V4 - Online softmax
__global__ void softmax_v4_online(
    const float *in,
    float *out,
    int64_t rows,
    int64_t cols
){
    uint idx = threadIdx.x;
    uint row = blockIdx.x;

    if(row >= rows) return;

    __shared__ float g_max;
    __shared__ float g_sum;
    __shared__ float warp_maxes[8];
    __shared__ float warp_sums[8];
    if(idx == 0){ g_max = -FLT_MAX; g_sum = 0.0f; }
    __syncthreads();

    float thread_max = -FLT_MAX;
    float thread_sum = 0.0f;

    uint stride = blockDim.x;
    for(uint i = idx; i < cols; i += stride){
        float x = in[row * cols + i];
        float new_max = fmaxf(thread_max, x);
        thread_sum = thread_sum * expf(thread_max - new_max) + expf(x - new_max);
        thread_max = new_max;
    }

    //* warp-level reduction of (max, sum) pairs using online merge
    for(int offset = 16; offset > 0; offset >>= 1){
        float other_max = __shfl_down_sync(0xffffffff, thread_max, offset);
        float other_sum = __shfl_down_sync(0xffffffff, thread_sum, offset);
        float new_max = fmaxf(thread_max, other_max);
        thread_sum = thread_sum * expf(thread_max - new_max)
                   + other_sum * expf(other_max - new_max);
        thread_max = new_max;
    }

    if(idx % 32 == 0){
        warp_maxes[idx / 32] = thread_max;
        warp_sums[idx / 32]  = thread_sum;
    }
    __syncthreads();

    if(idx == 0){
        float blk_max = warp_maxes[0];
        float blk_sum = warp_sums[0];
        for(int i = 1; i < blockDim.x / 32; i++){
            float wm = warp_maxes[i], ws = warp_sums[i];
            float new_max = fmaxf(blk_max, wm);
            blk_sum = blk_sum * expf(blk_max - new_max)
                    + ws      * expf(wm      - new_max);
            blk_max = new_max;
        }
        g_max = blk_max;
        g_sum = blk_sum;
    }
    __syncthreads();

    for(uint i = idx; i < cols; i += stride){
        out[row * cols + i] = expf(in[row * cols + i] - g_max) / g_sum;
    }
}

//* V5 - Online softmax with one warp per block
//* No shared memory needed — broadcast lane 0's result via __shfl_sync
__global__ void softmax_v5_warp(
    const float *in,
    float *out,
    uint64_t rows,
    uint64_t cols
){
    uint idx = threadIdx.x;
    uint row = blockIdx.x;

    if(row >= rows) return;

    float thread_max = -FLT_MAX;
    float thread_sum = 0.0f;

    uint stride = blockDim.x;
    for(uint i = idx; i < cols; i += stride){
        float x = in[row * cols + i];
        float new_max = fmaxf(thread_max, x);
        thread_sum = thread_sum * expf(thread_max - new_max) + expf(x - new_max);
        thread_max = new_max;
    }

    //* warp-level reduction of (max, sum) pairs using online merge
    for(int offset = 16; offset > 0; offset >>= 1){
        float other_max = __shfl_down_sync(0xffffffff, thread_max, offset);
        float other_sum = __shfl_down_sync(0xffffffff, thread_sum, offset);
        float new_max = fmaxf(thread_max, other_max);
        thread_sum = thread_sum * expf(thread_max - new_max)
                   + other_sum * expf(other_max - new_max);
        thread_max = new_max;
    }

    //* After shfl_down reduction only lane 0 has the correct final values.
    //* Broadcast them to all lanes so every thread can write its output elements.
    float blk_max = __shfl_sync(0xffffffff, thread_max, 0);
    float blk_sum = __shfl_sync(0xffffffff, thread_sum, 0);

    for(uint i = idx; i < cols; i += stride){
        out[row * cols + i] = expf(in[row * cols + i] - blk_max) / blk_sum;
    }
}



//* V6 - V4 with float4 vectorized loads/stores
//* Each thread issues one 128-bit LDG per loop iteration instead of four
//* separate 32-bit loads, cutting instruction count and memory-pipe pressure
//* by 4x.  Requires cols % 4 == 0 and 16-byte aligned row pointers (both
//* guaranteed when cudaMalloc is used and cols is a multiple of 4).
__global__ void softmax_v6_vec4(
    const float * __restrict__ in,
    float       * __restrict__ out,
    int64_t rows,
    int64_t cols
){
    uint idx = threadIdx.x;
    uint row = blockIdx.x;

    if(row >= rows) return;

    __shared__ float g_max;
    __shared__ float g_sum;
    __shared__ float warp_maxes[8];
    __shared__ float warp_sums[8];
    if(idx == 0){ g_max = -FLT_MAX; g_sum = 0.0f; }
    __syncthreads();

    float thread_max = -FLT_MAX;
    float thread_sum = 0.0f;

    const float4 *in4  = reinterpret_cast<const float4*>(in  + row * cols);
    float4 *out4 = reinterpret_cast<float4*>(out + row * cols);
    int64_t vec_cols = cols / 4;
    uint stride   = blockDim.x;

    //* accumulation pass — one 128-bit load, four online_merge calls per iteration
    for(uint i = idx; i < vec_cols; i += stride){
        float4 c = in4[i];
        online_merge(thread_max, thread_sum, c.x, 1.0f);
        online_merge(thread_max, thread_sum, c.y, 1.0f);
        online_merge(thread_max, thread_sum, c.z, 1.0f);
        online_merge(thread_max, thread_sum, c.w, 1.0f);
    }
    //* scalar tail for cols not divisible by 4
    for(int64_t i = vec_cols * 4 + idx; i < cols; i += stride)
        online_merge(thread_max, thread_sum, in[row * cols + i], 1.0f);

    //* warp + block reduction (identical structure to V4)
    warp_reduce(thread_max, thread_sum);

    if(idx % 32 == 0){
        warp_maxes[idx / 32] = thread_max;
        warp_sums[idx / 32]  = thread_sum;
    }
    __syncthreads();

    if(idx == 0){
        float blk_max = warp_maxes[0];
        float blk_sum = warp_sums[0];
        for(int i = 1; i < blockDim.x / 32; i++)
            online_merge(blk_max, blk_sum, warp_maxes[i], warp_sums[i]);
        g_max = blk_max;
        g_sum = blk_sum;
    }
    __syncthreads();

    //* output pass — one 128-bit store per iteration
    float inv_sum = 1.0f / g_sum;
    for(uint i = idx; i < vec_cols; i += stride){
        float4 c = in4[i];
        float4 r;
        r.x = __expf(c.x - g_max) * inv_sum;
        r.y = __expf(c.y - g_max) * inv_sum;
        r.z = __expf(c.z - g_max) * inv_sum;
        r.w = __expf(c.w - g_max) * inv_sum;
        out4[i] = r;
    }
    for(int64_t i = vec_cols * 4 + idx; i < cols; i += stride)
        out[row * cols + i] = __expf(in[row * cols + i] - g_max) * inv_sum;
}


//* ============================
//* CPU Reference
//* ============================

void softmax_cpu(
    const float *in, 
    float *out, 
    uint64_t rows, 
    uint64_t cols
){
    for(uint64_t r = 0; r < rows; r++){
        float sum = 0.0f;
        for(uint64_t c = 0; c < cols; c++)
            sum += expf(in[r * cols + c]);
        for(uint64_t c = 0; c < cols; c++)
            out[r * cols + c] = expf(in[r * cols + c]) / sum;
    }
}


//* ============================
//* Launch Configurations
//* ============================

void launch_softmax_v1(const float *in, float *out, uint64_t rows, uint64_t cols){
    dim3 BLOCK{256};
    dim3 GRID{((uint32_t)rows + BLOCK.x - 1) / BLOCK.x};
    softmax_v1_serial<<<GRID, BLOCK>>>(in, out, (int64_t)rows, (int64_t)cols);
}

void launch_softmax_v2(const float *in, float *out, uint64_t rows, uint64_t cols){
    dim3 BLOCK{256};
    dim3 GRID{(uint32_t)rows};
    softmax_v2_atomic<<<GRID, BLOCK>>>(in, out, (int64_t)rows, (int64_t)cols);
}

void launch_softmax_v3(const float *in, float *out, uint64_t rows, uint64_t cols){
    dim3 BLOCK{256};
    dim3 GRID{(uint32_t)rows};
    softmax_v3_stable<<<GRID, BLOCK>>>(in, out, (int64_t)rows, (int64_t)cols);
}

void launch_softmax_v4(const float *in, float *out, uint64_t rows, uint64_t cols){
    dim3 BLOCK{256};
    dim3 GRID{(uint32_t)rows};
    softmax_v4_online<<<GRID, BLOCK>>>(in, out, (int64_t)rows, (int64_t)cols);
}

void launch_softmax_v5(const float *in, float *out, uint64_t rows, uint64_t cols){
    dim3 BLOCK{32};
    dim3 GRID{(uint32_t)rows};
    softmax_v5_warp<<<GRID, BLOCK>>>(in, out, (int64_t)rows, (int64_t)cols);
}

void launch_softmax_v6(const float *in, float *out, uint64_t rows, uint64_t cols){
    dim3 BLOCK{256};
    dim3 GRID{(uint32_t)rows};
    softmax_v6_vec4<<<GRID, BLOCK>>>(in, out, (int64_t)rows, (int64_t)cols);
}



int main(){
    std::cout << "Benchmarking Softmax kernels — Blackwell SM_120\n";

    uint64_t ROWS   = 8192;
    uint64_t COLS   = 1024;
    uint64_t SIZE   = ROWS * COLS;
    uint64_t NBytes = SIZE * sizeof(float);

    std::vector<float> h_in(SIZE);
    std::vector<float> h_out(SIZE);
    std::vector<float> h_ref(SIZE);

    //* ── Load PyTorch reference data (falls back to random if files absent) ──
    auto fileMatchesSize = [](const std::string &p, size_t n_floats) -> bool {
        FILE *f = fopen(p.c_str(), "rb");
        if(!f) return false;
        fseek(f, 0, SEEK_END);
        size_t bytes = (size_t)ftell(f);
        fclose(f);
        return bytes == n_floats * sizeof(float);
    };
    bool has_ref = fileMatchesSize("data/softmax_inp.bin", SIZE);
    if(has_ref){
        loadBin("data/softmax_inp.bin", h_in.data(),  SIZE);
        loadBin("data/softmax_out.bin", h_ref.data(), SIZE);
        std::cout << "\nLoaded PyTorch reference from data/softmax_*.bin\n";
    } else {
        initVec(h_in);
        std::cout << "\nNo reference files found — using random data (benchmarks only)\n";
    }

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, NBytes));
    CUDA_CHECK(cudaMalloc(&d_out, NBytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), NBytes, cudaMemcpyHostToDevice));

    //* ── Correctness checks ───────────────────────────────────────────────
    if(has_ref){
        launch_softmax_v1(d_in, d_out, ROWS, COLS);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, NBytes, cudaMemcpyDeviceToHost));
        std::cout << "\nCorrectness V1 (serial vs PyTorch):  ";
        checkResult(h_ref.data(), h_out.data(), SIZE);
        reportPrecision("V1 precision", h_ref.data(), h_out.data(), SIZE);

        launch_softmax_v2(d_in, d_out, ROWS, COLS);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, NBytes, cudaMemcpyDeviceToHost));
        std::cout << "\nCorrectness V2 (atomic vs PyTorch):  ";
        checkResult(h_ref.data(), h_out.data(), SIZE);
        reportPrecision("V2 precision", h_ref.data(), h_out.data(), SIZE);

        launch_softmax_v3(d_in, d_out, ROWS, COLS);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, NBytes, cudaMemcpyDeviceToHost));
        std::cout << "\nCorrectness V3 (stable vs PyTorch):  ";
        checkResult(h_ref.data(), h_out.data(), SIZE);
        reportPrecision("V3 precision", h_ref.data(), h_out.data(), SIZE);

        launch_softmax_v4(d_in, d_out, ROWS, COLS);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, NBytes, cudaMemcpyDeviceToHost));
        std::cout << "\nCorrectness V4 (online vs PyTorch):  ";
        checkResult(h_ref.data(), h_out.data(), SIZE);
        reportPrecision("V4 precision", h_ref.data(), h_out.data(), SIZE);

        launch_softmax_v5(d_in, d_out, ROWS, COLS);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, NBytes, cudaMemcpyDeviceToHost));
        std::cout << "\nCorrectness V5 (warp vs PyTorch):    ";
        checkResult(h_ref.data(), h_out.data(), SIZE);
        reportPrecision("V5 precision", h_ref.data(), h_out.data(), SIZE);

        launch_softmax_v6(d_in, d_out, ROWS, COLS);
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, NBytes, cudaMemcpyDeviceToHost));
        std::cout << "\nCorrectness V6 (vec4 vs PyTorch):    ";
        checkResult(h_ref.data(), h_out.data(), SIZE);
        reportPrecision("V6 precision", h_ref.data(), h_out.data(), SIZE);
    }



    //* ── Benchmarks ───────────────────────────────────────────────────────
    //* FLOPs: 5 per element (2 exp + 1 add in sum pass, 1 exp + 1 div in output pass)
    //* Bandwidth: 2 reads + 1 write = 3× NBytes (all kernels here are two-pass).
    //* Triton's kernel is single-pass (1 read + 1 write = 2×), so GB/s numbers
    //* in baseline_softmax.py are NOT directly comparable — compare time (ms) instead.
    long long flops = 5LL * (long long)SIZE;
    size_t    bytes = 3 * NBytes;

    // KernelStats stats_v1 = benchmarkKernel(
    //     [&](){ launch_softmax_v1(d_in, d_out, ROWS, COLS); },
    //     100, 25, flops, bytes
    // );
    // displayStats("V1 — Serial (one thread per row)", stats_v1);

    // KernelStats stats_v2 = benchmarkKernel(
    //     [&](){ launch_softmax_v2(d_in, d_out, ROWS, COLS); },
    //     100, 25, flops, bytes
    // );
    // displayStats("V2 — AtomicAdd (one block per row)", stats_v2);

    // KernelStats stats_v3 = benchmarkKernel(
    //     [&](){ launch_softmax_v3(d_in, d_out, ROWS, COLS); },
    //     100, 25, flops, bytes
    // );
    // displayStats("V3 — AtomicAdd (one block per row) & stable", stats_v3);

    KernelStats stats_v4 = benchmarkKernel(
        [&](){ launch_softmax_v4(d_in, d_out, ROWS, COLS); },
        100, 25, flops, bytes
    );
    displayStats("V4 — Online (one block per row)", stats_v4);

    // KernelStats stats_v5 = benchmarkKernel(
    //     [&](){ launch_softmax_v5(d_in, d_out, ROWS, COLS); },
    //     100, 25, flops, bytes
    // );
    // displayStats("V5 — Online Warp (one warp per row)", stats_v5);

    KernelStats stats_v6 = benchmarkKernel(
        [&](){ launch_softmax_v6(d_in, d_out, ROWS, COLS); },
        100, 25, flops, bytes
    );
    displayStats("V6 — Vec4 (float4 loads + stores)", stats_v6);


    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}
