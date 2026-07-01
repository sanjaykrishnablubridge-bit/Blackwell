#include <cuda_runtime.h>
#include "utils/kernelBench.cuh"
#include <cfloat>
#include <algorithm>
#include <chrono>
#include <random>

//* ============================
//* Understanding
//* ============================
//* For one sample i with logits[vocab_size] and target index t:
//* float max_val = -INF;
//* for (int j = 0; j < vocab_size; j++)
//*     max_val = max(max_val, logits[j]);  -> Step 1: find max for stability

//* float sum_exp = 0.0f;
//* for (int j = 0; j < vocab_size; j++)
//*     sum_exp += expf(logits[j] - max_val);  -> Step 2: sum of exp(x - max)

//* float loss = logf(sum_exp) + max_val - logits[t];  -> Step 3: compute loss

//* loss = -log(softmax(logits)[target])
//*      = -(logits[target] - log(Σ exp(logits)))
//*      = log(Σ exp(logits)) - logits[target]

//* Stable form (subtract max before exp):
//*      = log(Σ exp(logits - max)) + max - logits[target]
//* ============================


//* ============================
//* Helpers
//* ============================
__device__ inline float warpReduceMax(float val){
    for(uint offset = 16; offset > 0; offset >>= 1){
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ inline float warpReduceSum(float val){
    for(uint offset = 16; offset > 0; offset >>= 1){
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

//* ============================
//* Kernel Implementations
//* ============================

//* V1 — one thread per sample, fully serial within the row
//* Bottleneck: each thread does all vocab_size work alone — massively under-utilizes GPU
__global__ void sparceCE_v1_serial(
    const float* logits,
    const float* targets,
    float* losses,
    int64_t batch_size,
    int64_t vocab_size
){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row >= batch_size) return;

    float max_val = -FLT_MAX;
    for(int64_t j = 0; j < vocab_size; j++)
        max_val = fmaxf(max_val, logits[row * vocab_size + j]);

    float sum_exp = 0.0f;
    for(int64_t j = 0; j < vocab_size; j++)
        sum_exp += expf(logits[row * vocab_size + j] - max_val);

    int t = (int)targets[row];
    losses[row] = logf(sum_exp) + max_val - logits[row * vocab_size + t];
}

//* V2 — one block per sample, threads stripe over vocab with warp reductions
//* Bottleneck: two global passes over logits, shared memory overhead
__global__ void sparceCE_v2_block(
    const float* logits,
    const float* targets,
    float* losses,
    int64_t batch_size,
    int64_t vocab_size
){
    uint idx = threadIdx.x;
    uint row = blockIdx.x;

    if(row >= (uint)batch_size) return;

    __shared__ float warpMax[8];
    __shared__ float warpSum[8];
    __shared__ float s_max;
    __shared__ float s_sum_exp;
    if(idx == 0){
        s_max = -FLT_MAX;
        s_sum_exp = 0.0f;
    }
    __syncthreads();

    //* Pass 1: find row max
    float current_max = -FLT_MAX;
    uint stride = blockDim.x;
    for(uint i = idx; i < vocab_size; i += stride){
        current_max = fmaxf(current_max, logits[row * vocab_size + i]);
    }

    float warp_max = warpReduceMax(current_max);
    if(idx % 32 == 0) warpMax[idx / 32] = warp_max;
    __syncthreads();

    if(idx == 0){
        float block_max = warpMax[0];
        for(uint i = 1; i < blockDim.x / 32; ++i){
            block_max = fmaxf(block_max, warpMax[i]);
        }
        s_max = block_max;
    }
    __syncthreads();

    //* Pass 2: sum exp(logit - max)
    float local_sum = 0.0f;
    for(uint i = idx; i < vocab_size; i += stride){
        local_sum += expf(logits[row * vocab_size + i] - s_max);
    }

    float warp_sum = warpReduceSum(local_sum);
    if(idx % 32 == 0) warpSum[idx / 32] = warp_sum;
    __syncthreads();

    if(idx == 0){
        float block_sum = 0.0f;
        for(uint i = 0; i < blockDim.x / 32; ++i){
            block_sum += warpSum[i];
        }
        s_sum_exp = block_sum;
    }
    __syncthreads();

    //* One loss value per sample
    if(idx == 0){
        int t = (int)targets[row];
        losses[row] = logf(s_sum_exp) + s_max - logits[row * vocab_size + t];
    }
}

//* V3 — online softmax: single pass, (max, sum) updated together per element
//* Eliminates the separate max pass from V2 — reads logits once instead of twice
__global__ void sparseCE_v3_online(
    const float *logits,
    const float* targets,
    float *losses,
    int64_t batch_size,
    int64_t vocab_size
){
    uint idx = threadIdx.x;
    uint row = blockIdx.x;

    if(row >= (uint)batch_size) return;

    __shared__ float warpMax[16];
    __shared__ float warpSum[16];
    __shared__ float s_max;
    __shared__ float s_sum;
    if(idx == 0){
        s_max = -FLT_MAX;
        s_sum = 0.0f;
    }
    __syncthreads();

    float thread_max = -FLT_MAX;
    float thread_sum = 0.0f;
    uint stride = blockDim.x;
    for(uint i = idx; i < vocab_size; i += stride){
        float x = logits[row * vocab_size + i];
        float new_max = fmaxf(thread_max, x);
        thread_sum = thread_sum * expf(thread_max - new_max) + expf(x - new_max);
        thread_max = new_max;
    }

    //* warp-level online merge: (max, sum) pairs must be merged together,
    //* not reduced independently — independent reduction gives wrong sum
    for(int offset = 16; offset > 0; offset >>= 1){
        float other_max = __shfl_down_sync(0xffffffff, thread_max, offset);
        float other_sum = __shfl_down_sync(0xffffffff, thread_sum, offset);
        float new_max = fmaxf(thread_max, other_max);
        thread_sum = thread_sum * expf(thread_max - new_max)
                   + other_sum * expf(other_max - new_max);
        thread_max = new_max;
    }

    if(idx % 32 == 0){
        warpMax[idx / 32] = thread_max;
        warpSum[idx / 32] = thread_sum;
    }
    __syncthreads();

    if(idx == 0){
        float block_max = warpMax[0];
        float block_sum = warpSum[0];
        for(uint i = 1; i < blockDim.x / 32; ++i){
            float new_max = fmaxf(block_max, warpMax[i]);
            block_sum = block_sum * expf(block_max - new_max)
                      + warpSum[i] * expf(warpMax[i] - new_max);
            block_max = new_max;
        }
        s_max = block_max;
        s_sum = block_sum;
    }
    __syncthreads();

    if(idx == 0){
        int t = (int)targets[row];
        losses[row] = logf(s_sum) + s_max - logits[row * vocab_size + t];
    }
}

//* V4 — online softmax with __expf (fast math: maps to ex2.approx in PTX, ~4x faster than expf)
//* Identical to V3 in structure; only exp calls are swapped to the intrinsic
__global__ void sparseCE_v4_fast_exp(
    const float *logits,
    const float* targets,
    float *losses,
    int64_t batch_size,
    int64_t vocab_size
){
    uint idx = threadIdx.x;
    uint row = blockIdx.x;

    if(row >= (uint)batch_size) return;

    __shared__ float warpMax[16];
    __shared__ float warpSum[16];
    __shared__ float s_max;
    __shared__ float s_sum;
    if(idx == 0){
        s_max = -FLT_MAX;
        s_sum = 0.0f;
    }
    __syncthreads();

    float thread_max = -FLT_MAX;
    float thread_sum = 0.0f;
    uint stride = blockDim.x;
    for(uint i = idx; i < vocab_size; i += stride){
        float x = logits[row * vocab_size + i];
        float new_max = fmaxf(thread_max, x);
        thread_sum = thread_sum * __expf(thread_max - new_max) + __expf(x - new_max);
        thread_max = new_max;
    }

    for(int offset = 16; offset > 0; offset >>= 1){
        float other_max = __shfl_down_sync(0xffffffff, thread_max, offset);
        float other_sum = __shfl_down_sync(0xffffffff, thread_sum, offset);
        float new_max = fmaxf(thread_max, other_max);
        thread_sum = thread_sum * __expf(thread_max - new_max)
                   + other_sum * __expf(other_max - new_max);
        thread_max = new_max;
    }

    if(idx % 32 == 0){
        warpMax[idx / 32] = thread_max;
        warpSum[idx / 32] = thread_sum;
    }
    __syncthreads();

    if(idx == 0){
        float block_max = warpMax[0];
        float block_sum = warpSum[0];
        for(uint i = 1; i < blockDim.x / 32; ++i){
            float new_max = fmaxf(block_max, warpMax[i]);
            block_sum = block_sum * __expf(block_max - new_max)
                      + warpSum[i] * __expf(warpMax[i] - new_max);
            block_max = new_max;
        }
        s_max = block_max;
        s_sum = block_sum;
    }
    __syncthreads();

    if(idx == 0){
        int t = (int)targets[row];
        losses[row] = logf(s_sum) + s_max - logits[row * vocab_size + t];
    }
}

//* V5 — online softmax with __expf + #pragma unroll 4 on the main accumulation loop
//* Unrolling 4x amortizes loop overhead and lets the compiler interleave loads/SFU calls
//* across iterations, hiding global-memory latency more aggressively than V4
__global__ void sparseCE_v5_unroll(
    const float *logits,
    const float* targets,
    float *losses,
    int64_t batch_size,
    int64_t vocab_size
){
    uint idx = threadIdx.x;
    uint row = blockIdx.x;

    if(row >= (uint)batch_size) return;

    __shared__ float warpMax[16];
    __shared__ float warpSum[16];
    __shared__ float s_max;
    __shared__ float s_sum;
    if(idx == 0){
        s_max = -FLT_MAX;
        s_sum = 0.0f;
    }
    __syncthreads();

    float thread_max = -FLT_MAX;
    float thread_sum = 0.0f;
    uint stride = blockDim.x;
    #pragma unroll 4
    for(uint i = idx; i < (uint)vocab_size; i += stride){
        float x = logits[row * vocab_size + i];
        float new_max = fmaxf(thread_max, x);
        thread_sum = thread_sum * __expf(thread_max - new_max) + __expf(x - new_max);
        thread_max = new_max;
    }

    for(int offset = 16; offset > 0; offset >>= 1){
        float other_max = __shfl_down_sync(0xffffffff, thread_max, offset);
        float other_sum = __shfl_down_sync(0xffffffff, thread_sum, offset);
        float new_max = fmaxf(thread_max, other_max);
        thread_sum = thread_sum * __expf(thread_max - new_max)
                   + other_sum * __expf(other_max - new_max);
        thread_max = new_max;
    }

    if(idx % 32 == 0){
        warpMax[idx / 32] = thread_max;
        warpSum[idx / 32] = thread_sum;
    }
    __syncthreads();

    if(idx == 0){
        float block_max = warpMax[0];
        float block_sum = warpSum[0];
        for(uint i = 1; i < blockDim.x / 32; ++i){
            float new_max = fmaxf(block_max, warpMax[i]);
            block_sum = block_sum * __expf(block_max - new_max)
                      + warpSum[i] * __expf(warpMax[i] - new_max);
            block_max = new_max;
        }
        s_max = block_max;
        s_sum = block_sum;
    }
    __syncthreads();

    if(idx == 0){
        int t = (int)targets[row];
        losses[row] = logf(s_sum) + s_max - logits[row * vocab_size + t];
    }
}

//* V6 — V5 + replace the serial O(16) cross-warp merge in thread 0 with a parallel shuffle
//* among the 16 warp leaders. All 16 leaders are threads 0-15, which are all in warp 0,
//* so a single __shfl_down_sync with mask 0x0000ffff and 4 steps replaces 15 serial merges.
//* No s_max/s_sum shared scalars needed — thread 0 writes the loss directly from its result.
//* Assumes BLOCK == 512 (16 warps); mask and offset bound are hardcoded to that shape.
__global__ void sparseCE_v6_warp_merge(
    const float *logits,
    const float* targets,
    float *losses,
    int64_t batch_size,
    int64_t vocab_size
){
    uint idx = threadIdx.x;
    uint row = blockIdx.x;

    if(row >= (uint)batch_size) return;

    __shared__ float warpMax[16];
    __shared__ float warpSum[16];

    float thread_max = -FLT_MAX;
    float thread_sum = 0.0f;
    uint stride = blockDim.x;
    #pragma unroll 4
    for(uint i = idx; i < (uint)vocab_size; i += stride){
        float x = logits[row * vocab_size + i];
        float new_max = fmaxf(thread_max, x);
        thread_sum = thread_sum * __expf(thread_max - new_max) + __expf(x - new_max);
        thread_max = new_max;
    }

    for(int offset = 16; offset > 0; offset >>= 1){
        float other_max = __shfl_down_sync(0xffffffff, thread_max, offset);
        float other_sum = __shfl_down_sync(0xffffffff, thread_sum, offset);
        float new_max = fmaxf(thread_max, other_max);
        thread_sum = thread_sum * __expf(thread_max - new_max)
                   + other_sum * __expf(other_max - new_max);
        thread_max = new_max;
    }

    if(idx % 32 == 0){
        warpMax[idx / 32] = thread_max;
        warpSum[idx / 32] = thread_sum;
    }
    __syncthreads();

    if(idx < 16){
        float wm = warpMax[idx];
        float ws = warpSum[idx];
        for(int offset = 8; offset > 0; offset >>= 1){
            float other_max = __shfl_down_sync(0x0000ffff, wm, offset);
            float other_sum = __shfl_down_sync(0x0000ffff, ws, offset);
            float new_max = fmaxf(wm, other_max);
            ws = ws * __expf(wm - new_max) + other_sum * __expf(other_max - new_max);
            wm = new_max;
        }
        if(idx == 0){
            int t = (int)targets[row];
            losses[row] = logf(ws) + wm - logits[row * vocab_size + t];
        }
    }
}

//* V7 — V6 + __ldg() on every logit read to route loads through the read-only data cache
//* (PTX ld.global.nc). The read-only cache is separate from L1 and never needs coherence
//* checks against stores, so the hardware can fill it more aggressively. Logits are never
//* written by this kernel, so nc is always safe here.
__global__ void sparseCE_v7_ldg(
    const float *logits,
    const float* targets,
    float *losses,
    int64_t batch_size,
    int64_t vocab_size
){
    uint idx = threadIdx.x;
    uint row = blockIdx.x;

    if(row >= (uint)batch_size) return;

    __shared__ float warpMax[16];
    __shared__ float warpSum[16];

    float thread_max = -FLT_MAX;
    float thread_sum = 0.0f;
    uint stride = blockDim.x;
    #pragma unroll 4
    for(uint i = idx; i < (uint)vocab_size; i += stride){
        float x = __ldg(&logits[row * vocab_size + i]);
        float new_max = fmaxf(thread_max, x);
        thread_sum = thread_sum * __expf(thread_max - new_max) + __expf(x - new_max);
        thread_max = new_max;
    }

    for(int offset = 16; offset > 0; offset >>= 1){
        float other_max = __shfl_down_sync(0xffffffff, thread_max, offset);
        float other_sum = __shfl_down_sync(0xffffffff, thread_sum, offset);
        float new_max = fmaxf(thread_max, other_max);
        thread_sum = thread_sum * __expf(thread_max - new_max)
                   + other_sum * __expf(other_max - new_max);
        thread_max = new_max;
    }

    if(idx % 32 == 0){
        warpMax[idx / 32] = thread_max;
        warpSum[idx / 32] = thread_sum;
    }
    __syncthreads();

    if(idx < 16){
        float wm = warpMax[idx];
        float ws = warpSum[idx];
        for(int offset = 8; offset > 0; offset >>= 1){
            float other_max = __shfl_down_sync(0x0000ffff, wm, offset);
            float other_sum = __shfl_down_sync(0x0000ffff, ws, offset);
            float new_max = fmaxf(wm, other_max);
            ws = ws * __expf(wm - new_max) + other_sum * __expf(other_max - new_max);
            wm = new_max;
        }
        if(idx == 0){
            int t = (int)targets[row];
            losses[row] = logf(ws) + wm - __ldg(&logits[row * vocab_size + t]);
        }
    }
}

//* ============================
//* CPU References
//* ============================

//* float32 reference — mirrors V1 serial exactly; use to verify logical correctness of V1
void sparseCE_cpu_f32(
    const float* logits,
    const float* targets,
    float* losses,
    int64_t batch_size,
    int64_t vocab_size
){
    for(int64_t b = 0; b < batch_size; b++){
        float max_val = -FLT_MAX;
        for(int64_t j = 0; j < vocab_size; j++)
            max_val = fmaxf(max_val, logits[b * vocab_size + j]);

        float sum_exp = 0.0f;
        for(int64_t j = 0; j < vocab_size; j++)
            sum_exp += expf(logits[b * vocab_size + j] - max_val);

        int t = (int)targets[b];
        losses[b] = logf(sum_exp) + max_val - logits[b * vocab_size + t];
    }
}

//* double reference — higher precision ground truth; use to verify V2+ which reduce
//* in a different order than the sequential CPU loop
void sparseCE_cpu_f64(
    const float* logits,
    const float* targets,
    float* losses,
    int64_t batch_size,
    int64_t vocab_size
){
    for(int64_t b = 0; b < batch_size; b++){
        double max_val = -FLT_MAX;
        for(int64_t j = 0; j < vocab_size; j++)
            max_val = std::max(max_val, (double)logits[b * vocab_size + j]);

        double sum_exp = 0.0;
        for(int64_t j = 0; j < vocab_size; j++)
            sum_exp += exp((double)logits[b * vocab_size + j] - max_val);

        int t = (int)targets[b];
        losses[b] = (float)(log(sum_exp) + max_val - (double)logits[b * vocab_size + t]);
    }
}

//* ============================
//* Kernel Launchers
//* ============================
void launch_sparseCE_v1(
    const float* logits,
    const float* targets,
    float* losses,
    int64_t batch_size,
    int64_t vocab_size
){
    dim3 BLOCK{256};
    dim3 GRID{((uint32_t)batch_size + BLOCK.x - 1) / BLOCK.x};
    sparceCE_v1_serial<<<GRID, BLOCK>>>(logits, targets, losses, batch_size, vocab_size);
}

void launch_sparseCE_v2(
    const float* logits,
    const float* targets,
    float* losses,
    int64_t batch_size,
    int64_t vocab_size
){
    dim3 BLOCK{256};
    dim3 GRID{(uint32_t)batch_size};
    sparceCE_v2_block<<<GRID, BLOCK>>>(logits, targets, losses, batch_size, vocab_size);
}

void launch_sparseCE_v3(
    const float* logits,
    const float* targets,
    float* losses,
    int64_t batch_size,
    int64_t vocab_size
){
    dim3 BLOCK{512};
    dim3 GRID{(uint32_t)batch_size};
    sparseCE_v3_online<<<GRID, BLOCK>>>(logits, targets, losses, batch_size, vocab_size);
}

void launch_sparseCE_v4(
    const float* logits,
    const float* targets,
    float* losses,
    int64_t batch_size,
    int64_t vocab_size
){
    dim3 BLOCK{512};
    dim3 GRID{(uint32_t)batch_size};
    sparseCE_v4_fast_exp<<<GRID, BLOCK>>>(logits, targets, losses, batch_size, vocab_size);
}

void launch_sparseCE_v5(
    const float* logits,
    const float* targets,
    float* losses,
    int64_t batch_size,
    int64_t vocab_size
){
    dim3 BLOCK{512};
    dim3 GRID{(uint32_t)batch_size};
    sparseCE_v5_unroll<<<GRID, BLOCK>>>(logits, targets, losses, batch_size, vocab_size);
}

void launch_sparseCE_v6(
    const float* logits,
    const float* targets,
    float* losses,
    int64_t batch_size,
    int64_t vocab_size
){
    dim3 BLOCK{512};
    dim3 GRID{(uint32_t)batch_size};
    sparseCE_v6_warp_merge<<<GRID, BLOCK>>>(logits, targets, losses, batch_size, vocab_size);
}

void launch_sparseCE_v7(
    const float* logits,
    const float* targets,
    float* losses,
    int64_t batch_size,
    int64_t vocab_size
){
    dim3 BLOCK{512};
    dim3 GRID{(uint32_t)batch_size};
    sparseCE_v7_ldg<<<GRID, BLOCK>>>(logits, targets, losses, batch_size, vocab_size);
}

int main(){
    std::cout << "Benchmarking Sparse Cross-Entropy kernels — Blackwell SM_120\n";

    int64_t BATCH = 8192;
    int64_t VOCAB = 50304;
    size_t logits_elems  = BATCH * VOCAB;
    size_t logits_bytes  = logits_elems * sizeof(float);
    size_t targets_bytes = BATCH * sizeof(float);
    size_t losses_bytes  = BATCH * sizeof(float);

    std::vector<float> h_logits(logits_elems);
    std::vector<float> h_targets(BATCH);
    std::vector<float> h_losses(BATCH);
    std::vector<float> h_ref(BATCH);

    //* ── Load PyTorch reference data (falls back to random + CPU refs if absent) ──
    auto fileMatchesSize = [](const std::string &p, size_t n_floats) -> bool {
        FILE *f = fopen(p.c_str(), "rb");
        if(!f) return false;
        fseek(f, 0, SEEK_END);
        size_t bytes = (size_t)ftell(f);
        fclose(f);
        return bytes == n_floats * sizeof(float);
    };
    bool has_ref = fileMatchesSize("data/sparsece_logits.bin", logits_elems);
    if(has_ref){
        loadBin("data/sparsece_logits.bin",  h_logits.data(),  logits_elems);
        loadBin("data/sparsece_targets.bin", h_targets.data(), (size_t)BATCH);
        loadBin("data/sparsece_losses.bin",  h_ref.data(),     (size_t)BATCH);
        std::cout << "\nLoaded PyTorch reference from data/sparsece_*.bin\n";
    } else {
        initVec(h_logits);
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> idx_dist(0, (int)VOCAB - 1);
        for(int64_t i = 0; i < BATCH; i++)
            h_targets[i] = (float)idx_dist(rng);
        std::cout << "\nNo reference files found — using random data (benchmarks only)\n";
    }

    float *d_logits, *d_targets, *d_losses;
    CUDA_CHECK(cudaMalloc(&d_logits,  logits_bytes));
    CUDA_CHECK(cudaMalloc(&d_targets, targets_bytes));
    CUDA_CHECK(cudaMalloc(&d_losses,  losses_bytes));
    CUDA_CHECK(cudaMemcpy(d_logits,  h_logits.data(),  logits_bytes,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_targets, h_targets.data(), targets_bytes, cudaMemcpyHostToDevice));

    //* ── Correctness checks ───────────────────────────────────────────────
    if(has_ref){
        launch_sparseCE_v1(d_logits, d_targets, d_losses, BATCH, VOCAB);
        CUDA_CHECK(cudaMemcpy(h_losses.data(), d_losses, losses_bytes, cudaMemcpyDeviceToHost));
        std::cout << "Correctness V1 (serial  vs PyTorch): ";
        checkResult(h_ref.data(), h_losses.data(), (size_t)BATCH);

        launch_sparseCE_v2(d_logits, d_targets, d_losses, BATCH, VOCAB);
        CUDA_CHECK(cudaMemcpy(h_losses.data(), d_losses, losses_bytes, cudaMemcpyDeviceToHost));
        std::cout << "Correctness V2 (block   vs PyTorch): ";
        checkResult(h_ref.data(), h_losses.data(), (size_t)BATCH);

        launch_sparseCE_v3(d_logits, d_targets, d_losses, BATCH, VOCAB);
        CUDA_CHECK(cudaMemcpy(h_losses.data(), d_losses, losses_bytes, cudaMemcpyDeviceToHost));
        std::cout << "Correctness V3 (online  vs PyTorch): ";
        checkResult(h_ref.data(), h_losses.data(), (size_t)BATCH);

        launch_sparseCE_v4(d_logits, d_targets, d_losses, BATCH, VOCAB);
        CUDA_CHECK(cudaMemcpy(h_losses.data(), d_losses, losses_bytes, cudaMemcpyDeviceToHost));
        std::cout << "Correctness V4 (fast_exp vs PyTorch):";
        checkResult(h_ref.data(), h_losses.data(), (size_t)BATCH);

        launch_sparseCE_v5(d_logits, d_targets, d_losses, BATCH, VOCAB);
        CUDA_CHECK(cudaMemcpy(h_losses.data(), d_losses, losses_bytes, cudaMemcpyDeviceToHost));
        std::cout << "Correctness V5 (unroll4  vs PyTorch):";
        checkResult(h_ref.data(), h_losses.data(), (size_t)BATCH);

        launch_sparseCE_v6(d_logits, d_targets, d_losses, BATCH, VOCAB);
        CUDA_CHECK(cudaMemcpy(h_losses.data(), d_losses, losses_bytes, cudaMemcpyDeviceToHost));
        std::cout << "Correctness V6 (warp_mrg vs PyTorch):";
        checkResult(h_ref.data(), h_losses.data(), (size_t)BATCH);

        launch_sparseCE_v7(d_logits, d_targets, d_losses, BATCH, VOCAB);
        CUDA_CHECK(cudaMemcpy(h_losses.data(), d_losses, losses_bytes, cudaMemcpyDeviceToHost));
        std::cout << "Correctness V7 (ldg      vs PyTorch):";
        checkResult(h_ref.data(), h_losses.data(), (size_t)BATCH);
    }

    //* ── Benchmarks ───────────────────────────────────────────────────────
    //* FLOPs: ~3 per element for all kernels (fmaxf + expf + add per element)
    long long flops = 3LL * BATCH * VOCAB;

    //* V1/V2: two-pass — reads logits twice
    // size_t bytes_two_pass = 2 * logits_bytes + targets_bytes + losses_bytes;
    //* V3: single-pass — reads logits once
    size_t bytes_one_pass = logits_bytes + targets_bytes + losses_bytes;

    // KernelStats stats_v1 = benchmarkKernel(
    //     [&](){ launch_sparseCE_v1(d_logits, d_targets, d_losses, BATCH, VOCAB); },
    //     100, 25, flops, bytes_two_pass
    // );
    // displayStats("V1 — Serial (one thread per sample)", stats_v1);

    // KernelStats stats_v2 = benchmarkKernel(
    //     [&](){ launch_sparseCE_v2(d_logits, d_targets, d_losses, BATCH, VOCAB); },
    //     100, 25, flops, bytes_two_pass
    // );
    // displayStats("V2 — Block  (one block per sample, warp reductions)", stats_v2);

    // KernelStats stats_v3 = benchmarkKernel(
    //     [&](){ launch_sparseCE_v3(d_logits, d_targets, d_losses, BATCH, VOCAB); },
    //     100, 25, flops, bytes_one_pass
    // );
    // displayStats("V3 — Online (single pass, fused max+sum)", stats_v3);

    // KernelStats stats_v4 = benchmarkKernel(
    //     [&](){ launch_sparseCE_v4(d_logits, d_targets, d_losses, BATCH, VOCAB); },
    //     100, 25, flops, bytes_one_pass
    // );
    // displayStats("V4 — Fast exp (__expf, ex2.approx)", stats_v4);

    KernelStats stats_v5 = benchmarkKernel(
        [&](){ launch_sparseCE_v5(d_logits, d_targets, d_losses, BATCH, VOCAB); },
        100, 25, flops, bytes_one_pass
    );
    displayStats("V5 — Unroll 4 (__expf + #pragma unroll 4)", stats_v5);

    KernelStats stats_v6 = benchmarkKernel(
        [&](){ launch_sparseCE_v6(d_logits, d_targets, d_losses, BATCH, VOCAB); },
        100, 25, flops, bytes_one_pass
    );
    displayStats("V6 — Warp merge (shuffle among 16 leaders)", stats_v6);

    KernelStats stats_v7 = benchmarkKernel(
        [&](){ launch_sparseCE_v7(d_logits, d_targets, d_losses, BATCH, VOCAB); },
        100, 25, flops, bytes_one_pass
    );
    displayStats("V7 — LDG      (__ldg on all logit reads)", stats_v7);

    CUDA_CHECK(cudaFree(d_logits));
    CUDA_CHECK(cudaFree(d_targets));
    CUDA_CHECK(cudaFree(d_losses));

    return 0;
}
