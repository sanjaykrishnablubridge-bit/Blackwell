#include <cuda_runtime.h>
#include "utils/kernelBench.cuh"
#include <chrono>
#include <random>

//* ============================
//* Kernel Implementations
//* ============================

//* V1 — naive, one thread per row
//* Each thread serially computes mean, variance, and the normalised output
//* for its assigned row.  Three passes over the row data — heavily memory bound.
__global__ void layernorm_v1_naive(
  const float *inp,
  float *out,
  const float *weight,
  const float* bias,
  float *mean,
  float *rstd,
  int N, // rows
  int C // cols
){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float eps = 1.0e-5f;

  if(idx < N){
    const float *x = inp + idx * C;

    float m = 0.0f;
    for(int i = 0; i < C; ++i) m += x[i];
    m /= C;

    float v = 0.0f;
    for(int i = 0; i < C; ++i){
      float diff = x[i] - m;
      v += diff * diff;
    }
    v /= C;

    float r = 1.0f / sqrtf(v + eps);  // sqrtf — stay in float

    float *y = out + idx * C;
    for(int i = 0; i < C; ++i)
      y[i] = (x[i] - m) * r * weight[i] + bias[i];

    mean[idx] = m;
    rstd[idx] = r;
  }
}

//* V2 — one block per row, threads cooperate via warp + block reductions
__global__ void layernorm_v2_block(
  const float *inp,
  float *out,
  const float *weight,
  const float* bias,
  float *mean,
  float *rstd,
  int N, // rows
  int C // cols
){
  uint idx = threadIdx.x;
  uint row = blockIdx.x;

  float eps = 1.0e-5f;
  if(row >= (uint)N) return;

  uint stride   = blockDim.x;
  uint nwarps   = blockDim.x / 32;

  __shared__ float s_mean[8];   // warp partial sums  (max 8 warps at 256 threads)
  __shared__ float s_var[8];    // warp partial variances
  __shared__ float s_m;         // broadcast mean
  __shared__ float s_r;         // broadcast rstd

  float sum = 0.0f;
  for(uint i = idx; i < C; i += stride)
    sum += inp[row * C + i];

  for(int offset = 16; offset > 0; offset >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, offset);

  if(idx % 32 == 0)
    s_mean[idx / 32] = sum;    
  __syncthreads();             

  if(idx == 0){                
    float blk_sum = 0.0f;
    for(uint i = 0; i < nwarps; ++i)
      blk_sum += s_mean[i];
    s_m = blk_sum / C;    
  }
  __syncthreads();

  float m = s_m;  

  float var = 0.0f;
  for(uint i = idx; i < C; i += stride){
    float delta = inp[row * C + i] - m;
    var += delta * delta;
  }

  for(int offset = 16; offset > 0; offset >>= 1)
    var += __shfl_down_sync(0xffffffff, var, offset);

  if(idx % 32 == 0)
    s_var[idx / 32] = var;
  __syncthreads();

  if(idx == 0){
    float blk_var = 0.0f;
    for(uint i = 0; i < nwarps; ++i)
      blk_var += s_var[i];
    s_r = 1.0f / sqrtf(blk_var / C + eps);
  }
  __syncthreads();

  float r = s_r;

  for(uint i = idx; i < C; i += stride)
    out[row * C + i] = (inp[row * C + i] - m) * r * weight[i] + bias[i];
  
  if(idx == 0){    
    mean[row] = m;
    rstd[row] = r;
  }
}

//* V5 — V2 with float4 vectorised loads/stores
//* Each thread issues one 128-bit LDG per loop iteration instead of four
//* separate 32-bit loads, cutting instruction count and memory-pipe pressure
//* by 4×.  Requires C % 4 == 0 (true for all standard hidden dims).
//* Structure is identical to V2; only the load/store loops change.
__global__ void layernorm_v5_vec4(
  const float * __restrict__ inp,
  float       * __restrict__ out,
  const float * __restrict__ weight,
  const float * __restrict__ bias,
  float *mean,
  float *rstd,
  int N,
  int C
){
  uint idx  = threadIdx.x;
  uint row  = blockIdx.x;
  float eps = 1.0e-5f;

  if(row >= (uint)N) return;

  uint stride    = blockDim.x;
  uint nwarps    = blockDim.x / 32;
  int  vec_cols  = C / 4;          // number of float4 elements per row

  __shared__ float s_mean[8];
  __shared__ float s_var[8];
  __shared__ float s_m;
  __shared__ float s_r;

  const float4 *inp4 = reinterpret_cast<const float4*>(inp + row * C);
  float4       *out4 = reinterpret_cast<float4*>(out + row * C);

  // ── pass 1: mean — 128-bit loads ─────────────────────────────────────────
  float sum = 0.0f;
  for(int i = idx; i < vec_cols; i += stride){
    float4 v = inp4[i];
    sum += v.x + v.y + v.z + v.w;
  }

  for(int offset = 16; offset > 0; offset >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, offset);

  if(idx % 32 == 0) s_mean[idx / 32] = sum;
  __syncthreads();

  if(idx == 0){
    float blk_sum = 0.0f;
    for(uint i = 0; i < nwarps; ++i) blk_sum += s_mean[i];
    s_m = blk_sum / C;
  }
  __syncthreads();

  float m = s_m;

  // ── pass 2: variance — 128-bit loads ─────────────────────────────────────
  float var = 0.0f;
  for(int i = idx; i < vec_cols; i += stride){
    float4 v = inp4[i];
    float dx = v.x - m, dy = v.y - m, dz = v.z - m, dw = v.w - m;
    var += dx*dx + dy*dy + dz*dz + dw*dw;
  }

  for(int offset = 16; offset > 0; offset >>= 1)
    var += __shfl_down_sync(0xffffffff, var, offset);

  if(idx % 32 == 0) s_var[idx / 32] = var;
  __syncthreads();

  if(idx == 0){
    float blk_var = 0.0f;
    for(uint i = 0; i < nwarps; ++i) blk_var += s_var[i];
    s_r = 1.0f / sqrtf(blk_var / C + eps);
  }
  __syncthreads();

  float r = s_r;

  // ── pass 3: output — 128-bit loads + stores ───────────────────────────────
  const float4 *w4 = reinterpret_cast<const float4*>(weight);
  const float4 *b4 = reinterpret_cast<const float4*>(bias);

  for(int i = idx; i < vec_cols; i += stride){
    float4 x = inp4[i];
    float4 w = w4[i];
    float4 b = b4[i];
    float4 o;
    o.x = (x.x - m) * r * w.x + b.x;
    o.y = (x.y - m) * r * w.y + b.y;
    o.z = (x.z - m) * r * w.z + b.z;
    o.w = (x.w - m) * r * w.w + b.w;
    out4[i] = o;
  }

  if(idx == 0){
    mean[row] = m;
    rstd[row] = r;
  }
}


//* V3 — Welford online algorithm, single HBM pass
//* Each thread runs serial Welford on its strided slice → local (mean, M2, count).
//* Local states are merged across the warp with the parallel Welford merge formula,
//* then across the block via shared memory.  inp is read only once from HBM.
__global__ void layernorm_v3_welford(
  const float *inp,
  float *out,
  const float *weight,
  const float* bias,
  float *mean,
  float *rstd,
  int N,
  int C
){
  uint idx  = threadIdx.x;
  uint row  = blockIdx.x;
  float eps = 1.0e-5f;

  if(row >= (uint)N) return;

  uint stride = blockDim.x;
  uint nwarps = blockDim.x / 32;

  // BUG 2 fixed: shared memory cannot be initialised inline; use 3 arrays for
  // per-warp (mean, M2, count) so thread 0 can merge them after the warp reduce
  __shared__ float s_mean[8];
  __shared__ float s_M2[8];
  __shared__ int   s_count[8];
  __shared__ float s_m;   // broadcast final mean
  __shared__ float s_r;   // broadcast final rstd

  // ── step 1: each thread runs serial Welford on its strided slice ──────────
  // BUG 3 fixed: count is a separate integer starting at 1, never divides by i
  // BUG 4 fixed: local state only — no shared writes inside this loop
  float t_mean  = 0.0f;
  float t_M2    = 0.0f;
  int   t_count = 0;
  for(uint i = idx; i < C; i += stride){
    float x     = inp[row * C + i];
    t_count    += 1;
    float delta = x - t_mean;
    t_mean     += delta / t_count;        // Welford mean update
    float delta2 = x - t_mean;
    t_M2       += delta * delta2;         // Welford M2 update
  }

  // ── step 2: warp-level parallel Welford merge ─────────────────────────────
  // Parallel Welford merge of (mean_a, M2_a, n_a) + (mean_b, M2_b, n_b):
  //   n   = n_a + n_b
  //   d   = mean_b - mean_a
  //   mean = mean_a + d * n_b / n
  //   M2  = M2_a + M2_b + d*d * n_a*n_b / n
  for(int offset = 16; offset > 0; offset >>= 1){
    float o_mean  = __shfl_down_sync(0xffffffff, t_mean,  offset);
    float o_M2    = __shfl_down_sync(0xffffffff, t_M2,    offset);
    int   o_count = __shfl_down_sync(0xffffffff, t_count, offset);

    int combined = t_count + o_count;
    if(combined > 0){
      float delta = o_mean - t_mean;
      t_mean  += delta * o_count / combined;
      t_M2    += o_M2 + delta * delta * (float)t_count * o_count / combined;
      t_count  = combined;
    }
  }

  // lane 0 of each warp writes its merged state
  if(idx % 32 == 0){
    s_mean[idx / 32]  = t_mean;
    s_M2[idx / 32]    = t_M2;
    s_count[idx / 32] = t_count;
  }
  __syncthreads();

  // ── step 3: thread 0 merges warp states and broadcasts ───────────────────
  if(idx == 0){
    float b_mean  = s_mean[0];
    float b_M2    = s_M2[0];
    int   b_count = s_count[0];
    for(uint i = 1; i < nwarps; ++i){
      float o_mean  = s_mean[i];
      float o_M2    = s_M2[i];
      int   o_count = s_count[i];
      int   combined = b_count + o_count;
      float delta    = o_mean - b_mean;
      b_mean  += delta * o_count / combined;
      b_M2    += o_M2 + delta * delta * (float)b_count * o_count / combined;
      b_count  = combined;
    }
    s_m = b_mean;
    s_r = 1.0f / sqrtf(b_M2 / C + eps);
  }
  __syncthreads();

  float m = s_m;
  float r = s_r;

  // ── step 4: output pass ───────────────────────────────────────────────────
  // BUG 5 fixed: (x - m) * r        BUG 6 fixed: weight[i] / bias[i] (1-D arrays)
  for(uint i = idx; i < C; i += stride)
    out[row * C + i] = (inp[row * C + i] - m) * r * weight[i] + bias[i];

  if(idx == 0){
    mean[row] = m;
    rstd[row] = r;
  }
}

//* V4 — shared-memory row cache: 1 HBM read + 1 HBM write
//*
//* V2 and V3 still touch HBM 3× because the output pass re-reads inp from HBM.
//* Triton avoids this by keeping x in registers — we do the same in CUDA by
//* loading the entire row into shared memory first, then all three passes
//* (mean, variance, output) read from smem, which is on-chip.
//*
//* smem layout (all floats, carved from one extern __shared__ allocation):
//*   [0 .. C-1]   s_inp   — the cached row
//*   [C .. C+7]   s_warp  — 8 warp-reduction scratch slots (reused for mean & var)
//*   [C+8]        s_m     — broadcast mean
//*   [C+9]        s_r     — broadcast rstd
//* Total: (C + 10) * sizeof(float) per block
__global__ void layernorm_v4_smem(
  const float * __restrict__ inp,
  float       * __restrict__ out,
  const float * __restrict__ weight,
  const float * __restrict__ bias,
  float *mean,
  float *rstd,
  int N,
  int C
){
  uint idx  = threadIdx.x;
  uint row  = blockIdx.x;
  float eps = 1.0e-5f;

  if(row >= (uint)N) return;

  uint stride = blockDim.x;
  uint nwarps = blockDim.x / 32;

  extern __shared__ float smem[];
  float *s_inp  = smem;          // C floats — cached row
  float *s_warp = smem + C;      // 8 floats — warp scratch (reused twice)
  float *s_m    = smem + C + 8;  // 1 float  — broadcast mean
  float *s_r    = smem + C + 9;  // 1 float  — broadcast rstd

  // ── step 1: load row into shared memory (single HBM read) ─────────────────
  for(uint i = idx; i < C; i += stride)
    s_inp[i] = inp[row * C + i];
  __syncthreads();

  // ── step 2: mean — warp + block reduction over s_inp ──────────────────────
  float sum = 0.0f;
  for(uint i = idx; i < C; i += stride)
    sum += s_inp[i];

  for(int offset = 16; offset > 0; offset >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, offset);

  if(idx % 32 == 0) s_warp[idx / 32] = sum;
  __syncthreads();

  if(idx == 0){
    float blk_sum = 0.0f;
    for(uint i = 0; i < nwarps; ++i) blk_sum += s_warp[i];
    *s_m = blk_sum / C;
  }
  __syncthreads();

  float m = *s_m;

  // ── step 3: variance — warp + block reduction over s_inp ──────────────────
  float var = 0.0f;
  for(uint i = idx; i < C; i += stride){
    float d = s_inp[i] - m;      // reads smem, not HBM
    var += d * d;
  }

  for(int offset = 16; offset > 0; offset >>= 1)
    var += __shfl_down_sync(0xffffffff, var, offset);

  if(idx % 32 == 0) s_warp[idx / 32] = var;
  __syncthreads();

  if(idx == 0){
    float blk_var = 0.0f;
    for(uint i = 0; i < nwarps; ++i) blk_var += s_warp[i];
    *s_r = 1.0f / sqrtf(blk_var / C + eps);
  }
  __syncthreads();

  float r = *s_r;

  // ── step 4: output — reads smem, writes HBM once ──────────────────────────
  for(uint i = idx; i < C; i += stride)
    out[row * C + i] = (s_inp[i] - m) * r * weight[i] + bias[i];

  if(idx == 0){
    mean[row] = m;
    rstd[row] = r;
  }
}


//* ============================
//* CPU Reference
//* ============================

void layernorm_cpu(
    const float *inp,
    float *out,
    const float *weight,
    const float *bias,
    float *mean,
    float *rstd,
    int N,
    int C
){
    const float eps = 1.0e-5f;
    for(int n = 0; n < N; ++n){
        const float *x = inp + n * C;

        float m = 0.0f;
        for(int i = 0; i < C; ++i) m += x[i];
        m /= C;

        float v = 0.0f;
        for(int i = 0; i < C; ++i){
            float d = x[i] - m;
            v += d * d;
        }
        v /= C;

        float r = 1.0f / sqrtf(v + eps);

        float *y = out + n * C;
        for(int i = 0; i < C; ++i)
            y[i] = (x[i] - m) * r * weight[i] + bias[i];

        mean[n] = m;
        rstd[n] = r;
    }
}


//* ============================
//* Launch Configurations
//* ============================

void launch_layernorm_v1(
    const float *inp, float *out,
    const float *weight, const float *bias,
    float *mean, float *rstd,
    int N, int C
){
    dim3 BLOCK{256};
    dim3 GRID{((uint32_t)N + BLOCK.x - 1) / BLOCK.x};
    layernorm_v1_naive<<<GRID, BLOCK>>>(inp, out, weight, bias, mean, rstd, N, C);
}

void launch_layernorm_v2(
    const float *inp, float *out,
    const float *weight, const float *bias,
    float *mean, float *rstd,
    int N, int C
){
    dim3 BLOCK{256};
    dim3 GRID{(uint32_t)N};
    layernorm_v2_block<<<GRID, BLOCK>>>(inp, out, weight, bias, mean, rstd, N, C);
}

void launch_layernorm_v5(
    const float *inp, float *out,
    const float *weight, const float *bias,
    float *mean, float *rstd,
    int N, int C
){
    dim3 BLOCK{256};
    dim3 GRID{(uint32_t)N};
    layernorm_v5_vec4<<<GRID, BLOCK>>>(inp, out, weight, bias, mean, rstd, N, C);
}

void launch_layernorm_v3(
    const float *inp, float *out,
    const float *weight, const float *bias,
    float *mean, float *rstd,
    int N, int C
){
    dim3 BLOCK{256};
    dim3 GRID{(uint32_t)N};
    layernorm_v3_welford<<<GRID, BLOCK>>>(inp, out, weight, bias, mean, rstd, N, C);
}

void launch_layernorm_v4(
    const float *inp, float *out,
    const float *weight, const float *bias,
    float *mean, float *rstd,
    int N, int C
){
    dim3 BLOCK{256};
    dim3 GRID{(uint32_t)N};
    size_t smem_bytes = (C + 10) * sizeof(float);
    layernorm_v4_smem<<<GRID, BLOCK, smem_bytes>>>(inp, out, weight, bias, mean, rstd, N, C);
}


//* ============================
//* Main
//* ============================

int main(int argc, char *argv[]){
    std::cout << "Benchmarking LayerNorm kernels — Blackwell SM_120\n";

    int N = (argc > 1) ? std::atoi(argv[1]) : 8192;
    int C = (argc > 2) ? std::atoi(argv[2]) : 1024;
    size_t SIZE   = (size_t)N * C;
    size_t NBytes = SIZE * sizeof(float);

    std::vector<float> h_inp(SIZE), h_weight(C), h_bias(C);
    std::vector<float> h_out(SIZE),  h_out_ref(SIZE);
    std::vector<float> h_mean(N),    h_mean_ref(N);
    std::vector<float> h_rstd(N),    h_rstd_ref(N);

    //* ── Load PyTorch reference data (falls back to random if files absent) ──
    auto fileMatchesSize = [](const std::string &p, size_t n_floats) -> bool {
        FILE *f = fopen(p.c_str(), "rb");
        if(!f) return false;
        fseek(f, 0, SEEK_END);
        size_t bytes = (size_t)ftell(f);
        fclose(f);
        return bytes == n_floats * sizeof(float);
    };
    bool has_ref = fileMatchesSize("data/layernorm_inp.bin", SIZE);
    if(has_ref){
        loadBin("data/layernorm_inp.bin",    h_inp.data(),     SIZE);
        loadBin("data/layernorm_weight.bin", h_weight.data(),  (size_t)C);
        loadBin("data/layernorm_bias.bin",   h_bias.data(),    (size_t)C);
        loadBin("data/layernorm_out.bin",    h_out_ref.data(), SIZE);
        loadBin("data/layernorm_mean.bin",   h_mean_ref.data(),(size_t)N);
        loadBin("data/layernorm_rstd.bin",   h_rstd_ref.data(),(size_t)N);
        std::cout << "\nLoaded PyTorch reference from data/layernorm_*.bin\n";
    } else {
        initVec(h_inp);
        initVec(h_weight);
        initVec(h_bias);
        std::cout << "\nNo reference files found — using random data (benchmarks only)\n";
    }

    //* ── Device allocations ───────────────────────────────────────────────
    float *d_inp, *d_out, *d_weight, *d_bias, *d_mean, *d_rstd;
    CUDA_CHECK(cudaMalloc(&d_inp,    NBytes));
    CUDA_CHECK(cudaMalloc(&d_out,    NBytes));
    CUDA_CHECK(cudaMalloc(&d_weight, C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias,   C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mean,   N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rstd,   N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_inp,    h_inp.data(),    NBytes,           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight.data(), C * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias,   h_bias.data(),   C * sizeof(float), cudaMemcpyHostToDevice));

    //* ── Correctness checks ───────────────────────────────────────────────
    if(has_ref){
    launch_layernorm_v1(d_inp, d_out, d_weight, d_bias, d_mean, d_rstd, N, C);
    CUDA_CHECK(cudaMemcpy(h_out.data(),  d_out,  NBytes,           cudaMemcpyDeviceToHost));
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

    launch_layernorm_v2(d_inp, d_out, d_weight, d_bias, d_mean, d_rstd, N, C);
    CUDA_CHECK(cudaMemcpy(h_out.data(),  d_out,  NBytes,            cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_mean.data(), d_mean, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_rstd.data(), d_rstd, N * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "\nCorrectness V2 (block vs PyTorch) — out:   ";
    checkResult(h_out_ref.data(), h_out.data(), SIZE);
    std::cout << "Correctness V2 (block vs PyTorch) — mean:  ";
    checkResult(h_mean_ref.data(), h_mean.data(), N);
    std::cout << "Correctness V2 (block vs PyTorch) — rstd:  ";
    checkResult(h_rstd_ref.data(), h_rstd.data(), N);
    reportPrecision("V1 out  precision", h_out_ref.data(),  h_out.data(),  SIZE);
    reportPrecision("V1 mean precision", h_mean_ref.data(), h_mean.data(), N);
    reportPrecision("V1 rstd precision", h_rstd_ref.data(), h_rstd.data(), N);
    } //* end if(has_ref)

    //* ── Benchmarks ───────────────────────────────────────────────────────
    //* FLOPs per row ≈ 8C  →  8*N*C total
    //* Bandwidth: V2 reads inp 3x (mean, var, output) + 1 write = 4*NBytes dominant
    long long flops = 8LL * N * C;
    size_t    bytes = 4 * NBytes;

    KernelStats stats_v1 = benchmarkKernel(
        [&](){ launch_layernorm_v1(d_inp, d_out, d_weight, d_bias, d_mean, d_rstd, N, C); },
        100, 25, flops, bytes
    );
    displayStats("V1 — Naive (one thread per row)", stats_v1);

    KernelStats stats_v2 = benchmarkKernel(
        [&](){ launch_layernorm_v2(d_inp, d_out, d_weight, d_bias, d_mean, d_rstd, N, C); },
        100, 25, flops, bytes
    );
    displayStats("V2 — Block cooperative (256 threads per row)", stats_v2);

    launch_layernorm_v5(d_inp, d_out, d_weight, d_bias, d_mean, d_rstd, N, C);
    CUDA_CHECK(cudaMemcpy(h_out.data(),  d_out,  NBytes,            cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_mean.data(), d_mean, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_rstd.data(), d_rstd, N * sizeof(float), cudaMemcpyDeviceToHost));

    if(has_ref){
    std::cout << "\nCorrectness V5 (vec4 vs PyTorch) — out:   ";
    checkResult(h_out_ref.data(), h_out.data(), SIZE);
    std::cout << "Correctness V5 (vec4 vs PyTorch) — mean:  ";
    checkResult(h_mean_ref.data(), h_mean.data(), N);
    std::cout << "Correctness V5 (vec4 vs PyTorch) — rstd:  ";
    checkResult(h_rstd_ref.data(), h_rstd.data(), N);
    reportPrecision("V1 out  precision", h_out_ref.data(),  h_out.data(),  SIZE);
    reportPrecision("V1 mean precision", h_mean_ref.data(), h_mean.data(), N);
    reportPrecision("V1 rstd precision", h_rstd_ref.data(), h_rstd.data(), N);
    }

    //* V5 same memory traffic as V2: 3 reads + 1 write = 4*NBytes
    KernelStats stats_v5 = benchmarkKernel(
        [&](){ launch_layernorm_v5(d_inp, d_out, d_weight, d_bias, d_mean, d_rstd, N, C); },
        100, 25, flops, bytes
    );
    displayStats("V5 — Vec4 (float4 loads+stores, V2 structure)", stats_v5);

    launch_layernorm_v3(d_inp, d_out, d_weight, d_bias, d_mean, d_rstd, N, C);
    CUDA_CHECK(cudaMemcpy(h_out.data(),  d_out,  NBytes,            cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_mean.data(), d_mean, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_rstd.data(), d_rstd, N * sizeof(float), cudaMemcpyDeviceToHost));

    if(has_ref){
    std::cout << "\nCorrectness V3 (welford vs PyTorch) — out:   ";
    checkResult(h_out_ref.data(), h_out.data(), SIZE);
    std::cout << "Correctness V3 (welford vs PyTorch) — mean:  ";
    checkResult(h_mean_ref.data(), h_mean.data(), N);
    std::cout << "Correctness V3 (welford vs PyTorch) — rstd:  ";
    checkResult(h_rstd_ref.data(), h_rstd.data(), N);
    reportPrecision("V1 out  precision", h_out_ref.data(),  h_out.data(),  SIZE);
    reportPrecision("V1 mean precision", h_mean_ref.data(), h_mean.data(), N);
    reportPrecision("V1 rstd precision", h_rstd_ref.data(), h_rstd.data(), N);
    }

    //* V3 reads inp once (Welford pass) + once (output pass) = 2*NBytes dominant
    size_t bytes_v3 = 2 * NBytes;
    KernelStats stats_v3 = benchmarkKernel(
        [&](){ launch_layernorm_v3(d_inp, d_out, d_weight, d_bias, d_mean, d_rstd, N, C); },
        100, 25, flops, bytes_v3
    );
    displayStats("V3 — Welford single-pass mean+var (256 threads per row)", stats_v3);

    launch_layernorm_v4(d_inp, d_out, d_weight, d_bias, d_mean, d_rstd, N, C);
    CUDA_CHECK(cudaMemcpy(h_out.data(),  d_out,  NBytes,            cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_mean.data(), d_mean, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_rstd.data(), d_rstd, N * sizeof(float), cudaMemcpyDeviceToHost));

    if(has_ref){
    std::cout << "\nCorrectness V4 (smem vs PyTorch) — out:   ";
    checkResult(h_out_ref.data(), h_out.data(), SIZE);
    std::cout << "Correctness V4 (smem vs PyTorch) — mean:  ";
    checkResult(h_mean_ref.data(), h_mean.data(), N);
    std::cout << "Correctness V4 (smem vs PyTorch) — rstd:  ";
    checkResult(h_rstd_ref.data(), h_rstd.data(), N);
    reportPrecision("V1 out  precision", h_out_ref.data(),  h_out.data(),  SIZE);
    reportPrecision("V1 mean precision", h_mean_ref.data(), h_mean.data(), N);
    reportPrecision("V1 rstd precision", h_rstd_ref.data(), h_rstd.data(), N);
    }

    //* V4: 1 HBM read (inp→smem) + 1 HBM write (out) = 2*NBytes
    size_t bytes_v4 = 2 * NBytes;
    KernelStats stats_v4 = benchmarkKernel(
        [&](){ launch_layernorm_v4(d_inp, d_out, d_weight, d_bias, d_mean, d_rstd, N, C); },
        100, 25, flops, bytes_v4
    );
    displayStats("V4 — Shared-mem row cache (1 HBM read + 1 write)", stats_v4);

    CUDA_CHECK(cudaFree(d_inp));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_mean));
    CUDA_CHECK(cudaFree(d_rstd));

    return 0;
}