# CUDA Kernel Optimization Techniques

## How to identify your bottleneck first

Before optimizing, profile with Nsight Compute and locate yourself on the roofline:

```
Arithmetic Intensity (AI) = FLOPs / Bytes moved

AI < ridge point  →  memory-bound
AI > ridge point  →  compute-bound
low occupancy + high latency stalls  →  latency-bound
```

Ridge point = Peak FLOPS / Peak Bandwidth. Everything below is a bandwidth problem. Everything above is a compute problem. A kernel can move between categories as you optimize it.

---

## Memory-Bound Kernels

The kernel is spending most of its time waiting for data from DRAM. The goal is to move fewer bytes, or move the same bytes faster.

### 1. Coalesced Global Memory Access

All 32 threads in a warp should access consecutive addresses so the hardware merges them into a single wide transaction. Strided or random access issues one transaction per thread.

```cuda
// good — threads 0..31 access consecutive floats → 1 transaction
float x = data[blockDim.x * blockIdx.x + threadIdx.x];

// bad — stride of N means 32 separate transactions
float x = data[threadIdx.x * N];
```

### 2. Vectorized Loads (`float4` / `float2`)

Replace scalar loads with vector loads to increase bytes-per-transaction and reduce instruction count. Requires 16-byte alignment.

```cuda
// scalar: 4 ld.global.f32 instructions
float a = data[i], b = data[i+1], c = data[i+2], d = data[i+3];

// vector: 1 ld.global.v4.f32 instruction, same bytes, 4x fewer instructions
float4 v = reinterpret_cast<float4*>(data)[i / 4];
```

### 3. `__ldg()` — Read-Only Data Cache

Routes loads through the read-only texture cache (`ld.global.nc` in PTX). Physically separate from L1, no coherence overhead. Use whenever the pointer is never written by the same kernel.

```cuda
float x = __ldg(&logits[row * vocab_size + i]);
```

The compiler may infer this from `const __restrict__` on the parameter. Check PTX to verify — if you already see `ld.global.nc`, `__ldg` adds nothing.

```cuda
__global__ void kernel(const float* __restrict__ data, ...) { ... }
```

### 4. Shared Memory Tiling

Load a tile from global memory into shared memory once, then let all threads in the block reuse it. Effective when the same data is read multiple times across threads.

```cuda
__shared__ float tile[TILE];
tile[threadIdx.x] = __ldg(&data[global_idx]);
__syncthreads();
// all threads now read from smem at L1 speed
float val = tile[some_index];
```

### 5. Avoid Shared Memory Bank Conflicts

Shared memory is divided into 32 banks (4-byte interleaved). When multiple threads in a warp access the same bank simultaneously, accesses serialize.

```cuda
// bad — all threads in a warp hit bank 0 (stride = 32 → same bank)
float v = smem[threadIdx.x * 32];

// fix — pad the array so stride does not align to a bank multiple
__shared__ float smem[ROWS][COLS + 1];  // +1 padding breaks alignment
```

### 6. Single-Pass Algorithms

Two passes over global memory costs 2× the bandwidth of one pass. Online algorithms (like online softmax) fuse what used to be separate passes.

```
Two-pass softmax:   pass 1 → find max,  pass 2 → sum exp   (2× BW)
Online softmax:     single pass → track (max, sum) together (1× BW)
```

### 7. L2 Cache Persistence (sm_80+)

Pin a frequently-reused buffer in L2 across kernel launches. Useful for weight matrices or lookup tables accessed by many consecutive kernels.

```cuda
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);

cudaAccessPolicyWindow win;
win.base_ptr   = weights_ptr;
win.num_bytes  = weight_bytes;
win.hitRatio   = 1.0f;
win.hitProp    = cudaAccessPropertyPersisting;
win.missProp   = cudaAccessPropertyStreaming;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &win);
```

### 8. Prefetching / Software Pipelining

Issue the load for the next iteration while computing on the current one. Hides global memory latency by overlapping it with arithmetic.

```cuda
float current = __ldg(&data[i]);
for (int i = idx; i < N; i += stride) {
    float next = __ldg(&data[i + stride]);  // prefetch next
    compute(current);
    current = next;
}
```

### 9. Streaming Loads for Write-Once Data

Mark data that is only read once as streaming so it bypasses L2 and does not evict more reusable data.

```cuda
// PTX level — use ld.global.cs (cache streaming)
// In CUDA C, no direct intrinsic; __ldg with __restrict__ approximates this
```

### 10. Minimize Global Atomic Pressure

Atomics to the same address serialize. Use warp/block-level reductions first, then a single atomic per block.

```cuda
// bad — all 512 threads atomically add to global
atomicAdd(&result, local_val);

// good — reduce within block first, then one atomic per block
local_val = blockReduce(local_val);
if (threadIdx.x == 0) atomicAdd(&result, local_val);
```

---

## Compute-Bound Kernels

The kernel's arithmetic throughput is the bottleneck. The goal is to do more useful work per clock, or reduce the number of instructions needed.

### 1. Fast Math Intrinsics

Device intrinsics map directly to single PTX instructions with ~1–4 cycle throughput. Standard math library functions go through error-correction paths that are 4–20× slower.

| Standard | Intrinsic | PTX instruction |
|---|---|---|
| `expf(x)` | `__expf(x)` | `ex2.approx.f32` |
| `logf(x)` | `__logf(x)` | `lg2.approx.f32` |
| `sinf(x)` | `__sinf(x)` | `sin.approx.f32` |
| `cosf(x)` | `__cosf(x)` | `cos.approx.f32` |
| `1.0f/x` | `__frcp_rn(x)` | `rcp.rn.f32` |
| `sqrtf(x)` | `__fsqrt_rn(x)` | `sqrt.rn.f32` |
| `rsqrtf(x)` | `__frsqrt_rn(x)` | `rsqrt.approx.f32` |

Accuracy trade-off: intrinsics have ~2 ULP error vs IEEE-754 compliance. For ML kernels this is almost always acceptable.

### 2. FMA — Fused Multiply-Add

`a * b + c` compiles to a single FMA instruction when using floats, giving two FLOPs for the price of one instruction and one rounding error instead of two.

```cuda
// compiler will fuse this automatically for floats
float result = a * b + c;

// explicit FMA intrinsic — forces fusion, no contracted expression ambiguity
float result = __fmaf_rn(a, b, c);
```

### 3. Instruction-Level Parallelism (ILP)

The warp scheduler can issue one instruction per clock per warp. If consecutive instructions have data dependencies, the warp stalls. Independent instructions allow the scheduler to fill those stall slots.

```cuda
// bad — chain of dependencies, each waits for previous
float a = data[i];
float b = a * 2.0f;
float c = b + 1.0f;

// good — four independent accumulators, scheduler can interleave them
float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
for (int i = idx; i < N; i += stride * 4) {
    s0 += data[i + 0 * stride];
    s1 += data[i + 1 * stride];
    s2 += data[i + 2 * stride];
    s3 += data[i + 3 * stride];
}
float sum = s0 + s1 + s2 + s3;
```

### 4. Loop Unrolling

Reduces loop overhead (branch, counter increment) and gives the compiler visibility into multiple iterations at once, enabling better instruction scheduling and ILP extraction.

```cuda
#pragma unroll 4
for (uint i = idx; i < vocab_size; i += stride) {
    // compiler emits 4 copies of the body, merges loop overhead
}

#pragma unroll   // fully unroll if trip count is known at compile time
for (int i = 0; i < 5; i++) { ... }
```

### 5. Mixed Precision (FP16 / BF16)

FP16/BF16 operations have 2× the throughput of FP32 on most modern SMs, and half the memory bandwidth cost.

```cuda
#include <cuda_fp16.h>
__half a = __float2half(1.0f);
__half b = __float2half(2.0f);
__half c = __hadd(a, b);        // FP16 add
__half d = __hmul(a, b);        // FP16 multiply
```

For entire tensor operations, use Tensor Cores via WMMA or CUTLASS which provide 8–16× FP16 throughput over scalar FP32.

### 6. Tensor Cores (WMMA API)

Tensor Cores perform D = A × B + C on small matrix fragments in a single instruction. Available from Volta (sm_70) onwards.

```cuda
#include <mma.h>
using namespace nvcuda::wmma;

fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
fragment<accumulator, 16, 16, 16, float> c_frag;

load_matrix_sync(a_frag, a_ptr, 16);
load_matrix_sync(b_frag, b_ptr, 16);
fill_fragment(c_frag, 0.0f);
mma_sync(c_frag, a_frag, b_frag, c_frag);
store_matrix_sync(c_ptr, c_frag, 16, mem_row_major);
```

### 7. Avoid Integer Division and Modulo

Integer division compiles to a multi-instruction sequence (~20 cycles). Replace with bit operations where the divisor is a power of two.

```cuda
// bad — compiler emits integer divide (~20 cycles)
int warp_id = threadIdx.x / 32;
int lane_id = threadIdx.x % 32;

// good — bit ops (1 cycle each)
int warp_id = threadIdx.x >> 5;
int lane_id = threadIdx.x & 31;
```

### 8. Strength Reduction

Replace expensive operations with cheaper equivalents that produce the same result.

```cuda
// expensive
float y = powf(x, 2.0f);   // general power function

// cheap
float y = x * x;            // one multiply

// expensive
float y = expf(x * logf(2.0f));   // exp(x * ln2)

// cheap — __exp2f maps to ex2.approx directly
float y = __exp2f(x);
```

### 9. Warp-Level Primitives

Warp shuffle operations move data between threads in a warp with zero latency and no shared memory. Use them to replace shared-memory reductions inside a warp.

```cuda
// sum reduction within a warp — no shared memory, no __syncthreads
for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xffffffff, val, offset);
// lane 0 now holds the sum of all 32 lanes

// broadcast from lane 0 to all lanes
val = __shfl_sync(0xffffffff, val, 0);

// XOR butterfly — useful for prefix scans
val = __shfl_xor_sync(0xffffffff, val, offset);
```

### 10. Compiler Flags

```bash
--use_fast_math          # enables __expf, __sinf, etc. for all math calls automatically
-O3                      # maximum optimization
--ftz=true               # flush denormals to zero (avoids slow denormal handling)
--prec-div=false         # use fast approximate division
--prec-sqrt=false        # use fast approximate sqrt
```

---

## Latency-Bound Kernels

The kernel has enough arithmetic to do but threads are stalling waiting for instructions to complete (memory latency, dependency chains, synchronization). The goal is to hide latency by keeping the warp scheduler busy.

### 1. Increase Occupancy

Occupancy = active warps / max warps per SM. Higher occupancy gives the scheduler more warps to switch to while waiting for a stall to resolve.

**What limits occupancy:**
- Register usage per thread (most common)
- Shared memory per block
- Block size too small (too few threads → too few warps)

```cuda
// query theoretical occupancy for a given kernel + block size
int min_grid, block_size;
cudaOccupancyMaxPotentialBlockSize(&min_grid, &block_size, my_kernel, 0, 0);

// limit register usage
__launch_bounds__(256, 4)   // max 256 threads/block, min 4 blocks/SM
__global__ void kernel(...) { ... }
```

### 2. Minimize `__syncthreads()`

Every `__syncthreads()` stalls all warps in the block. Reduce their frequency by restructuring data flow or using warp-level primitives which synchronize implicitly within a warp.

```cuda
// bad — sync after every stage
load_tile();  __syncthreads();
compute_1();  __syncthreads();
compute_2();  __syncthreads();
write_tile(); __syncthreads();

// better — merge stages so fewer syncs are needed
load_and_compute();  __syncthreads();  // one sync instead of four
write_tile();
```

### 3. Double Buffering in Shared Memory

Overlap the load of the next tile with computation on the current tile. Requires two shared memory buffers and careful sync placement.

```cuda
__shared__ float buf[2][TILE];

// load tile 0 to buf[0]
buf[0][threadIdx.x] = __ldg(&data[0 + threadIdx.x]);
__syncthreads();

int cur = 0, nxt = 1;
for (int tile = 1; tile < num_tiles; tile++) {
    // async prefetch next tile into buf[nxt]
    buf[nxt][threadIdx.x] = __ldg(&data[tile * TILE + threadIdx.x]);
    compute(buf[cur]);    // compute on current tile while next loads
    __syncthreads();
    cur ^= 1; nxt ^= 1;
}
compute(buf[cur]);
```

### 4. `cuda::pipeline` / `cp.async` (sm_80+)

Asynchronous copy from global to shared memory without stalling the warp. The warp continues executing while the copy happens in the background.

```cuda
#include <cuda/pipeline>

__shared__ float smem[TILE];
cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

// issue async copy — warp does not stall
cuda::memcpy_async(smem + threadIdx.x,
                   data  + global_idx,
                   sizeof(float), pipe);

// do other work here while copy is in flight
other_computation();

// wait for the copy to complete
pipe.consumer_wait();
__syncthreads();

// now safe to use smem
```

### 5. Persistent Kernels

Launch exactly `SM_count × target_occupancy` blocks and have each block loop over all tiles internally. Eliminates kernel launch overhead and wave quantization tail effects.

```cuda
__global__ void persistent_kernel(float* data, int num_tiles) {
    // each block claims tiles via atomic counter
    __shared__ int tile_idx;
    while (true) {
        if (threadIdx.x == 0)
            tile_idx = atomicAdd(&global_tile_counter, 1);
        __syncthreads();
        if (tile_idx >= num_tiles) break;
        process_tile(data, tile_idx);
    }
}

// launch exactly enough blocks to fill the GPU
int blocks = sm_count * target_blocks_per_sm;
persistent_kernel<<<blocks, BLOCK_SIZE>>>(data, num_tiles);
```

### 6. Warp Divergence Elimination

When threads in the same warp take different branches, the hardware executes both paths serially and masks inactive threads. All threads taking the same path doubles throughput.

```cuda
// bad — half the warp goes each way → 2× cost
if (threadIdx.x % 2 == 0)
    do_even();
else
    do_odd();

// good — both halves are contiguous warps, no intra-warp divergence
if (threadIdx.x < 16)
    do_even();
else
    do_odd();

// best — restructure so the condition is uniform across the warp
```

### 7. Register Pressure Management

Too many live registers per thread reduces the number of warps that can be resident on an SM, killing occupancy and latency hiding.

```cuda
// check register usage
nvcc -Xptxas -v mykernel.cu
// output: "registers: 64" — if > 32, occupancy starts dropping on most SMs

// force limit — spills to local memory if needed (local memory = global speed)
__launch_bounds__(128, 8)   // tells compiler to target 8 blocks/SM
__global__ void kernel(...) { ... }
```

### 8. Reduce Synchronization Scope

Use the narrowest synchronization primitive that is correct. Block sync is the most expensive; warp sync and thread sync are cheaper.

```cuda
__syncthreads();              // entire block — all warps stall
__syncwarp(0xffffffff);       // single warp only — others continue
__threadfence_block();        // memory fence without execution sync
```

### 9. Instruction Mix Balance

The SM has separate pipelines for different instruction types (FP32 ALU, SFU, load/store, integer). An imbalanced mix leaves some pipelines idle while others are saturated.

```
If __expf calls dominate: SFU is the bottleneck → reduce exp calls or overlap with ALU work
If loads dominate:        LSU is the bottleneck → you're memory-bound, not latency-bound
If integer indexing dominates: INT pipe is the bottleneck → precompute indices, use pointer arithmetic
```

### 10. Overlap Compute and Memory Transfers (Streams)

Use CUDA streams to overlap H2D transfers, kernel execution, and D2H transfers across different batches.

```cuda
for (int i = 0; i < num_chunks; i++) {
    int s = i % 2;   // ping-pong between two streams
    cudaMemcpyAsync(d_in[s],  h_in  + i * chunk, bytes, H2D, streams[s]);
    kernel<<<grid, block, 0, streams[s]>>>(d_in[s], d_out[s]);
    cudaMemcpyAsync(h_out + i * chunk, d_out[s], bytes, D2H, streams[s]);
}
```

---

## Quick Reference: Which technique targets which bottleneck

| Technique | Memory-Bound | Compute-Bound | Latency-Bound |
|---|:---:|:---:|:---:|
| Coalesced access | ✓ | | |
| `float4` vectorized loads | ✓ | | |
| `__ldg` / read-only cache | ✓ | | |
| Shared memory tiling | ✓ | | ✓ |
| Single-pass algorithms | ✓ | | |
| L2 persistence | ✓ | | |
| Prefetching / `cp.async` | ✓ | | ✓ |
| `__expf` / fast intrinsics | | ✓ | |
| FMA | | ✓ | |
| ILP / multiple accumulators | | ✓ | ✓ |
| `#pragma unroll` | ✓ | ✓ | |
| FP16 / BF16 | ✓ | ✓ | |
| Tensor Cores | | ✓ | |
| Avoid int divide | | ✓ | |
| Warp shuffle primitives | | ✓ | ✓ |
| Increase occupancy | | | ✓ |
| Reduce `__syncthreads` | | | ✓ |
| Double buffering | | | ✓ |
| Persistent kernels | | | ✓ |
| Eliminate warp divergence | | ✓ | ✓ |
| Register pressure control | | | ✓ |
| CUDA streams overlap | ✓ | | ✓ |

---

## The optimization loop (applies to all three categories)

```
1. Profile with Nsight Compute → identify bottleneck category
2. Pick the highest-impact technique from the relevant section above
3. Implement as a new kernel version (don't modify the previous one)
4. Benchmark → measure the delta
5. Check if the bottleneck category shifted (common after a big win)
6. Repeat
```

Never guess. Measure first, optimize second, measure again.
