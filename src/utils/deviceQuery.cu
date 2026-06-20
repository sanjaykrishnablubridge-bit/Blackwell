// deviceQueryDeep.cu
// Comprehensive GPU property query for kernel optimization
// Compile: nvcc -o deviceQueryDeep deviceQueryDeep.cu
// Run:     ./deviceQueryDeep

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
static void check(cudaError_t e, const char *file, int line) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error %s:%d — %s\n", file, line, cudaGetErrorString(e));
        exit(1);
    }
}
#define CHECK(x) check((x), __FILE__, __LINE__)

static int getAttr(cudaDeviceAttr attr, int dev) {
    int v = 0;
    cudaDeviceGetAttribute(&v, attr, dev);
    return v;
}

static const char *archName(int major, int minor) {
    if (major == 12 && minor == 0) return "Blackwell (B100/B200)";
    if (major == 10 && minor == 2) return "Blackwell (GB10x)";
    if (major == 9  && minor == 0) return "Hopper (H100/H200)";
    if (major == 8  && minor == 9) return "Ada Lovelace (L40/L4)";
    if (major == 8  && minor == 6) return "Ampere (A10/A30/RTX 30xx)";
    if (major == 8  && minor == 0) return "Ampere (A100)";
    if (major == 7  && minor == 5) return "Turing (T4/RTX 20xx)";
    if (major == 7  && minor == 0) return "Volta (V100)";
    return "Unknown";
}

// CUDA core count per SM per architecture
static int coresPerSM(int major, int minor) {
    // (major*10 + minor)
    switch (major * 10 + minor) {
        case 100: case 102: return 128; // Blackwell
        case  90:           return 128; // Hopper
        case  89: case  86: return 128; // Ada / Ampere-GA10x
        case  80:           return 64;  // Ampere A100
        case  75:           return 64;  // Turing
        case  70:           return 64;  // Volta
        case  61:           return 128; // Pascal GP104
        case  60:           return 64;  // Pascal GP100
        default:            return -1;
    }
}

// FP64 cores per SM
static int fp64CoresPerSM(int major, int minor) {
    switch (major * 10 + minor) {
        case 100: case 102: return 64;  // Blackwell
        case  90:           return 64;  // Hopper
        case  80:           return 32;  // Ampere A100
        case  70:           return 32;  // Volta
        case  60:           return 32;  // Pascal GP100
        default:            return 2;   // Most consumer GPUs 1/64 rate
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Cache size lookup tables (architecture-specific static values)
// ─────────────────────────────────────────────────────────────────────────────
// static int l1CacheKBPerSM(int major, int minor) {
//     // Unified L1 data cache + shared memory pool size per SM
//     switch (major * 10 + minor) {
//         case 100: case 102: return 256; // Blackwell
//         case  90:           return 256; // Hopper
//         case  80:           return 192; // Ampere A100
//         case  86: case  89: return 128; // Ampere GA10x / Ada
//         case  75:           return  96; // Turing
//         case  70:           return 128; // Volta
//         case  61: case  60: return  64; // Pascal
//         default:            return  -1;
//     }
// }

// static int tmemKBPerSM(int major, int minor) {
//     // Tensor Memory — dedicated SRAM for WGMMA (Blackwell SM 10.x only)
//     if (major == 10 || major == 12) return 128;
//     return 0;
// }

// ─────────────────────────────────────────────────────────────────────────────
// Latency benchmark: pointer-chase kernels + orchestrator
// ─────────────────────────────────────────────────────────────────────────────

// Build a sequential pointer chain in host memory.
// arr[i*stride] = ((i+1) % (n/stride)) * stride  →  serial dependency chain.
// static void buildPointerChain(uint32_t *arr, uint32_t n, uint32_t stride) {
//     uint32_t nodes = n / stride;
//     memset(arr, 0, (size_t)n * sizeof(uint32_t));
//     for (uint32_t i = 0; i < nodes; i++)
//         arr[i * stride] = ((i + 1) % nodes) * stride;
// }

// GPU-side hash chain init — avoids a large host malloc for the DRAM test.
// Uses Knuth multiplicative hash to scatter accesses across the full array.
__global__ void kern_init_hash_chain(uint32_t *arr, uint32_t nodes, uint32_t stride) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nodes) return;
    arr[i * stride] = (uint32_t)(((uint64_t)(i + 1) * 2654435769ULL) % nodes) * stride;
}

#define BENCH_ITERS  4096u
#define DRAM_ITERS  (1u << 19)   // 512K — enough to thrash a 40 MB L2

// Global memory pointer chase (parameterised iteration count).
__global__ void kern_global_lat(const uint32_t * __restrict__ arr,
                                 uint32_t iters, uint64_t *out) {
    if (threadIdx.x) return;
    uint32_t idx = 0;
    uint64_t t0 = clock64();
    #pragma unroll 1
    for (uint32_t i = 0; i < iters; i++) idx = arr[idx];
    uint64_t t1 = clock64();
    *out = (t1 - t0) / iters;
    if (idx == 0xFFFFFFFFu) out[1] = idx;   // prevent DCE
}

// Shared memory pointer chase (8-element / 32-byte stride, 2 KB working set).
__global__ void kern_shared_lat(uint64_t *out) {
    const uint32_t N = 512, S = 8;
    __shared__ uint32_t smem[N];
    if (threadIdx.x == 0) {
        uint32_t nodes = N / S;
        for (uint32_t i = 0; i < nodes; i++) smem[i * S] = ((i + 1) % nodes) * S;
    }
    __syncthreads();
    if (threadIdx.x) return;
    uint32_t idx = 0;
    uint64_t t0 = clock64();
    #pragma unroll 1
    for (uint32_t i = 0; i < BENCH_ITERS; i++) idx = smem[idx];
    uint64_t t1 = clock64();
    *out = (t1 - t0) / BENCH_ITERS;
    if (idx == 0xFFFFFFFFu) smem[0] = 0;
}

// Constant memory read chain (fits in the 64 KB constant cache).
#define CONST_N 4096u
__constant__ uint32_t d_constArr[CONST_N];

__global__ void kern_const_lat(uint64_t *out) {
    if (threadIdx.x) return;
    uint32_t idx = 0;
    uint64_t t0 = clock64();
    #pragma unroll 1
    for (uint32_t i = 0; i < BENCH_ITERS; i++) idx = d_constArr[idx];
    uint64_t t1 = clock64();
    *out = (t1 - t0) / BENCH_ITERS;
    if (idx == 0xFFFFFFFFu) out[1] = idx;
}

// Texture memory latency via texture object (L1 texture cache hit).
__global__ void kern_tex_lat(cudaTextureObject_t tex, uint64_t *out) {
    if (threadIdx.x) return;
    uint32_t idx = 0;
    uint64_t t0 = clock64();
    #pragma unroll 1
    for (uint32_t i = 0; i < BENCH_ITERS; i++)
        idx = tex1Dfetch<uint32_t>(tex, (int)idx);
    uint64_t t1 = clock64();
    *out = (t1 - t0) / BENCH_ITERS;
    if (idx == 0xFFFFFFFFu) (void)tex1Dfetch<uint32_t>(tex, 0);
}

// Local memory latency — volatile + large array forces register spill to
// local (per-thread global) memory even at -O2.
__global__ void kern_local_lat(uint64_t *out) {
    if (threadIdx.x) return;
    volatile uint32_t loc[512];
    const uint32_t N = 512, S = 8;
    for (uint32_t i = 0; i < N / S; i++) loc[i * S] = ((i + 1) % (N / S)) * S;
    uint32_t idx = 0;
    uint64_t t0 = clock64();
    #pragma unroll 1
    for (uint32_t i = 0; i < BENCH_ITERS; i++) idx = loc[idx];
    uint64_t t1 = clock64();
    *out = (t1 - t0) / BENCH_ITERS;
    if (idx == 0xFFFFFFFFu) loc[0] = 0;
    *out = (t1 - t0) / BENCH_ITERS;  // second write uses idx indirectly
}

// ── Print helper ─────────────────────────────────────────────────────────────
#define PRINT_LAT(label, cyc, ghz) \
    printf("│  %-30s  %5llu cycles  /  %6.1f ns\n", \
           label, (unsigned long long)(cyc), (cyc) / (ghz))

// static void runLatencyBench(int dev, int smClkKHz) {
//     double clkGHz = smClkKHz / 1e6;
//     const uint32_t STRIDE = 32;   // 128 bytes = one cache line

//     uint64_t *d_out, h_out;
//     CHECK(cudaMalloc(&d_out, 2 * sizeof(uint64_t)));   // [0]=result [1]=DCE sink

//     printf("\n├─ MEMORY LATENCY BENCHMARK ────────────────────────────────\n");
//     printf("│  Pointer-chase, single thread, 128-B stride, %.3f GHz\n", clkGHz);
//     printf("│  (compile with -lineinfo and use Nsight to get exact pipeline latency)\n│\n");

//     // ── Global: L1 hit — 8 KB working set ──────────────────────────────────
//     {
//         const uint32_t N = 2048;
//         uint32_t *h = (uint32_t*)malloc(N * sizeof(uint32_t));
//         uint32_t *d; buildPointerChain(h, N, STRIDE);
//         CHECK(cudaMalloc(&d, N * sizeof(uint32_t)));
//         CHECK(cudaMemcpy(d, h, N * sizeof(uint32_t), cudaMemcpyHostToDevice));
//         kern_global_lat<<<1,32>>>(d, BENCH_ITERS, d_out); CHECK(cudaDeviceSynchronize());
//         kern_global_lat<<<1,32>>>(d, BENCH_ITERS, d_out); CHECK(cudaDeviceSynchronize());
//         CHECK(cudaMemcpy(&h_out, d_out, sizeof(uint64_t), cudaMemcpyDeviceToHost));
//         PRINT_LAT("Global  L1 hit   (  8 KB):", h_out, clkGHz);
//         free(h); CHECK(cudaFree(d));
//     }

//     // ── Global: L2 hit — 8 MB working set ──────────────────────────────────
//     {
//         const uint32_t N = 2u * 1024u * 1024u;   // 8 MB
//         uint32_t *h = (uint32_t*)malloc((size_t)N * sizeof(uint32_t));
//         uint32_t *d; buildPointerChain(h, N, STRIDE);
//         CHECK(cudaMalloc(&d, (size_t)N * sizeof(uint32_t)));
//         CHECK(cudaMemcpy(d, h, (size_t)N * sizeof(uint32_t), cudaMemcpyHostToDevice));
//         kern_global_lat<<<1,32>>>(d, BENCH_ITERS, d_out); CHECK(cudaDeviceSynchronize()); // warm-up → fills L2
//         kern_global_lat<<<1,32>>>(d, BENCH_ITERS, d_out); CHECK(cudaDeviceSynchronize());
//         CHECK(cudaMemcpy(&h_out, d_out, sizeof(uint64_t), cudaMemcpyDeviceToHost));
//         PRINT_LAT("Global  L2 hit   (  8 MB):", h_out, clkGHz);
//         free(h); CHECK(cudaFree(d));
//     }

//     // ── Global: DRAM — 256 MB device-only allocation, hash chain ───────────
//     // 256 MB / 128 B per line = 2M cache lines >> typical L2 (≤80 MB / 128 B = 640K lines)
//     // DRAM_ITERS = 512K ensures the accessed set exceeds L2 capacity each pass.
//     {
//         const uint32_t N     = 64u * 1024u * 1024u;  // 64M uint32 = 256 MB
//         const uint32_t NODES = N / STRIDE;
//         uint32_t *d;
//         CHECK(cudaMalloc(&d, (size_t)N * sizeof(uint32_t)));
//         kern_init_hash_chain<<<(NODES+255)/256, 256>>>(d, NODES, STRIDE);
//         CHECK(cudaDeviceSynchronize());
//         kern_global_lat<<<1,32>>>(d, DRAM_ITERS, d_out); CHECK(cudaDeviceSynchronize());
//         kern_global_lat<<<1,32>>>(d, DRAM_ITERS, d_out); CHECK(cudaDeviceSynchronize());
//         CHECK(cudaMemcpy(&h_out, d_out, sizeof(uint64_t), cudaMemcpyDeviceToHost));
//         PRINT_LAT("Global  DRAM     (256 MB):", h_out, clkGHz);
//         CHECK(cudaFree(d));
//     }

//     // ── Shared memory ───────────────────────────────────────────────────────
//     kern_shared_lat<<<1,64>>>(d_out); CHECK(cudaDeviceSynchronize());
//     kern_shared_lat<<<1,64>>>(d_out); CHECK(cudaDeviceSynchronize());
//     CHECK(cudaMemcpy(&h_out, d_out, sizeof(uint64_t), cudaMemcpyDeviceToHost));
//     PRINT_LAT("Shared  (no bank conflict):", h_out, clkGHz);

//     // ── Constant memory ─────────────────────────────────────────────────────
//     {
//         uint32_t h_const[CONST_N];
//         buildPointerChain(h_const, CONST_N, 8);   // stride 8 within 16 KB const
//         CHECK(cudaMemcpyToSymbol(d_constArr, h_const, CONST_N * sizeof(uint32_t)));
//         kern_const_lat<<<1,32>>>(d_out); CHECK(cudaDeviceSynchronize());
//         kern_const_lat<<<1,32>>>(d_out); CHECK(cudaDeviceSynchronize());
//         CHECK(cudaMemcpy(&h_out, d_out, sizeof(uint64_t), cudaMemcpyDeviceToHost));
//         PRINT_LAT("Constant (L1 const cache):", h_out, clkGHz);
//     }

//     // ── Texture memory ──────────────────────────────────────────────────────
//     {
//         const uint32_t N = 2048;
//         uint32_t *h = (uint32_t*)malloc(N * sizeof(uint32_t));
//         uint32_t *d; buildPointerChain(h, N, STRIDE);
//         CHECK(cudaMalloc(&d, N * sizeof(uint32_t)));
//         CHECK(cudaMemcpy(d, h, N * sizeof(uint32_t), cudaMemcpyHostToDevice));

//         cudaResourceDesc rd = {};
//         rd.resType                = cudaResourceTypeLinear;
//         rd.res.linear.devPtr      = d;
//         rd.res.linear.desc        = cudaCreateChannelDesc<uint32_t>();
//         rd.res.linear.sizeInBytes = N * sizeof(uint32_t);
//         cudaTextureDesc td = {};
//         td.readMode = cudaReadModeElementType;
//         cudaTextureObject_t tex = 0;
//         CHECK(cudaCreateTextureObject(&tex, &rd, &td, nullptr));

//         kern_tex_lat<<<1,32>>>(tex, d_out); CHECK(cudaDeviceSynchronize());
//         kern_tex_lat<<<1,32>>>(tex, d_out); CHECK(cudaDeviceSynchronize());
//         CHECK(cudaMemcpy(&h_out, d_out, sizeof(uint64_t), cudaMemcpyDeviceToHost));
//         PRINT_LAT("Texture (L1 tex cache):", h_out, clkGHz);
//         CHECK(cudaDestroyTextureObject(tex));
//         free(h); CHECK(cudaFree(d));
//     }

//     // ── Local memory (register spill → global addr space) ──────────────────
//     kern_local_lat<<<1,32>>>(d_out); CHECK(cudaDeviceSynchronize());
//     kern_local_lat<<<1,32>>>(d_out); CHECK(cudaDeviceSynchronize());
//     CHECK(cudaMemcpy(&h_out, d_out, sizeof(uint64_t), cudaMemcpyDeviceToHost));
//     PRINT_LAT("Local   (reg-spill):", h_out, clkGHz);
//     printf("│    Use -Xptxas -v to confirm .local spill in PTX\n");

//     // ── TMEM — Tensor Memory (Blackwell only) ───────────────────────────────
//     printf("│\n");
//     printf("│  TMEM (Tensor Memory, SM 10.x Blackwell):\n");
//     printf("│    Not addressable from standard CUDA C — requires tcgen05.ld/st PTX.\n");
//     printf("│    Accessible only via WGMMA warp-group matrix ops (≈SM 10.x+).\n");
//     printf("│    Expected latency: ~20–30 cycles (similar to shared memory).\n");

//     CHECK(cudaFree(d_out));
// }

#undef BENCH_ITERS
#undef DRAM_ITERS
#undef PRINT_LAT

// Tensor core generations
static const char *tensorCoreGen(int major, int minor) {
    if (major == 10 || major == 12) return "5th gen (FP4/FP8/FP16/BF16/TF32/FP64 + Block Scaling)";
    if (major == 9)  return "4th gen (FP8/FP16/BF16/TF32/FP64 + Sparsity)";
    if (major == 8)  return "3rd gen (TF32/BF16/FP16/INT8/INT4/FP64)";
    if (major == 7)  return "2nd gen (FP16/INT8/INT4/INT1)";
    return "None";
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char **argv) {

    int devCount = 0;
    CHECK(cudaGetDeviceCount(&devCount));
    printf("═══════════════════════════════════════════════════════════\n");
    printf("  CUDA Deep Device Query  —  %d device(s) found\n", devCount);
    printf("═══════════════════════════════════════════════════════════\n\n");

    for (int dev = 0; dev < devCount; dev++) {
        cudaDeviceProp p;
        CHECK(cudaGetDeviceProperties(&p, dev));

        int sm = p.major * 10 + p.minor;
        int cores_per_sm  = coresPerSM(p.major, p.minor);
        int fp64_per_sm   = fp64CoresPerSM(p.major, p.minor);
        int total_fp32    = (cores_per_sm > 0) ? cores_per_sm * p.multiProcessorCount : -1;
        int total_fp64    = fp64_per_sm * p.multiProcessorCount;

        // Memory bandwidth (GB/s) = busWidth(bits)/8 * clockRate(kHz)*1000 * 2 (DDR) / 1e9
        int memClkKHz   = getAttr(cudaDevAttrMemoryClockRate,     dev);
        int smClkKHz    = getAttr(cudaDevAttrClockRate,           dev);
        double memBW_GBs = (double)p.memoryBusWidth / 8.0 * (double)memClkKHz * 1e3 * 2.0 / 1e9;

        // Theoretical peak TFLOPS
        double clkGHz     = smClkKHz / 1e6;
        double fp32_tflops = (total_fp32 > 0) ? (2.0 * total_fp32 * clkGHz / 1e3) : -1;
        double fp64_tflops = 2.0 * total_fp64 * clkGHz / 1e3;

        // ── Section 1: Identity ──────────────────────────────────────────────
        printf("╔══════════════════════════════════════════════════════════╗\n");
        printf("║  Device %d: %-46s║\n", dev, p.name);
        printf("╚══════════════════════════════════════════════════════════╝\n");

        printf("\n┌─ IDENTITY ────────────────────────────────────────────────\n");
        printf("|  Name:                      %s\n", p.name);
        printf("│  Architecture:              %s\n", archName(p.major, p.minor));
        printf("│  Compute Capability:        %d.%d (SM_%d)\n", p.major, p.minor, sm);
        printf("│  UUID:                      ");
        for (int i = 0; i < 16; i++) printf("%02x%s", (unsigned char)p.uuid.bytes[i],
                                            (i==3||i==5||i==7||i==9) ? "-" : "");
        printf("\n");
        printf("│  PCI Bus / Device / Domain: %d / %d / %d\n", p.pciBusID, p.pciDeviceID, p.pciDomainID);
        printf("│  Integrated GPU:            %s\n", p.integrated ? "Yes" : "No");
        printf("│  Kernel Exec Timeout:       %s\n",
               getAttr(cudaDevAttrKernelExecTimeout, dev) ? "Enabled (display GPU)" : "Disabled");
        printf("│  TCC Driver:                %s\n", p.tccDriver ? "Yes" : "No");
        int computeMode = getAttr(cudaDevAttrComputeMode, dev);
        printf("│  Compute Mode:              %d (%s)\n", computeMode,
               computeMode == 0 ? "Default (shared)" :
               computeMode == 2 ? "Exclusive process" :
               computeMode == 3 ? "Prohibited" : "Exclusive thread");

        // ── Section 2: SM / Execution Units ─────────────────────────────────
        printf("\n├─ STREAMING MULTIPROCESSORS ───────────────────────────────\n");
        printf("│  SM Count:                  %d\n", p.multiProcessorCount);
        printf("│  FP32 Cores / SM:           %d  →  %d total\n", cores_per_sm, total_fp32);
        printf("│  FP64 Cores / SM:           %d  →  %d total\n", fp64_per_sm,  total_fp64);
        printf("│  Tensor Core Generation:    %s\n", tensorCoreGen(p.major, p.minor));
        printf("│  SM Clock:                  %.3f GHz  (boost: same value if no boost query)\n", clkGHz);
        printf("│  Theoretical FP32 Peak:     %.2f TFLOPS\n", fp32_tflops);
        printf("│  Theoretical FP64 Peak:     %.2f TFLOPS\n", fp64_tflops);
        printf("│  Warp Size:                 %d threads\n", p.warpSize);

        // ── Section 3: Occupancy Limits ─────────────────────────────────────
        printf("\n├─ OCCUPANCY & SCHEDULING ──────────────────────────────────\n");
        int maxWarpsPerSM   = p.maxThreadsPerMultiProcessor / p.warpSize;
        int maxBlocksPerSM  = getAttr(cudaDevAttrMaxBlocksPerMultiprocessor, dev);
        printf("│  Max Threads / SM:          %d\n",  p.maxThreadsPerMultiProcessor);
        printf("│  Max Warps / SM:            %d  ← occupancy ceiling\n", maxWarpsPerSM);
        printf("│  Max Blocks / SM:           %d\n",  maxBlocksPerSM);
        printf("│  Max Threads / Block:       %d\n",  p.maxThreadsPerBlock);
        printf("│  Max Thread Dims (x,y,z):   (%d, %d, %d)\n",
               p.maxThreadsDim[0], p.maxThreadsDim[1], p.maxThreadsDim[2]);
        printf("│  Max Grid Dims  (x,y,z):    (%d, %d, %d)\n",
               p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2]);

        // Register file
        printf("│\n│  ─ Register File ─\n");
        printf("│  Regs / Block:              %d\n", p.regsPerBlock);
        printf("│  Regs / SM:                 %d\n", p.regsPerMultiprocessor);
        printf("│  Regs / Thread (to hit max warp occ.):\n");
        // regs/thread limit = regsPerSM / maxWarpsPerSM / warpSize
        int regs_for_full_occ = p.regsPerMultiprocessor / maxWarpsPerSM / p.warpSize;
        printf("│    ≤ %d regs/thread for full warp occupancy\n", regs_for_full_occ);

        // Shared memory
        printf("│\n│  ─ Shared Memory Occupancy Limits ─\n");
        size_t smemPerSM  = p.sharedMemPerMultiprocessor;
        size_t smemPerBlk = p.sharedMemPerBlockOptin; // max with opt-in (cudaFuncSetAttribute)
        printf("│  Shared Mem / SM:           %zu KB  (%zu bytes)\n", smemPerSM/1024, smemPerSM);
        printf("│  Shared Mem / Block (default): %zu KB\n", p.sharedMemPerBlock/1024);
        printf("│  Shared Mem / Block (max optin): %zu KB  ← use cudaFuncSetAttribute\n", smemPerBlk/1024);
        if (smemPerBlk > 0) {
            printf("│  Max smem/thread for full block occ. (%d threads):\n", p.maxThreadsPerBlock);
            printf("│    %.1f bytes/thread  (smemPerSM / maxBlocks / maxThreads)\n",
                   (double)smemPerSM / maxBlocksPerSM / p.maxThreadsPerBlock);
        }

        // ── Section 4: Memory Hierarchy ─────────────────────────────────────
        printf("\n├─ MEMORY HIERARCHY ────────────────────────────────────────\n");
        printf("│  Global Memory:             %.2f GB  (%zu bytes)\n",
               p.totalGlobalMem / 1e9, p.totalGlobalMem);
        printf("│  Memory Bus Width:          %d bits\n",   p.memoryBusWidth);
        printf("│  Memory Clock:              %.3f GHz\n",  memClkKHz / 1e6);
        printf("│  Peak Memory Bandwidth:     %.1f GB/s\n", memBW_GBs);
        printf("│  ECC Enabled:               %s\n", p.ECCEnabled ? "Yes (reduces effective BW ~6%%)" : "No");
        printf("│\n");
        printf("│  L2 Cache:                  %d MB  (%d bytes)\n",
               p.l2CacheSize / (1024*1024), p.l2CacheSize);

        int l2PersistMax = getAttr(cudaDevAttrMaxPersistingL2CacheSize, dev);
        if (l2PersistMax > 0)
            printf("│  L2 Persisting Cache Max:   %d MB  ← cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize)\n",
                   l2PersistMax / (1024*1024));

        int l1CacheSupported = getAttr(cudaDevAttrLocalL1CacheSupported,   dev);
        int glbCacheSupported= getAttr(cudaDevAttrGlobalL1CacheSupported,  dev);
        printf("│  L1 Cache (local/stack):    %s\n", l1CacheSupported  ? "Supported" : "Not supported");
        printf("│  L1 Cache (global loads):   %s\n", glbCacheSupported ? "Supported" : "Not supported");

        // Shared mem bank config API removed in CUDA 13; fixed at 4 bytes on Volta+
        printf("│\n│  Shared Memory Banks:       32 banks (fixed)\n");
        printf("│  Bank Width:                4 bytes (fixed on SM 7.0+, config API removed in CUDA 13)\n");

        printf("│  Constant Memory:           %zu KB\n",   p.totalConstMem/1024);
        printf("│  Texture Alignment:         %zu bytes\n", p.textureAlignment);
        printf("│  Surface Alignment:         %zu bytes\n", p.surfaceAlignment);

        // ── Section 5: Transfer / PCIe ──────────────────────────────────────
        printf("\n├─ TRANSFER & INTERCONNECT ─────────────────────────────────\n");
        printf("│  Async Engine Count:        %d  (copy engines)\n", p.asyncEngineCount);
        printf("│  Concurrent Kernels:        %s\n", p.concurrentKernels ? "Yes" : "No");
        printf("│  Concurrent Managed Access: %s\n", p.concurrentManagedAccess ? "Yes" : "No");
        printf("│  Can Map Host Memory:       %s\n", p.canMapHostMemory ? "Yes" : "No");
        printf("│  Unified Addressing (UVA):  %s\n", p.unifiedAddressing ? "Yes" : "No");
        printf("│  Managed Memory:            %s\n", p.managedMemory ? "Yes" : "No");
        printf("│  Direct Managed Mem Access: %s\n", p.directManagedMemAccessFromHost ? "Yes (host can access without migration)" : "No");
        printf("│  Host Native Atomic Supported: %s\n", p.hostNativeAtomicSupported ? "Yes" : "No");
        printf("│  Page-able Mem Access:      %s\n", p.pageableMemoryAccess ? "Yes" : "No");
        printf("│  Page-able Mem via USB:     %s\n", p.pageableMemoryAccessUsesHostPageTables ? "Yes" : "No");

        int nvlinkSupport = 0;
#ifdef cudaDevAttrNvlinkSupported
        nvlinkSupport = getAttr(cudaDevAttrNvlinkSupported, dev);
#endif
        printf("│  NVLink Supported:          %s\n", nvlinkSupport ? "Yes" : "No / Not queryable via this attr");

        // ── Section 6: Advanced Execution Features ──────────────────────────
        printf("\n├─ ADVANCED EXECUTION FEATURES ─────────────────────────────\n");

        int coopLaunch = getAttr(cudaDevAttrCooperativeLaunch, dev);
        printf("│  Cooperative Launch:        %s  ← grid-sync across all blocks\n",
               coopLaunch ? "Yes" : "No");

        int streamPriorities = getAttr(cudaDevAttrStreamPrioritiesSupported, dev);
        int prioLo, prioHi;
        cudaDeviceGetStreamPriorityRange(&prioLo, &prioHi);
        printf("│  Stream Priorities:         %s  (range: %d to %d, %d = highest)\n",
               streamPriorities ? "Yes" : "No", prioLo, prioHi, prioHi);

        int memPoolSupport = getAttr(cudaDevAttrMemoryPoolsSupported,        dev);
        int memPoolRdma    = getAttr(cudaDevAttrGPUDirectRDMASupported,      dev);
        printf("│  Memory Pools (async alloc): %s\n", memPoolSupport ? "Yes  ← cudaMallocAsync" : "No");
        printf("│  GPUDirect RDMA:            %s\n",  memPoolRdma    ? "Yes" : "No");

        int flushRemoteWrites = getAttr(cudaDevAttrGPUDirectRDMAFlushWritesOptions, dev);
        if (memPoolRdma)
            printf("│    Flush Writes Options:    0x%x\n", flushRemoteWrites);

        int sparseSupport = 0;
#ifdef cudaDevAttrSparseCudaArraySupported
        sparseSupport = getAttr(cudaDevAttrSparseCudaArraySupported, dev);
#endif
        printf("│  Sparse CUDA Arrays:        %s\n", sparseSupport ? "Yes" : "No");

        int deferredMappingSupport = 0;
#ifdef cudaDevAttrDeferredMappingCudaArraySupported
        deferredMappingSupport = getAttr(cudaDevAttrDeferredMappingCudaArraySupported, dev);
#endif
        printf("│  Deferred Mapping Arrays:   %s\n", deferredMappingSupport ? "Yes" : "No");

        int hostRegReadOnly = 0;
#ifdef cudaDevAttrHostRegisterReadOnlySupported
        hostRegReadOnly = getAttr(cudaDevAttrHostRegisterReadOnlySupported, dev);
#endif
        printf("│  Host Register Read-Only:   %s\n", hostRegReadOnly ? "Yes" : "No");

        // ── Section 7: Warp-level Features ──────────────────────────────────
        printf("\n├─ WARP-LEVEL & INSTRUCTION FEATURES ───────────────────────\n");
        printf("│  Warp Size:                 %d\n", p.warpSize);
        printf("│  Max Warps In-flight / SM:  %d\n", maxWarpsPerSM);
        printf("│\n");
        printf("│  Warp-level primitives available (SM %d.%d):\n", p.major, p.minor);
        if (sm >= 70) printf("│    __shfl_sync, __ballot_sync, __any_sync, __all_sync — Yes\n");
        if (sm >= 80) printf("│    __reduce_add_sync, __reduce_and_sync etc.          — Yes\n");
        if (sm >= 90) printf("│    warpgroup async descriptors (WGMMA)                — Yes\n");
        if (sm >= 100)printf("│    Blackwell warp-group matrix ops (WGMMA ext.)       — Yes\n");

        int preemption = getAttr(cudaDevAttrComputePreemptionSupported, dev);
        printf("│  Compute Preemption:        %s  ← instruction-level on Pascal+\n",
               preemption ? "Yes" : "No");

        // ── Section 8: Texture / Surface ────────────────────────────────────
        printf("\n├─ TEXTURE & SURFACE ───────────────────────────────────────\n");
        printf("│  Max 1D Texture:            %d\n",        p.maxTexture1D);
        printf("│  Max 2D Texture:            (%d, %d)\n",  p.maxTexture2D[0],  p.maxTexture2D[1]);
        printf("│  Max 3D Texture:            (%d, %d, %d)\n",
               p.maxTexture3D[0], p.maxTexture3D[1], p.maxTexture3D[2]);
        printf("│  Max 1D Layered:            (%d layers, size %d)\n",
               p.maxTexture1DLayered[1], p.maxTexture1DLayered[0]);
        printf("│  Max 2D Layered:            (%d, %d) × %d layers\n",
               p.maxTexture2DLayered[0], p.maxTexture2DLayered[1], p.maxTexture2DLayered[2]);
        printf("│  Max Cubemap Texture:       %d\n",        p.maxTextureCubemap);
        printf("│  Texture Pitch Alignment:   %zu bytes\n", p.texturePitchAlignment);
        printf("│  Max 1D Surface:            %d\n",        p.maxSurface1D);
        printf("│  Max 2D Surface:            (%d, %d)\n",  p.maxSurface2D[0], p.maxSurface2D[1]);

        printf("\n├─────────────────────────────────────────────────────────────\n\n");
    }
    
    return 0;
}