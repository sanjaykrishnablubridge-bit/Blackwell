// gpu_specs.cu — Comprehensive GPU spec dump for CUDA kernel optimization.
// Build:  nvcc -O2 gpu_specs.cu -o gpu_specs -lcuda -lnvidia-ml
// Run:    ./gpu_specs | tee gpu_specs.txt

#include <cuda_runtime.h>
#include <cuda.h>
#include <nvml.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr,"CUDA err %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); std::exit(1);} } while(0)
#define CKD(x) do { CUresult e = (x); if (e != CUDA_SUCCESS) { \
    const char* s; cuGetErrorString(e,&s); fprintf(stderr,"DRV err %s:%d: %s\n",__FILE__,__LINE__,s); } } while(0)

// static const char* mem_type(int t) {
//     switch (t) { default: return "unknown"; }
// }

int main() {
    int dev_count = 0;
    CK(cudaGetDeviceCount(&dev_count));
    printf("=================================================================\n");
    printf("  GPU SPECIFICATIONS — %d device(s) detected\n", dev_count);
    printf("=================================================================\n\n");

    CKD(cuInit(0));
    nvmlInit_v2();

    for (int d = 0; d < dev_count; ++d) {
        cudaDeviceProp p{};
        CK(cudaGetDeviceProperties(&p, d));
        CK(cudaSetDevice(d));

        printf("------------------------- Device %d -------------------------\n", d);
        printf("Name                        : %s\n", p.name);
        printf("Compute capability (SM)     : %d.%d  (sm_%d%d)\n", p.major, p.minor, p.major, p.minor);
        printf("CUDA driver / runtime       : ");
        int drv=0, rt=0; cudaDriverGetVersion(&drv); cudaRuntimeGetVersion(&rt);
        printf("%d.%d / %d.%d\n", drv/1000, (drv%100)/10, rt/1000, (rt%100)/10);

        // ---- Streaming Multiprocessors / cores ----
        printf("\n[ Streaming Multiprocessors ]\n");
        printf("  SM count                  : %d\n", p.multiProcessorCount);
        printf("  Max threads / SM          : %d\n", p.maxThreadsPerMultiProcessor);
        printf("  Max blocks / SM           : %d\n", p.maxBlocksPerMultiProcessor);
        printf("  Max warps / SM            : %d\n", p.maxThreadsPerMultiProcessor / p.warpSize);
        printf("  Warp size                 : %d\n", p.warpSize);
        printf("  Max threads / block       : %d\n", p.maxThreadsPerBlock);
        printf("  Max block dims            : %d x %d x %d\n", p.maxThreadsDim[0],p.maxThreadsDim[1],p.maxThreadsDim[2]);
        printf("  Max grid dims             : %d x %d x %d\n", p.maxGridSize[0],p.maxGridSize[1],p.maxGridSize[2]);
        printf("  Registers / block         : %d\n", p.regsPerBlock);
        printf("  Registers / SM            : %d\n", p.regsPerMultiprocessor);

        // ---- Shared memory / L1 ----
        printf("\n[ Shared Memory & L1 ]\n");
        printf("  Shared mem / block (def)  : %zu KB\n", p.sharedMemPerBlock/1024);
        printf("  Shared mem / block (opt)  : %zu KB  (cudaFuncSetAttribute MaxDynamicSharedMemorySize)\n",
               p.sharedMemPerBlockOptin/1024);
        printf("  Shared mem / SM           : %zu KB\n", p.sharedMemPerMultiprocessor/1024);
        printf("  L1 cache supported        : %s\n", p.localL1CacheSupported ? "yes" : "no");
        printf("  Global L1 cache supported : %s\n", p.globalL1CacheSupported ? "yes" : "no");

        // ---- L2 cache ----
        printf("\n[ L2 Cache ]\n");
        printf("  L2 cache size             : %d KB (%.2f MB)\n", p.l2CacheSize/1024, p.l2CacheSize/(1024.0*1024.0));
        printf("  Persisting L2 max         : %d KB\n", p.persistingL2CacheMaxSize/1024);
        printf("  Access policy max window  : %d KB\n", p.accessPolicyMaxWindowSize/1024);

        // ---- Global memory (HBM/GDDR) ----
        printf("\n[ Global Memory ]\n");
        printf("  Total global memory       : %.2f GB\n", p.totalGlobalMem/(1024.0*1024.0*1024.0));
        printf("  Memory bus width          : %d bits\n", p.memoryBusWidth);
        int memClkKHz = 0;
        cudaDeviceGetAttribute(&memClkKHz, cudaDevAttrMemoryClockRate, d);
        printf("  Memory clock rate         : %d MHz\n", memClkKHz/1000);
        double bw = 2.0 * (memClkKHz*1000.0) * (p.memoryBusWidth/8.0) / 1e9;
        printf("  Theoretical BW (DDR x2)   : %.1f GB/s\n", bw);
        printf("  ECC enabled               : %s\n", p.ECCEnabled ? "yes" : "no");
        printf("  Unified addressing        : %s\n", p.unifiedAddressing ? "yes" : "no");
        printf("  Managed memory            : %s\n", p.managedMemory ? "yes" : "no");
        printf("  Concurrent managed access : %s\n", p.concurrentManagedAccess ? "yes" : "no");
        printf("  Pageable mem access       : %s\n", p.pageableMemoryAccess ? "yes" : "no");

        // ---- Memory type / variant via NVML ----
        nvmlDevice_t nv;
        if (nvmlDeviceGetHandleByIndex_v2(d, &nv) == NVML_SUCCESS) {
            nvmlMemory_t m{};
            if (nvmlDeviceGetMemoryInfo(nv, &m) == NVML_SUCCESS) {
                printf("  NVML total / free / used  : %.2f / %.2f / %.2f GB\n",
                    m.total/1e9, m.free/1e9, m.used/1e9);
            }
            nvmlPciInfo_t pci{};
            nvmlDeviceGetPciInfo_v3(nv, &pci); // best-effort
            unsigned int link=0, gen=0;
            if (nvmlDeviceGetCurrPcieLinkWidth(nv,&link)==NVML_SUCCESS &&
                nvmlDeviceGetCurrPcieLinkGeneration(nv,&gen)==NVML_SUCCESS) {
                printf("  PCIe link                 : Gen%u x%u\n", gen, link);
            }
            unsigned int sm_clk=0, mem_clk=0, gfx_clk=0;
            nvmlDeviceGetClockInfo(nv, NVML_CLOCK_SM, &sm_clk);
            nvmlDeviceGetClockInfo(nv, NVML_CLOCK_MEM, &mem_clk);
            nvmlDeviceGetClockInfo(nv, NVML_CLOCK_GRAPHICS, &gfx_clk);
            printf("  Current clocks (SM/MEM/GFX): %u / %u / %u MHz\n", sm_clk, mem_clk, gfx_clk);
            unsigned int max_sm=0, max_mem=0;
            nvmlDeviceGetMaxClockInfo(nv, NVML_CLOCK_SM, &max_sm);
            nvmlDeviceGetMaxClockInfo(nv, NVML_CLOCK_MEM, &max_mem);
            printf("  Max clocks (SM/MEM)       : %u / %u MHz\n", max_sm, max_mem);
            unsigned int power=0, plimit=0;
            nvmlDeviceGetPowerUsage(nv,&power);
            nvmlDeviceGetPowerManagementLimit(nv,&plimit);
            printf("  Power (cur / limit)       : %.1f / %.1f W\n", power/1000.0, plimit/1000.0);
            unsigned int temp=0;
            nvmlDeviceGetTemperature(nv, NVML_TEMPERATURE_GPU, &temp);
            printf("  GPU temperature           : %u C\n", temp);
        }

        // ---- Compute features ----
        printf("\n[ Compute Features ]\n");
        printf("  Concurrent kernels        : %s\n", p.concurrentKernels ? "yes" : "no");
        printf("  Async engine count        : %d\n", p.asyncEngineCount);
        printf("  Cooperative launch        : %s\n", p.cooperativeLaunch ? "yes" : "no");
        printf("  Stream priorities         : %s\n", p.streamPrioritiesSupported ? "yes" : "no");
        printf("  Compute preemption        : %s\n", p.computePreemptionSupported ? "yes" : "no");
        printf("  Tensor cores (sm>=7.0)    : %s\n", (p.major>=7) ? "yes" : "no");
        printf("  Async copy (cp.async)     : %s\n", (p.major>=8) ? "yes (Ampere+)" : "no");
        printf("  TMA (Hopper+)             : %s\n", (p.major>=9) ? "yes" : "no");
        printf("  Distributed shared mem    : %s\n", (p.major>=9) ? "yes (Hopper+)" : "no");

        // ---- Theoretical FLOPS (rough) ----
        // FP32 cores per SM: Volta 64, Turing 64, Ampere GA100 64, Ampere GA10x 128, Ada 128, Hopper 128, Blackwell 128
        int fp32_per_sm = 128;
        if (p.major == 7) fp32_per_sm = 64;                  // V100/T4
        else if (p.major == 8 && p.minor == 0) fp32_per_sm = 64; // A100
        // SM clock (kHz -> GHz)
        int smClkKHz = 0;
        cudaDeviceGetAttribute(&smClkKHz, cudaDevAttrClockRate, d);
        double clk_ghz = smClkKHz / 1.0e6;
        double fp32_tflops = 2.0 * p.multiProcessorCount * fp32_per_sm * clk_ghz / 1000.0;
        printf("\n[ Theoretical Peak (rough) ]\n");
        printf("  FP32 cores / SM (assumed) : %d\n", fp32_per_sm);
        printf("  GPU boost clock           : %.2f GHz\n", clk_ghz);
        printf("  FP32 peak                 : %.1f TFLOPS\n", fp32_tflops);
        printf("  Memory bandwidth peak     : %.1f GB/s\n", bw);
        printf("  Arithmetic intensity ridge: %.1f FLOP/byte (fp32 peak / bw)\n",
               fp32_tflops*1e3 / bw);

        printf("\n");
    }

    nvmlShutdown();
    return 0;
}