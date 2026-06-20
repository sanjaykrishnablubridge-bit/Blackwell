#include <cuda.h>
#include <cuda_runtime.h>
#include "utils/kernelBench.cuh"
#include <cmath>
#include <chrono>
#include <random>

// ── Embedded PTX (src/ptx/sm120/gelu.ptx) ────────────────────────────────────
// Loaded at runtime via the Driver API so no separate file path is needed.
static const char *PTX_SRC = R"PTX(
.version 9.0
.target sm_120
.address_size 64

.visible .entry gelu_kernel(
    .param .u64 .ptr.global.align 4 inp,
    .param .u64 .ptr.global.align 4 out,
    .param .u64 N
)
{
    .reg .pred    %p0;
    .reg .b32     %r<3>;
    .reg .f32     %f<10>;
    .reg .b64     %rd<10>;

    ld.param.u64    %rd0, [inp];
    ld.param.u64    %rd1, [out];
    ld.param.u64    %rd2, [N];

    mov.u32         %r0, %tid.x;
    mov.u32         %r1, %ctaid.x;
    mov.u32         %r2, %ntid.x;
    mad.lo.u32      %r0, %r1, %r2, %r0;
    cvt.u64.u32     %rd3, %r0;

    setp.ge.u64     %p0, %rd3, %rd2;
    @%p0 bra        EXIT_LABEL;

    shl.b64         %rd4, %rd3, 2;
    add.u64         %rd5, %rd0, %rd4;
    add.u64         %rd6, %rd1, %rd4;

    ld.global.f32   %f1, [%rd5];

    mul.f32         %f2, %f1, %f1;
    mul.f32         %f3, %f2, %f1;
    mul.f32         %f4, %f3, 0f3D37549A;
    add.f32         %f5, %f1, %f4;
    mul.f32         %f6, %f5, 0f3F4C422A;
    tanh.approx.f32 %f7, %f6;
    add.f32         %f8, %f7, 0f3F800000;
    mul.f32         %f9, %f1, 0f3F000000;
    mul.f32         %f1, %f9, %f8;

    st.global.f32   [%rd6], %f1;

EXIT_LABEL:
    ret;
}
)PTX";


// ── CUDA C++ kernel — mirrors the PTX instruction-for-instruction ─────────────
//
// Uses __int_as_float() with the same hex literals the PTX uses so the
// compiler cannot silently pick different constant values.
//   0f3D37549A → 0.044715f
//   0f3F4C422A → sqrt(2/π) ≈ 0.79788456f
//   0f3F800000 → 1.0f
//   0f3F000000 → 0.5f
//
// --use_fast_math (set in CMakeLists) makes tanhf() emit tanh.approx.f32,
// exactly matching the PTX instruction.
__global__ void gelu_cuda_kernel(const float *inp, float *out, size_t N){
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;

    const float c_044715  = __int_as_float(0x3D37549A);  // 0.044715
    const float c_sqrt2pi = __int_as_float(0x3F4C422A);  // sqrt(2/π)
    const float c_one     = __int_as_float(0x3F800000);  // 1.0
    const float c_half    = __int_as_float(0x3F000000);  // 0.5

    float x  = inp[idx];
    float x2 = x * x;
    float x3 = x2 * x;
    float inner = c_sqrt2pi * (x + c_044715 * x3);
    out[idx]    = (c_half * x) * (c_one + tanhf(inner));
}


// ── CPU reference ─────────────────────────────────────────────────────────────
// Use memcpy to reinterpret the same hex bit patterns used in the PTX/CUDA kernel.
static inline float bits_to_float(uint32_t bits){
    float f; memcpy(&f, &bits, 4); return f;
}

void gelu_cpu(const float *inp, float *out, size_t N){
    const float c_044715  = bits_to_float(0x3D37549A);
    const float c_sqrt2pi = bits_to_float(0x3F4C422A);
    for(size_t i = 0; i < N; ++i){
        float x     = inp[i];
        float inner = c_sqrt2pi * (x + c_044715 * x * x * x);
        out[i]      = 0.5f * x * (1.0f + tanhf(inner));
    }
}


// ── PTX kernel wrapper via Driver API ─────────────────────────────────────────
struct PTXKernel {
    CUmodule   mod;
    CUfunction fn;

    PTXKernel(){
        // cudaFree(0) forces the runtime to initialise the primary context;
        // the driver API then reuses that same context automatically.
        CUDA_CHECK(cudaFree(0));

        CUresult r;
        r = cuModuleLoadData(&mod, PTX_SRC);
        if(r != CUDA_SUCCESS){
            const char *err; cuGetErrorString(r, &err);
            fprintf(stderr, "cuModuleLoadData failed: %s\n", err); exit(1);
        }
        r = cuModuleGetFunction(&fn, mod, "gelu_kernel");
        if(r != CUDA_SUCCESS){
            const char *err; cuGetErrorString(r, &err);
            fprintf(stderr, "cuModuleGetFunction failed: %s\n", err); exit(1);
        }
    }

    ~PTXKernel(){ cuModuleUnload(mod); }

    void launch(const float *inp, float *out, size_t N, int block_size = 256){
        int grid = ((int)N + block_size - 1) / block_size;
        void *args[] = { (void*)&inp, (void*)&out, (void*)&N };
        cuLaunchKernel(fn, grid, 1, 1, block_size, 1, 1, 0, 0, args, nullptr);
    }
};


int main(){
    std::cout << "Benchmarking GELU: PTX vs CUDA C++  — Blackwell SM_120\n";

    const size_t N      = 1 << 24;   // 16 M elements
    const size_t NBytes = N * sizeof(float);

    std::vector<float> h_inp(N), h_out_cuda(N), h_out_ptx(N), h_ref(N);

    // ── Load PyTorch reference data (falls back to random if files absent) ────
    auto fileMatchesSize = [](const std::string &p, size_t n_floats) -> bool {
        FILE *f = fopen(p.c_str(), "rb");
        if(!f) return false;
        fseek(f, 0, SEEK_END);
        size_t bytes = (size_t)ftell(f);
        fclose(f);
        return bytes == n_floats * sizeof(float);
    };
    bool has_ref = fileMatchesSize("data/gelu_inp.bin", N);
    if(has_ref){
        loadBin("data/gelu_inp.bin", h_inp.data(), N);
        loadBin("data/gelu_out.bin", h_ref.data(), N);
        std::cout << "\nLoaded PyTorch reference from data/gelu_*.bin\n";
    } else {
        initVec(h_inp);
        std::cout << "\nNo reference files found — using random data (benchmarks only)\n";
    }

    // ── Device allocations ────────────────────────────────────────────────────
    float *d_inp, *d_out;
    CUDA_CHECK(cudaMalloc(&d_inp, NBytes));
    CUDA_CHECK(cudaMalloc(&d_out, NBytes));
    CUDA_CHECK(cudaMemcpy(d_inp, h_inp.data(), NBytes, cudaMemcpyHostToDevice));

    // ── Load PTX ──────────────────────────────────────────────────────────────
    PTXKernel ptx;

    // ── Correctness ───────────────────────────────────────────────────────────
    int block = 256, grid = ((int)N + block - 1) / block;
    gelu_cuda_kernel<<<grid, block>>>(d_inp, d_out, N);
    CUDA_CHECK(cudaMemcpy(h_out_cuda.data(), d_out, NBytes, cudaMemcpyDeviceToHost));

    ptx.launch(d_inp, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out_ptx.data(), d_out, NBytes, cudaMemcpyDeviceToHost));

    if(has_ref){
        std::cout << "\nCorrectness CUDA kernel vs PyTorch: ";
        checkResult(h_ref.data(), h_out_cuda.data(), N);
        reportPrecision("CUDA precision", h_ref.data(), h_out_cuda.data(), N);

        std::cout << "Correctness PTX  kernel vs PyTorch: ";
        checkResult(h_ref.data(), h_out_ptx.data(), N);
        reportPrecision("PTX  precision", h_ref.data(), h_out_ptx.data(), N);
    }

    std::cout << "Correctness PTX  kernel vs CUDA:    ";
    checkResult(h_out_cuda.data(), h_out_ptx.data(), N);

    // ── Benchmarks ────────────────────────────────────────────────────────────
    // FLOPs: 7 per element (2 mul for x^2/x^3, 1 mul for 0.044715*x^3,
    //         1 add, 1 mul for sqrt2pi, 1 tanh, 1 fma for final)  ≈ 8 FLOPs
    // Bandwidth: 1 read + 1 write = 2 * NBytes
    long long flops = 8LL * (long long)N;
    size_t    bytes = 2 * NBytes;

    KernelStats stats_cuda = benchmarkKernel(
        [&](){ gelu_cuda_kernel<<<grid, block>>>(d_inp, d_out, N); },
        100, 25, flops, bytes
    );
    displayStats("GELU — CUDA C++ kernel", stats_cuda);

    KernelStats stats_ptx = benchmarkKernel(
        [&](){ ptx.launch(d_inp, d_out, N); },
        100, 25, flops, bytes
    );
    displayStats("GELU — PTX kernel", stats_ptx);

    CUDA_CHECK(cudaFree(d_inp));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
