#pragma once

#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include "kernelUtils.cuh"

struct KernelStats {
    float avg_ms;
    float median_ms;
    float std_dev_ms;
    float min_ms;
    float max_ms;
    double tflops;
    double bandwidth_gb_s;
};

/**
 * @brief Benchmarks a CUDA kernel call.
 * 
 * @tparam Func A lambda or function object that launches the kernel.
 * @param kernel_launch A function object that performs the kernel launch.
 * @param num_iters Number of iterations to run for timing.
 * @param num_warmup Number of warm-up iterations (not timed).
 * @param total_flops Total floating point operations performed per kernel launch.
 * @param total_bytes Total bytes moved (read + written) per kernel launch.
 * @return KernelStats Statistics about the execution time and performance.
 */
template<typename Func>
KernelStats benchmarkKernel(Func kernel_launch, int num_iters = 100, int num_warmup = 25, long long total_flops = 0, size_t total_bytes = 0) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Flush buffer: larger than L2 on any current Blackwell/Hopper GPU (up to ~100 MB).
    // Written to between every timed iteration so the kernel's working set is evicted
    // from L2 before each measurement — mirrors triton.testing.do_bench fast_flush=True.
    constexpr size_t kFlushBytes = 256ULL * 1024 * 1024;
    int *flush_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&flush_buf, kFlushBytes));

    // Warm-up (no flush needed — we want cache-warm behaviour during warmup)
    for (int i = 0; i < num_warmup; ++i) {
        kernel_launch();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> times;
    times.reserve(num_iters);

    for (int i = 0; i < num_iters; ++i) {
        // Evict working set from L2 before each timed rep
        CUDA_CHECK(cudaMemset(flush_buf, 0, kFlushBytes));

        CUDA_CHECK(cudaEventRecord(start));
        kernel_launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times.push_back(ms);
    }

    CUDA_CHECK(cudaFree(flush_buf));

    // Sort to find median and min/max
    std::sort(times.begin(), times.end());

    float sum = std::accumulate(times.begin(), times.end(), 0.0f);
    float avg = sum / num_iters;
    float median = (num_iters % 2 == 0) ? (times[num_iters / 2 - 1] + times[num_iters / 2]) / 2.0f : times[num_iters / 2];
    
    float sq_sum = std::inner_product(times.begin(), times.end(), times.begin(), 0.0f);
    float std_dev = std::sqrt(sq_sum / num_iters - avg * avg);

    KernelStats stats;
    stats.avg_ms = avg;
    stats.median_ms = median;
    stats.std_dev_ms = std_dev;
    stats.min_ms = times.front();
    stats.max_ms = times.back();
    
    // TFLOPS = (FLOPs / 10^12) / (ms / 1000) = FLOPs / (ms * 10^9)
    stats.tflops = (total_flops > 0) ? (static_cast<double>(total_flops) / (avg * 1e9)) : 0.0;
    
    // Bandwidth = (Bytes / 10^9) / (ms / 1000) = (Bytes / ms) / 10^6
    stats.bandwidth_gb_s = (total_bytes > 0) ? (static_cast<double>(total_bytes) / (avg * 1e6)) : 0.0;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return stats;
}

inline void displayStats(const std::string& name, const KernelStats& stats) {
    std::cout << "\n----------------------------------------------------------\n";
    std::cout << "Benchmark: " << name << "\n";
    std::cout << "----------------------------------------------------------\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Average Time:  " << std::setw(10) << stats.avg_ms << " ms\n";
    std::cout << "Median Time:   " << std::setw(10) << stats.median_ms << " ms\n";
    std::cout << "Std Dev:       " << std::setw(10) << stats.std_dev_ms << " ms\n";
    std::cout << "Min Time:      " << std::setw(10) << stats.min_ms << " ms\n";
    std::cout << "Max Time:      " << std::setw(10) << stats.max_ms << " ms\n";
    
    if (stats.tflops > 0) {
        std::cout << "Performance:   " << std::setw(10) << stats.tflops << " TFLOPS\n";
    }
    if (stats.bandwidth_gb_s > 0) {
        std::cout << "Bandwidth:     " << std::setw(10) << stats.bandwidth_gb_s << " GB/s\n";
    }
    std::cout << "----------------------------------------------------------\n";
}
