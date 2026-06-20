# CUDA Kernel Development and Optimization Guidelines

## 1. Performance Measurement

### 1.1 Measurement Methodology

All kernel timing must be performed via CUDA events so that only GPU compute time is captured — that is, the time from when the GPU begins executing the kernel to when it completes. This ensures a fair, device-only comparison that excludes host-side scheduling noise.

Timing must be taken from the high-level API call, not from the raw kernel launch. When a kernel runs as part of a real workload, the latency of the CUDA API call itself (e.g. `cudnnConvolutionForward`, `torch.nn.functional.scaled_dot_product_attention`) is included in every step. Measuring only the raw kernel compute time and reporting a win against a competitor is misleading if the API overhead reverses that advantage in practice.

### 1.2 Cache Flushing

The L1 cache must be flushed before every individual timing measurement. A kernel launch can benefit from data left hot in L1 by the previous launch, artificially lowering its measured latency. Without flushing, results are not reproducible across launch order and cannot be compared honestly.

### 1.3 Warmup and Iteration Protocol

A proper warmup phase must precede all timing. Cold launches incur driver initialization and JIT overhead that do not reflect steady-state performance. After warmup, a sufficient number of iterations must be run and the following statistics recorded for each kernel version:

| Metric | Description |
|---|---|
| Mean time | Average latency across all iterations |
| Median time | Primary comparison metric — robust to outliers |
| Standard deviation | Indicator of measurement stability |
| Min time | Best-case hardware ceiling |
| Max time | Worst-case latency bound |
| Throughput | Achieved TFLOPS |
| Bandwidth | Achieved GB/s |

The median is the primary figure used for comparison. Multiple independent runs should be conducted and their medians averaged to further reduce variance.

### 1.4 Competitor Benchmarking

When comparing against a competitor kernel (e.g. PyTorch, cuDNN, CUTLASS reference), both kernels must receive identical inputs and be measured using the same CUDA events protocol. This ensures a true apples-to-apples comparison of GPU compute time.

---

## 2. Correctness Verification

### 2.1 Ground Truth Reference

A host-side (CPU) reference implementation must be written for every kernel. This implementation is the ground truth. Because FP32 arithmetic on GPUs is susceptible to precision loss from accumulation order and tensor core rounding, a numerical tolerance of **1e-7 to 1e-8** is used as the correctness threshold for element-wise comparison against the reference.

### 2.2 Error Metrics

Three error metrics are computed for every kernel output against the ground truth:

**Mean Absolute Error (MAE)**
```
MAE = mean(|output - reference|)
```
Measures the average magnitude of error across all elements. A low MAE indicates the kernel is close to the reference on average, but can mask large isolated errors.

**Maximum Absolute Error (MaxAE)**
```
MaxAE = max(|output - reference|)
```
Measures the worst-case deviation at any single element. This is the strictest correctness signal — a kernel passes only if its MaxAE is within the defined threshold.

**Relative Error**
```
RelErr = mean(|output - reference| / (|reference| + epsilon))
```
Normalizes the error by the magnitude of the reference value. Useful for identifying precision loss on large-magnitude outputs that would look acceptable in absolute terms but are proportionally inaccurate.

### 2.3 Correctness Gates

A kernel must pass both of the following gates before any performance comparison is made:

1. All element-wise values pass the absolute threshold (1e-7 or 1e-8) against the host reference.
2. MAE, MaxAE, and relative error are within acceptable bounds compared to the equivalent PyTorch kernel receiving the same inputs.

No performance result from a kernel that has not passed both gates is considered valid.

---

## 3. Development Workflow

### 3.1 Problem Scoping

When a kernel is assigned, the first step is to record the target problem size (batch, sequence length, head dim, dtype, etc.) and benchmark the latency of PyTorch and any other relevant competitor kernels at that size. These numbers define the performance target.

### 3.2 Baseline Implementation (Version 0)

Before any optimization, write the simplest possible implementation that produces correct results. This is Version 0. It does not need to be fast — it needs to be correct and readable. Version 0 serves as the functional reference for all subsequent versions and as the first data point in the performance progression.

### 3.3 Roofline Analysis

Once Version 0 is running and correct, run an Nsight Compute profile and perform a roofline analysis. This determines whether the kernel is compute bound or memory bandwidth bound at the target problem size, which in turn determines which class of optimizations will be effective. No optimization work should begin without this analysis — optimizing the wrong bottleneck wastes time.

### 3.4 Incremental Optimization

Optimizations are applied one phase at a time. Each phase constitutes a new versioned kernel. Every version must be measured for both latency and precision before the next phase begins. Typical optimization phases, applied in order of expected impact:

1. Algorithm selection (e.g. tiling strategy, online softmax, split-K)
2. Shared memory layout and padding (bank conflict elimination)
3. Async memory pipelines (cp.async double buffering)
4. Tensor core utilization (WMMA / warp-level MMA)
5. Instruction-level optimizations (vectorized loads, loop unrolling)
6. Occupancy tuning (register pressure, launch bounds)

The priority and applicability of each phase is determined by the roofline result. A memory-bound kernel benefits most from phases 1–3; a compute-bound kernel benefits most from phases 4–6.

### 3.5 Algorithm Selection Principle

The best algorithm is problem-size dependent. An approach that is optimal for sequence length 512 may not be optimal for sequence length 4096. This is why Version 0 must exist and be built upon incrementally — the version history provides empirical data across all optimization decisions, making it possible to select the best version for a given problem size rather than assuming a single implementation is universally optimal.

---

## 4. Profiling Standards

### 4.1 Per-Version Profiling

An Nsight Compute report must be generated at the following stages:

- After Version 0 (baseline roofline)
- After each major algorithmic change
- For the final version
- For the competitor kernel being beaten

### 4.2 Key Metrics

The following metrics must be recorded and interpreted for each profile:

| Category | Metric | What it tells you |
|---|---|---|
| Compute | `sm__pipe_tensor_op_hmma_cycles_active` | Tensor core utilization |
| Compute | Achieved TFLOPS vs peak | Distance from compute roof |
| Memory | `l1tex__t_bytes` | L1 / shared memory traffic |
| Memory | `dram__bytes` | HBM traffic |
| Memory | Achieved GB/s vs peak | Distance from memory roof |
| Roofline | Arithmetic intensity | Whether kernel is memory or compute bound |
| Occupancy | Active warps / max warps | Warp-level latency hiding effectiveness |

### 4.3 Developing Profiling Intuition

Raw metric numbers are not sufficient. The engineer must develop intuition for why a kernel is underperforming based on the combination of metrics. For example: high DRAM traffic with low tensor core utilization indicates memory boundedness; high register usage with low occupancy indicates register pressure limiting warp count. Each optimization decision should be traceable back to a specific metric signal from the profile.

---

## 5. Repository and Version Control

### 5.1 Separate Development Repository

All kernel versions are maintained in a dedicated development repository, separate from the main production repository. Every version is preserved — no version is deleted or overwritten. The commit history serves as the optimization log.

### 5.2 Integration Criteria

A kernel is integrated into the main repository only when all of the following conditions are met:

- It produces correct results within the defined precision thresholds against both the host reference and the PyTorch baseline
- Its median latency strictly beats the competitor at the target problem size
- A full Nsight Compute profile exists for both the new kernel and the competitor
- A final documentation report has been completed (see Section 6)

No kernel is integrated on the basis of partial results or incomplete profiling.

---

## 6. Final Documentation

Upon completion of a kernel, a report must be produced as a Jupyter notebook containing the following sections:

1. **Problem definition** — target operation, problem size, dtype, hardware
2. **Competitor baseline** — latency, throughput, bandwidth, and correctness metrics for the reference kernel
3. **Version history** — latency and precision table across all versions with a brief description of what changed in each
4. **Roofline analysis** — annotated roofline chart showing the progression of versions from baseline to final
5. **Final kernel profile** — full Nsight Compute metrics with interpretation
6. **Correctness report** — MAE, MaxAE, and relative error vs both host reference and PyTorch
7. **Conclusion** — summary of which version is recommended for which problem sizes, with the justification

The notebook must be self-contained and reproducible — running it end to end on the target hardware must regenerate all results from scratch.