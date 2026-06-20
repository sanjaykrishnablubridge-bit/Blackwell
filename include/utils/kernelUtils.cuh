#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <random>
#include <iterator>
#include <stdlib.h>
#include <time.h>
#include <cuda_bf16.h>

//* Helper function to initialize random values to a vector
template<typename T>
void initVec(std::vector<T>& vec){
  size_t n = vec.size();
  
  std::random_device rd;
  std::mt19937 mersenne_engine(rd());

  std::uniform_real_distribution<T> dist(0.0f,1.0f);
  auto gen = [&]() { return dist(mersenne_engine); };

  std::generate(vec.begin(), vec.end(), gen);
}

//* Helper function to display the vector
template<typename T>
void displayVector(std::vector<T> vec, size_t digits = 10){
  for(size_t i = 0; i < digits; ++i){
    std::cout << vec[i] << " ";
  } std::cout << "\n";
}

//* Helper function to display a matrix
void displayMatrix(float* Mat, int rows, int cols){
  float *iM = Mat;
  printf("Matrix: {%d, %d}\n", rows, cols);
    for(int iy = 0; iy < cols; iy++){
      for(int ix = 0; ix < rows; ++ix){
        printf("%f ", iM[ix]);
      }
      iM += cols;
      printf("\n");
    }
    printf("\n");
}

#include <cmath>

// //* report max/mean absolute error and max relative error between two buffers
// void reportPrecision(const char *label, float *ref, float *got, size_t n){
//   double max_abs = 0, sum_abs = 0, max_rel = 0;
//   for(size_t i = 0; i < n; ++i){
//     double diff = std::abs((double)ref[i] - (double)got[i]);
//     double rel  = diff / (1e-8 + 1e-5 * std::abs((double)ref[i]));
//     max_abs  = std::max(max_abs, diff);
//     max_rel  = std::max(max_rel, rel);
//     sum_abs += diff;
//   }
//   std::cout << std::scientific << std::setprecision(3);
//   std::cout << label
//             << "  max_abs=" << max_abs
//             << "  mean_abs=" << sum_abs / n
//             << "  max_rel=" << max_rel << "\n";
//   std::cout << std::defaultfloat;
// }

//* report max/mean absolute error and max relative error between two buffers
void reportPrecision(const char *label, float *ref, float *got, size_t n){
  double max_abs = 0, sum_abs = 0, max_rel = 0;
  for(size_t i = 0; i < n; ++i){
    double diff = std::abs((double)ref[i] - (double)got[i]);
    double denom = std::abs((double)ref[i]);
    double rel  = (denom > 1e-5) ? diff / denom : diff;
    max_abs  = std::max(max_abs, diff);
    max_rel  = std::max(max_rel, rel);
    sum_abs += diff;
  }
  std::cout << std::scientific << std::setprecision(3);
  std::cout << label
            << "  max_abs=" << max_abs
            << "  mean_abs=" << sum_abs / n
            << "  max_rel=" << max_rel << "\n";
  std::cout << std::defaultfloat;
}

//* helper function to verify the result
//* uses |a-b| <= atol + rtol*|b|  (matches torch.testing.assert_close defaults for float32)
//* atol=1e-5, rtol=1.3e-6 — tolerates float32 non-associativity from parallel reductions
void checkResult(float *hostRef, float *gpuRef, const size_t n,
                 float atol = 1e-5f, float rtol = 1.3e-6f){
  bool match = true;
  for(size_t i = 0; i < n; ++i){
    float diff = std::abs(hostRef[i] - gpuRef[i]);
    float tol  = atol + rtol * std::abs(hostRef[i]);
    if(diff > tol){
      match = false;
      std::cout << "\033[31m" << "Values don't match at index " << i << "!\n";
      std::cout << std::fixed << std::setprecision(8);
      std::cout << "Ref: " << hostRef[i] << ", GPU: " << gpuRef[i]
                << "  diff=" << diff << " tol=" << tol << "\n" << "\033[0m";
      break;
    }
  }
  if(match){
    std::cout << "\033[32m" << "Values match!\n" << "\033[0m";
  }
}

//* helper function to check the error status
#define CUDA_CHECK(call){ \
  const cudaError_t status = call; \
  if(status != cudaSuccess){ \
    printf("Error: %s : %d\n", __FILE__, __LINE__); \
    printf("code : %d , reason: %s\n", status, cudaGetErrorString(status)); \
    exit(1); \
  } \
}

//* helper function to initailize data 
void initPtr(float *ptr, int size){
  std::random_device rd;
  std::mt19937 mersenne_engine(rd());

  std::uniform_real_distribution<float> dist(0.0f,1.0f);
  auto gen = [&]() { return dist(mersenne_engine); };

  for(int i = 0; i < size; ++i){
    ptr[i] = (float)gen();
  }
}

//* helper function to initialize data in bfloat16
//* values are drawn in float [0,1) then rounded to bf16 (matches the float initPtr range)
void initPtr(__nv_bfloat16 *ptr, int size){
  std::random_device rd;
  std::mt19937 mersenne_engine(rd());

  std::uniform_real_distribution<float> dist(0.0f,1.0f);
  auto gen = [&]() { return dist(mersenne_engine); };

  for(int i = 0; i < size; ++i){
    ptr[i] = __float2bfloat16(gen());
  }
}

//* helper function to allocate memory on GPU
template<typename T>
void allocateDevice(T** d_ptr, size_t size) {
    if (size == 0) {
        *d_ptr = nullptr;
        return;
    }
    CUDA_CHECK(cudaMalloc(d_ptr, size));
}

//* helper function to transfer data from Host to Device
template<typename T>
void copyToDevice(T* d_ptr, const T* h_ptr, size_t size) {
    if (size == 0) return;
    if (d_ptr == nullptr || h_ptr == nullptr) {
        fprintf(stderr, "Error: Null pointer in copyToDevice\n");
        exit(1);
    }
    CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice));
}

//* helper function to transfer data from Device to Host
template<typename T>
void copyToHost(T* h_ptr, const T* d_ptr, size_t size) {
    if (size == 0) return;
    if (d_ptr == nullptr || h_ptr == nullptr) {
        fprintf(stderr, "Error: Null pointer in copyToHost\n");
        exit(1);
    }
    CUDA_CHECK(cudaMemcpy(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost));
}

//* Legacy support for old names (to avoid breaking other files if any)
inline void to_gpu(float *h_ptr, float *&d_ptr, size_t NBytes){
  allocateDevice(&d_ptr, NBytes);
  copyToDevice(d_ptr, h_ptr, NBytes);
}

inline void to_cpu(float *h_ptr, float *d_ptr, size_t NBytes){
  copyToHost(h_ptr, d_ptr, NBytes);
}

//* Load a raw float32 binary file written by numpy/torch (.numpy().tofile())
inline void loadBin(const char *path, float *buf, size_t n){
  FILE *f = fopen(path, "rb");
  if(!f){ fprintf(stderr, "loadBin: cannot open %s\n", path); exit(1); }
  size_t read = fread(buf, sizeof(float), n, f);
  fclose(f);
  if(read != n){
    fprintf(stderr, "loadBin: expected %zu floats, got %zu from %s\n", n, read, path);
    exit(1);
  }
}