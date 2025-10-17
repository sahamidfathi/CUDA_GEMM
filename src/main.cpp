#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cuda_runtime.h>
#include "gemm.h"
#include "utils.h"

#define CUDA_CHECK(call) do { \
  cudaError_t err = (call); \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(1); \
  } \
} while(0)

int main(int argc, char** argv) {
  int N = 1024;
  int iters = 10;
  const char* mode = "shared";

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) N = atoi(argv[++i]);
    else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) iters = atoi(argv[++i]);
    else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) mode = argv[++i];
    else if (strcmp(argv[i], "--help") == 0) {
      printf("Usage: %s [--size N] [--iters iters] [--mode naive|shared|reg|regprefetch]\n", argv[0]);
      return 0;
    }
  }

  printf("GEMM modular demo: N=%d iters=%d mode=%s\n", N, iters, mode);

  size_t bytes = (size_t)N * N * sizeof(float);
  float *hA = (float*)malloc(bytes), *hB = (float*)malloc(bytes), *hC = (float*)malloc(bytes), *hC_ref = (float*)malloc(bytes);
  fill_random(hA, N, 1234);
  fill_random(hB, N, 4321);
  memset(hC, 0, bytes);
  memset(hC_ref, 0, bytes);

  float *dA = nullptr, *dB = nullptr, *dC = nullptr;
  CUDA_CHECK(cudaMalloc(&dA, bytes));
  CUDA_CHECK(cudaMalloc(&dB, bytes));
  CUDA_CHECK(cudaMalloc(&dC, bytes));

  CUDA_CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(dC, 0, bytes));

  double avg_ms = 0.0;
  if (strcmp(mode, "naive") == 0) {
    avg_ms = launch_gemm_naive(dA, dB, dC, N, iters);
  } else if (strcmp(mode, "shared") == 0) {
    avg_ms = launch_gemm_tiled(dA, dB, dC, N, iters);
  } else if (strcmp(mode, "reg") == 0) {
    avg_ms = launch_gemm_reg(dA, dB, dC, N, iters);
  } else if (strcmp(mode, "regprefetch") == 0) {
    avg_ms = launch_gemm_reg_async(dA, dB, dC, N, iters);
  } else {
    // other modes
  }

  CUDA_CHECK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));

  if (N <= 512) {
    printf("Validating against CPU reference (N <= 512)...\n");
    cpu_gemm_ref(hA, hB, hC_ref, N);
    float maxe = max_abs_err(hC, hC_ref, N);
    printf("max_abs_error = %g\n", maxe);
    if (maxe > 1e-2f) {
      printf("WARNING: large numeric difference\n");
    } else {
      printf("PASS: results close to CPU\n");
    }
  } else {
    printf("Skipping CPU validation for N > 512.\n");
  }

  double flops = 2.0 * (double)N * N * N; // one multiplication and almost one addition for N times for N*N elements of C
  double secs = 1e-3 * avg_ms;
  double gflops = flops / secs / 1e9;
  printf("Average kernel time: %.3f ms\n", avg_ms);
  printf("GFLOPS = %.3f\n", gflops);

  CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
  free(hA); free(hB); free(hC); free(hC_ref);
  return 0;
}

