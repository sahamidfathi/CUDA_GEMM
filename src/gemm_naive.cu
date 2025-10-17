#include <cuda_runtime.h>
#include <cstdio>
#include "gemm.h"

// block size for the naive kernel, TILE_NAIVE*TILE_NAIVE
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

// error check macro
#define CUDA_CHECK(call) do { \
  cudaError_t err = (call); \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(1); \
  } \
} while(0)

// naive gemm kernel: each thread directly uses the global memory.
__global__ void gemm_naive_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int N) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= N || col >= N) return;

  float acc = 0.0f;
  for (int k = 0; k < N; ++k) {
    float a = A[row * N + k];
    float b = B[k * N + col];
    acc += a * b;
  }
  C[row * N + col] = acc;
}

extern "C" double launch_gemm_naive(const float* dA, const float* dB, float* dC, int N, int iter) {
  constexpr int BS = BLOCK_SIZE;
  dim3 block(BS, BS);
  dim3 grid((N + BS - 1) / BS, (N + BS - 1) / BS);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaDeviceSynchronize());

  // warmup run
  gemm_naive_kernel<<<grid, block>>>(dA, dB, dC, N);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iter; ++i)
    gemm_naive_kernel<<<grid, block>>>(dA, dB, dC, N);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return (double)ms / iter;
}




















