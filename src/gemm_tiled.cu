#include <cuda_runtime.h>
#include <cstdio>
#include "gemm.h"

// tile size for tiled kernel, tune
#ifndef SIMPLE_TS
#define SIMPLE_TS 16
#endif

// error check macro
#define CUDA_CHECK(call) do { \
  cudaError_t err = (call); \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(1); \
  } \
} while(0)

/* for better visualization:
Matrix A (64x64)           Matrix B (64x64)           Matrix C (64x64)
+---+---+---+---+          +---+---+---+---+          +---+---+---+---+
|A00|A01|A02|A03|          |B00|B01|B02|B03|          |C00|C01|C02|C03|
+---+---+---+---+          +---+---+---+---+          +---+---+---+---+
|A10|A11|A12|A13|          |B10|B11|B12|B13|          |C10|C11|C12|C13|
+---+---+---+---+          +---+---+---+---+          +---+---+---+---+
|A20|A21|A22|A23|          |B20|B21|B22|B23|          |C20|C21|C22|C23|
+---+---+---+---+          +---+---+---+---+          +---+---+---+---+
|A30|A31|A32|A33|          |B30|B31|B32|B33|          |C30|C31|C32|C33|
+---+---+---+---+          +---+---+---+---+          +---+---+---+---+
each threadblock computes a tile in C, e.g. C11. for that it needs to 
multiply row A10..A13 by col B01..B31.
*/

// tiled kernel: each thread computes one element of a TSxTS block
__global__ void gemm_tiled_simple_kernel(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C,
                                         int N) {
  constexpr int TS = SIMPLE_TS;
  __shared__ float As[TS][TS + 1];
  __shared__ float Bs[TS][TS + 1];

  int bx = blockIdx.x, by = blockIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y;

  int row = by * TS + ty;
  int col = bx * TS + tx;

  float acc = 0.0f;

  for (int k0 = 0; k0 < N; k0 += TS) {
    // load A tiles into As
    if (row < N && (k0 + tx) < N)
      As[ty][tx] = A[row * N + (k0 + tx)];
    else
      As[ty][tx] = 0.0f;
      
    // load B tiles into Bs
    if ((k0 + ty) < N && col < N)
      Bs[ty][tx] = B[(k0 + ty) * N + col];
    else
      Bs[ty][tx] = 0.0f;

    __syncthreads();

    #pragma unroll
    for (int k = 0; k < TS; ++k) {
      acc += As[ty][k] * Bs[k][tx];
    }

    __syncthreads();
  }

  if (row < N && col < N) C[row * N + col] = acc;
}

extern "C" double launch_gemm_tiled(const float* dA, const float* dB, float* dC, int N, int iter) {
  constexpr int TS = SIMPLE_TS;
  dim3 block(TS, TS);
  dim3 grid((N + TS - 1) / TS, (N + TS - 1) / TS);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaDeviceSynchronize());

  // warmup run
  gemm_tiled_simple_kernel<<<grid, block>>>(dA, dB, dC, N);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iter; ++i)
    gemm_tiled_simple_kernel<<<grid, block>>>(dA, dB, dC, N);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return (double)ms / iter;
}


