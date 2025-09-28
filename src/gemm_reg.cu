#include <cstdio>
#include <cuda_runtime.h>
#include "gemm.h"

// these macros need to be set in the Makefile or default values are used.
#ifndef REG_TS
#define REG_TS 64
#endif
#ifndef MR
#define MR 8
#endif
#ifndef NR
#define NR 8
#endif

static_assert(REG_TS % MR == 0 && REG_TS % NR == 0, "REG_TS must be divisible by MR and NR");

#define CUDA_CHECK(call) do { \
  cudaError_t err = (call); \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(1); \
  } \
} while(0)

// register-block kernel: each thread computes MRxNR microtile held in registers
__global__ void gemm_reg_kernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int N) {
  constexpr int TS = REG_TS;
  constexpr int _MR = MR;
  constexpr int _NR = NR;

  __shared__ float As[TS][TS + 1];
  __shared__ float Bs[TS][TS + 1];

  int bx = blockIdx.x, by = blockIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y;

  int blockRow = by * TS;
  int blockCol = bx * TS;

  int row0 = blockRow + ty * _MR;
  int col0 = blockCol + tx * _NR;

  float regs[_MR][_NR];
  #pragma unroll
  for (int i = 0; i < _MR; ++i)
    for (int j = 0; j < _NR; ++j)
      regs[i][j] = 0.0f;

  for (int k0 = 0; k0 < N; k0 += TS) {
    // each thread writes its small sub-block to shared memory
    #pragma unroll
    for (int i = 0; i < _MR; ++i) {
      #pragma unroll
      for (int j = 0; j < _NR; ++j) {
        int aRow = row0 + i;
        int aCol = k0 + tx * _NR + j;
        if (aRow < N && aCol < N)
          As[ty * _MR + i][tx * _NR + j] = A[aRow * N + aCol];
        else
          As[ty * _MR + i][tx * _NR + j] = 0.0f;

        int bRow = k0 + ty * _MR + i;
        int bCol = col0 + j;
        if (bRow < N && bCol < N)
          Bs[ty * _MR + i][tx * _NR + j] = B[bRow * N + bCol];
        else
          Bs[ty * _MR + i][tx * _NR + j] = 0.0f;
      }
    }

    __syncthreads();

    for (int k = 0; k < TS; ++k) {
      float avals[_MR];
      float bvals[_NR];
      #pragma unroll
      for (int i = 0; i < _MR; ++i) avals[i] = As[ty * _MR + i][k];
      #pragma unroll
      for (int j = 0; j < _NR; ++j) bvals[j] = Bs[k][tx * _NR + j];

      #pragma unroll
      for (int i = 0; i < _MR; ++i) {
        #pragma unroll
        for (int j = 0; j < _NR; ++j) {
          regs[i][j] += avals[i] * bvals[j];
        }
      }
    }

    __syncthreads();
  }

  // write back
  for (int i = 0; i < _MR; ++i) {
    for (int j = 0; j < _NR; ++j) {
      int r = row0 + i;
      int c = col0 + j;
      if (r < N && c < N) {
        C[r * N + c] = regs[i][j];
      }
    }
  }
}

extern "C" double launch_gemm_reg(const float* dA, const float* dB, float* dC, int N, int iter) {
  constexpr int TS = REG_TS;
  dim3 block(TS / NR, TS / MR);
  dim3 grid((N + TS - 1) / TS, (N + TS - 1) / TS);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaDeviceSynchronize());

  // warmup
  gemm_reg_kernel<<<grid, block>>>(dA, dB, dC, N);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iter; ++i)
    gemm_reg_kernel<<<grid, block>>>(dA, dB, dC, N);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return (double)ms / iter;
}

