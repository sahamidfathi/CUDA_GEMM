#include <cstdio>
#include <cuda_runtime.h>
#include "gemm.h"

// can be set in the Makefile
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

// Asynchronous-prefetch style + double-buffering kernel
// Each thread computes MRxNR micro-tile in registers.
// Uses two shared buffers As[2] and Bs[2] and alternates between them.
// A, B, C are N*N matrices.
__global__ void gemm_reg_async_kernel(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ C,
                                      int N) {
  constexpr int TS = REG_TS;
  constexpr int _MR = MR;
  constexpr int _NR = NR;

  // double-buffered shared memory: 2 buffers of TS x (TS+1) each (pad avoids bank conflicts)
  __shared__ float As_buf[2][TS][TS + 1];
  __shared__ float Bs_buf[2][TS][TS + 1];

  int bx = blockIdx.x, by = blockIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y;

  int blockRow = by * TS;
  int blockCol = bx * TS;

  int row0 = blockRow + ty * _MR;
  int col0 = blockCol + tx * _NR;

  // register tile
  float regs[_MR][_NR];
  #pragma unroll
  for (int i = 0; i < _MR; ++i)
    for (int j = 0; j < _NR; ++j)
      regs[i][j] = 0.0f;

  // number of K-tiles
  int Ktiles = (N + TS - 1) / TS;

  // current buffer index: 0 or 1
  int cur = 0;

  // initial prefetch of tile ktile = 0 into buffer cur
  int ktile = 0;
  if (ktile < Ktiles) {
    int k0 = ktile * TS;
    // each thread writes its small sub-block to shared memory cur
    #pragma unroll
    for (int i = 0; i < _MR; ++i) {
      #pragma unroll
      for (int j = 0; j < _NR; ++j) {
        int aRow = row0 + i;
        int aCol = k0 + tx * _NR + j;
        if (aRow < N && aCol < N)
          As_buf[cur][ty * _MR + i][tx * _NR + j] = A[aRow * N + aCol];
        else
          As_buf[cur][ty * _MR + i][tx * _NR + j] = 0.0f;

        int bRow = k0 + ty * _MR + i;
        int bCol = col0 + j;
        if (bRow < N && bCol < N)
          Bs_buf[cur][ty * _MR + i][tx * _NR + j] = B[bRow * N + bCol];
        else
          Bs_buf[cur][ty * _MR + i][tx * _NR + j] = 0.0f;
      }
    }
  }
  // make sure the initial load is visible to all threads
  __syncthreads();

  // iterate over k-tiles. For each iteration:
  //  - compute using buffer cur
  //  - prefetch next tile into buffer next
  //  - swap cur and next
  for (ktile = 0; ktile < Ktiles; ++ktile) {
    // compute on current buffer: TS entries along k dimension for this tile
    #pragma unroll
    for (int k = 0; k < TS; ++k) {
      // gather avals and bvals from shared (current buffer)
      float avals[_MR];
      float bvals[_NR];

      #pragma unroll
      for (int i = 0; i < _MR; ++i) {
        int a_r = ty * _MR + i;
        avals[i] = As_buf[cur][a_r][k];
      }
      #pragma unroll
      for (int j = 0; j < _NR; ++j) {
        int b_c = tx * _NR + j;
        bvals[j] = Bs_buf[cur][k][b_c];
      }

      #pragma unroll
      for (int i = 0; i < _MR; ++i)
        #pragma unroll
        for (int j = 0; j < _NR; ++j)
          regs[i][j] += avals[i] * bvals[j];
    }

    // prefetch the next tile into other buffer (overlapping latency with previous arithmetic)
    int next = cur ^ 1;
    int nextKtile = ktile + 1;
    if (nextKtile < Ktiles) {
      int k0 = nextKtile * TS;
      // load next tile into shared memory 'next'
      #pragma unroll
      for (int i = 0; i < _MR; ++i) {
        #pragma unroll
        for (int j = 0; j < _NR; ++j) {
          int aRow = row0 + i;
          int aCol = k0 + tx * _NR + j;
          if (aRow < N && aCol < N)
            As_buf[next][ty * _MR + i][tx * _NR + j] = A[aRow * N + aCol];
          else
            As_buf[next][ty * _MR + i][tx * _NR + j] = 0.0f;

          int bRow = k0 + ty * _MR + i;
          int bCol = col0 + j;
          if (bRow < N && bCol < N)
            Bs_buf[next][ty * _MR + i][tx * _NR + j] = B[bRow * N + bCol];
          else
            Bs_buf[next][ty * _MR + i][tx * _NR + j] = 0.0f;
        }
      }
    }

    // ensure we have completed writing the 'next' buffer before it's read next iteration
    __syncthreads();
    
    cur = next;
  }

  // write back result matrix C
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

extern "C" double launch_gemm_reg_async(const float* dA, const float* dB, float* dC, int N, int iter) {
  constexpr int TS = REG_TS;
  dim3 block(TS / NR, TS / MR);
  dim3 grid((N + TS - 1) / TS, (N + TS - 1) / TS);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaDeviceSynchronize());

  // warmup
  gemm_reg_async_kernel<<<grid, block>>>(dA, dB, dC, N);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iter; ++i)
    gemm_reg_async_kernel<<<grid, block>>>(dA, dB, dC, N);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return (double)ms / iter;
}

