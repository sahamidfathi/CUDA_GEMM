#ifndef GEMM_H
#define GEMM_H

#include <cstddef>

// avoid name mangling in case different compilers are used.
extern "C" {

// dA, dB, dC: device pointers, N: matrix size, iter: number of iterations, returns ave. ms per iteration.

double launch_gemm_naive(const float* dA, const float* dB, float* dC, int N, int iter);

double launch_gemm_tiled(const float* dA, const float* dB, float* dC, int N, int iter);

double launch_gemm_reg(const float* dA, const float* dB, float* dC, int N, int iter);

double launch_gemm_reg_async(const float* dA, const float* dB, float* dC, int N, int iter);
}

#endif // GEMM_H

