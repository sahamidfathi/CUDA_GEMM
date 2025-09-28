#ifndef UTILS_H
#define UTILS_H

#include <cstddef>

// host utilities
void cpu_gemm_ref(const float* A, const float* B, float* C, int N);
void fill_random(float* M, int N, unsigned seed = 1221);
float max_abs_err(const float* A, const float* B, int N);

#endif // UTILS_H

