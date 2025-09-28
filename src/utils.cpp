#include <random>
#include <cmath>
#include "utils.h"

// CPU GEMM reference
void cpu_gemm_ref(const float* A, const float* B, float* C, int N) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      double sum = 0.0;
      for (int k = 0; k < N; ++k) {
        sum += static_cast<double>(A[i * N + k]) * static_cast<double>(B[k * N + j]);
      }
      C[i * N + j] = static_cast<float>(sum);
    }
  }
}

void fill_random(float* M, int N, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  const size_t size = static_cast<size_t>(N) * N;
  for (size_t i = 0; i < size; ++i) M[i] = dist(rng);
}

float max_abs_err(const float* A, const float* B, int N) {
  float maxe = 0.0f;
  const size_t size = static_cast<size_t>(N) * N;
  for (size_t i = 0; i < size; ++i) {
    float e = fabsf(A[i] - B[i]);
    if (e > maxe) maxe = e;
  }
  return maxe;
}

