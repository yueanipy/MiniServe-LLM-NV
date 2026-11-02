#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

// 教学版归约工具：调用方提供至少 THREADS 个 float shared-memory 槽。
// 每一轮都同步，清晰优先；Phase 8 后续可在 correctness 不变时迭代 warp shuffle。
template <int THREADS>
__device__ __forceinline__ float miniserve_block_reduce_sum(
    float value,
    float* shared) {
  shared[threadIdx.x] = value;
  __syncthreads();
  for (int offset = THREADS / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      shared[threadIdx.x] += shared[threadIdx.x + offset];
    }
    __syncthreads();
  }
  return shared[0];
}
