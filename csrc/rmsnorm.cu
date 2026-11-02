#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <limits>
#include <type_traits>

namespace {

constexpr int kThreads = 256;

template <typename scalar_t, bool UseDouble>
__global__ void rmsnorm_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int64_t hidden_size,
    float eps) {
  // 大 hidden size 用 FP64 累积，避免深层链上的舍入漂移；小模型保留既有
  // FP32 路径，以维持已冻结的 SmolLM2/Qwen3-0.6B 数值行为。
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t base = row * hidden_size;
  using Accum = std::conditional_t<UseDouble, double, float>;
  Accum square_sum = Accum(0);
  for (int64_t column = threadIdx.x; column < hidden_size; column += blockDim.x) {
    const Accum value = static_cast<Accum>(x[base + column]);
    square_sum += value * value;
  }

  __shared__ Accum partial[kThreads];
  partial[threadIdx.x] = square_sum;
  __syncthreads();
  // blockDim 固定 256，是 2 的幂；树形归约不会访问越界 shared memory。
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      partial[threadIdx.x] += partial[threadIdx.x + offset];
    }
    __syncthreads();
  }
  const float inverse_rms = UseDouble
      ? static_cast<float>(
            rsqrt(partial[0] / static_cast<Accum>(hidden_size) + static_cast<Accum>(eps)))
      : rsqrtf(partial[0] / static_cast<Accum>(hidden_size) + static_cast<Accum>(eps));
  for (int64_t column = threadIdx.x; column < hidden_size; column += blockDim.x) {
    const float normalized = static_cast<float>(x[base + column]) * inverse_rms;
    // Match Transformers Qwen3RMSNorm: round the normalized value back to the
    // model dtype before multiplying by the same-dtype weight.
    const scalar_t rounded = static_cast<scalar_t>(normalized);
    output[base + column] = rounded * weight[column];
  }
}

template <typename scalar_t>
__global__ void rmsnorm_warp_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    int64_t hidden_size,
    float eps) {
  // Match PyTorch's contiguous FP32 mean reduction used by HF Qwen3RMSNorm:
  // one warp per output row, four independent vector-lane accumulators, then
  // the fixed shuffle tree.  hidden_size must be divisible by four.
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t base = row * hidden_size;
  const int64_t vector_count = hidden_size / 4;
  const int lane = threadIdx.x;
  float accumulator0 = 0.0f;
  float accumulator1 = 0.0f;
  float accumulator2 = 0.0f;
  float accumulator3 = 0.0f;
  for (int64_t vector_index = lane; vector_index < vector_count; vector_index += 32) {
    const int64_t column = vector_index * 4;
    const float value0 = static_cast<float>(x[base + column]);
    const float value1 = static_cast<float>(x[base + column + 1]);
    const float value2 = static_cast<float>(x[base + column + 2]);
    const float value3 = static_cast<float>(x[base + column + 3]);
    accumulator0 += value0 * value0;
    accumulator1 += value1 * value1;
    accumulator2 += value2 * value2;
    accumulator3 += value3 * value3;
  }
  float square_sum = accumulator0 + accumulator1;
  square_sum += accumulator2;
  square_sum += accumulator3;
  for (int offset = 16; offset > 0; offset >>= 1) {
    square_sum += __shfl_down_sync(0xffffffffu, square_sum, offset);
  }
  float inverse_rms = 0.0f;
  if (lane == 0) {
    const float variance =
        square_sum * (1.0f / static_cast<float>(hidden_size));
    inverse_rms = rsqrtf(variance + eps);
  }
  inverse_rms = __shfl_sync(0xffffffffu, inverse_rms, 0);

  for (int64_t vector_index = lane; vector_index < vector_count; vector_index += 32) {
    const int64_t column = vector_index * 4;
#pragma unroll
    for (int item = 0; item < 4; ++item) {
      const float normalized =
          static_cast<float>(x[base + column + item]) * inverse_rms;
      const scalar_t rounded = static_cast<scalar_t>(normalized);
      output[base + column + item] = rounded * weight[column + item];
    }
  }
}

at::Tensor rmsnorm_cuda(
    const at::Tensor& x,
    const at::Tensor& weight,
    double eps) {
  TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "rmsnorm expects CUDA tensors");
  TORCH_CHECK(x.device() == weight.device(), "x and weight must share one CUDA device");
  TORCH_CHECK(x.is_contiguous() && weight.is_contiguous(), "x and weight must be contiguous");
  TORCH_CHECK(x.dim() >= 1 && x.numel() > 0, "x must be non-empty and have at least one dimension");
  TORCH_CHECK(weight.dim() == 1, "weight must be one-dimensional");
  TORCH_CHECK(x.size(-1) == weight.numel(), "weight length must equal x hidden size");
  TORCH_CHECK(x.scalar_type() == weight.scalar_type(), "x and weight dtype must match");
  TORCH_CHECK(
      x.scalar_type() == at::ScalarType::Float ||
          x.scalar_type() == at::ScalarType::Half ||
          x.scalar_type() == at::ScalarType::BFloat16,
      "rmsnorm supports float32, float16, and bfloat16 only");
  TORCH_CHECK(eps > 0.0, "eps must be positive");

  c10::cuda::CUDAGuard guard(x.device());
  // The accumulator width is selected in the launch branch above; the output
  // multiplication order matches the HF Qwen3RMSNorm reference for every dtype.
  at::Tensor output = at::empty_like(x);
  const int64_t hidden_size = x.size(-1);
  const int64_t rows = x.numel() / hidden_size;
  TORCH_CHECK(rows <= static_cast<int64_t>(std::numeric_limits<int>::max()), "too many RMSNorm rows");
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.device().index());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x.scalar_type(),
      "miniserve_rmsnorm_cuda",
      [&] {
        if (hidden_size >= 128 && hidden_size % 4 == 0) {
          rmsnorm_warp_kernel<scalar_t><<<static_cast<int>(rows), 32, 0, stream>>>(
              x.data_ptr<scalar_t>(),
              weight.data_ptr<scalar_t>(),
              output.data_ptr<scalar_t>(),
              hidden_size,
              static_cast<float>(eps));
        } else if (hidden_size <= 256 || hidden_size > 1024) {
          rmsnorm_kernel<scalar_t, true><<<static_cast<int>(rows), kThreads, 0, stream>>>(
              x.data_ptr<scalar_t>(),
              weight.data_ptr<scalar_t>(),
              output.data_ptr<scalar_t>(),
              hidden_size,
              static_cast<float>(eps));
        } else {
          rmsnorm_kernel<scalar_t, false><<<static_cast<int>(rows), kThreads, 0, stream>>>(
              x.data_ptr<scalar_t>(),
              weight.data_ptr<scalar_t>(),
              output.data_ptr<scalar_t>(),
              hidden_size,
              static_cast<float>(eps));
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace

TORCH_LIBRARY_IMPL(miniserve, CUDA, m) {
  m.impl("rmsnorm", &rmsnorm_cuda);
}
