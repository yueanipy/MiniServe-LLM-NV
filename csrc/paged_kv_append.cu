#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

namespace {

template <typename scalar_t>
__global__ void paged_kv_append_graph_kernel(
    scalar_t* __restrict__ key_pool,
    scalar_t* __restrict__ value_pool,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const int64_t* __restrict__ slot_mapping,
    int layer_index,
    int num_blocks,
    int num_kv_heads,
    int block_size,
    int head_dim,
    int tokens) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t elements = static_cast<int64_t>(tokens) * num_kv_heads * head_dim;
  if (linear >= elements) {
    return;
  }
  const int dimension = linear % head_dim;
  const int64_t temp = linear / head_dim;
  const int kv_head = temp % num_kv_heads;
  const int token = temp / num_kv_heads;
  const int64_t slot = slot_mapping[token];
  const int physical_block = static_cast<int>(slot / block_size);
  const int offset = static_cast<int>(slot % block_size);
  const int64_t pool_index =
      (((((static_cast<int64_t>(layer_index) * num_blocks + physical_block) * num_kv_heads +
           kv_head) *
          block_size +
          offset) *
         head_dim) +
       dimension);
  key_pool[pool_index] = key[linear];
  value_pool[pool_index] = value[linear];
}

at::Tensor paged_kv_append_graph_cuda(
    at::Tensor key_pool,
    at::Tensor value_pool,
    int64_t layer_index,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& slot_mapping) {
  TORCH_CHECK(
      key_pool.is_cuda() && value_pool.is_cuda() && key.is_cuda() && value.is_cuda() &&
          slot_mapping.is_cuda(),
      "paged_kv_append_graph expects CUDA tensors");
  TORCH_CHECK(
      key_pool.device() == value_pool.device() && key_pool.device() == key.device() &&
          key_pool.device() == value.device() && key_pool.device() == slot_mapping.device(),
      "paged_kv_append_graph tensors must share one device");
  TORCH_CHECK(
      key_pool.is_contiguous() && value_pool.is_contiguous() && key.is_contiguous() &&
          value.is_contiguous() && slot_mapping.is_contiguous(),
      "paged_kv_append_graph tensors must be contiguous");
  TORCH_CHECK(
      key_pool.dim() == 5 && key_pool.sizes() == value_pool.sizes(),
      "paged K/V pool must have matching [layers,blocks,kv_heads,block_size,head_dim]");
  TORCH_CHECK(
      key.dim() == 3 && key.sizes() == value.sizes(),
      "new K/V must have matching [tokens,kv_heads,head_dim]");
  TORCH_CHECK(
      key.size(1) == key_pool.size(2) && key.size(2) == key_pool.size(4),
      "new K/V shape does not match pool layout");
  TORCH_CHECK(
      slot_mapping.dim() == 1 && slot_mapping.size(0) == key.size(0) &&
          slot_mapping.scalar_type() == at::ScalarType::Long,
      "slot_mapping must be int64 [tokens]");
  TORCH_CHECK(
      layer_index >= 0 && layer_index < key_pool.size(0),
      "layer_index is outside the paged pool");
  TORCH_CHECK(
      key_pool.scalar_type() == value_pool.scalar_type() &&
          key_pool.scalar_type() == key.scalar_type() && key.scalar_type() == value.scalar_type(),
      "paged K/V append dtype mismatch");
  TORCH_CHECK(
      key.scalar_type() == at::ScalarType::Float || key.scalar_type() == at::ScalarType::Half ||
          key.scalar_type() == at::ScalarType::BFloat16,
      "paged_kv_append_graph supports float32, float16, and bfloat16 only");

  c10::cuda::CUDAGuard guard(key_pool.device());
  const int64_t elements = key.numel();
  constexpr int threads = 256;
  const int blocks = static_cast<int>((elements + threads - 1) / threads);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(key_pool.device().index());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      key.scalar_type(),
      "miniserve_paged_kv_append_graph_cuda",
      [&] {
        paged_kv_append_graph_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            key_pool.data_ptr<scalar_t>(),
            value_pool.data_ptr<scalar_t>(),
            key.data_ptr<scalar_t>(),
            value.data_ptr<scalar_t>(),
            slot_mapping.data_ptr<int64_t>(),
            static_cast<int>(layer_index),
            static_cast<int>(key_pool.size(1)),
            static_cast<int>(key_pool.size(2)),
            static_cast<int>(key_pool.size(3)),
            static_cast<int>(key_pool.size(4)),
            static_cast<int>(key.size(0)));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return key_pool;
}

}  // namespace

TORCH_LIBRARY_IMPL(miniserve, CUDA, m) {
  m.impl("paged_kv_append_graph", &paged_kv_append_graph_cuda);
}
