#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cmath>
#include <cfloat>
#include <limits>

#include "cuda_utils.cuh"

namespace {

template <typename scalar_t, int kHeadDim>
__global__ void paged_decode_attention_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key_cache,
    const scalar_t* __restrict__ value_cache,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ sequence_lengths,
    scalar_t* __restrict__ output,
    int num_query_heads,
    int num_kv_heads,
    int block_size,
    int max_blocks_per_sequence,
    float scale) {
  // 一个 CTA 负责一个 (batch, query_head)，线程与 head_dim 一一对应。
  const int pair_index = static_cast<int>(blockIdx.x);
  const int batch_index = pair_index / num_query_heads;
  const int query_head = pair_index % num_query_heads;
  const int group_size = num_query_heads / num_kv_heads;
  const int kv_head = query_head / group_size;
  const int sequence_length = sequence_lengths[batch_index];
  const int dimension = static_cast<int>(threadIdx.x);

  extern __shared__ float shared[];
  float* partial = shared;
  float* scores = shared + kHeadDim;

  const int64_t query_base =
      (static_cast<int64_t>(batch_index) * num_query_heads + query_head) * kHeadDim;
  const float query_value = static_cast<float>(query[query_base + dimension]);

  // V1 naive/stable：逐 token 计算 Q·K，block reduction 后把 score 放入 shared。
  for (int position = 0; position < sequence_length; ++position) {
    const int logical_block = position / block_size;
    const int offset = position % block_size;
    const int physical_block =
        block_tables[batch_index * max_blocks_per_sequence + logical_block];
    const int64_t cache_index =
        (((static_cast<int64_t>(physical_block) * num_kv_heads + kv_head) * block_size + offset) *
         kHeadDim) +
        dimension;
    const float product = query_value * static_cast<float>(key_cache[cache_index]);
    const float dot = miniserve_block_reduce_sum<kHeadDim>(product, partial);
    if (dimension == 0) {
      scores[position] = dot * scale;
    }
    __syncthreads();
  }

  // thread 0 使用稳定 softmax 计算 max 与 denominator；shared 中只保留 POD。
  if (dimension == 0) {
    float maximum = -FLT_MAX;
    for (int position = 0; position < sequence_length; ++position) {
      maximum = fmaxf(maximum, scores[position]);
    }
    float denominator = 0.0f;
    for (int position = 0; position < sequence_length; ++position) {
      denominator += expf(scores[position] - maximum);
    }
    partial[0] = maximum;
    partial[1] = denominator;
  }
  __syncthreads();

  const float maximum = partial[0];
  const float denominator = partial[1];
  float weighted_value = 0.0f;
  for (int position = 0; position < sequence_length; ++position) {
    const int logical_block = position / block_size;
    const int offset = position % block_size;
    const int physical_block =
        block_tables[batch_index * max_blocks_per_sequence + logical_block];
    const int64_t cache_index =
        (((static_cast<int64_t>(physical_block) * num_kv_heads + kv_head) * block_size + offset) *
         kHeadDim) +
        dimension;
    const float probability = expf(scores[position] - maximum) / denominator;
    weighted_value += probability * static_cast<float>(value_cache[cache_index]);
  }
  output[query_base + dimension] = static_cast<scalar_t>(weighted_value);
}

at::Tensor paged_decode_attention_cuda_impl(
    const at::Tensor& query,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const at::Tensor& block_tables,
    const at::Tensor& sequence_lengths,
    double scale,
    bool validate_metadata_values) {
  TORCH_CHECK(
      query.is_cuda() && key_cache.is_cuda() && value_cache.is_cuda() &&
          block_tables.is_cuda() && sequence_lengths.is_cuda(),
      "paged_decode_attention expects CUDA tensors");
  TORCH_CHECK(
      query.device() == key_cache.device() && query.device() == value_cache.device() &&
          query.device() == block_tables.device() && query.device() == sequence_lengths.device(),
      "all paged_decode_attention tensors must share one CUDA device");
  TORCH_CHECK(
      query.is_contiguous() && key_cache.is_contiguous() && value_cache.is_contiguous() &&
          block_tables.is_contiguous() && sequence_lengths.is_contiguous(),
      "paged_decode_attention tensors must be contiguous");
  TORCH_CHECK(query.dim() == 3, "query must be [batch, query_heads, head_dim]");
  TORCH_CHECK(
      key_cache.dim() == 4 && key_cache.sizes() == value_cache.sizes(),
      "K/V cache must have matching [blocks, kv_heads, block_size, head_dim] shapes");
  TORCH_CHECK(block_tables.dim() == 2, "block_tables must be [batch, max_blocks]");
  TORCH_CHECK(sequence_lengths.dim() == 1, "sequence_lengths must be [batch]");
  TORCH_CHECK(query.size(0) > 0 && query.size(1) > 0, "query batch and heads must be positive");
  const int head_dim = static_cast<int>(query.size(2));
  TORCH_CHECK(
      head_dim == 64 || head_dim == 128,
      "paged_decode_attention requires head_dim=64 or 128");
  TORCH_CHECK(key_cache.size(3) == head_dim, "cache head_dim must equal query head_dim");
  TORCH_CHECK(
      key_cache.size(0) > 0 && key_cache.size(1) > 0 && key_cache.size(2) > 0,
      "cache dimensions must be positive");
  TORCH_CHECK(
      query.size(0) == block_tables.size(0) && query.size(0) == sequence_lengths.size(0),
      "metadata batch must equal query batch");
  TORCH_CHECK(block_tables.size(1) > 0, "block_tables must contain at least one block column");
  TORCH_CHECK(
      query.size(1) % key_cache.size(1) == 0,
      "query heads must be divisible by KV heads for GQA");
  TORCH_CHECK(
      query.scalar_type() == key_cache.scalar_type() &&
          query.scalar_type() == value_cache.scalar_type(),
      "query and K/V cache dtype must match");
  TORCH_CHECK(
      query.scalar_type() == at::ScalarType::Float ||
          query.scalar_type() == at::ScalarType::Half ||
          query.scalar_type() == at::ScalarType::BFloat16,
      "paged_decode_attention supports float32, float16, and bfloat16 only");
  TORCH_CHECK(
      block_tables.scalar_type() == at::ScalarType::Int &&
          sequence_lengths.scalar_type() == at::ScalarType::Int,
      "block_tables and sequence_lengths must be int32");
  TORCH_CHECK(std::isfinite(scale) && scale > 0.0, "attention scale must be finite and positive");

  c10::cuda::CUDAGuard guard(query.device());
  // V1 wrapper deliberately validates metadata values before launch.  This synchronizes the
  // current stream but prevents invalid allocator metadata from becoming an illegal memory access.
  int maximum_length = static_cast<int>(block_tables.size(1) * key_cache.size(2));
  if (validate_metadata_values) {
    const int minimum_length = sequence_lengths.min().item<int>();
    maximum_length = sequence_lengths.max().item<int>();
    TORCH_CHECK(minimum_length > 0, "sequence_lengths must be positive");
    TORCH_CHECK(
        maximum_length <= block_tables.size(1) * key_cache.size(2),
        "sequence length exceeds block table capacity");
    const int minimum_block = block_tables.min().item<int>();
    const int maximum_block = block_tables.max().item<int>();
    TORCH_CHECK(minimum_block >= 0, "block table contains a negative physical block ID");
    TORCH_CHECK(
        maximum_block < key_cache.size(0),
        "block table contains an out-of-range physical block ID");
  }

  TORCH_CHECK(
      query.size(0) * query.size(1) <= std::numeric_limits<int>::max(),
      "paged decode grid is too large");
  const int grid = static_cast<int>(query.size(0) * query.size(1));
  at::Tensor output = at::empty_like(query);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(query.device().index());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query.scalar_type(),
      "miniserve_paged_decode_attention_cuda",
      [&] {
        if (head_dim == 64) {
          const int shared_bytes = (64 + maximum_length) * static_cast<int>(sizeof(float));
          paged_decode_attention_kernel<scalar_t, 64><<<grid, 64, shared_bytes, stream>>>(
              query.data_ptr<scalar_t>(), key_cache.data_ptr<scalar_t>(),
              value_cache.data_ptr<scalar_t>(), block_tables.data_ptr<int32_t>(),
              sequence_lengths.data_ptr<int32_t>(), output.data_ptr<scalar_t>(),
              static_cast<int>(query.size(1)), static_cast<int>(key_cache.size(1)),
              static_cast<int>(key_cache.size(2)), static_cast<int>(block_tables.size(1)),
              static_cast<float>(scale));
        } else {
          const int shared_bytes = (128 + maximum_length) * static_cast<int>(sizeof(float));
          paged_decode_attention_kernel<scalar_t, 128><<<grid, 128, shared_bytes, stream>>>(
              query.data_ptr<scalar_t>(), key_cache.data_ptr<scalar_t>(),
              value_cache.data_ptr<scalar_t>(), block_tables.data_ptr<int32_t>(),
              sequence_lengths.data_ptr<int32_t>(), output.data_ptr<scalar_t>(),
              static_cast<int>(query.size(1)), static_cast<int>(key_cache.size(1)),
              static_cast<int>(key_cache.size(2)), static_cast<int>(block_tables.size(1)),
              static_cast<float>(scale));
        }
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor paged_decode_attention_cuda(
    const at::Tensor& query,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const at::Tensor& block_tables,
    const at::Tensor& sequence_lengths,
    double scale) {
  return paged_decode_attention_cuda_impl(
      query, key_cache, value_cache, block_tables, sequence_lengths, scale, true);
}

at::Tensor paged_decode_attention_graph_cuda(
    const at::Tensor& query,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    const at::Tensor& block_tables,
    const at::Tensor& sequence_lengths,
    double scale) {
  // 仅 GraphManager 可调用；值域校验必须在 copy 到 static buffers 之前完成。
  return paged_decode_attention_cuda_impl(
      query, key_cache, value_cache, block_tables, sequence_lengths, scale, false);
}

}  // namespace

TORCH_LIBRARY_IMPL(miniserve, CUDA, m) {
  m.impl("paged_decode_attention", &paged_decode_attention_cuda);
  m.impl("paged_decode_attention_graph", &paged_decode_attention_graph_cuda);
}
