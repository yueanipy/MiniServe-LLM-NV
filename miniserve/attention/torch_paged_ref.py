"""Paged storage + gather 的 PyTorch reference attention backend。"""

from __future__ import annotations

import math

import torch

from miniserve.kv.paged import PagedKVCacheBackend

from .torch_contiguous import TorchContiguousAttention


def paged_decode_attention_torch(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    sequence_lengths: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """清晰优先的单层 paged decode oracle，支持 CPU 或 CUDA。"""

    if query.ndim != 3:
        raise ValueError("query must be [batch, query_heads, head_dim].")
    if key_cache.ndim != 4 or key_cache.shape != value_cache.shape:
        raise ValueError("K/V cache must share [blocks, kv_heads, block_size, head_dim].")
    if block_tables.ndim != 2 or sequence_lengths.ndim != 1:
        raise ValueError("block tables and sequence lengths have invalid rank.")
    batch, query_heads, head_dim = query.shape
    num_blocks, kv_heads, block_size, cache_head_dim = key_cache.shape
    if head_dim != cache_head_dim or head_dim <= 0:
        raise ValueError("query and cache head_dim must match and be positive.")
    if batch != block_tables.shape[0] or batch != sequence_lengths.shape[0]:
        raise ValueError("metadata batch must equal query batch.")
    if query_heads % kv_heads != 0:
        raise ValueError("query heads must be divisible by KV heads.")
    if query.dtype != key_cache.dtype or query.dtype != value_cache.dtype:
        raise ValueError("query and cache dtype must match.")
    if query.device != key_cache.device or query.device != value_cache.device:
        raise ValueError("query and cache device must match.")
    if block_tables.dtype != torch.int32 or sequence_lengths.dtype != torch.int32:
        raise ValueError("paged metadata must use int32.")
    if block_tables.device != query.device or sequence_lengths.device != query.device:
        raise ValueError("paged metadata must share the query device.")
    if not all(item.is_contiguous() for item in (query, key_cache, value_cache, block_tables, sequence_lengths)):
        raise ValueError("paged decode tensors must be contiguous.")
    scale = head_dim**-0.5 if scale is None else float(scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("attention scale must be finite and positive.")

    output = torch.empty_like(query)
    group_size = query_heads // kv_heads
    # item() 只出现在慢速 oracle/测试；生产 CUDA backend 不使用此 Python 循环。
    for batch_index in range(batch):
        sequence_length = int(sequence_lengths[batch_index].item())
        if sequence_length <= 0 or sequence_length > block_tables.shape[1] * block_size:
            raise ValueError("sequence length exceeds paged metadata capacity.")
        for query_head in range(query_heads):
            kv_head = query_head // group_size
            keys: list[torch.Tensor] = []
            values: list[torch.Tensor] = []
            for position in range(sequence_length):
                logical_block, offset = divmod(position, block_size)
                physical_block = int(block_tables[batch_index, logical_block].item())
                if not 0 <= physical_block < num_blocks:
                    raise ValueError("block table contains an out-of-range physical block ID.")
                keys.append(key_cache[physical_block, kv_head, offset])
                values.append(value_cache[physical_block, kv_head, offset])
            key_matrix = torch.stack(keys).float()
            value_matrix = torch.stack(values).float()
            scores = torch.mv(key_matrix, query[batch_index, query_head].float()) * scale
            probabilities = torch.softmax(scores, dim=0, dtype=torch.float32)
            output[batch_index, query_head] = torch.mv(value_matrix.transpose(0, 1), probabilities).to(
                query.dtype
            )
    return output


class TorchPagedReferenceAttention(TorchContiguousAttention):
    """复用同一 Qwen attention 数学，只把历史 K/V 来源替换为 paged gather。"""

    name = "torch_paged_ref"

    def __init__(self, kv_backend: PagedKVCacheBackend) -> None:
        if not isinstance(kv_backend, PagedKVCacheBackend):
            raise TypeError("TorchPagedReferenceAttention requires PagedKVCacheBackend.")
        super().__init__(kv_backend)

    def decode(
        self,
        query: torch.Tensor,
        *,
        layer_index: int,
        block_tables: torch.Tensor,
        sequence_lengths: torch.Tensor,
    ) -> torch.Tensor:
        return paged_decode_attention_torch(
            query,
            self.kv_backend.key_pool[layer_index],
            self.kv_backend.value_pool[layer_index],
            block_tables,
            sequence_lengths,
            query.shape[-1] ** -0.5,
        )
