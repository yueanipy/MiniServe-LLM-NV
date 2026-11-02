"""每请求预分配连续 K/V Tensor 的低复杂度 reference backend。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .layout import KVLayout


@dataclass
class _ContiguousAllocation:
    key: torch.Tensor
    value: torch.Tensor


class ContiguousKVCacheBackend:
    """用显存浪费换取最直接的 position→storage 对应关系。"""

    def __init__(self, layout: KVLayout, *, max_sequences: int = 1) -> None:
        if max_sequences <= 0:
            raise ValueError("max_sequences must be positive.")
        self.layout = layout
        self.max_sequences = max_sequences
        self._allocations: dict[str, _ContiguousAllocation] = {}

    @property
    def allocated_sequences(self) -> int:
        return len(self._allocations)

    def has_sequence(self, request_id: str) -> bool:
        return request_id in self._allocations

    def allocate_sequence(self, request_id: str) -> None:
        if not request_id:
            raise ValueError("request_id must not be empty.")
        if request_id in self._allocations:
            raise KeyError(f"Duplicate KV allocation: {request_id}")
        if len(self._allocations) >= self.max_sequences:
            raise RuntimeError("No free contiguous KV sequence slot.")
        shape = self.layout.sequence_shape
        self._allocations[request_id] = _ContiguousAllocation(
            key=torch.empty(shape, dtype=self.layout.dtype, device=self.layout.device),
            value=torch.empty(shape, dtype=self.layout.dtype, device=self.layout.device),
        )

    def append(
        self,
        request_id: str,
        layer_index: int,
        start: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        allocation = self._allocation(request_id)
        if not 0 <= layer_index < self.layout.num_layers:
            raise IndexError("layer_index is outside KV layout.")
        expected_tail = (self.layout.num_kv_heads, self.layout.head_dim)
        if key.ndim != 3 or value.ndim != 3 or tuple(key.shape[1:]) != expected_tail:
            raise ValueError("K/V append tensors must be [tokens, kv_heads, head_dim].")
        if key.shape != value.shape or key.dtype != self.layout.dtype or value.dtype != self.layout.dtype:
            raise ValueError("K/V append shape and dtype must match the canonical layout.")
        if key.device != self.layout.device or value.device != self.layout.device:
            raise ValueError("K/V append tensors must be on the cache device.")
        end = start + key.shape[0]
        if start < 0 or end > self.layout.max_model_len:
            raise ValueError("K/V append range exceeds max_model_len.")
        allocation.key[layer_index, start:end].copy_(key)
        allocation.value[layer_index, start:end].copy_(value)

    def view(self, request_id: str, layer_index: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        allocation = self._allocation(request_id)
        if not 0 <= layer_index < self.layout.num_layers:
            raise IndexError("layer_index is outside KV layout.")
        if not 0 < end <= self.layout.max_model_len:
            raise ValueError("KV view end is outside allocated context.")
        # 返回 view 而不是 clone；attention 只读，生命周期由 backend allocation 拥有。
        return allocation.key[layer_index, :end], allocation.value[layer_index, :end]

    def release_sequence(self, request_id: str) -> None:
        if request_id not in self._allocations:
            raise KeyError(f"Unknown or already released KV sequence: {request_id}")
        del self._allocations[request_id]

    def _allocation(self, request_id: str) -> _ContiguousAllocation:
        try:
            return self._allocations[request_id]
        except KeyError as exc:
            raise KeyError(f"Unknown KV sequence: {request_id}") from exc
