"""Scheduler 可见容量快照与执行期私有 KV metadata。"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Protocol

import torch

from .layout import KVLayout


@dataclass(frozen=True)
class KVCapacitySnapshot:
    """不暴露物理 Tensor/ID 的容量摘要。

    Phase 2 把一个预分配 sequence slot 视作一个 block；Phase 5 paged backend
    会把 block_size 改为真实 token block 大小，而 Scheduler 合同不变。
    """

    total_blocks: int
    free_blocks: int
    cached_blocks: int
    block_size: int

    def __post_init__(self) -> None:
        values = (self.total_blocks, self.free_blocks, self.cached_blocks, self.block_size)
        if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
            raise ValueError("KV capacity fields must be integers.")
        if self.total_blocks <= 0 or self.block_size <= 0:
            raise ValueError("KV total_blocks and block_size must be positive.")
        if not 0 <= self.free_blocks <= self.total_blocks:
            raise ValueError("KV free_blocks is outside total capacity.")
        if not 0 <= self.cached_blocks <= self.total_blocks:
            raise ValueError("KV cached_blocks is outside total capacity.")


@dataclass(frozen=True)
class KVSequenceExecutionMetadata:
    request_id: str
    context_len_before_step: int
    context_len_after_step: int
    token_start: int
    token_end: int
    block_table: tuple[int, ...] = ()
    slot_mapping: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty.")
        numeric = (
            self.context_len_before_step,
            self.context_len_after_step,
            self.token_start,
            self.token_end,
        )
        if any(not isinstance(value, int) or isinstance(value, bool) for value in numeric):
            raise ValueError("KV context/token positions must be integers.")
        if not 0 <= self.context_len_before_step < self.context_len_after_step:
            raise ValueError("KV context must advance by at least one token.")
        if self.token_start != self.context_len_before_step:
            raise ValueError("New token range must begin at committed context length.")
        if self.token_end != self.context_len_after_step:
            raise ValueError("New token range must end at pending context length.")
        if not isinstance(self.block_table, tuple) or not isinstance(self.slot_mapping, tuple):
            raise TypeError("KV block_table and slot_mapping must be immutable tuples.")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in (*self.block_table, *self.slot_mapping)
        ):
            raise ValueError("KV physical metadata must contain non-negative integers.")
        if self.slot_mapping and len(self.slot_mapping) != self.token_end - self.token_start:
            raise ValueError("KV slot_mapping length must equal the new token count.")


@dataclass(frozen=True)
class KVStepContext:
    step_id: int
    sequence_metadata: Mapping[str, KVSequenceExecutionMetadata]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.step_id, int)
            or isinstance(self.step_id, bool)
            or self.step_id < 0
            or not self.sequence_metadata
        ):
            raise ValueError("KVStepContext requires a non-negative step and metadata.")
        copied = dict(self.sequence_metadata)
        if set(copied) != {metadata.request_id for metadata in copied.values()}:
            raise ValueError("KV metadata keys must equal request IDs.")
        object.__setattr__(self, "sequence_metadata", MappingProxyType(copied))


class KVCacheBackend(Protocol):
    layout: KVLayout
    max_sequences: int

    def allocate_sequence(self, request_id: str) -> None: ...

    def append(
        self,
        request_id: str,
        layer_index: int,
        start: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None: ...

    def view(self, request_id: str, layer_index: int, end: int) -> tuple[torch.Tensor, torch.Tensor]: ...

    def release_sequence(self, request_id: str) -> None: ...

    def has_sequence(self, request_id: str) -> bool: ...
