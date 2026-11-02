"""一次性 K/V pool、事务 reservation 与慢速 gather 的 paged backend。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .allocator import BlockAllocator
from .block_table import BlockTable
from .layout import KVLayout


@dataclass
class _PagedSequence:
    table: BlockTable


@dataclass(frozen=True)
class _PendingReservation:
    request_id: str
    step_id: int
    old_block_count: int
    new_blocks: tuple[int, ...]
    tentative_table: tuple[int, ...]


class PagedKVCacheBackend:
    """Torch reference backend；pool 在初始化时只分配一次。"""

    def __init__(self, layout: KVLayout, *, max_sequences: int) -> None:
        if layout.num_blocks is None:
            raise ValueError("PagedKVCacheBackend requires layout.num_blocks.")
        if max_sequences <= 0:
            raise ValueError("max_sequences must be positive.")
        self.layout = layout
        self.max_sequences = max_sequences
        self.allocator = BlockAllocator(layout.num_blocks)
        shape = layout.paged_pool_shape
        self.key_pool = torch.empty(shape, dtype=layout.dtype, device=layout.device)
        self.value_pool = torch.empty(shape, dtype=layout.dtype, device=layout.device)
        self._sequences: dict[str, _PagedSequence] = {}
        self._pending: dict[tuple[int, str], _PendingReservation] = {}

    @property
    def allocated_sequences(self) -> int:
        return len(self._sequences)

    def has_sequence(self, request_id: str) -> bool:
        return request_id in self._sequences

    def allocate_sequence(self, request_id: str) -> None:
        if request_id in self._sequences:
            raise KeyError(f"Duplicate paged sequence: {request_id}")
        if len(self._sequences) >= self.max_sequences:
            raise RuntimeError("No free paged sequence admission slot.")
        self._sequences[request_id] = _PagedSequence(BlockTable(self.layout.block_size))

    def reserve(self, request_id: str, new_total_tokens: int, step_id: int) -> tuple[int, ...]:
        sequence = self._sequence(request_id)
        pending_key = (step_id, request_id)
        if pending_key in self._pending:
            raise RuntimeError("Paged request/step is already pending.")
        required = sequence.table.required_blocks(new_total_tokens)
        missing = required - len(sequence.table.physical_block_ids)
        if missing < 0:
            raise RuntimeError("Paged reservation cannot shrink a committed table.")
        new_blocks = tuple(self.allocator.allocate(missing))
        tentative = sequence.table.snapshot() + new_blocks
        self._pending[pending_key] = _PendingReservation(
            request_id, step_id, len(sequence.table.physical_block_ids), new_blocks, tentative
        )
        return tentative

    def commit_reservation(self, step_id: int) -> None:
        keys = sorted(key for key in self._pending if key[0] == step_id)
        if not keys:
            raise KeyError(f"Unknown paged reservation step: {step_id}")
        # 先验证本 step 的全部 reservation，再进行任何 pop/append。否则第二个请求
        # 验证失败时，第一个请求可能已经提交，调用方无法安全 abort 整个 batch。
        for key in keys:
            pending = self._pending[key]
            sequence = self._sequence(pending.request_id)
            if len(sequence.table.physical_block_ids) != pending.old_block_count:
                raise RuntimeError("Committed block table changed during pending reservation.")
        for key in keys:
            pending = self._pending.pop(key)
            sequence = self._sequence(pending.request_id)
            sequence.table.append_blocks(list(pending.new_blocks))

    def abort_reservation(self, step_id: int) -> None:
        keys = sorted(key for key in self._pending if key[0] == step_id)
        if not keys:
            raise KeyError(f"Unknown paged reservation step: {step_id}")
        # allocator.release 会先完整校验输入；合并释放保证多请求 abort 不会部分完成。
        blocks = tuple(
            block
            for key in keys
            for block in self._pending[key].new_blocks
        )
        if blocks:
            self.allocator.release(blocks)
        for key in keys:
            self._pending.pop(key)

    def block_table(self, request_id: str, *, step_id: int | None = None) -> tuple[int, ...]:
        if step_id is not None and (step_id, request_id) in self._pending:
            return self._pending[(step_id, request_id)].tentative_table
        return self._sequence(request_id).table.snapshot()

    def has_pending_step(self, step_id: int) -> bool:
        return any(key[0] == step_id for key in self._pending)

    def sequence_block_ids(self) -> dict[str, tuple[int, ...]]:
        """每个活动序列当前已提交的 physical block（只读快照，用于容量记账）。"""

        return {
            request_id: sequence.table.snapshot()
            for request_id, sequence in self._sequences.items()
        }

    def pending_block_ids(self) -> tuple[int, ...]:
        """已从 allocator 取得、尚未 commit 的 reservation blocks（去重）。"""

        return tuple(
            sorted({block for pending in self._pending.values() for block in pending.new_blocks})
        )

    def slot_for(self, request_id: str, token_position: int, *, step_id: int | None = None) -> int:
        table = BlockTable(self.layout.block_size, list(self.block_table(request_id, step_id=step_id)))
        return table.slot(token_position)

    def append(
        self,
        request_id: str,
        layer_index: int,
        start: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        if key.shape != value.shape or key.ndim != 3:
            raise ValueError("Paged append expects matching [tokens,kv_heads,head_dim].")
        if tuple(key.shape[1:]) != (self.layout.num_kv_heads, self.layout.head_dim):
            raise ValueError("Paged append shape does not match layout.")
        if (
            key.dtype != self.layout.dtype
            or value.dtype != self.layout.dtype
            or key.device != self.layout.device
            or value.device != self.layout.device
        ):
            raise ValueError("Paged append dtype/device mismatch.")
        pending_step = self._pending_step_for(request_id)
        for local, position in enumerate(range(start, start + key.shape[0])):
            table = BlockTable(
                self.layout.block_size,
                list(self.block_table(request_id, step_id=pending_step)),
            )
            _, physical, offset = table.resolve(position)
            self.key_pool[layer_index, physical, :, offset, :].copy_(key[local])
            self.value_pool[layer_index, physical, :, offset, :].copy_(value[local])

    def view(self, request_id: str, layer_index: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        """与 contiguous protocol 对齐的慢速 gather，用于 Torch attention oracle。"""

        if end <= 0:
            raise ValueError("Paged view requires positive context length.")
        pending_step = self._pending_step_for(request_id, required=False)
        table = BlockTable(
            self.layout.block_size,
            list(self.block_table(request_id, step_id=pending_step)),
        )
        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        for position in range(end):
            _, physical, offset = table.resolve(position)
            keys.append(self.key_pool[layer_index, physical, :, offset, :])
            values.append(self.value_pool[layer_index, physical, :, offset, :])
        return torch.stack(keys, dim=0), torch.stack(values, dim=0)

    def release_sequence(self, request_id: str) -> None:
        if any(pending.request_id == request_id for pending in self._pending.values()):
            raise RuntimeError("Cannot release sequence with pending paged reservation.")
        sequence = self._sequence(request_id)
        blocks = sequence.table.snapshot()
        if blocks:
            self.allocator.release(blocks)
        del self._sequences[request_id]

    def attach_blocks(self, request_id: str, block_ids: tuple[int, ...]) -> None:
        sequence = self._sequence(request_id)
        if sequence.table.physical_block_ids:
            raise RuntimeError("Prefix blocks can only attach to an empty sequence table.")
        sequence.table.append_blocks(list(block_ids))

    def required_new_blocks(self, request_id: str, new_total_tokens: int) -> int:
        sequence = self._sequence(request_id)
        return max(
            0,
            sequence.table.required_blocks(new_total_tokens)
            - len(sequence.table.physical_block_ids),
        )

    def _sequence(self, request_id: str) -> _PagedSequence:
        try:
            return self._sequences[request_id]
        except KeyError as exc:
            raise KeyError(f"Unknown paged sequence: {request_id}") from exc

    def _pending_step_for(self, request_id: str, *, required: bool = True) -> int | None:
        matching = [key[0] for key, pending in self._pending.items() if pending.request_id == request_id]
        if len(matching) > 1:
            raise RuntimeError("A sequence has multiple pending paged steps.")
        if required and not matching:
            raise RuntimeError("Paged append requires a pending reservation.")
        return matching[0] if matching else None
