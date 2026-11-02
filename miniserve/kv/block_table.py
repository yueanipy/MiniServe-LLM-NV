"""逻辑 block index 到 physical block ID 的请求私有映射。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BlockTable:
    block_size: int
    physical_block_ids: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.block_size, int) or isinstance(self.block_size, bool) or self.block_size <= 0:
            raise ValueError("block_size must be a positive integer.")
        if any(
            not isinstance(block_id, int) or isinstance(block_id, bool) or block_id < 0
            for block_id in self.physical_block_ids
        ):
            raise ValueError("Physical block IDs must be non-negative integers.")
        if len(self.physical_block_ids) != len(set(self.physical_block_ids)):
            raise ValueError("One sequence block table cannot duplicate a physical ID.")

    def required_blocks(self, token_count: int) -> int:
        if not isinstance(token_count, int) or isinstance(token_count, bool) or token_count < 0:
            raise ValueError("token_count must be a non-negative integer.")
        return (token_count + self.block_size - 1) // self.block_size

    def append_blocks(self, block_ids: list[int]) -> None:
        if not isinstance(block_ids, list) or any(
            not isinstance(block_id, int) or isinstance(block_id, bool) or block_id < 0
            for block_id in block_ids
        ):
            raise ValueError("Appended physical block IDs must be a list of non-negative integers.")
        if len(block_ids) != len(set(block_ids)) or set(block_ids) & set(self.physical_block_ids):
            raise ValueError("Appended physical block IDs must be unique.")
        self.physical_block_ids.extend(block_ids)

    def resolve(self, token_position: int) -> tuple[int, int, int]:
        if not isinstance(token_position, int) or isinstance(token_position, bool) or token_position < 0:
            raise ValueError("token_position must be a non-negative integer.")
        logical = token_position // self.block_size
        offset = token_position % self.block_size
        if logical >= len(self.physical_block_ids):
            raise IndexError("Token position has no reserved physical block.")
        return logical, self.physical_block_ids[logical], offset

    def slot(self, token_position: int) -> int:
        _, physical, offset = self.resolve(token_position)
        return physical * self.block_size + offset

    def snapshot(self) -> tuple[int, ...]:
        return tuple(self.physical_block_ids)
