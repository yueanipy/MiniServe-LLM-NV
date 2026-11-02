"""确定性 physical KV block allocator 与引用计数。"""

from __future__ import annotations

from collections.abc import Sequence


class OutOfBlocksError(RuntimeError):
    pass


class BlockAllocator:
    """最小 ID 优先分配；每次公开操作后强制验证总数守恒。"""

    def __init__(self, num_blocks: int) -> None:
        if not isinstance(num_blocks, int) or isinstance(num_blocks, bool) or num_blocks <= 0:
            raise ValueError("num_blocks must be a positive integer.")
        self.num_blocks = num_blocks
        self._free_ids: set[int] = set(range(num_blocks))
        self._refcounts = [0] * num_blocks
        self.assert_invariants()

    def allocate(self, count: int) -> list[int]:
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("Allocation count must be a non-negative integer.")
        if count > len(self._free_ids):
            raise OutOfBlocksError(
                f"OUT_OF_BLOCKS: requested={count}, free={len(self._free_ids)}"
            )
        # 先完整选出再修改，保证 OOM 前后 allocator 原子不变。
        selected = sorted(self._free_ids)[:count]
        for block_id in selected:
            self._free_ids.remove(block_id)
            self._refcounts[block_id] = 1
        self.assert_invariants()
        return selected

    def retain(self, block_ids: Sequence[int]) -> None:
        ids = self._validate_ids(block_ids, unique=True)
        if any(self._refcounts[block_id] == 0 for block_id in ids):
            raise RuntimeError("Cannot retain a free block.")
        for block_id in ids:
            self._refcounts[block_id] += 1
        self.assert_invariants()

    def release(self, block_ids: Sequence[int]) -> None:
        ids = self._validate_ids(block_ids, unique=True)
        if any(self._refcounts[block_id] == 0 for block_id in ids):
            raise RuntimeError("Double free or release of a free block.")
        for block_id in ids:
            self._refcounts[block_id] -= 1
            if self._refcounts[block_id] == 0:
                self._free_ids.add(block_id)
        self.assert_invariants()

    def free_count(self) -> int:
        return len(self._free_ids)

    def refcount(self, block_id: int) -> int:
        self._validate_ids((block_id,), unique=True)
        return self._refcounts[block_id]

    def free_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._free_ids))

    def snapshot(self) -> dict[str, object]:
        return {"free_ids": list(self.free_ids()), "refcounts": list(self._refcounts)}

    def assert_invariants(self) -> None:
        if len(self._refcounts) != self.num_blocks:
            raise AssertionError("Refcount table size changed.")
        for block_id, refcount in enumerate(self._refcounts):
            if refcount < 0:
                raise AssertionError("Negative block refcount.")
            if (block_id in self._free_ids) != (refcount == 0):
                raise AssertionError("Free list and refcount disagree.")
        allocated_unique = sum(refcount > 0 for refcount in self._refcounts)
        if self.free_count() + allocated_unique != self.num_blocks:
            raise AssertionError("Block total is not conserved.")

    def _validate_ids(self, block_ids: Sequence[int], *, unique: bool) -> tuple[int, ...]:
        ids = tuple(block_ids)
        if unique and len(ids) != len(set(ids)):
            raise ValueError("Duplicate block ID in one operation.")
        if any(not isinstance(block_id, int) or isinstance(block_id, bool) for block_id in ids):
            raise TypeError("Block IDs must be integers.")
        if any(block_id < 0 or block_id >= self.num_blocks for block_id in ids):
            raise IndexError("Block ID outside allocator.")
        return ids
