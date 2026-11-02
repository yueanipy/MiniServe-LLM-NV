"""Physical block allocator 与 BlockTable 边界单元测试。"""

from __future__ import annotations

import pytest

from miniserve.kv.allocator import BlockAllocator, OutOfBlocksError
from miniserve.kv.block_table import BlockTable

pytestmark = pytest.mark.unit_cpu


def test_allocate_retain_release_and_total_conservation() -> None:
    allocator = BlockAllocator(6)
    blocks = allocator.allocate(2)
    assert blocks == [0, 1]
    allocator.retain(blocks)
    assert [allocator.refcount(block) for block in blocks] == [2, 2]
    allocator.release(blocks)
    assert allocator.free_count() == 4
    allocator.release(blocks)
    assert allocator.free_count() == 6
    allocator.assert_invariants()


def test_out_of_blocks_and_invalid_release_are_atomic() -> None:
    allocator = BlockAllocator(3)
    allocator.allocate(2)
    before = allocator.snapshot()
    with pytest.raises(OutOfBlocksError, match="OUT_OF_BLOCKS"):
        allocator.allocate(2)
    assert allocator.snapshot() == before
    with pytest.raises(RuntimeError, match="free block"):
        allocator.release((2,))
    assert allocator.snapshot() == before
    with pytest.raises(ValueError, match="Duplicate"):
        allocator.release((0, 0))


@pytest.mark.parametrize("value", [True, 1.5, "2", -1])
def test_allocator_rejects_coercible_or_invalid_counts(value) -> None:
    with pytest.raises((TypeError, ValueError)):
        BlockAllocator(value)
    allocator = BlockAllocator(2)
    with pytest.raises(ValueError, match="non-negative integer"):
        allocator.allocate(value)


@pytest.mark.parametrize(
    "table",
    [
        BlockTable,
    ],
)
def test_block_table_rejects_invalid_physical_metadata(table) -> None:
    for block_size, block_ids in ((True, [0]), (16, [-1]), (16, [True]), (16, [1.5])):
        with pytest.raises(ValueError):
            table(block_size, block_ids)
    valid = table(16, [0])
    with pytest.raises(ValueError):
        valid.append_blocks([True])
    with pytest.raises(ValueError):
        valid.resolve(1.5)


@pytest.mark.parametrize(
    ("position", "expected"),
    [
        (0, (0, 5, 0)),
        (15, (0, 5, 15)),
        (16, (1, 2, 0)),
        (17, (1, 2, 1)),
        (31, (1, 2, 15)),
        (32, (2, 7, 0)),
        (33, (2, 7, 1)),
    ],
)
def test_noncontiguous_block_table_boundaries(position: int, expected: tuple[int, int, int]) -> None:
    table = BlockTable(16, [5, 2, 7])
    assert table.resolve(position) == expected
    assert table.slot(position) == expected[1] * 16 + expected[2]
