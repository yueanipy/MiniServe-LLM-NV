"""M0 KV 容量分类记账 fixtures：固定 pool != used bytes，cache-only 不是泄漏。"""

from __future__ import annotations

import pytest
import torch

from miniserve.engine.plan import ExecutionPlan, PrefillItem
from miniserve.kv import ContiguousKVCacheBackend, KVCacheManager, KVLayout
from miniserve.kv.paged import PagedKVCacheBackend
from miniserve.kv.prefix_cache import PrefixCache, PrefixNamespace

pytestmark = pytest.mark.unit_cpu


def _paged_manager(*, with_cache: bool = False, num_blocks: int = 8) -> tuple[KVCacheManager, PagedKVCacheBackend]:
    layout = KVLayout(1, 64, 2, 64, torch.float16, "cpu", num_blocks=num_blocks, block_size=4)
    backend = PagedKVCacheBackend(layout, max_sequences=3)
    cache = None
    if with_cache:
        cache = PrefixCache(
            namespace=PrefixNamespace("rev", "rev", "float16", "rope"),
            block_size=4,
            max_entries=8,
        )
    return KVCacheManager(backend, prefix_cache=cache), backend


def _commit_prefill(manager: KVCacheManager, request_id: str, start: int, end: int, step_id: int) -> None:
    if not manager.has_sequence(request_id):
        manager.register_sequence(request_id)
    context = manager.prepare_step(
        ExecutionPlan(step_id, (PrefillItem(request_id, start, end),), (), end - start)
    )
    manager.commit_step(context)


def test_fixed_pool_bytes_never_equals_used_bytes() -> None:
    manager, backend = _paged_manager()
    empty = manager.usage_snapshot()
    assert empty.pool_bytes == backend.allocator.num_blocks * backend.layout.bytes_per_block()
    assert empty.used_bytes == 0
    assert empty.as_dict()["fixed_pool_equals_used_bytes"] is False

    _commit_prefill(manager, "a", 0, 10, step_id=0)
    usage = manager.usage_snapshot()
    # 10 tokens / block_size 4 -> 3 个 physical block；pool 仍是 8 块。
    assert usage.active_blocks == 3
    assert usage.free_blocks == 5
    assert usage.used_blocks == 3
    assert usage.pool_bytes == 8 * backend.layout.bytes_per_block()
    assert usage.used_bytes == 3 * backend.layout.bytes_per_block()
    assert usage.pool_bytes != usage.used_bytes
    assert usage.logical_tokens == 10
    assert usage.logical_payload_bytes == 10 * backend.layout.bytes_per_token()
    assert usage.as_dict()["conservation_holds"] is True


def test_pending_and_active_blocks_are_separate_categories() -> None:
    manager, backend = _paged_manager()
    _commit_prefill(manager, "a", 0, 10, step_id=0)
    manager.register_sequence("b")
    pending_context = manager.prepare_step(
        ExecutionPlan(1, (PrefillItem("b", 0, 9),), (), 9)
    )  # 不 commit
    usage = manager.usage_snapshot()
    assert usage.active_blocks == 3
    assert usage.pending_blocks == 3
    assert usage.free_blocks == 2
    assert usage.used_blocks == 6
    assert usage.as_dict()["conservation_holds"] is True
    manager.abort_step(pending_context)
    after_abort = manager.usage_snapshot()
    assert after_abort.pending_blocks == 0
    assert after_abort.free_blocks == 5


def test_cache_only_blocks_are_retained_not_leaked_and_evict_restores_free() -> None:
    manager, backend = _paged_manager(with_cache=True)
    tokens = tuple(range(8))  # 8 tokens = 2 full blocks（block_size=4）
    _commit_prefill(manager, "a", 0, 8, step_id=0)
    manager.cache_prefix("a", tokens)
    active_and_cached = manager.usage_snapshot()
    # 同一批块同时被活动序列与 cache 引用：只计入 active，不重复计数。
    assert active_and_cached.active_blocks == 2
    assert active_and_cached.shared_active_cache_blocks == 2
    assert active_and_cached.cache_only_blocks == 0
    assert active_and_cached.used_blocks == 2

    manager.release_sequence("a")
    cache_only = manager.usage_snapshot()
    assert cache_only.active_blocks == 0
    assert cache_only.cache_only_blocks == 2
    assert cache_only.free_blocks == backend.allocator.num_blocks - 2
    snapshot = manager.capacity_snapshot()
    assert snapshot.free_blocks == backend.allocator.num_blocks - 2
    assert snapshot.cached_blocks == 2
    # cache-only 是复用状态而不是泄漏：显式驱逐后全部回到 free。
    manager.evict_all_cache_only()
    drained = manager.usage_snapshot()
    assert drained.cache_only_blocks == 0
    assert drained.free_blocks == backend.allocator.num_blocks
    assert drained.used_bytes == 0
    drained_snapshot = manager.capacity_snapshot()
    assert drained_snapshot.free_blocks == backend.allocator.num_blocks
    assert drained_snapshot.cached_blocks == 0


def test_prefix_full_hit_recomputes_the_tail_block() -> None:
    manager, _backend = _paged_manager(with_cache=True)
    tokens = tuple(range(8))
    _commit_prefill(manager, "a", 0, 8, step_id=0)
    manager.cache_prefix("a", tokens)
    manager.register_sequence("b")
    match = manager.attach_prefix("b", tokens)
    # 完整 prompt 命中必须回退最后一块用于 logits，而不是复用全部 8 个 token。
    assert match.matched_tokens == 4
    assert len(match.physical_block_ids) == 1


def test_contiguous_reports_reserved_payload_vs_logical_usage() -> None:
    layout = KVLayout(2, 32, 2, 8, torch.float16, "cpu")
    backend = ContiguousKVCacheBackend(layout, max_sequences=1)
    manager = KVCacheManager(backend)
    _commit_prefill(manager, "a", 0, 20, step_id=0)
    usage = manager.usage_snapshot()
    assert usage.backend_kind == "contiguous"
    assert usage.pool_bytes == layout.bytes_per_sequence()
    assert usage.used_bytes == layout.bytes_per_sequence()
    assert usage.logical_tokens == 20
    assert usage.logical_payload_bytes == 20 * layout.bytes_per_token()
    assert usage.logical_payload_bytes < usage.used_bytes
    assert usage.as_dict()["fixed_pool_equals_used_bytes"] is True
