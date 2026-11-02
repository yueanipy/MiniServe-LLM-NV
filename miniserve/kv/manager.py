"""KV 逻辑长度的事务式 prepare/commit/abort 管理器。"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from miniserve.engine.plan import ExecutionPlan

from .base import (
    KVCacheBackend,
    KVCapacitySnapshot,
    KVSequenceExecutionMetadata,
    KVStepContext,
)
from .paged import PagedKVCacheBackend
from .prefix_cache import PrefixCache, PrefixMatch


@dataclass
class _SequenceKVState:
    request_id: str
    committed_tokens: int = 0


@dataclass(frozen=True)
class KVCapacityUsage:
    """一次容量记账快照（M0 fixtures 的 canonical 账本）。

    块的分类互斥且不重复计费：
    - ``free``：allocator 未分配；
    - ``active``：被至少一个活动序列的 block table 引用；
    - ``pending``：本 step 已 reserve、尚未 commit（不在任何 block table 中）；
    - ``cache_only``：只被 prefix cache 保留、没有任何活动序列引用；
    - ``shared_active_cache``：同时被活动序列与 cache 引用（只计入 active，不重复）。

    固定 pool 的字节数不随请求变化；``used_bytes`` 是可变量，二者绝不能混为一谈。
    cache-only 是合法的复用状态，不是泄漏：``evict_all_cache_only`` 后应全部回到 free。
    """

    backend_kind: str
    total_blocks: int
    block_size: int
    bytes_per_block: int
    pool_bytes: int
    free_blocks: int
    active_blocks: int
    pending_blocks: int
    cache_only_blocks: int
    shared_active_cache_blocks: int
    used_blocks: int
    used_bytes: int
    logical_tokens: int
    logical_payload_bytes: int

    def as_dict(self) -> dict[str, object]:
        return {
            "backend_kind": self.backend_kind,
            "total_blocks": self.total_blocks,
            "block_size": self.block_size,
            "bytes_per_block": self.bytes_per_block,
            "pool_bytes": self.pool_bytes,
            "free_blocks": self.free_blocks,
            "active_blocks": self.active_blocks,
            "pending_blocks": self.pending_blocks,
            "cache_only_blocks": self.cache_only_blocks,
            "shared_active_cache_blocks": self.shared_active_cache_blocks,
            "used_blocks": self.used_blocks,
            "used_bytes": self.used_bytes,
            "logical_tokens": self.logical_tokens,
            "logical_payload_bytes": self.logical_payload_bytes,
            "fixed_pool_equals_used_bytes": self.pool_bytes == self.used_bytes,
            "conservation_holds": self.used_blocks + self.free_blocks == self.total_blocks,
        }


class KVCacheManager:
    """物理写入可先发生，但只有 commit 才推进可见 context length。"""

    def __init__(
        self,
        backend: KVCacheBackend,
        *,
        prefix_cache: PrefixCache | None = None,
        max_finalized_steps: int = 1024,
    ) -> None:
        if max_finalized_steps <= 0:
            raise ValueError("max_finalized_steps must be positive.")
        self.backend = backend
        self.prefix_cache = prefix_cache
        self._sequences: dict[str, _SequenceKVState] = {}
        self._pending: dict[int, KVStepContext] = {}
        self._max_finalized_steps = max_finalized_steps
        self._max_finalized_step_id = -1
        self._finalized_steps: OrderedDict[int, str] = OrderedDict()

    def register_sequence(self, request_id: str) -> None:
        if request_id in self._sequences:
            raise KeyError(f"Duplicate KV sequence: {request_id}")
        self.backend.allocate_sequence(request_id)
        self._sequences[request_id] = _SequenceKVState(request_id)

    def capacity_snapshot(self) -> KVCapacitySnapshot:
        if isinstance(self.backend, PagedKVCacheBackend):
            return KVCapacitySnapshot(
                total_blocks=self.backend.allocator.num_blocks,
                free_blocks=self.backend.allocator.free_count(),
                cached_blocks=self.reclaimable_cache_only_blocks(),
                block_size=self.backend.layout.block_size,
            )
        allocated = len(self._sequences)
        return KVCapacitySnapshot(
            total_blocks=self.backend.max_sequences,
            free_blocks=self.backend.max_sequences - allocated,
            cached_blocks=0,
            block_size=self.backend.layout.max_model_len,
        )

    def prepare_step(self, plan: ExecutionPlan) -> KVStepContext:
        if plan.empty:
            raise ValueError("Cannot prepare KV for an empty plan.")
        if plan.step_id in self._pending or plan.step_id <= self._max_finalized_step_id:
            raise RuntimeError(f"KV step_id was already used: {plan.step_id}")
        metadata: dict[str, KVSequenceExecutionMetadata] = {}
        for item in plan.prefill_items:
            state = self._state(item.request_id)
            before = state.committed_tokens
            if item.token_start != before:
                raise RuntimeError("Prefill range does not begin at committed KV length.")
            self._add_metadata(metadata, item.request_id, before, item.token_end)
        for item in plan.decode_items:
            state = self._state(item.request_id)
            before = state.committed_tokens
            if item.position != before:
                raise RuntimeError("Decode position does not equal committed KV length.")
            self._add_metadata(metadata, item.request_id, before, before + 1)
        if isinstance(self.backend, PagedKVCacheBackend):
            paged_metadata: dict[str, KVSequenceExecutionMetadata] = {}
            required = sum(
                self.backend.required_new_blocks(request_id, item.context_len_after_step)
                for request_id, item in metadata.items()
            )
            self._evict_cache_until_free(required)
            try:
                for request_id, item_metadata in metadata.items():
                    table = self.backend.reserve(
                        request_id,
                        item_metadata.context_len_after_step,
                        plan.step_id,
                    )
                    slots = tuple(
                        self.backend.slot_for(request_id, position, step_id=plan.step_id)
                        for position in range(item_metadata.token_start, item_metadata.token_end)
                    )
                    paged_metadata[request_id] = KVSequenceExecutionMetadata(
                        request_id=request_id,
                        context_len_before_step=item_metadata.context_len_before_step,
                        context_len_after_step=item_metadata.context_len_after_step,
                        token_start=item_metadata.token_start,
                        token_end=item_metadata.token_end,
                        block_table=table,
                        slot_mapping=slots,
                    )
            except Exception:
                if self.backend.has_pending_step(plan.step_id):
                    self.backend.abort_reservation(plan.step_id)
                raise
            metadata = paged_metadata
        context = KVStepContext(step_id=plan.step_id, sequence_metadata=metadata)
        self._pending[plan.step_id] = context
        return context

    def commit_step(self, context: KVStepContext) -> None:
        pending = self._get_pending(context)
        # 所有可能失败的验证都必须发生在 backend/逻辑状态改变之前。
        for request_id, metadata in pending.sequence_metadata.items():
            state = self._state(request_id)
            if state.committed_tokens != metadata.context_len_before_step:
                raise RuntimeError("Committed KV length changed after prepare.")
        if isinstance(self.backend, PagedKVCacheBackend):
            self.backend.commit_reservation(context.step_id)
        for request_id, metadata in pending.sequence_metadata.items():
            state = self._state(request_id)
            state.committed_tokens = metadata.context_len_after_step
        self._finalize_pending(context.step_id, outcome="committed")

    def abort_step(self, context: KVStepContext) -> None:
        # Contiguous storage 可能已有未提交写入；逻辑长度不变，重试会覆盖相同 slots。
        self._get_pending(context)
        if isinstance(self.backend, PagedKVCacheBackend):
            self.backend.abort_reservation(context.step_id)
        self._finalize_pending(context.step_id, outcome="aborted")

    def release_sequence(self, request_id: str) -> None:
        self._state(request_id)
        if any(request_id in context.sequence_metadata for context in self._pending.values()):
            raise RuntimeError("Cannot release a sequence with a pending KV step.")
        self.backend.release_sequence(request_id)
        del self._sequences[request_id]

    def committed_tokens(self, request_id: str) -> int:
        return self._state(request_id).committed_tokens

    def has_sequence(self, request_id: str) -> bool:
        return request_id in self._sequences

    @property
    def available_sequence_slots(self) -> int:
        """Return backend admission slots not currently owned by requests."""

        return max(0, self.backend.max_sequences - len(self._sequences))

    def attach_prefix(self, request_id: str, token_ids: tuple[int, ...]) -> PrefixMatch:
        if self.prefix_cache is None or not isinstance(self.backend, PagedKVCacheBackend):
            return PrefixMatch(0, ())
        state = self._state(request_id)
        if state.committed_tokens != 0:
            raise RuntimeError("Prefix lookup requires an empty sequence state.")
        match = self.prefix_cache.lookup(token_ids)
        if match.physical_block_ids:
            self.backend.allocator.retain(match.physical_block_ids)
            self.backend.attach_blocks(request_id, match.physical_block_ids)
            state.committed_tokens = match.matched_tokens
        return match

    def cache_prefix(self, request_id: str, token_ids: tuple[int, ...]) -> None:
        if self.prefix_cache is None or not isinstance(self.backend, PagedKVCacheBackend):
            return
        table = self.backend.block_table(request_id)
        full_blocks = len(token_ids) // self.backend.layout.block_size
        added, evicted = self.prefix_cache.insert(token_ids, table[:full_blocks])
        for entry in added:
            self.backend.allocator.retain((entry.physical_block_id,))
        for entry in evicted:
            self.backend.allocator.release((entry.physical_block_id,))

    def evict_all_cache_only(self) -> None:
        if self.prefix_cache is None or not isinstance(self.backend, PagedKVCacheBackend):
            return
        while True:
            entry = self.prefix_cache.evict_one(
                lambda candidate: self.backend.allocator.refcount(candidate.physical_block_id) == 1
            )
            if entry is None:
                break
            self.backend.allocator.release((entry.physical_block_id,))

    def reclaimable_cache_only_blocks(self) -> int:
        """Count cache-only blocks that one eviction can return to free."""

        if self.prefix_cache is None or not isinstance(self.backend, PagedKVCacheBackend):
            return 0
        sequence_blocks = {
            block
            for blocks in self.backend.sequence_block_ids().values()
            for block in blocks
        }
        pending = set(self.backend.pending_block_ids())
        cached = set(self.prefix_cache.physical_block_ids())
        cache_only = cached - sequence_blocks - pending
        return sum(
            self.backend.allocator.refcount(block) == 1 for block in cache_only
        )

    def _evict_cache_until_free(self, required_blocks: int) -> None:
        if not isinstance(self.backend, PagedKVCacheBackend):
            return
        while self.backend.allocator.free_count() < required_blocks and self.prefix_cache is not None:
            entry = self.prefix_cache.evict_one(
                lambda candidate: self.backend.allocator.refcount(candidate.physical_block_id) == 1
            )
            if entry is None:
                break
            self.backend.allocator.release((entry.physical_block_id,))

    def usage_snapshot(self) -> KVCapacityUsage:
        """容量分类记账；paged 与 contiguous 使用同一字段语义。"""

        layout = self.backend.layout
        if isinstance(self.backend, PagedKVCacheBackend):
            allocator = self.backend.allocator
            sequence_blocks = {
                block
                for blocks in self.backend.sequence_block_ids().values()
                for block in blocks
            }
            pending = set(self.backend.pending_block_ids())
            cached = set() if self.prefix_cache is None else set(self.prefix_cache.physical_block_ids())
            cache_only = cached - sequence_blocks - pending
            shared = cached & sequence_blocks
            used = sequence_blocks | pending | cache_only
            logical_tokens = sum(state.committed_tokens for state in self._sequences.values())
            bytes_per_block = layout.bytes_per_block()
            return KVCapacityUsage(
                backend_kind="paged",
                total_blocks=allocator.num_blocks,
                block_size=layout.block_size,
                bytes_per_block=bytes_per_block,
                pool_bytes=allocator.num_blocks * bytes_per_block,
                free_blocks=allocator.free_count(),
                active_blocks=len(sequence_blocks),
                pending_blocks=len(pending),
                cache_only_blocks=len(cache_only),
                shared_active_cache_blocks=len(shared),
                used_blocks=len(used),
                used_bytes=len(used) * bytes_per_block,
                logical_tokens=logical_tokens,
                logical_payload_bytes=logical_tokens * layout.bytes_per_token(),
            )
        # contiguous：每请求预留 max_model_len 的完整 K/V；reserved 才是 pool 口径。
        bytes_per_token = layout.bytes_per_token()
        reserved_bytes = len(self._sequences) * layout.bytes_per_sequence()
        logical_tokens = sum(state.committed_tokens for state in self._sequences.values())
        return KVCapacityUsage(
            backend_kind="contiguous",
            total_blocks=self.backend.max_sequences,
            block_size=layout.max_model_len,
            bytes_per_block=layout.bytes_per_sequence(),
            pool_bytes=reserved_bytes,
            free_blocks=self.backend.max_sequences - len(self._sequences),
            active_blocks=len(self._sequences),
            pending_blocks=0,
            cache_only_blocks=0,
            shared_active_cache_blocks=0,
            used_blocks=len(self._sequences),
            used_bytes=reserved_bytes,
            logical_tokens=logical_tokens,
            logical_payload_bytes=logical_tokens * bytes_per_token,
        )

    def assert_consistent(self, request_id: str, computed_tokens: int) -> None:
        committed = self.committed_tokens(request_id)
        if committed != computed_tokens:
            raise AssertionError(
                f"KV/sequence token mismatch for {request_id}: committed={committed}, computed={computed_tokens}"
            )

    def _add_metadata(
        self,
        metadata: dict[str, KVSequenceExecutionMetadata],
        request_id: str,
        before: int,
        after: int,
    ) -> None:
        if request_id in metadata:
            raise RuntimeError("A request appears twice in one KV step.")
        if after > self.backend.layout.max_model_len:
            raise RuntimeError("CONTEXT_CAPACITY_EXCEEDED")
        metadata[request_id] = KVSequenceExecutionMetadata(
            request_id=request_id,
            context_len_before_step=before,
            context_len_after_step=after,
            token_start=before,
            token_end=after,
        )

    def _get_pending(self, context: KVStepContext) -> KVStepContext:
        if context.step_id in self._finalized_steps:
            raise RuntimeError(
                f"KV step {context.step_id} was already {self._finalized_steps[context.step_id]}."
            )
        pending = self._pending.get(context.step_id)
        if pending is None and context.step_id <= self._max_finalized_step_id:
            raise RuntimeError(f"KV step {context.step_id} was already finalized.")
        if pending is None or pending is not context:
            raise RuntimeError("Unknown or mismatched pending KV context.")
        return pending

    def _finalize_pending(self, step_id: int, *, outcome: str) -> None:
        del self._pending[step_id]
        self._finalized_steps[step_id] = outcome
        self._finalized_steps.move_to_end(step_id)
        self._max_finalized_step_id = max(self._max_finalized_step_id, step_id)
        while len(self._finalized_steps) > self._max_finalized_steps:
            self._finalized_steps.popitem(last=False)

    @property
    def retained_finalized_step_count(self) -> int:
        """最近事务结果的诊断窗口大小；旧 step 仍由高水位拒绝复用。"""

        return len(self._finalized_steps)

    def _state(self, request_id: str) -> _SequenceKVState:
        try:
            return self._sequences[request_id]
        except KeyError as exc:
            raise KeyError(f"Unknown KV sequence: {request_id}") from exc
