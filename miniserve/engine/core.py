"""Phase 1 从请求提交到结束的唯一 Engine 主链。"""

from __future__ import annotations

import copy
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Callable

from miniserve.config.runtime_config import RuntimeConfig
from miniserve.kv.base import KVStepContext
from miniserve.kv.manager import KVCacheManager
from miniserve.kv.paged import PagedKVCacheBackend

from .batch_builder import BatchBuilder
from .events import EngineEvent, EngineEventKind, ResultPublisher
from .faults import (
    EngineFatalError,
    EngineStateError,
    FaultKind,
    classify_fault,
    is_out_of_memory,
)
from .executor import ExecutorOutput, ModelExecutor
from .request import EngineRequest, SamplingParams
from .scheduler import ContinuousBatchScheduler, SingleRequestScheduler
from .state import RequestStatus, SequenceState, SequenceStore, SequenceView
from .plan import ExecutionPlan


@dataclass(frozen=True)
class EngineResult:
    request_id: str
    status: RequestStatus
    generated_token_ids: tuple[int, ...]
    finish_reason: str | None
    error_message: str | None


@dataclass(frozen=True)
class _StagedRequest:
    request_id: str
    sequence: SequenceState
    token_event: EngineEvent | None


def validate_request_sampling_scope(
    config: RuntimeConfig,
    sampling_params: SamplingParams,
) -> None:
    """Reject unsupported C1 sampling before request state or KV is created."""

    if (
        config.engine.attention_backend == "flashinfer_paged"
        and sampling_params.temperature != 0.0
    ):
        raise ValueError(
            "C1_EXPERIMENTAL_GREEDY_ONLY: flashinfer_paged supports "
            "temperature=0 greedy sampling only."
        )


class EngineCore:
    """不含 HTTP、tokenizer 或 HF AutoModel 的同步学习版 Engine。"""

    def __init__(
        self,
        runtime_config: RuntimeConfig,
        executor: ModelExecutor,
        *,
        kv_manager: KVCacheManager | None = None,
        publisher: ResultPublisher | None = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        max_retained_results: int = 1024,
        decode_trace_capacity: int = 0,
    ) -> None:
        if max_retained_results <= 0:
            raise ValueError("max_retained_results must be positive.")
        if (
            not isinstance(decode_trace_capacity, int)
            or isinstance(decode_trace_capacity, bool)
            or decode_trace_capacity < 0
        ):
            raise ValueError("decode_trace_capacity must be a non-negative integer.")
        # M0：按独立语义开关做 capability validation；Phase 强绑定仅保留为兼容回归入口。
        runtime_config.validate_capabilities()
        engine_config = runtime_config.engine
        if engine_config.kv_backend == "recompute":
            if kv_manager is not None:
                raise ValueError("Recompute EngineCore must not receive a KV manager.")
            self._continuous = False
        elif engine_config.kv_backend == "contiguous":
            if kv_manager is None:
                raise ValueError("Contiguous EngineCore requires a KV manager.")
            self._continuous = engine_config.max_num_seqs > 1
        else:
            if kv_manager is None:
                raise ValueError("Paged EngineCore requires a KV manager.")
            self._continuous = True
        if executor.device.type != "cuda":
            # 正式 Profile L 集成 Gate 必须是真 GPU；小型 CPU correctness 测试直接测下层模块。
            raise RuntimeError("Phase 1 EngineCore requires a CUDA model.")
        self.config = runtime_config
        self.executor = executor
        self.executor.model.set_rmsnorm_backend(runtime_config.engine.rmsnorm_backend)
        if runtime_config.engine.decode_execution == "cuda_graph":
            if runtime_config.engine.batched_graph:
                enable_graph = getattr(
                    self.executor.attention_backend,
                    "enable_batched_graph",
                    None,
                )
                if enable_graph is None:
                    raise RuntimeError(
                        "Selected attention backend cannot enable batched CUDA Graph."
                    )
                enable_graph(
                    model=self.executor.model,
                    max_model_len=runtime_config.model.max_model_len,
                )
            else:
                enable_graph = getattr(
                    self.executor.attention_backend,
                    "enable_cuda_graph",
                    None,
                )
                if enable_graph is None:
                    raise RuntimeError("Selected attention backend cannot enable CUDA Graph.")
                enable_graph(
                    max_model_len=runtime_config.model.max_model_len,
                )
        self.kv_manager = kv_manager
        self.publisher = publisher or ResultPublisher(
            max_retained_requests=max_retained_results
        )
        self.store = SequenceStore()
        if self._continuous:
            self.scheduler = ContinuousBatchScheduler(
                # Paged B=1 is still a continuous engine.  The scheduler requires
                # at least two policy slots, while the backend may expose one.
                max_num_seqs=max(2, runtime_config.engine.max_num_seqs),
                max_num_batched_tokens=runtime_config.engine.max_num_batched_tokens,
                enable_chunked_prefill=runtime_config.engine.enable_chunked_prefill,
                prefill_chunk_size=runtime_config.engine.prefill_chunk_size,
            )
        else:
            self.scheduler = SingleRequestScheduler(
                max_num_batched_tokens=runtime_config.engine.max_num_batched_tokens,
                enable_chunked_prefill=runtime_config.engine.enable_chunked_prefill,
                prefill_chunk_size=runtime_config.engine.prefill_chunk_size,
            )
        self.batch_builder = BatchBuilder(
            self.store,
            device=executor.device,
            decode_batching=engine_config.decode_batching,
            block_size=(
                kv_manager.backend.layout.block_size
                if kv_manager is not None
                and isinstance(kv_manager.backend, PagedKVCacheBackend)
                else None
            ),
            physical_block_count=(
                kv_manager.backend.layout.num_blocks
                if kv_manager is not None
                and isinstance(kv_manager.backend, PagedKVCacheBackend)
                else None
            ),
        )
        self._clock_ns = clock_ns
        self.last_output: ExecutorOutput | None = None
        self.last_outputs: tuple[ExecutorOutput, ...] = ()
        self.last_plan: ExecutionPlan | None = None
        self.last_execution_order: tuple[str, ...] = ()
        self.decode_forward_count = 0
        self.decode_forward_row_total = 0
        self._decode_forward_row_histogram: dict[int, int] = {}
        self._decode_forward_trace_capacity = decode_trace_capacity
        self._decode_forward_trace: deque[int] | None = (
            deque(maxlen=decode_trace_capacity)
            if decode_trace_capacity > 0
            else None
        )
        self._decode_forward_trace_dropped = 0
        self.prefill_forward_count = 0
        self.prefill_input_tokens = 0
        set_forward_observer = getattr(self.executor, "set_forward_observer", None)
        if set_forward_observer is not None:
            set_forward_observer(self._record_forward_completed)
        self._max_retained_results = max_retained_results
        self._finalized_results: OrderedDict[str, EngineResult] = OrderedDict()

    def submit(self, request: EngineRequest) -> SequenceView:
        if request.request_id in self._finalized_results or self.store.contains(request.request_id):
            raise KeyError(f"Duplicate request_id: {request.request_id}")
        validate_request_sampling_scope(self.config, request.sampling_params)
        if not self._continuous and self.store.active():
            raise RuntimeError("Phase 1 accepts only one active request.")
        if self._continuous and len(self.store.active()) >= self.config.engine.max_pending_requests:
            raise RuntimeError("max_pending_requests capacity reached.")
        requested_total = len(request.prompt_token_ids) + request.sampling_params.max_new_tokens
        if requested_total > self.config.model.max_model_len:
            raise ValueError(
                "prompt_tokens + max_new_tokens exceeds max_model_len "
                f"({requested_total} > {self.config.model.max_model_len})."
            )
        if self.kv_manager is not None and isinstance(
            self.kv_manager.backend, PagedKVCacheBackend
        ):
            block_size = self.kv_manager.backend.layout.block_size
            required_blocks = (requested_total + block_size - 1) // block_size
            if required_blocks > self.kv_manager.backend.allocator.num_blocks:
                raise ValueError(
                    "CONTEXT_CAPACITY_EXCEEDED: request requires "
                    f"{required_blocks} blocks, pool has {self.kv_manager.backend.allocator.num_blocks}."
                )
        if any(token >= self.config.model.vocab_size for token in request.prompt_token_ids):
            raise ValueError("Prompt contains a token outside the vocabulary.")
        if request.sampling_params.top_k > self.config.model.vocab_size:
            raise ValueError("top_k exceeds the model vocabulary size.")
        if any(
            token >= self.config.model.vocab_size
            for token in request.sampling_params.stop_token_ids
        ):
            raise ValueError("stop_token_ids contains a token outside the vocabulary.")
        if (
            self._continuous
            and not self.config.engine.enable_chunked_prefill
            and len(request.prompt_token_ids) > self.config.engine.max_num_batched_tokens
        ):
            raise ValueError(
                "CHUNKED_PREFILL_DISABLED: prompt length "
                f"{len(request.prompt_token_ids)} exceeds max_num_batched_tokens "
                f"{self.config.engine.max_num_batched_tokens}; enable chunked prefill "
                "or provide a larger explicit budget."
            )
        sequence = SequenceState.from_request(request)
        # Continuous waiting requests may outnumber execution slots. Prefix-enabled
        # requests attach immediately only while a backend slot is actually free.
        should_register_kv = self.kv_manager is not None and (
            not self._continuous
            or (
                self.config.engine.enable_prefix_cache
                and self.kv_manager.available_sequence_slots > 0
            )
        )
        if should_register_kv:
            self.kv_manager.register_sequence(request.request_id)
        try:
            if (
                self.kv_manager is not None
                and self.config.engine.enable_prefix_cache
                and self.kv_manager.has_sequence(request.request_id)
            ):
                match = self.kv_manager.attach_prefix(
                    request.request_id, request.prompt_token_ids
                )
                if match.matched_tokens:
                    sequence.apply_prefix_match(match.matched_tokens)
            self.store.add(sequence)
        except Exception:
            if self.kv_manager is not None and self.kv_manager.has_sequence(request.request_id):
                self.kv_manager.release_sequence(request.request_id)
            raise
        return sequence.snapshot()

    def validate_sampling_scope(self, sampling_params: SamplingParams) -> None:
        """Expose the same fail-closed C1 sampling boundary to the API facade."""

        validate_request_sampling_scope(self.config, sampling_params)

    def step(self) -> SequenceView | tuple[SequenceView, ...] | None:
        snapshots = self.step_multi()
        if not snapshots:
            return None
        return snapshots[0] if len(snapshots) == 1 else snapshots

    def step_multi(self) -> tuple[SequenceView, ...]:
        """执行一个多请求原子 step；Phase 1/2 仍自然退化为一个 item。"""

        if self._continuous:
            self._cleanup_terminal_sequences()
            if self.config.engine.enable_prefix_cache:
                self._admit_prefix_waiters()
        views = self.store.snapshot_schedulable()
        if self._continuous:
            assert self.kv_manager is not None
            if self.config.engine.enable_prefix_cache:
                views = tuple(
                    view
                    for view in views
                    if view.status is not RequestStatus.WAITING
                    or self.kv_manager.has_sequence(view.request_id)
                )
            plan = self.scheduler.schedule(views, self.kv_manager.capacity_snapshot())
        else:
            plan = self.scheduler.schedule(views)
        if plan.empty:
            self.last_plan = plan
            self.last_execution_order = ()
            return ()
        self.last_plan = plan

        request_ids = tuple(
            item.request_id for item in (*plan.decode_items, *plan.prefill_items)
        )
        kv_step: KVStepContext | None = None
        kv_committed = False
        commit_invariant = False
        try:
            if self.kv_manager is None:
                batch = self.batch_builder.prepare_recompute(plan)
                outputs = (self.executor.execute(batch),)
                self.last_execution_order = request_ids
            else:
                # WAITING 请求只在本轮被实际接纳时占用 contiguous slot。
                for item in plan.prefill_items:
                    if not self.kv_manager.has_sequence(item.request_id):
                        self.kv_manager.register_sequence(item.request_id)
                kv_step = self.kv_manager.prepare_step(plan)
                if self._continuous:
                    prepared = self.batch_builder.prepare_contiguous_step(plan, kv_step)
                    step_output = self.executor.execute_step(prepared)
                    outputs = step_output.outputs
                    self.last_execution_order = step_output.execution_order
                else:
                    batch = self.batch_builder.prepare_contiguous(plan, kv_step)
                    outputs = (self.executor.execute(batch),)
                    self.last_execution_order = request_ids
            if not outputs:
                raise RuntimeError("Executor returned no outputs for a non-empty plan.")
            self.last_output = outputs[-1]
            self.last_outputs = outputs
            output_tokens = self._map_output_tokens(request_ids, outputs)
            staged = self._stage_plan_updates(plan, request_ids, output_tokens)

            if self.kv_manager is not None:
                assert kv_step is not None
                self.kv_manager.commit_step(kv_step)
                kv_committed = True
            commit_invariant = kv_committed
            self.executor.commit_staged_samples(outputs)
            for item in staged:
                self.store.replace(item.sequence)
            if self.kv_manager is not None:
                for item in staged:
                    self.kv_manager.assert_consistent(
                        item.request_id, item.sequence.computed_tokens
                    )
            if self.config.engine.enable_prefix_cache and self.kv_manager is not None:
                prefill_ids = {item.request_id for item in plan.prefill_items}
                for item in staged:
                    if (
                        item.request_id in prefill_ids
                        and item.sequence.prefill_cursor
                        == len(item.sequence.prompt_token_ids)
                    ):
                        self.kv_manager.cache_prefix(
                            item.request_id, item.sequence.prompt_token_ids
                        )
            for item in staged:
                if item.token_event is not None:
                    self.publisher.publish(item.token_event)
            snapshots = [item.sequence.snapshot() for item in staged]
        except Exception as exc:
            if commit_invariant:
                commit_fault = classify_fault(exc)
                if commit_fault is FaultKind.FATAL_CONTEXT:
                    raise EngineFatalError(
                        "COMMIT_INVARIANT FATAL_CONTEXT: "
                        f"{type(exc).__name__}: {exc}",
                        original=exc,
                    ) from exc
                raise EngineStateError(
                    f"COMMIT_INVARIANT FATAL_STATE: {type(exc).__name__}: {exc}",
                    original=exc,
                ) from exc
            fault = classify_fault(exc)
            fatal = fault is not FaultKind.RECOVERABLE
            rollback_invariant: Exception | None = None
            # fatal：不在坏 CUDA context 上做任何 GPU 清理/重用，也不 abort KV reservation
            # （allocator 记账随后随进程退出丢弃）；只做 CPU 侧终态收敛并向上抛出。
            if not fatal and self.kv_manager is not None and kv_step is not None and not kv_committed:
                try:
                    self.kv_manager.abort_step(kv_step)
                except Exception as abort_exc:
                    rollback_invariant = abort_exc
            reason = f"{type(exc).__name__}: {exc}"
            if fault is FaultKind.FATAL_STATE:
                reason = f"FATAL_STATE: {reason}"
            elif fatal:
                reason = f"FATAL_CONTEXT: {reason}"
            elif is_out_of_memory(exc):
                reason = f"CUDA_OOM: {reason}"
            snapshots = []
            for request_id in request_ids:
                if not self.store.contains(request_id):
                    continue
                sequence = self.store.get(request_id)
                if sequence.status not in {
                    RequestStatus.FINISHED,
                    RequestStatus.FAILED,
                    RequestStatus.CANCELLED,
                }:
                    sequence.fail(reason, now_ns=self._clock_ns())
                snapshots.append(sequence.snapshot())
            if fault is FaultKind.FATAL_STATE:
                raise EngineStateError(reason, original=exc) from exc
            if fatal:
                # fail-stop 由 EngineWorker 执行：停止 admission、readiness=false、
                # 发布活动请求终态并退出进程。
                raise EngineFatalError(reason, original=exc) from exc
            if rollback_invariant is not None:
                # 回滚本身失败证明 KV 记账/物理状态已不能作为安全恢复边界。
                # 这不是可重试的普通失败，也不能借用 CUDA fatal 的原因名。
                rollback_fault = classify_fault(rollback_invariant)
                rollback_category = (
                    "FATAL_CONTEXT"
                    if rollback_fault is FaultKind.FATAL_CONTEXT
                    else "FATAL_STATE"
                )
                state_reason = (
                    f"ROLLBACK_INVARIANT {rollback_category}: "
                    f"{type(exc).__name__}: {exc}; "
                    f"KV abort also failed: {type(rollback_invariant).__name__}: "
                    f"{rollback_invariant}"
                )
                if rollback_fault is FaultKind.FATAL_CONTEXT:
                    raise EngineFatalError(
                        state_reason, original=rollback_invariant
                    ) from rollback_invariant
                raise EngineStateError(
                    state_reason, original=rollback_invariant
                ) from rollback_invariant

        result = tuple(snapshots)
        if self._continuous:
            self._cleanup_terminal_sequences()
        return result

    def _record_decode_forward(self, rows: int) -> None:
        if not isinstance(rows, int) or isinstance(rows, bool) or rows <= 0:
            raise ValueError("Decode forward rows must be a positive integer.")
        self.decode_forward_count += 1
        self.decode_forward_row_total += rows
        self._decode_forward_row_histogram[rows] = (
            self._decode_forward_row_histogram.get(rows, 0) + 1
        )
        trace = self._decode_forward_trace
        if trace is None:
            return
        if len(trace) == self._decode_forward_trace_capacity:
            self._decode_forward_trace_dropped += 1
        trace.append(rows)

    def _record_forward_completed(
        self, mode: str, decode_rows: int, prefill_input_tokens: int
    ) -> None:
        """Record a model forward after it returns, independent of plan commit."""

        if mode == "decode":
            if prefill_input_tokens != 0:
                raise ValueError("Decode forward cannot carry prefill token accounting.")
            self._record_decode_forward(decode_rows)
            return
        if mode == "prefill":
            if decode_rows != 0 or prefill_input_tokens <= 0:
                raise ValueError("Invalid completed prefill forward observation.")
            self.prefill_forward_count += 1
            self.prefill_input_tokens += prefill_input_tokens
            return
        raise ValueError(f"Unknown completed forward mode: {mode!r}")

    @property
    def decode_forward_rows(self) -> list[int]:
        """Return only the bounded diagnostic trace; production defaults to empty."""

        trace = self._decode_forward_trace
        return [] if trace is None else list(trace)

    @property
    def decode_forward_row_histogram(self) -> dict[int, int]:
        return dict(sorted(self._decode_forward_row_histogram.items()))

    @property
    def decode_forward_trace_state(self) -> dict[str, int]:
        return {
            "capacity": self._decode_forward_trace_capacity,
            "retained": len(self.decode_forward_rows),
            "dropped": self._decode_forward_trace_dropped,
            "total_forwards": self.decode_forward_count,
            "total_rows": self.decode_forward_row_total,
        }

    def _map_output_tokens(
        self,
        request_ids: tuple[str, ...],
        outputs: tuple[ExecutorOutput, ...],
    ) -> dict[str, int | None]:
        """把 executor 的 batch/单行输出拆成稳定 request_id 映射。"""

        mapped: dict[str, int | None] = {}
        for output in outputs:
            if not output.request_ids:
                raise RuntimeError("Executor output has no request IDs.")
            if output.next_token_ids and len(output.next_token_ids) != len(output.request_ids):
                raise RuntimeError("Executor output token count does not match request rows.")
            if not output.next_token_ids:
                if len(output.request_ids) != 1:
                    raise RuntimeError("A no-sample output must contain exactly one prefill row.")
                tokens: tuple[int | None, ...] = (None,)
            else:
                tokens = tuple(output.next_token_ids)
            for request_id, token in zip(output.request_ids, tokens):
                if request_id in mapped:
                    raise RuntimeError(f"Executor returned duplicate request row: {request_id}")
                if request_id not in request_ids:
                    raise RuntimeError(f"Executor returned an unexpected request: {request_id}")
                mapped[request_id] = token
        if set(mapped) != set(request_ids):
            raise RuntimeError("Executor outputs do not cover the complete ExecutionPlan.")
        return mapped

    def _stage_plan_updates(
        self,
        plan: ExecutionPlan,
        request_ids: tuple[str, ...],
        output_tokens: dict[str, int | None],
    ) -> tuple[_StagedRequest, ...]:
        """在 commit 前构造完整 CPU 状态和事件，不触碰真实 store/RNG。"""

        prefill_end = {item.request_id: item.token_end for item in plan.prefill_items}
        prefill_ids = set(prefill_end)
        staged: list[_StagedRequest] = []
        for request_id in request_ids:
            work = copy.deepcopy(self.store.get(request_id))
            token = output_tokens[request_id]
            if request_id in prefill_ids:
                if work.status is RequestStatus.WAITING:
                    work.begin_prefill()
                work.commit_prefill(prefill_end[request_id])
            else:
                work.mark_decode_input_computed()

            if token is None:
                if (
                    request_id not in prefill_ids
                    or work.prefill_cursor == len(work.prompt_token_ids)
                ):
                    raise RuntimeError("Executor suppressed sampling for a final/decode batch.")
                staged.append(_StagedRequest(request_id, work, None))
                continue
            if not 0 <= token < self.config.model.vocab_size:
                raise RuntimeError("Executor sampled a token outside the model vocabulary.")
            now_ns = self._clock_ns()
            work.append_sampled_token(token, now_ns=now_ns)
            event = EngineEvent(
                request_id,
                EngineEventKind.TOKEN,
                token_id=token,
                timestamp_ns=now_ns,
            )
            if reason := work.should_finish():
                work.finish(reason, now_ns=self._clock_ns())
            staged.append(_StagedRequest(request_id, work, event))
        return tuple(staged)

    def run_until_terminal(self, request_id: str) -> SequenceState:
        if self._continuous:
            raise RuntimeError("Use ResultPublisher/EngineWorker for continuous requests.")
        sequence = self.store.get(request_id)
        prefill_take = len(sequence.prompt_token_ids)
        if self.config.engine.enable_chunked_prefill:
            prefill_take = min(prefill_take, self.config.engine.max_num_batched_tokens)
        if self.config.engine.prefill_chunk_size is not None:
            prefill_take = min(prefill_take, self.config.engine.prefill_chunk_size)
        prefill_steps = (
            len(sequence.prompt_token_ids) + max(1, prefill_take) - 1
        ) // max(1, prefill_take)
        max_steps = prefill_steps + sequence.sampling_params.max_new_tokens + 1
        for _ in range(max_steps):
            # Staged commits replace the mutable SequenceState object in the store.
            # Re-read the canonical object each step; holding the original snapshot
            # would make run_until_terminal observe a stale WAITING/DECODING state.
            sequence = self.store.get(request_id)
            if sequence.status in {RequestStatus.FINISHED, RequestStatus.FAILED, RequestStatus.CANCELLED}:
                return sequence
            self.step()
        sequence = self.store.get(request_id)
        sequence.fail("Engine exceeded the bounded Phase 1 step count.", now_ns=self._clock_ns())
        return sequence

    def cancel(self, request_id: str, *, reason: str = "cancelled") -> SequenceView:
        if request_id in self._finalized_results:
            raise RuntimeError("Request is already finalized.")
        sequence = self.store.get(request_id)
        sequence.cancel(now_ns=self._clock_ns(), reason=reason)
        snapshot = sequence.snapshot()
        if self._continuous:
            self._cleanup_terminal_sequences()
        return snapshot

    def finalize_sequence(self, request_id: str) -> EngineResult:
        if request_id in self._finalized_results:
            return self._finalized_results[request_id]
        sequence = self.store.get(request_id)
        if sequence.status not in {RequestStatus.FINISHED, RequestStatus.FAILED, RequestStatus.CANCELLED}:
            raise RuntimeError("Cannot finalize a non-terminal sequence.")
        return self._finalize_state(sequence)

    def _finalize_state(self, sequence: SequenceState) -> EngineResult:
        request_id = sequence.request_id
        result = EngineResult(
            request_id=request_id,
            status=sequence.status,
            generated_token_ids=tuple(sequence.generated_token_ids),
            finish_reason=sequence.finish_reason,
            error_message=sequence.error_message,
        )
        if self.kv_manager is not None and self.kv_manager.has_sequence(request_id):
            self.kv_manager.assert_consistent(request_id, sequence.computed_tokens)
            self.kv_manager.release_sequence(request_id)
        self.executor.release_request(request_id)
        terminal_event = None
        if not self.publisher.is_terminal(request_id):
            terminal_reason = (
                sequence.error_message
                if sequence.status is RequestStatus.FAILED
                else sequence.finish_reason
            )
            # release() 会把 mutable SequenceState 改成 RELEASED；必须先冻结合法终态。
            terminal_event = EngineEvent(
                request_id,
                EngineEventKind.TERMINAL,
                status=sequence.status,
                reason=terminal_reason,
                timestamp_ns=sequence.finish_ns or self._clock_ns(),
            )
        # ResultPublisher 的 subscriber 可能在另一个线程/事件循环中立即处理 terminal。
        # 必须先完成 store cleanup 并登记 final result，保证“观察到 terminal”蕴含
        # finalized_result(request_id) 已可读，不能留下 HTTP 已返回但结果仍为 None 的 race。
        sequence.release()
        self.store.remove(request_id)
        self._finalized_results[request_id] = result
        self._finalized_results.move_to_end(request_id)
        while len(self._finalized_results) > self._max_retained_results:
            self._finalized_results.popitem(last=False)
        if terminal_event is not None:
            self.publisher.publish(terminal_event)
        return result

    def _cleanup_terminal_sequences(self) -> None:
        for sequence in self.store.terminal():
            self._finalize_state(sequence)

    def _admit_prefix_waiters(self) -> None:
        """Attach prefixes only after a continuous request owns a KV slot."""

        assert self.kv_manager is not None
        for view in self.store.snapshot_schedulable():
            if self.kv_manager.available_sequence_slots <= 0:
                break
            if view.status is not RequestStatus.WAITING:
                continue
            if self.kv_manager.has_sequence(view.request_id):
                continue
            sequence = self.store.get(view.request_id)
            self.kv_manager.register_sequence(view.request_id)
            try:
                match = self.kv_manager.attach_prefix(
                    view.request_id, sequence.prompt_token_ids
                )
                if match.matched_tokens:
                    sequence.apply_prefix_match(match.matched_tokens)
            except Exception:
                self.kv_manager.release_sequence(view.request_id)
                raise

    def finalized_result(self, request_id: str) -> EngineResult | None:
        return self._finalized_results.get(request_id)

    @property
    def retained_finalized_result_count(self) -> int:
        """返回当前可查询的最近终态结果数，不包含活动请求。"""

        return len(self._finalized_results)
