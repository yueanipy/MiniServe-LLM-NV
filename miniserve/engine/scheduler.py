"""Phase 1 的单请求、整段 prefill Scheduler。"""

from __future__ import annotations

from collections.abc import Sequence

from .plan import DecodeItem, ExecutionPlan, PrefillItem
from .state import RequestStatus, SequenceView
from miniserve.kv.base import KVCapacitySnapshot


def _validate_prefill_chunk(
    *,
    max_num_batched_tokens: int,
    enable_chunked_prefill: bool,
    prefill_chunk_size: int | None,
) -> None:
    if prefill_chunk_size is None:
        return
    if (
        not isinstance(prefill_chunk_size, int)
        or isinstance(prefill_chunk_size, bool)
        or prefill_chunk_size <= 0
    ):
        raise ValueError("prefill_chunk_size must be a positive integer or None.")
    if not enable_chunked_prefill:
        raise ValueError("prefill_chunk_size requires enable_chunked_prefill=True.")
    if prefill_chunk_size > max_num_batched_tokens:
        raise ValueError(
            "prefill_chunk_size cannot exceed max_num_batched_tokens."
        )


class SingleRequestScheduler:
    """每个 step 最多产生一个 prefill 或 decode item。"""

    def __init__(
        self,
        *,
        max_num_batched_tokens: int,
        enable_chunked_prefill: bool = False,
        prefill_chunk_size: int | None = None,
    ) -> None:
        if max_num_batched_tokens <= 0:
            raise ValueError("max_num_batched_tokens must be positive.")
        _validate_prefill_chunk(
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=enable_chunked_prefill,
            prefill_chunk_size=prefill_chunk_size,
        )
        self._max_num_batched_tokens = max_num_batched_tokens
        self._enable_chunked_prefill = enable_chunked_prefill
        self._prefill_chunk_size = prefill_chunk_size
        self._next_step_id = 0

    def schedule(self, sequences: Sequence[SequenceView]) -> ExecutionPlan:
        """只读 immutable views 并返回 plan；不持有或修改 SequenceStore。"""

        step_id = self._next_step_id
        self._next_step_id += 1
        if not sequences:
            return ExecutionPlan.empty_plan(step_id)
        if len(sequences) != 1:
            raise RuntimeError("Phase 1 scheduler supports exactly one active request.")

        sequence = sequences[0]
        if sequence.status is RequestStatus.WAITING:
            if (
                not self._enable_chunked_prefill
                and sequence.prompt_len > self._max_num_batched_tokens
            ):
                raise RuntimeError("Prompt exceeds the Phase 1 per-step token budget.")

        if sequence.status in {RequestStatus.WAITING, RequestStatus.PREFILLING}:
            remaining = sequence.prompt_len - sequence.prefill_cursor
            take = remaining
            if self._enable_chunked_prefill:
                take = min(take, self._max_num_batched_tokens)
            if self._prefill_chunk_size is not None:
                take = min(take, self._prefill_chunk_size)
            if take <= 0:
                raise RuntimeError("Prefillable sequence has no remaining prompt tokens.")
            item = PrefillItem(
                request_id=sequence.request_id,
                token_start=sequence.prefill_cursor,
                token_end=sequence.prefill_cursor + take,
            )
            return ExecutionPlan(
                step_id=step_id,
                prefill_items=(item,),
                scheduled_tokens=take,
            )
        if sequence.status is RequestStatus.DECODING:
            if sequence.next_input_token_id is None:
                raise RuntimeError("DECODING SequenceView has no pending input token.")
            item = DecodeItem(
                request_id=sequence.request_id,
                input_token_id=sequence.next_input_token_id,
                position=sequence.computed_tokens,
            )
            return ExecutionPlan(step_id=step_id, decode_items=(item,), scheduled_tokens=1)
        raise RuntimeError(f"Unexpected active request status: {sequence.status.name}")


class ContinuousBatchScheduler:
    """Phase 3 FIFO、decode-first、整段 prefill 的确定性 reference policy。"""

    def __init__(
        self,
        *,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        enable_chunked_prefill: bool = False,
        prefill_chunk_size: int | None = None,
    ) -> None:
        if max_num_seqs < 2 or max_num_batched_tokens <= 0:
            raise ValueError("Continuous scheduler requires >=2 seqs and a positive token budget.")
        _validate_prefill_chunk(
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=enable_chunked_prefill,
            prefill_chunk_size=prefill_chunk_size,
        )
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.enable_chunked_prefill = enable_chunked_prefill
        self.prefill_chunk_size = prefill_chunk_size
        self._next_step_id = 0

    def schedule(
        self,
        sequences: Sequence[SequenceView],
        kv: KVCapacitySnapshot,
    ) -> ExecutionPlan:
        step_id = self._next_step_id
        self._next_step_id += 1
        ordered = tuple(sorted(sequences, key=lambda item: (item.arrival_ns, item.request_id)))
        token_budget = self.max_num_batched_tokens
        seq_budget = self.max_num_seqs
        decode_items: list[DecodeItem] = []
        prefill_items: list[PrefillItem] = []
        # Cache-only blocks are reclaimable under the same prefix-cache lock.
        # prepare_step performs the physical eviction before reservation.
        estimated_free_blocks = kv.free_blocks + kv.cached_blocks

        prefill_candidates = [
            sequence
            for sequence in ordered
            if sequence.status in {RequestStatus.WAITING, RequestStatus.PREFILLING}
        ]
        can_admit_waiting = kv.free_blocks > 0
        has_progressable_prefill = any(
            sequence.status is RequestStatus.PREFILLING
            or (sequence.status is RequestStatus.WAITING and can_admit_waiting)
            for sequence in prefill_candidates
        )
        reserved_seq = 1 if self.enable_chunked_prefill and has_progressable_prefill else 0
        reserved_token = reserved_seq

        # 已持有 KV slot 的 decode 请求优先；每请求本 step 只推进一个 token。
        for sequence in ordered:
            if sequence.status is not RequestStatus.DECODING:
                continue
            if seq_budget <= reserved_seq or token_budget <= reserved_token:
                break
            if sequence.next_input_token_id is None:
                raise RuntimeError("DECODING view has no pending input token.")
            extra_blocks = (
                (sequence.computed_tokens + 1 + kv.block_size - 1) // kv.block_size
                - (sequence.computed_tokens + kv.block_size - 1) // kv.block_size
            )
            if extra_blocks > estimated_free_blocks:
                continue
            decode_items.append(
                DecodeItem(sequence.request_id, sequence.next_input_token_id, sequence.computed_tokens)
            )
            seq_budget -= 1
            token_budget -= 1
            estimated_free_blocks -= extra_blocks

        # PREFILLING 已有 allocation；WAITING 只可使用 capacity snapshot 中的空 slot。
        for sequence in ordered:
            if sequence.status not in {RequestStatus.WAITING, RequestStatus.PREFILLING}:
                continue
            remaining = sequence.prompt_len - sequence.prefill_cursor
            if remaining <= 0:
                raise RuntimeError("Prefillable sequence has no remaining prompt tokens.")
            if seq_budget == 0 or token_budget == 0:
                continue
            if sequence.status is RequestStatus.WAITING:
                pass
            take = min(remaining, token_budget) if self.enable_chunked_prefill else remaining
            if self.prefill_chunk_size is not None:
                take = min(take, self.prefill_chunk_size)
            if take <= 0 or (not self.enable_chunked_prefill and take > token_budget):
                continue
            context_before = sequence.computed_tokens
            while take > 0:
                context_after = context_before + take
                extra_blocks = (
                    (context_after + kv.block_size - 1) // kv.block_size
                    - (context_before + kv.block_size - 1) // kv.block_size
                )
                if extra_blocks <= estimated_free_blocks:
                    break
                if not self.enable_chunked_prefill:
                    take = 0
                    break
                take -= 1
            if take <= 0:
                continue
            prefill_items.append(
                PrefillItem(
                    sequence.request_id,
                    sequence.prefill_cursor,
                    sequence.prefill_cursor + take,
                )
            )
            seq_budget -= 1
            token_budget -= take
            estimated_free_blocks -= extra_blocks

        return ExecutionPlan(
            step_id=step_id,
            prefill_items=tuple(prefill_items),
            decode_items=tuple(decode_items),
            scheduled_tokens=self.max_num_batched_tokens - token_budget,
        )
