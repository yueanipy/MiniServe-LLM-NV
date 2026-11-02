"""单请求的 canonical token 状态与可执行不变量。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from .request import EngineRequest, SamplingParams


class RequestStatus(Enum):
    WAITING = auto()
    PREFILLING = auto()
    DECODING = auto()
    FINISHED = auto()
    FAILED = auto()
    CANCELLED = auto()
    RELEASED = auto()


@dataclass(frozen=True)
class SequenceView:
    """Scheduler 只读快照，避免 policy 意外修改真实状态。"""

    request_id: str
    status: RequestStatus
    prompt_len: int
    prefill_cursor: int
    computed_tokens: int
    total_tokens: int
    num_generated: int
    max_new_tokens: int
    arrival_ns: int
    next_input_token_id: int | None


@dataclass
class SequenceState:
    """一个请求的 token 生命周期。

    DECODING 状态始终恰好保留一个“已采样、尚未 forward”的 token，因此：
    ``total_tokens == computed_tokens + 1``。这是 Phase 2 接入 KV 时最重要的边界。
    """

    request_id: str
    prompt_token_ids: tuple[int, ...]
    generated_token_ids: list[int]
    status: RequestStatus
    prefill_cursor: int
    computed_tokens: int
    sampling_params: SamplingParams
    arrival_ns: int
    first_token_ns: int | None = None
    finish_ns: int | None = None
    finish_reason: str | None = None
    error_message: str | None = None

    @classmethod
    def from_request(cls, request: EngineRequest) -> "SequenceState":
        state = cls(
            request_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            generated_token_ids=[],
            status=RequestStatus.WAITING,
            prefill_cursor=0,
            computed_tokens=0,
            sampling_params=request.sampling_params,
            arrival_ns=request.arrival_ns,
        )
        state.assert_invariants()
        return state

    @property
    def all_tokens(self) -> tuple[int, ...]:
        return self.prompt_token_ids + tuple(self.generated_token_ids)

    @property
    def total_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.generated_token_ids)

    def snapshot(self) -> SequenceView:
        self.assert_invariants()
        return SequenceView(
            request_id=self.request_id,
            status=self.status,
            prompt_len=len(self.prompt_token_ids),
            prefill_cursor=self.prefill_cursor,
            computed_tokens=self.computed_tokens,
            total_tokens=self.total_tokens,
            num_generated=len(self.generated_token_ids),
            max_new_tokens=self.sampling_params.max_new_tokens,
            arrival_ns=self.arrival_ns,
            next_input_token_id=(
                self.all_tokens[self.computed_tokens]
                if self.status is RequestStatus.DECODING
                else None
            ),
        )

    def begin_prefill(self) -> None:
        if self.status is not RequestStatus.WAITING:
            raise RuntimeError("Only a waiting sequence can begin prefill.")
        self.status = RequestStatus.PREFILLING
        self.assert_invariants()

    def apply_prefix_match(self, matched_tokens: int) -> None:
        if self.status is not RequestStatus.WAITING:
            raise RuntimeError("Prefix match must be applied before scheduling.")
        if matched_tokens < 0 or matched_tokens >= len(self.prompt_token_ids):
            raise ValueError("Prefix match must leave at least one prompt token to prefill.")
        self.prefill_cursor = matched_tokens
        self.computed_tokens = matched_tokens
        self.assert_invariants()

    def mark_prompt_computed(self) -> None:
        self.commit_prefill(len(self.prompt_token_ids))

    def commit_prefill(self, token_end: int) -> None:
        """成功提交一个 prompt chunk；中间 chunk 保持 PREFILLING 且不采样。"""

        if self.status is not RequestStatus.PREFILLING:
            raise RuntimeError("Prompt completion requires PREFILLING state.")
        if not self.prefill_cursor < token_end <= len(self.prompt_token_ids):
            raise RuntimeError("Prefill commit must advance within the prompt.")
        self.prefill_cursor = token_end
        self.computed_tokens = token_end
        self.assert_invariants()

    def mark_decode_input_computed(self) -> None:
        if self.status is not RequestStatus.DECODING or self.computed_tokens != self.total_tokens - 1:
            raise RuntimeError("Decode must consume exactly the pending sampled token.")
        self.computed_tokens += 1

    def append_sampled_token(self, token_id: int, *, now_ns: int) -> None:
        if self.status not in {RequestStatus.PREFILLING, RequestStatus.DECODING}:
            raise RuntimeError("A token can only be sampled after model execution.")
        if self.computed_tokens != self.total_tokens:
            raise RuntimeError("All current tokens must be computed before sampling the next token.")
        if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0:
            raise ValueError("token_id must be a non-negative integer.")
        self._validate_timestamp(now_ns)
        self.generated_token_ids.append(token_id)
        if self.first_token_ns is None:
            self.first_token_ns = now_ns
        self.status = RequestStatus.DECODING
        self.assert_invariants()

    def should_finish(self) -> str | None:
        if not self.generated_token_ids:
            return None
        if self.generated_token_ids[-1] in self.sampling_params.stop_token_ids:
            return "stop"
        if len(self.generated_token_ids) >= self.sampling_params.max_new_tokens:
            return "length"
        return None

    def finish(self, reason: str, *, now_ns: int) -> None:
        if self.status is not RequestStatus.DECODING:
            raise RuntimeError("Only a decoding sequence can finish normally.")
        if reason not in {"stop", "length"}:
            raise ValueError("Normal finish reason must be 'stop' or 'length'.")
        self._validate_timestamp(now_ns)
        self.status = RequestStatus.FINISHED
        self.finish_reason = reason
        self.finish_ns = now_ns
        self.assert_invariants()

    def cancel(self, *, now_ns: int, reason: str = "cancelled") -> None:
        if self.status in {RequestStatus.FINISHED, RequestStatus.FAILED, RequestStatus.CANCELLED, RequestStatus.RELEASED}:
            raise RuntimeError("Terminal sequence cannot be cancelled again.")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("cancel reason must not be empty.")
        self._validate_timestamp(now_ns)
        self.status = RequestStatus.CANCELLED
        self.finish_reason = reason
        self.finish_ns = now_ns
        self.assert_invariants()

    def fail(self, message: str, *, now_ns: int) -> None:
        if self.status in {RequestStatus.FINISHED, RequestStatus.FAILED, RequestStatus.CANCELLED, RequestStatus.RELEASED}:
            raise RuntimeError("Terminal sequence cannot transition to FAILED.")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("failure message must not be empty.")
        self._validate_timestamp(now_ns)
        self.status = RequestStatus.FAILED
        self.finish_reason = "error"
        self.error_message = message
        self.finish_ns = now_ns
        self.assert_invariants()

    def release(self) -> None:
        if self.status not in {RequestStatus.FINISHED, RequestStatus.FAILED, RequestStatus.CANCELLED}:
            raise RuntimeError("Only a terminal sequence can be released.")
        self.status = RequestStatus.RELEASED
        self.assert_invariants()

    def assert_invariants(self) -> None:
        prompt_len = len(self.prompt_token_ids)
        if not self.request_id or prompt_len == 0:
            raise AssertionError("Sequence identity and prompt must be non-empty.")
        if not 0 <= self.prefill_cursor <= prompt_len:
            raise AssertionError("prefill_cursor is outside the prompt.")
        if not 0 <= self.computed_tokens <= self.total_tokens:
            raise AssertionError("computed_tokens is outside all_tokens.")
        if len(self.generated_token_ids) > self.sampling_params.max_new_tokens:
            raise AssertionError("generated tokens exceed max_new_tokens.")
        if self.generated_token_ids and self.prefill_cursor != prompt_len:
            raise AssertionError("Generation cannot start before the whole prompt is prefetched.")
        if self.status is RequestStatus.WAITING:
            if self.computed_tokens != self.prefill_cursor or self.generated_token_ids:
                raise AssertionError("WAITING state may only contain an attached prefix.")
        elif self.status is RequestStatus.PREFILLING:
            if self.generated_token_ids or self.computed_tokens != self.prefill_cursor:
                raise AssertionError("PREFILLING only advances prompt tokens.")
        elif self.status in {RequestStatus.DECODING, RequestStatus.FINISHED}:
            if not self.generated_token_ids or self.computed_tokens != self.total_tokens - 1:
                raise AssertionError("Decode/finish must retain one sampled, uncomputed token.")
        if self.status is RequestStatus.FINISHED and self.finish_reason not in {"stop", "length"}:
            raise AssertionError("FINISHED state requires a normal finish reason.")
        if self.status is RequestStatus.CANCELLED and not self.finish_reason:
            raise AssertionError("CANCELLED state requires a reason.")
        if self.status is RequestStatus.FAILED and (
            self.finish_reason != "error" or not self.error_message
        ):
            raise AssertionError("FAILED state requires a separate diagnostic message.")
        if self.status in {
            RequestStatus.FINISHED,
            RequestStatus.FAILED,
            RequestStatus.CANCELLED,
            RequestStatus.RELEASED,
        } and self.finish_ns is None:
            raise AssertionError("Terminal/released state requires finish_ns.")

    def _validate_timestamp(self, now_ns: int) -> None:
        if not isinstance(now_ns, int) or isinstance(now_ns, bool) or now_ns < 0:
            raise ValueError("now_ns must be a non-negative integer.")


@dataclass
class SequenceStore:
    """以 request_id 为唯一键保存 mutable state。"""

    _sequences: dict[str, SequenceState] = field(default_factory=dict)

    def add(self, sequence: SequenceState) -> None:
        if sequence.request_id in self._sequences:
            raise KeyError(f"Duplicate request_id: {sequence.request_id}")
        sequence.assert_invariants()
        self._sequences[sequence.request_id] = sequence

    def get(self, request_id: str) -> SequenceState:
        try:
            return self._sequences[request_id]
        except KeyError as exc:
            raise KeyError(f"Unknown request_id: {request_id}") from exc

    def remove(self, request_id: str) -> SequenceState:
        return self._sequences.pop(request_id)

    def replace(self, sequence: SequenceState) -> None:
        """原子替换一个已存在序列的 staged CPU 状态。"""

        if sequence.request_id not in self._sequences:
            raise KeyError(f"Unknown request_id: {sequence.request_id}")
        sequence.assert_invariants()
        self._sequences[sequence.request_id] = sequence

    def snapshots(self) -> tuple[SequenceView, ...]:
        return tuple(sequence.snapshot() for sequence in self._sequences.values())

    def snapshot_schedulable(self) -> tuple[SequenceView, ...]:
        """按 arrival/request ID 稳定排序的 immutable active views。"""

        return tuple(
            sequence.snapshot()
            for sequence in sorted(
                self.active(), key=lambda item: (item.arrival_ns, item.request_id)
            )
        )

    def contains(self, request_id: str) -> bool:
        return request_id in self._sequences

    def active(self) -> tuple[SequenceState, ...]:
        terminal = {RequestStatus.FINISHED, RequestStatus.FAILED, RequestStatus.CANCELLED, RequestStatus.RELEASED}
        return tuple(sequence for sequence in self._sequences.values() if sequence.status not in terminal)

    def terminal(self) -> tuple[SequenceState, ...]:
        statuses = {RequestStatus.FINISHED, RequestStatus.FAILED, RequestStatus.CANCELLED}
        return tuple(sequence for sequence in self._sequences.values() if sequence.status in statuses)
