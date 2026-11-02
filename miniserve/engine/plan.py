"""Scheduler 到 BatchBuilder 之间不含 Tensor 的纯描述计划。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrefillItem:
    request_id: str
    token_start: int
    token_end: int

    def __post_init__(self) -> None:
        if (
            not self.request_id
            or not isinstance(self.token_start, int)
            or isinstance(self.token_start, bool)
            or not isinstance(self.token_end, int)
            or isinstance(self.token_end, bool)
            or self.token_start < 0
            or self.token_end <= self.token_start
        ):
            raise ValueError("Invalid prefill item.")


@dataclass(frozen=True)
class DecodeItem:
    request_id: str
    input_token_id: int
    position: int

    def __post_init__(self) -> None:
        if (
            not self.request_id
            or not isinstance(self.input_token_id, int)
            or isinstance(self.input_token_id, bool)
            or not isinstance(self.position, int)
            or isinstance(self.position, bool)
            or self.input_token_id < 0
            or self.position < 0
        ):
            raise ValueError("Invalid decode item.")


@dataclass(frozen=True)
class ExecutionPlan:
    step_id: int
    prefill_items: tuple[PrefillItem, ...] = ()
    decode_items: tuple[DecodeItem, ...] = ()
    scheduled_tokens: int = 0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.step_id, int)
            or isinstance(self.step_id, bool)
            or not isinstance(self.scheduled_tokens, int)
            or isinstance(self.scheduled_tokens, bool)
            or self.step_id < 0
            or self.scheduled_tokens < 0
        ):
            raise ValueError("step_id and scheduled_tokens must be non-negative.")
        if not isinstance(self.prefill_items, tuple) or not isinstance(self.decode_items, tuple):
            raise TypeError("ExecutionPlan items must be immutable tuples.")
        if any(not isinstance(item, PrefillItem) for item in self.prefill_items) or any(
            not isinstance(item, DecodeItem) for item in self.decode_items
        ):
            raise TypeError("ExecutionPlan contains an invalid item type.")
        request_ids = tuple(
            item.request_id for item in (*self.prefill_items, *self.decode_items)
        )
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("A request may appear only once in one ExecutionPlan.")
        expected = sum(item.token_end - item.token_start for item in self.prefill_items) + len(self.decode_items)
        if self.scheduled_tokens != expected:
            raise ValueError("scheduled_tokens does not match plan items.")

    @property
    def empty(self) -> bool:
        return not self.prefill_items and not self.decode_items

    @classmethod
    def empty_plan(cls, step_id: int) -> "ExecutionPlan":
        return cls(step_id=step_id)
