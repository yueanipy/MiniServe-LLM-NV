"""请求、状态机、token 边界和 ExecutionPlan 的纯 CPU 测试。"""

from __future__ import annotations

import pytest

from miniserve.engine.plan import DecodeItem, ExecutionPlan, PrefillItem
from miniserve.engine.request import EngineRequest, SamplingParams
from miniserve.engine.state import RequestStatus, SequenceState, SequenceStore

pytestmark = pytest.mark.unit_cpu


def _request(max_new_tokens: int = 2) -> EngineRequest:
    return EngineRequest(
        request_id="request-1",
        prompt_token_ids=(10, 20, 30),
        sampling_params=SamplingParams(max_new_tokens=max_new_tokens, stop_token_ids=(99,)),
        arrival_ns=100,
    )


def test_request_rejects_empty_prompt_and_phase1_validator_rejects_non_greedy() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        EngineRequest("bad", (), SamplingParams(), 0)
    request = EngineRequest("sampling", (1,), SamplingParams(temperature=0.7), 0)
    with pytest.raises(ValueError, match="greedy"):
        request.sampling_params.validate_phase1()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("request_id", 1),
        ("request_id", "   "),
        ("prompt_token_ids", [1]),
        ("sampling_params", object()),
        ("arrival_ns", True),
        ("arrival_ns", 1.5),
    ],
)
def test_request_rejects_coercible_contract_values(field, value) -> None:
    values = {
        "request_id": "strict",
        "prompt_token_ids": (1,),
        "sampling_params": SamplingParams(),
        "arrival_ns": 1,
    }
    values[field] = value
    with pytest.raises(ValueError):
        EngineRequest(**values)


def test_prefill_and_decode_preserve_pending_token_invariant() -> None:
    state = SequenceState.from_request(_request())
    state.begin_prefill()
    state.mark_prompt_computed()
    state.append_sampled_token(42, now_ns=200)

    assert state.status is RequestStatus.DECODING
    assert state.all_tokens == (10, 20, 30, 42)
    assert state.computed_tokens == 3
    assert state.total_tokens == state.computed_tokens + 1
    assert state.snapshot().num_generated == 1

    state.mark_decode_input_computed()
    state.append_sampled_token(43, now_ns=300)
    assert state.should_finish() == "length"
    state.finish("length", now_ns=301)
    assert state.status is RequestStatus.FINISHED
    assert state.computed_tokens == state.total_tokens - 1


def test_stop_token_finishes_without_forwarding_stop_token() -> None:
    state = SequenceState.from_request(_request(max_new_tokens=4))
    state.begin_prefill()
    state.mark_prompt_computed()
    state.append_sampled_token(99, now_ns=200)
    assert state.should_finish() == "stop"
    state.finish("stop", now_ns=201)
    assert state.computed_tokens == 3
    assert state.total_tokens == 4


def test_illegal_decode_progress_and_duplicate_store_id_fail_loudly() -> None:
    state = SequenceState.from_request(_request())
    with pytest.raises(RuntimeError, match="pending sampled token"):
        state.mark_decode_input_computed()
    store = SequenceStore()
    store.add(state)
    with pytest.raises(KeyError, match="Duplicate"):
        store.add(SequenceState.from_request(_request()))


def test_partial_prefill_advances_cursor_without_sampling() -> None:
    state = SequenceState.from_request(_request())
    state.begin_prefill()
    state.commit_prefill(2)
    assert state.status is RequestStatus.PREFILLING
    assert state.prefill_cursor == state.computed_tokens == 2
    assert state.generated_token_ids == []
    state.commit_prefill(3)
    state.append_sampled_token(42, now_ns=200)
    assert state.status is RequestStatus.DECODING


def test_execution_plan_recomputes_scheduled_token_count() -> None:
    prefill = ExecutionPlan(
        step_id=0,
        prefill_items=(PrefillItem("request-1", 0, 3),),
        scheduled_tokens=3,
    )
    assert not prefill.empty

    decode = ExecutionPlan(
        step_id=1,
        decode_items=(DecodeItem("request-1", input_token_id=42, position=3),),
        scheduled_tokens=1,
    )
    assert decode.decode_items[0].position == 3

    with pytest.raises(ValueError, match="does not match"):
        ExecutionPlan(step_id=2, prefill_items=(PrefillItem("r", 0, 2),), scheduled_tokens=1)


def test_execution_plan_rejects_mutable_duplicate_and_coercible_metadata() -> None:
    with pytest.raises(TypeError, match="immutable tuples"):
        ExecutionPlan(0, [PrefillItem("r", 0, 1)], (), 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="only once"):
        ExecutionPlan(
            0,
            (PrefillItem("r", 0, 1),),
            (DecodeItem("r", 1, 1),),
            2,
        )
    for invalid in (True, 1.5):
        with pytest.raises(ValueError):
            DecodeItem("r", invalid, 0)  # type: ignore[arg-type]


def test_failed_state_keeps_diagnostic_message_separate_from_finish_category() -> None:
    state = SequenceState.from_request(_request())
    state.fail("CUDA kernel rejected head_dim=64", now_ns=200)
    assert state.status is RequestStatus.FAILED
    assert state.finish_reason == "error"
    assert state.error_message == "CUDA kernel rejected head_dim=64"


def test_invalid_terminal_inputs_do_not_partially_mutate_state() -> None:
    state = SequenceState.from_request(_request())
    with pytest.raises(ValueError, match="cancel reason"):
        state.cancel(now_ns=200, reason="   ")
    assert state.status is RequestStatus.WAITING
    assert state.finish_reason is state.finish_ns is None

    with pytest.raises(ValueError, match="failure message"):
        state.fail("", now_ns=200)
    assert state.status is RequestStatus.WAITING

    state.fail("first failure", now_ns=200)
    with pytest.raises(RuntimeError, match="Terminal"):
        state.fail("second failure", now_ns=201)
    assert state.error_message == "first failure"


def test_invalid_token_finish_reason_and_timestamp_fail_before_mutation() -> None:
    state = SequenceState.from_request(_request())
    state.begin_prefill()
    state.mark_prompt_computed()
    for invalid in (True, 1.5, -1):
        with pytest.raises(ValueError, match="token_id"):
            state.append_sampled_token(invalid, now_ns=200)  # type: ignore[arg-type]
    assert state.generated_token_ids == []

    state.append_sampled_token(42, now_ns=200)
    with pytest.raises(ValueError, match="finish reason"):
        state.finish("unknown", now_ns=201)
    assert state.status is RequestStatus.DECODING
    with pytest.raises(ValueError, match="now_ns"):
        state.finish("length", now_ns=-1)
    assert state.status is RequestStatus.DECODING
