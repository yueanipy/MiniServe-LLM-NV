"""Public prefill defaults and early whole-prompt admission checks."""

from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace

import pytest

from miniserve.api.bootstrap import _resolve_max_num_batched_tokens
from miniserve.api.server import build_cli_parser
from miniserve.engine.core import EngineCore
from miniserve.engine.request import EngineRequest, SamplingParams
from miniserve.engine.state import SequenceStore

pytestmark = pytest.mark.unit_cpu


def test_public_cli_defaults_to_whole_prefill() -> None:
    args = build_cli_parser().parse_args([])
    assert args.enable_chunked_prefill is False
    assert args.max_num_batched_tokens is None


def test_public_cli_accepts_explicit_chunk_and_budget() -> None:
    args = build_cli_parser().parse_args(
        ["--chunked-prefill", "--max-num-batched-tokens", "32"]
    )
    assert args.enable_chunked_prefill is True
    assert args.max_num_batched_tokens == 32


def test_public_cli_keeps_legacy_no_chunk_switch() -> None:
    args = build_cli_parser().parse_args(["--no-chunked-prefill"])
    assert args.enable_chunked_prefill is False


def test_public_cli_rejects_conflicting_chunk_switches() -> None:
    with pytest.raises(SystemExit):
        build_cli_parser().parse_args(
            ["--chunked-prefill", "--no-chunked-prefill"]
        )


def test_unspecified_budget_resolves_to_context_limit() -> None:
    assert _resolve_max_num_batched_tokens(64, None) == 64
    assert _resolve_max_num_batched_tokens(64, 16) == 16


def test_unsplittable_prompt_rejects_before_state_or_kv_registration() -> None:
    class _FakeKVManager:
        def __init__(self) -> None:
            self.backend = object()
            self.register_calls = 0

        def register_sequence(self, request_id: str) -> None:
            self.register_calls += 1

    core = object.__new__(EngineCore)
    core._continuous = True
    core.config = SimpleNamespace(
        engine=SimpleNamespace(
            attention_backend="cuda_paged",
            enable_chunked_prefill=False,
            max_num_batched_tokens=4,
            max_pending_requests=4,
        ),
        model=SimpleNamespace(max_model_len=64, vocab_size=1000),
    )
    core.store = SequenceStore()
    core.kv_manager = _FakeKVManager()
    core._finalized_results = OrderedDict()
    request = EngineRequest(
        "whole-prefill-reject",
        (1, 2, 3, 4, 5),
        SamplingParams(max_new_tokens=1),
        0,
    )

    with pytest.raises(ValueError, match="CHUNKED_PREFILL_DISABLED"):
        core.submit(request)

    assert not core.store.contains(request.request_id)
    assert core.kv_manager.register_calls == 0


@pytest.mark.parametrize("stream", [False, True])
def test_public_budget_rejection_releases_real_api_worker_control_chain(stream) -> None:
    from fastapi.testclient import TestClient

    from miniserve.api.runtime import ServiceRuntime
    from miniserve.api.server import create_app
    from miniserve.engine.async_engine import AsyncEngine
    from miniserve.engine.events import ResultPublisher
    from miniserve.engine.worker import EngineWorker

    class Tokenizer:
        eos_token_id = None

        def apply_chat_template(self, messages, **kwargs):
            return [1, 2, 3, 4, 5]

        def decode(self, tokens, **kwargs):
            return ""

    class Manager:
        backend = object()
        register_calls = 0

        def register_sequence(self, request_id):
            self.register_calls += 1

    core = object.__new__(EngineCore)
    core._continuous = True
    core.config = SimpleNamespace(
        engine=SimpleNamespace(
            attention_backend="cuda_paged",
            enable_chunked_prefill=False,
            enable_prefix_cache=False,
            max_num_batched_tokens=4,
            max_pending_requests=4,
        ),
        model=SimpleNamespace(max_model_len=64, vocab_size=1000),
    )
    core.store = SequenceStore()
    core.kv_manager = Manager()
    core._finalized_results = OrderedDict()
    core.publisher = ResultPublisher()
    executed = []

    def no_forward():
        assert not core.store.active(), "invalid admission reached execution"
        return ()

    core.step_multi = no_forward
    core.executor = SimpleNamespace(execute=lambda: executed.append(True))
    worker = EngineWorker(core, max_commands=4)
    runtime = ServiceRuntime(
        AsyncEngine(worker), Tokenizer(), "budget-test", 64, 4,
        vocab_size=1000, whole_prefill_token_budget=4,
    )
    with TestClient(create_app(runtime)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "budget-test",
                "messages": [{"role": "user", "content": "too long"}],
                "max_tokens": 1,
                "temperature": 0,
                "stream": stream,
            },
        )
        assert runtime.pending_requests == 0
        assert runtime.bridge.active_request_ids == ()
        assert core.store.active() == ()
        assert core.kv_manager.register_calls == 0
        assert executed == []
        assert worker.fatal_error is None
        assert client.get("/health/ready").status_code == 200
        assert response.status_code == 400, response.text
        assert "CHUNKED_PREFILL_DISABLED" in response.json()["error"]["message"]
        assert runtime.metrics.summary()["requests_started_total"] == 0
    assert not worker.is_alive


@pytest.mark.parametrize("budget", [0, -1, True, 65])
def test_public_runtime_rejects_invalid_whole_prefill_budget(budget) -> None:
    from miniserve.api.runtime import ServiceRuntime

    with pytest.raises(ValueError, match="whole_prefill_token_budget"):
        ServiceRuntime(object(), object(), "budget-test", 64, 4,
                       whole_prefill_token_budget=budget)
