"""Phase 10 对外请求 schema 的支持/拒绝矩阵。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from miniserve.api.schemas import ChatCompletionRequest

pytestmark = pytest.mark.unit_cpu


def _body(**changes):
    value = {"model": "Qwen/Qwen3-0.6B", "messages": [{"role": "user", "content": "你好"}]}
    value.update(changes)
    return value


def test_schema_accepts_the_frozen_generation_fields() -> None:
    request = ChatCompletionRequest.model_validate(
        _body(
            max_tokens=2,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            seed=7,
            ignore_eos=True,
            stream=True,
        )
    )
    assert request.max_tokens == 2
    assert request.seed == 7
    assert request.stream is True
    assert request.ignore_eos is True


def test_schema_omits_model_specific_defaults_until_runtime_resolution() -> None:
    request = ChatCompletionRequest.model_validate(_body())
    assert (request.temperature, request.top_k, request.top_p) == (None, None, None)


@pytest.mark.parametrize(
    "changes",
    [
        {"messages": []},
        {"messages": [{"role": "tool", "content": "x"}]},
        {"max_tokens": 0},
        {"temperature": -0.1},
        {"top_p": 0},
        {"top_k": -1},
        {"seed": -1},
        {"stop": "END"},
        {"stop": ["END"]},
        {"frequency_penalty": 1.0},
    ],
)
def test_schema_rejects_invalid_or_unimplemented_fields(changes) -> None:
    with pytest.raises(ValidationError):
        ChatCompletionRequest.model_validate(_body(**changes))
