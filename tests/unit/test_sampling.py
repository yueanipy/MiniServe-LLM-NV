"""Phase 10 sampling 参数、过滤顺序和边界的纯 CPU 测试。"""

from __future__ import annotations

import pytest
import torch

from miniserve.engine.request import SamplingParams
from miniserve.model.sampling import PerRequestSampler, _filter_top_k, sample_logits

pytestmark = pytest.mark.unit_cpu


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"temperature": float("nan")}, "temperature"),
        ({"temperature": True}, "temperature"),
        ({"temperature": "1"}, "temperature"),
        ({"temperature": 2.1}, "temperature"),
        ({"top_p": 0.0}, "top_p"),
        ({"top_p": True}, "top_p"),
        ({"top_k": -1}, "top_k"),
        ({"top_k": True}, "top_k"),
        ({"top_k": 1.5}, "top_k"),
        ({"max_new_tokens": True}, "max_new_tokens"),
        ({"max_new_tokens": 1.5}, "max_new_tokens"),
        ({"stop_token_ids": [1]}, "stop_token_ids"),
        ({"seed": -1}, "seed"),
        ({"seed": True}, "seed"),
    ],
)
def test_sampling_params_reject_invalid_boundaries(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        SamplingParams(**kwargs)


def test_greedy_is_argmax_and_does_not_require_generator() -> None:
    logits = torch.tensor([[0.1, 4.0, 3.0]])
    actual = sample_logits(logits, SamplingParams(temperature=0.0))
    assert actual.tolist() == [1]


def test_top_k_then_top_p_keeps_only_the_expected_candidate() -> None:
    logits = torch.tensor([[5.0, 4.0, 3.0, 2.0]])
    generator = torch.Generator().manual_seed(7)
    params = SamplingParams(temperature=1.0, top_k=2, top_p=0.6, seed=7)
    samples = [int(sample_logits(logits, params, generator=generator).item()) for _ in range(20)]
    assert samples == [0] * 20


def test_top_k_cannot_exceed_vocabulary() -> None:
    with pytest.raises(ValueError, match="vocabulary"):
        sample_logits(
            torch.zeros((1, 3)),
            SamplingParams(temperature=1.0, top_k=4),
            generator=torch.Generator().manual_seed(1),
        )


def test_sampler_release_is_idempotent() -> None:
    sampler = PerRequestSampler()
    params = SamplingParams(temperature=1.0, seed=3)
    sampler.sample("r", torch.tensor([[1.0, 1.0]]), params)
    assert sampler.active_request_ids == ("r",)
    sampler.release("r")
    sampler.release("r")
    assert sampler.active_request_ids == ()


def test_staged_sampling_does_not_advance_rng_until_commit() -> None:
    sampler = PerRequestSampler()
    logits = torch.tensor([[0.1, 1.0, 2.0, 3.0]])
    params = SamplingParams(temperature=1.0, seed=17)
    first = sampler.stage("staged", logits, params)
    second = sampler.stage("staged", logits, params)
    assert first.token_id == second.token_id
    assert sampler.active_request_ids == ()

    reference = torch.Generator().manual_seed(17)
    expected_first = int(sample_logits(logits, params, generator=reference).item())
    expected_second = int(sample_logits(logits, params, generator=reference).item())
    assert first.token_id == expected_first
    sampler.commit((first,))
    assert sampler.active_request_ids == ("staged",)
    next_token = sampler.stage("staged", logits, params).token_id
    assert next_token == expected_second


@pytest.mark.parametrize(
    "logits",
    [
        torch.tensor([[0.0, float("nan")]]),
        torch.tensor([[0.0, float("inf")]]),
        torch.tensor([[0.0, float("-inf")]]),
    ],
)
@pytest.mark.parametrize("temperature", [0.0, 0.7])
def test_nonfinite_logits_fail_instead_of_silently_selecting_a_token(logits, temperature) -> None:
    generator = None if temperature == 0 else torch.Generator().manual_seed(1)
    with pytest.raises(ValueError, match="finite"):
        sample_logits(logits, SamplingParams(temperature=temperature), generator=generator)


def test_trusted_serving_path_can_skip_the_syncing_finite_scan(monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        pytest.fail("torch.isfinite must not run when validation is explicitly disabled")

    monkeypatch.setattr(torch, "isfinite", fail_if_called)
    actual = sample_logits(
        torch.tensor([[0.0, 2.0]]),
        SamplingParams(temperature=0.0),
        validate_finite=False,
    )
    assert actual.tolist() == [1]


def test_integer_logits_are_rejected_at_the_model_executor_boundary() -> None:
    with pytest.raises(TypeError, match="floating-point"):
        sample_logits(torch.tensor([[1, 2, 3]]), SamplingParams())


def test_top_k_keeps_exactly_k_candidates_even_when_logits_are_tied() -> None:
    filtered = _filter_top_k(torch.tensor([[5.0, 5.0, 5.0, 1.0]]), 2)
    assert int(torch.isfinite(filtered).sum().item()) == 2


def test_sampling_supports_leading_batch_dimensions() -> None:
    logits = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    actual = sample_logits(logits, SamplingParams(temperature=0.0))
    assert actual.shape == (2, 3)
    assert actual.tolist() == [[3, 3, 3], [3, 3, 3]]
