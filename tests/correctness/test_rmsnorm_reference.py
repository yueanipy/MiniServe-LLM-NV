"""Phase 7 的独立 Torch RMSNorm oracle；不加载自定义 CUDA 扩展。"""

from __future__ import annotations

import pytest
import torch

from miniserve.model.layers import RMSNorm, rmsnorm_torch

pytestmark = [pytest.mark.correctness, pytest.mark.unit_cpu]


@pytest.mark.parametrize("shape", [(1, 8), (7, 16), (2, 3, 32)])
def test_rmsnorm_torch_matches_direct_fp32_accumulator_formula(shape: tuple[int, ...]) -> None:
    generator = torch.Generator().manual_seed(sum(shape) + 17)
    values = torch.randn(shape, generator=generator, dtype=torch.float32)
    weight = torch.linspace(0.5, 1.5, shape[-1], dtype=torch.float32)
    expected = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + 1e-6) * weight

    torch.testing.assert_close(rmsnorm_torch(values, weight, 1e-6), expected, atol=0.0, rtol=0.0)


def test_rmsnorm_module_torch_backend_uses_frozen_reference() -> None:
    module = RMSNorm(8, 1e-6, dtype=torch.float32)
    module.weight.data.copy_(torch.linspace(0.75, 1.25, 8))
    values = torch.tensor([[0.0, 1.0, -1.0, 1e-4, -1e-4, 8.0, -8.0, 2.0]])
    torch.testing.assert_close(
        module(values),
        rmsnorm_torch(values, module.weight, module.variance_epsilon),
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    ("values", "weight", "eps", "message"),
    [
        (torch.ones(2, 8), torch.ones(7), 1e-6, "hidden dimension"),
        (torch.ones(2, 8), torch.ones(1, 8), 1e-6, "one-dimensional"),
        (torch.ones(2, 8), torch.ones(8), 0.0, "positive"),
    ],
)
def test_rmsnorm_torch_rejects_invalid_contract(values, weight, eps: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        rmsnorm_torch(values, weight, eps)
