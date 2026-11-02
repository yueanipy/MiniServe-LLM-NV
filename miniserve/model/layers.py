"""Qwen3/Llama dense decoder 共用的 RMSNorm 与 bias-free gated MLP。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def rmsnorm_torch(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Phase 7 永久保留的 PyTorch 数值参考实现。

    平方和在 FP32 中计算，避免 FP16/BF16 直接累加溢出或丢失精度。最后先把
    normalized 转回输入 dtype，再按 Qwen3/Hugging Face 4.57.1 的顺序乘权重。
    该函数不调用自定义算子，因此 CUDA kernel 出错时仍有独立 oracle。
    """

    if hidden_states.ndim < 1 or hidden_states.shape[-1] != weight.numel():
        raise ValueError("RMSNorm weight length must equal the input hidden dimension.")
    if weight.ndim != 1:
        raise ValueError("RMSNorm weight must be one-dimensional.")
    if hidden_states.device != weight.device or hidden_states.dtype != weight.dtype:
        raise ValueError("RMSNorm input and weight must share device and dtype.")
    if eps <= 0.0:
        raise ValueError("RMSNorm eps must be positive.")
    input_dtype = hidden_states.dtype
    values = hidden_states.to(torch.float32)
    variance = values.pow(2).mean(dim=-1, keepdim=True)
    normalized = values * torch.rsqrt(variance + eps)
    return weight * normalized.to(input_dtype)


class RMSNorm(nn.Module):
    """使用 FP32 累加器的 Qwen3 RMSNorm。

    平方与均值若直接在 FP16 中计算容易溢出或丢精度，因此先升到 FP32，
    归一化后再回到输入 dtype。权重随后按 PyTorch 的类型提升规则相乘。
    """

    def __init__(self, hidden_size: int, eps: float, *, device=None, dtype=None, backend: str = "torch") -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.variance_epsilon = eps
        self.backend = backend

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.backend == "cuda":
            if not hidden_states.is_cuda:
                raise RuntimeError("CUDA RMSNorm backend requires CUDA hidden states.")
            from miniserve._native import rmsnorm

            return rmsnorm(hidden_states.contiguous(), self.weight.contiguous(), self.variance_epsilon)
        if self.backend != "torch":
            raise RuntimeError(f"Unknown RMSNorm backend: {self.backend}")
        return rmsnorm_torch(hidden_states, self.weight, self.variance_epsilon)

    def set_backend(self, backend: str) -> None:
        if backend not in {"torch", "cuda"}:
            raise ValueError("RMSNorm backend must be torch or cuda.")
        if backend == "cuda":
            from miniserve._native import load_native_ops

            load_native_ops()
        self.backend = backend


class Qwen3MLP(nn.Module):
    """``down(silu(gate(x)) * up(x))``，三个投影均无 bias。"""

    def __init__(self, config, *, device=None, dtype=None) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False, device=device, dtype=dtype
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False, device=device, dtype=dtype
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False, device=device, dtype=dtype
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
