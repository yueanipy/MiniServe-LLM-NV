"""Qwen3 与非 interleaved Llama 共用的显式绝对位置 RoPE。"""

from __future__ import annotations

import torch


def build_rope_cache(
    positions: torch.Tensor,
    *,
    head_dim: int,
    rope_theta: float,
    output_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """为绝对 ``positions [batch, seq]`` 计算 FP32 cos/sin 后转回模型 dtype。"""

    if positions.ndim != 2:
        raise ValueError("positions must have shape [batch, sequence].")
    if head_dim <= 0 or head_dim % 2:
        raise ValueError("head_dim must be a positive even number.")
    if rope_theta <= 0:
        raise ValueError("rope_theta must be positive.")

    frequencies = torch.arange(0, head_dim, 2, device=positions.device, dtype=torch.float32)
    inverse_frequency = 1.0 / (rope_theta ** (frequencies / head_dim))
    # [batch, seq, head_dim/2]；显式 FP32 与 HF 默认 RoPE 的 autocast-off 语义一致。
    angles = positions.to(torch.float32).unsqueeze(-1) * inverse_frequency.view(1, 1, -1)
    embedding = torch.cat((angles, angles), dim=-1)
    return embedding.cos().to(output_dtype), embedding.sin().to(output_dtype)


def rotate_half(values: torch.Tensor) -> torch.Tensor:
    first, second = values.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """把 ``[batch, seq, dim]`` cache 广播到 ``[batch, heads, seq, dim]``。"""

    if query.ndim != 4 or key.ndim != 4 or cos.ndim != 3 or sin.ndim != 3:
        raise ValueError("Unexpected RoPE tensor rank.")
    cosine = cos.unsqueeze(1)
    sine = sin.unsqueeze(1)
    return (
        query * cosine + rotate_half(query) * sine,
        key * cosine + rotate_half(key) * sine,
    )
