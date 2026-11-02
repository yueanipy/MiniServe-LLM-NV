"""PyTorch eager full-history attention reference backend。"""

from __future__ import annotations

import torch

from miniserve.kv.base import KVCacheBackend
from miniserve.kv.base import KVSequenceExecutionMetadata
from miniserve.model.qwen3 import Qwen3ForCausalLM


def build_absolute_causal_mask(
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """允许条件唯一为 ``key_abs <= query_abs``，返回 additive mask。"""

    if query_positions.ndim != 2 or query_positions.dtype != torch.long:
        raise ValueError("query_positions must be int64 [batch, query].")
    if key_positions.ndim == 1:
        keys = key_positions.view(1, 1, 1, -1)
    elif key_positions.ndim == 2 and key_positions.shape[0] == query_positions.shape[0]:
        keys = key_positions[:, None, None, :]
    else:
        raise ValueError("key_positions must be int64 [key] or [batch, key].")
    if key_positions.dtype != torch.long or key_positions.device != query_positions.device:
        raise ValueError("query/key positions must be int64 on the same device.")
    allowed = keys <= query_positions[:, None, :, None]
    zero = torch.zeros((), dtype=dtype, device=query_positions.device)
    blocked = torch.full((), torch.finfo(dtype).min, dtype=dtype, device=query_positions.device)
    return torch.where(allowed, zero, blocked)


class TorchContiguousAttention:
    """名称为最终 backend 名；Phase 1 方法明确标注没有 KV cache。"""

    name = "torch_contiguous"

    def __init__(self, kv_backend: KVCacheBackend | None = None) -> None:
        self.kv_backend = kv_backend

    def forward_recompute(
        self,
        model: Qwen3ForCausalLM,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        logits, _ = model.forward_recompute(input_ids, positions)
        return logits

    def forward_with_kv(
        self,
        model: Qwen3ForCausalLM,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        *,
        request_id: str,
        context_len_before: int,
        mode: str = "decode",
        block_tables: torch.Tensor | None = None,
        seq_lens: torch.Tensor | None = None,
        slot_mapping: torch.Tensor | None = None,
        host_input_token_ids: tuple[int, ...] | None = None,
        kv_metadata: KVSequenceExecutionMetadata | None = None,
    ) -> torch.Tensor:
        if self.kv_backend is None:
            raise RuntimeError("TorchContiguousAttention has no configured KV backend.")
        logits = model.forward_with_kv(
            input_ids,
            positions,
            kv_backend=self.kv_backend,
            request_id=request_id,
            context_len_before=context_len_before,
        )
        return logits
