"""Engine 与具体 attention/KV 实现之间的最小协议。"""

from __future__ import annotations

from typing import Protocol

import torch

from miniserve.kv.base import KVSequenceExecutionMetadata
from miniserve.model.qwen3 import Qwen3ForCausalLM


class AttentionBackend(Protocol):
    """Phase 1 只有 recompute；后续阶段在此增加 KV 上下文而不改 EngineCore。"""

    name: str

    def forward_recompute(
        self,
        model: Qwen3ForCausalLM,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor: ...

    def forward_with_kv(
        self,
        model: Qwen3ForCausalLM,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        *,
        request_id: str,
        context_len_before: int,
        mode: str,
        block_tables: torch.Tensor | None,
        seq_lens: torch.Tensor,
        slot_mapping: torch.Tensor | None,
        host_input_token_ids: tuple[int, ...] | None,
        kv_metadata: KVSequenceExecutionMetadata | None,
    ) -> torch.Tensor: ...

    def forward_flat_decode(
        self,
        model: Qwen3ForCausalLM,
        batch,
        *,
        metadata: tuple[KVSequenceExecutionMetadata, ...],
        host_input_token_ids: tuple[int, ...],
    ) -> torch.Tensor: ...
