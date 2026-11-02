"""Phase 8：prefill 保持 Torch，只有 decode 调用自定义 paged CUDA op。"""

from __future__ import annotations

import torch

from miniserve.kv.base import KVSequenceExecutionMetadata
from miniserve.kv.paged import PagedKVCacheBackend
from miniserve.engine.flat_batch import FlatDecodeBatch, validate_flat_decode_provenance
from miniserve.model.qwen3 import Qwen3ForCausalLM

from .torch_paged_ref import TorchPagedReferenceAttention


class _PrevalidatedDecodeAttention:
    """只在同一 CPU provenance 已验证的 forward 内调用无值检查 CUDA op。"""

    def __init__(self, backend: "CudaPagedDecodeAttention") -> None:
        self._backend = backend

    def decode(self, query: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._backend.decode_graph_safe(query, **kwargs)


class CudaPagedDecodeAttention(TorchPagedReferenceAttention):
    name = "cuda_paged"

    def __init__(self, kv_backend: PagedKVCacheBackend) -> None:
        super().__init__(kv_backend)
        from miniserve._native import load_native_ops

        # 配置选择 CUDA 时在启动阶段失败；不等第一条请求再悄悄 fallback。
        load_native_ops()
        self.graph_manager = None
        self.batched_graph_manager = None
        self._prevalidated_eager_forwards = 0

    @property
    def prevalidated_eager_forwards(self) -> int:
        return self._prevalidated_eager_forwards

    def enable_cuda_graph(
        self,
        *,
        max_model_len: int,
    ) -> None:
        from miniserve.engine.graph import FullDecodeGraphManager

        max_blocks = (max_model_len + self.kv_backend.layout.block_size - 1) // self.kv_backend.layout.block_size
        self.graph_manager = FullDecodeGraphManager(
            max_blocks_per_sequence=max_blocks,
            max_model_len=max_model_len,
        )

    def enable_batched_graph(
        self,
        *,
        model,
        max_model_len: int,
    ) -> None:
        from miniserve.engine.graph import (
            EXACT_BATCHED_GRAPH_BUCKETS,
            BatchedDecodeGraphManager,
        )

        max_blocks = (max_model_len + self.kv_backend.layout.block_size - 1) // self.kv_backend.layout.block_size
        manager = BatchedDecodeGraphManager(
            buckets=EXACT_BATCHED_GRAPH_BUCKETS,
            max_blocks_per_sequence=max_blocks,
            model_vocab_size=model.config.vocab_size,
        )
        manager.prepare(model, self, self.kv_backend)
        self.batched_graph_manager = manager

    def decode(
        self,
        query: torch.Tensor,
        *,
        layer_index: int,
        block_tables: torch.Tensor,
        sequence_lengths: torch.Tensor,
    ) -> torch.Tensor:
        from miniserve._native import paged_decode_attention

        return paged_decode_attention(
            query,
            self.kv_backend.key_pool[layer_index],
            self.kv_backend.value_pool[layer_index],
            block_tables,
            sequence_lengths,
            query.shape[-1] ** -0.5,
        )

    def decode_graph_safe(
        self,
        query: torch.Tensor,
        *,
        layer_index: int,
        block_tables: torch.Tensor,
        sequence_lengths: torch.Tensor,
    ) -> torch.Tensor:
        from miniserve._native import paged_decode_attention_graph

        return paged_decode_attention_graph(
            query,
            self.kv_backend.key_pool[layer_index],
            self.kv_backend.value_pool[layer_index],
            block_tables,
            sequence_lengths,
            query.shape[-1] ** -0.5,
        )

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
        if mode not in {"prefill", "decode"}:
            raise ValueError(f"Unknown attention execution mode: {mode}")
        if mode == "prefill":
            # 即使 chunk size 恰好是 1，也必须保留带 causal mask 的 prefill 数学。
            return super().forward_with_kv(
                model,
                input_ids,
                positions,
                request_id=request_id,
                context_len_before=context_len_before,
                mode=mode,
                block_tables=block_tables,
                seq_lens=seq_lens,
                host_input_token_ids=host_input_token_ids,
                kv_metadata=kv_metadata,
            )
        if input_ids.shape != (1, 1) or block_tables is None or seq_lens is None:
            raise RuntimeError("CUDA paged decode requires one token and packed paged metadata.")
        if self.graph_manager is not None:
            if slot_mapping is None:
                raise RuntimeError("Full decode CUDA Graph requires slot_mapping.")
            output, reason = self.graph_manager.run(
                model,
                self,
                self.kv_backend,
                input_ids,
                positions,
                block_tables,
                seq_lens,
                slot_mapping,
                host_input_token_ids=host_input_token_ids,
                host_metadata=kv_metadata,
            )
            if output is not None:
                return output
            assert reason is not None
        decode_attention = self
        if host_input_token_ids is not None or kv_metadata is not None:
            if host_input_token_ids is None or kv_metadata is None:
                raise ValueError("Host token IDs and KV metadata must be provided together.")
            from miniserve.engine.graph import validate_host_decode_metadata

            validate_host_decode_metadata(
                model,
                self.kv_backend,
                host_input_token_ids,
                kv_metadata,
                max_blocks_per_sequence=block_tables.shape[1],
                max_model_len=int(model.config.max_model_len),
            )
            # serving eager fallback 已由 CPU provenance 完成一次值校验；各层只保留
            # C++ shape/dtype/device 检查，不再各做四次 .item() 同步。
            decode_attention = _PrevalidatedDecodeAttention(self)
            self._prevalidated_eager_forwards += 1
        return model.forward_with_kv(
            input_ids,
            positions,
            kv_backend=self.kv_backend,
            request_id=request_id,
            context_len_before=context_len_before,
            decode_attention=decode_attention,
            block_tables=block_tables,
            sequence_lengths=seq_lens,
        )

    def forward_flat_decode(
        self,
        model: Qwen3ForCausalLM,
        batch: FlatDecodeBatch,
        *,
        metadata: tuple[KVSequenceExecutionMetadata, ...],
        host_input_token_ids: tuple[int, ...],
    ) -> torch.Tensor:
        """执行一次真正的 B 行 paged decode；host provenance 已在本调用前校验。"""

        if self.graph_manager is not None:
            raise RuntimeError("Batched decode cannot use the legacy CUDA Graph path.")
        validate_flat_decode_provenance(
            batch,
            metadata,
            host_input_token_ids,
            physical_block_count=self.kv_backend.allocator.num_blocks,
            vocab_size=model.config.vocab_size,
            max_model_len=model.config.max_model_len,
        )
        if self.batched_graph_manager is not None:
            output, reason = self.batched_graph_manager.run(batch, self.kv_backend)
            if output is not None:
                return output
            assert reason is not None
        return model.forward_with_kv_batch(
            batch.input_ids.reshape(batch.batch_size, 1),
            batch.positions.reshape(batch.batch_size, 1),
            kv_backend=self.kv_backend,
            decode_attention=self,
            block_tables=batch.block_tables,
            sequence_lengths=batch.context_lens,
            slot_mapping=batch.slot_mapping,
        )
