"""Phase 9 CUDA Graph bucket、静态 buffers 与 paged-decode capture/replay。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from miniserve.kv.base import KVSequenceExecutionMetadata

EXACT_BATCHED_GRAPH_BUCKETS = (1, 8, 16)


def validate_host_decode_metadata(
    model,
    kv_backend,
    host_input_token_ids: tuple[int, ...],
    metadata: KVSequenceExecutionMetadata,
    *,
    max_blocks_per_sequence: int,
    max_model_len: int | None,
) -> None:
    """校验由 KVManager 生成、随后复制到 CUDA tensor 的 CPU provenance。

    该函数只读不可变 tuple/int，不触发 CUDA 同步；它验证的字段覆盖 paged
    attention 与 KV append 中参与地址计算的 sequence/table/slot 以及 token 边界。
    """

    if not isinstance(host_input_token_ids, tuple) or len(host_input_token_ids) != 1:
        raise ValueError("Host decode metadata requires exactly one input token.")
    if not isinstance(metadata, KVSequenceExecutionMetadata):
        raise TypeError("Host decode metadata must come from KVCacheManager.")
    key_pool = kv_backend.key_pool
    value_pool = kv_backend.value_pool
    if key_pool.shape != value_pool.shape or key_pool.ndim != 5:
        raise ValueError("Full decode Graph requires matching 5-D paged K/V pools.")
    block_size = int(kv_backend.layout.block_size)
    if int(key_pool.shape[3]) != block_size:
        raise ValueError("Full decode Graph K/V pool block size does not match layout.")
    num_blocks = int(key_pool.shape[1])
    sequence_length = metadata.context_len_after_step
    if metadata.context_len_before_step + 1 != sequence_length:
        raise ValueError("Host decode metadata must advance exactly one token.")
    exact_max_length = (
        max_model_len
        if max_model_len is not None
        else max_blocks_per_sequence * block_size
    )
    if sequence_length <= 0 or sequence_length > exact_max_length:
        raise ValueError("Full decode Graph sequence length is outside static capacity.")
    input_token_id = host_input_token_ids[0]
    if not isinstance(input_token_id, int) or isinstance(input_token_id, bool) or input_token_id < 0:
        raise ValueError("Host decode input token must be a non-negative integer.")
    vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
    if vocab_size is not None and not 0 <= input_token_id < int(vocab_size):
        raise ValueError("Full decode Graph input token is outside the model vocabulary.")
    required_blocks = (sequence_length + block_size - 1) // block_size
    if required_blocks > max_blocks_per_sequence or len(metadata.block_table) < required_blocks:
        raise ValueError("Full decode Graph block table is shorter than sequence length.")
    active_tables = metadata.block_table[:required_blocks]
    if not active_tables or min(active_tables) < 0 or max(active_tables) >= num_blocks:
        raise ValueError("Full decode Graph block table contains an invalid physical block ID.")
    if len(metadata.slot_mapping) != 1:
        raise ValueError("Host decode metadata requires exactly one slot mapping.")
    slot = metadata.slot_mapping[0]
    if slot < 0 or slot >= num_blocks * block_size:
        raise ValueError("Full decode Graph slot mapping is outside the K/V pool.")
    position = metadata.context_len_before_step
    logical_block, offset = divmod(position, block_size)
    expected_slot = active_tables[logical_block] * block_size + offset
    if slot != expected_slot:
        raise ValueError("Full decode Graph slot mapping does not match block table/position.")


def select_bucket(actual_batch: int, buckets: tuple[int, ...]) -> int | None:
    """返回第一个能容纳 actual batch 的 bucket；无可用 bucket 时返回 None。"""

    if actual_batch <= 0:
        raise ValueError("actual_batch must be positive.")
    if tuple(sorted(set(buckets))) != buckets or any(item <= 0 for item in buckets):
        raise ValueError("CUDA Graph buckets must be positive, sorted, and unique.")
    return next((item for item in buckets if item >= actual_batch), None)


def graph_fallback_reason(actual_batch: int, buckets: tuple[int, ...]) -> str | None:
    bucket = select_bucket(actual_batch, buckets)
    return None if bucket is not None else f"batch={actual_batch} exceeds largest CUDA Graph bucket"


@dataclass(frozen=True)
class GraphKey:
    bucket: int
    dtype: str
    device: str
    model_revision: str
    backend_signature: str
    layer_index: int


@dataclass
class GraphBuffers:
    query: torch.Tensor
    block_tables: torch.Tensor
    sequence_lengths: torch.Tensor
    output: torch.Tensor | None = None


@dataclass(frozen=True)
class GraphStats:
    captures: int
    replays: int
    fallbacks: int
    entries: int
    last_fallback_reason: str | None


@dataclass
class _GraphEntry:
    key: GraphKey
    graph: torch.cuda.CUDAGraph
    buffers: GraphBuffers
    key_cache_data_ptr: int
    value_cache_data_ptr: int


class GraphManager:
    """为每层/bucket 捕获一个 graph-safe paged-decode operator。

    当前阶段捕获 attention operator，而不是整个 Qwen layer/model。这样仍能真实验证
    static buffers、capture/replay、bucket/fallback 和 stable K/V pointers，同时避免把
    尚未 tensor 化的 KV append slot/context host 状态偷偷固化进整模型 Graph。
    """

    def __init__(
        self,
        *,
        buckets: tuple[int, ...],
        max_blocks_per_sequence: int,
        model_revision: str,
        backend_signature: str = "cuda_paged_v1",
        validate_metadata_values: bool = True,
    ) -> None:
        if max_blocks_per_sequence <= 0:
            raise ValueError("max_blocks_per_sequence must be positive.")
        select_bucket(1, buckets)
        self.buckets = buckets
        self.max_blocks_per_sequence = max_blocks_per_sequence
        self.model_revision = model_revision
        self.backend_signature = backend_signature
        self.validate_metadata_values = validate_metadata_values
        self._entries: dict[GraphKey, _GraphEntry] = {}
        self._captures = 0
        self._replays = 0
        self._fallbacks = 0
        self._last_fallback_reason: str | None = None

    @property
    def stats(self) -> GraphStats:
        return GraphStats(
            self._captures,
            self._replays,
            self._fallbacks,
            len(self._entries),
            self._last_fallback_reason,
        )

    def clear(self) -> None:
        """释放 Python 对 Graph/buffer 的所有权；eager backend 不受影响。"""

        self._entries.clear()

    def fallback(self, reason: str) -> None:
        self._fallbacks += 1
        self._last_fallback_reason = reason

    def run_paged_decode(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        sequence_lengths: torch.Tensor,
        scale: float,
        *,
        layer_index: int,
    ) -> tuple[torch.Tensor | None, str | None]:
        if not query.is_cuda:
            return None, "CUDA Graph requires CUDA query tensors"
        actual_batch = query.shape[0]
        reason = graph_fallback_reason(actual_batch, self.buckets)
        if reason is not None:
            self.fallback(reason)
            return None, reason
        bucket = select_bucket(actual_batch, self.buckets)
        assert bucket is not None
        self._validate_metadata(
            query, key_cache, value_cache, block_tables, sequence_lengths
        )
        key = GraphKey(
            bucket,
            str(query.dtype),
            str(query.device),
            self.model_revision,
            self.backend_signature,
            layer_index,
        )
        entry = self._entries.get(key)
        if entry is None:
            entry = self._capture(
                key,
                query,
                key_cache,
                value_cache,
                block_tables,
                sequence_lengths,
                scale,
            )
            self._entries[key] = entry
            self._captures += 1
        elif (
            entry.key_cache_data_ptr != key_cache.data_ptr()
            or entry.value_cache_data_ptr != value_cache.data_ptr()
        ):
            raise RuntimeError("CUDA Graph K/V pool pointer changed after capture.")
        self._copy_inputs(entry.buffers, query, block_tables, sequence_lengths)
        entry.graph.replay()
        self._replays += 1
        assert entry.buffers.output is not None
        return entry.buffers.output[:actual_batch], None

    def _validate_metadata(self, query, key_cache, value_cache, tables, lengths) -> None:
        if query.ndim != 3 or query.shape[1:] != (16, 128):
            raise ValueError("Phase 9 Graph expects query [B,16,128].")
        if key_cache.shape != value_cache.shape or key_cache.ndim != 4:
            raise ValueError("Graph K/V cache shape mismatch.")
        if tuple(key_cache.shape[1:]) != (8, 16, 128):
            raise ValueError("Phase 9 Graph expects cache [blocks,8,16,128].")
        if tables.ndim != 2 or lengths.ndim != 1 or tables.shape[0] != query.shape[0]:
            raise ValueError("Graph metadata batch/rank mismatch.")
        if lengths.shape[0] != query.shape[0]:
            raise ValueError("Graph sequence_lengths batch mismatch.")
        if tables.shape[1] > self.max_blocks_per_sequence:
            raise ValueError("Graph block table exceeds configured static capacity.")
        if tables.dtype != torch.int32 or lengths.dtype != torch.int32:
            raise ValueError("Graph metadata must be int32.")
        if any(item.device != query.device for item in (key_cache, value_cache, tables, lengths)):
            raise ValueError("Graph tensors must share one CUDA device.")
        if query.dtype != key_cache.dtype or query.dtype != value_cache.dtype:
            raise ValueError("Graph query/cache dtype mismatch.")
        if self.validate_metadata_values:
            minimum_length = int(lengths.min().item())
            maximum_length = int(lengths.max().item())
            if minimum_length <= 0 or maximum_length > self.max_blocks_per_sequence * 16:
                raise ValueError("Graph sequence length is outside static capacity.")
            minimum_block = int(tables.min().item())
            maximum_block = int(tables.max().item())
            if minimum_block < 0 or maximum_block >= key_cache.shape[0]:
                raise ValueError("Graph block table contains an invalid physical block ID.")

    def _new_buffers(self, key: GraphKey, query: torch.Tensor) -> GraphBuffers:
        return GraphBuffers(
            query=torch.zeros(
                key.bucket,
                query.shape[1],
                query.shape[2],
                dtype=query.dtype,
                device=query.device,
            ),
            block_tables=torch.zeros(
                key.bucket,
                self.max_blocks_per_sequence,
                dtype=torch.int32,
                device=query.device,
            ),
            sequence_lengths=torch.ones(
                key.bucket, dtype=torch.int32, device=query.device
            ),
        )

    def _copy_inputs(self, buffers, query, tables, lengths) -> None:
        actual = query.shape[0]
        # Buffers 创建时已把 padding 初始化为 query=0/table=0/length=1。实际 rows
        # 只需三次 copy；未使用的 table columns 和 bucket padding 不会被 kernel 读取。
        buffers.query[:actual].copy_(query)
        buffers.block_tables[:actual, : tables.shape[1]].copy_(tables)
        buffers.sequence_lengths[:actual].copy_(lengths)

    def _capture(self, key, query, key_cache, value_cache, tables, lengths, scale):
        from miniserve._native import paged_decode_attention_graph

        buffers = self._new_buffers(key, query)
        self._copy_inputs(buffers, query, tables, lengths)
        # PyTorch 推荐在 side stream 上预热，避免 lazy allocator/module work 进入 capture。
        warmup_stream = torch.cuda.Stream(device=query.device)
        warmup_stream.wait_stream(torch.cuda.current_stream(query.device))
        with torch.cuda.stream(warmup_stream):
            for _ in range(2):
                paged_decode_attention_graph(
                    buffers.query,
                    key_cache,
                    value_cache,
                    buffers.block_tables,
                    buffers.sequence_lengths,
                    scale,
                )
        torch.cuda.current_stream(query.device).wait_stream(warmup_stream)
        torch.cuda.synchronize(query.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            buffers.output = paged_decode_attention_graph(
                buffers.query,
                key_cache,
                value_cache,
                buffers.block_tables,
                buffers.sequence_lengths,
                scale,
            )
        return _GraphEntry(
            key,
            graph,
            buffers,
            key_cache.data_ptr(),
            value_cache.data_ptr(),
        )


@dataclass
class FullDecodeGraphBuffers:
    input_ids: torch.Tensor
    positions: torch.Tensor
    block_tables: torch.Tensor
    sequence_lengths: torch.Tensor
    slot_mapping: torch.Tensor
    output_logits: torch.Tensor | None = None


@dataclass
class _FullDecodeGraphEntry:
    graph: torch.cuda.CUDAGraph
    buffers: FullDecodeGraphBuffers
    key_pool_data_ptr: int
    value_pool_data_ptr: int


class FullDecodeGraphManager:
    """捕获单请求完整 Qwen decode forward，dynamic state 全部进入 static buffers。"""

    def __init__(
        self,
        *,
        max_blocks_per_sequence: int,
        max_model_len: int | None = None,
    ) -> None:
        if max_blocks_per_sequence <= 0:
            raise ValueError("max_blocks_per_sequence must be positive.")
        if max_model_len is not None and (
            not isinstance(max_model_len, int)
            or isinstance(max_model_len, bool)
            or max_model_len <= 0
        ):
            raise ValueError("max_model_len must be a positive integer when provided.")
        self.max_blocks_per_sequence = max_blocks_per_sequence
        self.max_model_len = max_model_len
        self._entry: _FullDecodeGraphEntry | None = None
        self._captures = 0
        self._replays = 0
        self._fallbacks = 0
        self._last_fallback_reason: str | None = None
        self._host_metadata_validations = 0
        self._device_metadata_validations = 0

    @property
    def stats(self) -> GraphStats:
        return GraphStats(
            self._captures,
            self._replays,
            self._fallbacks,
            0 if self._entry is None else 1,
            self._last_fallback_reason,
        )

    def clear(self) -> None:
        self._entry = None

    @property
    def metadata_validation_counts(self) -> dict[str, int]:
        """区分 serving CPU provenance 与兼容调用的 device-readback 校验。"""

        return {
            "host": self._host_metadata_validations,
            "device_readback": self._device_metadata_validations,
        }

    def run(
        self,
        model,
        attention_backend,
        kv_backend,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        block_tables: torch.Tensor,
        sequence_lengths: torch.Tensor,
        slot_mapping: torch.Tensor,
        *,
        host_input_token_ids: tuple[int, ...] | None = None,
        host_metadata: KVSequenceExecutionMetadata | None = None,
    ) -> tuple[torch.Tensor | None, str | None]:
        reason = self._fallback_reason(
            input_ids, positions, block_tables, sequence_lengths, slot_mapping
        )
        if reason is not None:
            self._fallbacks += 1
            self._last_fallback_reason = reason
            return None, reason
        self._validate_metadata_values(
            model,
            kv_backend,
            input_ids,
            positions,
            block_tables,
            sequence_lengths,
            slot_mapping,
            host_input_token_ids=host_input_token_ids,
            host_metadata=host_metadata,
        )
        if self._entry is None:
            self._entry = self._capture(
                model,
                attention_backend,
                kv_backend,
                input_ids,
                positions,
                block_tables,
                sequence_lengths,
                slot_mapping,
            )
            self._captures += 1
        elif (
            self._entry.key_pool_data_ptr != kv_backend.key_pool.data_ptr()
            or self._entry.value_pool_data_ptr != kv_backend.value_pool.data_ptr()
        ):
            raise RuntimeError("Full decode Graph K/V pool pointer changed after capture.")
        self._copy_inputs(
            self._entry.buffers,
            input_ids,
            positions,
            block_tables,
            sequence_lengths,
            slot_mapping,
        )
        self._entry.graph.replay()
        self._replays += 1
        assert self._entry.buffers.output_logits is not None
        return self._entry.buffers.output_logits, None

    def _fallback_reason(self, input_ids, positions, tables, lengths, slots) -> str | None:
        if not input_ids.is_cuda:
            return "full decode Graph requires CUDA tensors"
        if input_ids.shape != (1, 1) or positions.shape != (1, 1):
            return "full decode Graph currently supports one request and one token"
        if tables.ndim != 2 or tables.shape[0] != 1 or tables.shape[1] > self.max_blocks_per_sequence:
            return "block table does not fit the full decode Graph buffer"
        if lengths.shape != (1,) or slots.shape != (1,):
            return "sequence length/slot mapping does not fit full decode Graph"
        if input_ids.dtype != torch.long or positions.dtype != torch.long:
            return "input_ids/positions must be int64"
        if tables.dtype != torch.int32 or lengths.dtype != torch.int32 or slots.dtype != torch.int64:
            return "full decode Graph metadata dtype mismatch"
        if any(item.device != input_ids.device for item in (positions, tables, lengths, slots)):
            return "full decode Graph tensors must share one device"
        return None

    def _validate_metadata_values(
        self,
        model,
        kv_backend,
        input_ids,
        positions,
        tables,
        lengths,
        slots,
        *,
        host_input_token_ids: tuple[int, ...] | None = None,
        host_metadata: KVSequenceExecutionMetadata | None = None,
    ) -> None:
        """在 capture/replay 外校验所有会参与 CUDA 地址计算的 metadata。

        这里每个 decode forward 只同步一次小型 metadata，而不是让每一层的
        graph-safe CUDA op 各自做 ``.item()``。非法 allocator metadata 会在任何
        kernel launch 前变成确定的 Python ``ValueError``。
        """

        key_pool = kv_backend.key_pool
        value_pool = kv_backend.value_pool
        if key_pool.shape != value_pool.shape or key_pool.ndim != 5:
            raise ValueError("Full decode Graph requires matching 5-D paged K/V pools.")
        block_size = int(kv_backend.layout.block_size)
        if int(key_pool.shape[3]) != block_size:
            raise ValueError("Full decode Graph K/V pool block size does not match layout.")
        if key_pool.device != tables.device or value_pool.device != tables.device:
            raise ValueError("Full decode Graph metadata and K/V pools must share one device.")
        num_blocks = int(key_pool.shape[1])
        if (host_input_token_ids is None) != (host_metadata is None):
            raise ValueError("Host token IDs and KV metadata must be provided together.")
        if host_input_token_ids is not None and host_metadata is not None:
            if tables.shape[1] != len(host_metadata.block_table):
                raise ValueError("CUDA block-table tensor shape disagrees with host metadata.")
            validate_host_decode_metadata(
                model,
                kv_backend,
                host_input_token_ids,
                host_metadata,
                max_blocks_per_sequence=self.max_blocks_per_sequence,
                max_model_len=self.max_model_len,
            )
            self._host_metadata_validations += 1
            return
        self._device_metadata_validations += 1
        # 四种 tiny metadata 先在设备端打包，再只做一次 D2H copy/synchronize。
        packed = torch.cat(
            (
                positions.reshape(-1).to(dtype=torch.long),
                lengths.reshape(-1).to(dtype=torch.long),
                slots.reshape(-1),
                input_ids.reshape(-1),
                tables.reshape(-1).to(dtype=torch.long),
            )
        ).detach().cpu()
        position = int(packed[0].item())
        sequence_length = int(packed[1].item())
        slot = int(packed[2].item())
        input_token_id = int(packed[3].item())
        table_values = packed[4:].reshape(tables.shape)
        exact_max_length = (
            self.max_model_len
            if self.max_model_len is not None
            else self.max_blocks_per_sequence * block_size
        )
        if sequence_length <= 0 or sequence_length > exact_max_length:
            raise ValueError("Full decode Graph sequence length is outside static capacity.")
        vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
        if vocab_size is not None and not 0 <= input_token_id < int(vocab_size):
            raise ValueError("Full decode Graph input token is outside the model vocabulary.")
        if position != sequence_length - 1:
            raise ValueError("Full decode Graph position must equal sequence_length - 1.")
        required_blocks = (sequence_length + block_size - 1) // block_size
        if tables.shape[1] < required_blocks:
            raise ValueError("Full decode Graph block table is shorter than sequence length.")
        active_tables = table_values[:, :required_blocks]
        minimum_block = int(active_tables.min().item())
        maximum_block = int(active_tables.max().item())
        if minimum_block < 0 or maximum_block >= num_blocks:
            raise ValueError("Full decode Graph block table contains an invalid physical block ID.")
        if slot < 0 or slot >= num_blocks * block_size:
            raise ValueError("Full decode Graph slot mapping is outside the K/V pool.")
        logical_block, offset = divmod(position, block_size)
        expected_slot = int(active_tables[0, logical_block].item()) * block_size + offset
        if slot != expected_slot:
            raise ValueError("Full decode Graph slot mapping does not match block table/position.")

    def _new_buffers(self, input_ids: torch.Tensor) -> FullDecodeGraphBuffers:
        device = input_ids.device
        return FullDecodeGraphBuffers(
            input_ids=torch.zeros((1, 1), dtype=torch.long, device=device),
            positions=torch.zeros((1, 1), dtype=torch.long, device=device),
            block_tables=torch.zeros(
                (1, self.max_blocks_per_sequence), dtype=torch.int32, device=device
            ),
            sequence_lengths=torch.ones((1,), dtype=torch.int32, device=device),
            slot_mapping=torch.zeros((1,), dtype=torch.int64, device=device),
        )

    def _copy_inputs(self, buffers, input_ids, positions, tables, lengths, slots) -> None:
        buffers.input_ids.copy_(input_ids)
        buffers.positions.copy_(positions)
        buffers.block_tables[:, : tables.shape[1]].copy_(tables)
        buffers.sequence_lengths.copy_(lengths)
        buffers.slot_mapping.copy_(slots)

    def _forward(self, model, attention_backend, kv_backend, buffers):
        return model.forward_with_kv(
            buffers.input_ids,
            buffers.positions,
            kv_backend=kv_backend,
            request_id="__full_decode_graph__",
            context_len_before=0,
            decode_attention=attention_backend,
            block_tables=buffers.block_tables,
            sequence_lengths=buffers.sequence_lengths,
            graph_slot_mapping=buffers.slot_mapping,
            graph_capture=True,
        )

    def _capture(
        self,
        model,
        attention_backend,
        kv_backend,
        input_ids,
        positions,
        tables,
        lengths,
        slots,
    ) -> _FullDecodeGraphEntry:
        buffers = self._new_buffers(input_ids)
        self._copy_inputs(buffers, input_ids, positions, tables, lengths, slots)
        warmup_stream = torch.cuda.Stream(device=input_ids.device)
        warmup_stream.wait_stream(torch.cuda.current_stream(input_ids.device))
        with torch.cuda.stream(warmup_stream), torch.inference_mode():
            for _ in range(2):
                self._forward(model, attention_backend, kv_backend, buffers)
        torch.cuda.current_stream(input_ids.device).wait_stream(warmup_stream)
        torch.cuda.synchronize(input_ids.device)
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph):
            buffers.output_logits = self._forward(
                model, attention_backend, kv_backend, buffers
            )
        return _FullDecodeGraphEntry(
            graph,
            buffers,
            kv_backend.key_pool.data_ptr(),
            kv_backend.value_pool.data_ptr(),
        )


@dataclass
class BatchedDecodeGraphBuffers:
    input_ids: torch.Tensor
    positions: torch.Tensor
    block_tables: torch.Tensor
    sequence_lengths: torch.Tensor
    slot_mapping: torch.Tensor
    output_logits: torch.Tensor | None = None


@dataclass
class _BatchedDecodeGraphEntry:
    graph: torch.cuda.CUDAGraph
    buffers: BatchedDecodeGraphBuffers
    key_pool_data_ptr: int
    value_pool_data_ptr: int


class BatchedDecodeGraphManager:
    """Capture exact-B full model decode graphs without padding.

    The manager is deliberately startup-only: it captures every supported bucket
    before request admission and never captures in the request hot path. The
    legacy single-request ``FullDecodeGraphManager`` remains separate.
    """

    def __init__(
        self,
        *,
        buckets: tuple[int, ...] = EXACT_BATCHED_GRAPH_BUCKETS,
        max_blocks_per_sequence: int,
        model_vocab_size: int,
    ) -> None:
        if not buckets or tuple(sorted(set(buckets))) != buckets:
            raise ValueError("Batched Graph buckets must be sorted and unique.")
        if any(bucket <= 0 for bucket in buckets):
            raise ValueError("Batched Graph buckets must be positive.")
        if max_blocks_per_sequence <= 0 or model_vocab_size <= 0:
            raise ValueError("Batched Graph dimensions must be positive.")
        self.buckets = buckets
        self.max_blocks_per_sequence = max_blocks_per_sequence
        self.model_vocab_size = model_vocab_size
        self._entries: dict[int, _BatchedDecodeGraphEntry] = {}
        self._captures = 0
        self._replays = 0
        self._fallbacks = 0
        self._last_fallback_reason: str | None = None

    @property
    def stats(self) -> dict[str, object]:
        return {
            "captures": self._captures,
            "replays": self._replays,
            "fallbacks": self._fallbacks,
            "entries": len(self._entries),
            "buckets": list(self.buckets),
            "last_fallback_reason": self._last_fallback_reason,
        }

    def clear(self) -> None:
        self._entries.clear()

    def prepare(self, model, attention_backend, kv_backend) -> None:
        """Capture all exact buckets while no user sequence owns KV blocks."""

        if getattr(kv_backend, "_sequences", None):
            raise RuntimeError("Batched Graph capture requires an empty KV backend.")
        if getattr(kv_backend, "_pending", None):
            raise RuntimeError("Batched Graph capture cannot run with pending KV steps.")
        allocator = getattr(kv_backend, "allocator", None)
        if allocator is None:
            raise RuntimeError("Batched Graph requires the paged KV allocator.")
        for bucket in self.buckets:
            scratch = allocator.allocate(bucket)
            try:
                buffers = self._new_buffers(bucket, kv_backend.layout.device)
                buffers.block_tables.zero_()
                scratch_device = torch.tensor(
                    scratch, dtype=torch.int32, device=kv_backend.layout.device
                )
                buffers.block_tables[:, 0].copy_(scratch_device)
                buffers.sequence_lengths.fill_(1)
                buffers.slot_mapping.copy_(
                    scratch_device.to(dtype=torch.int64)
                    * int(kv_backend.layout.block_size)
                )
                buffers.input_ids.zero_()
                buffers.positions.zero_()
                entry = self._capture(
                    model,
                    attention_backend,
                    kv_backend,
                    buffers,
                )
                self._entries[bucket] = entry
                self._captures += 1
            finally:
                allocator.release(scratch)
        torch.cuda.synchronize(kv_backend.layout.device)
        if allocator.free_count() != allocator.num_blocks:
            raise RuntimeError(
                "Batched Graph scratch blocks were not fully returned to the allocator."
            )

    def _new_buffers(
        self, bucket: int, device: torch.device | str
    ) -> BatchedDecodeGraphBuffers:
        return BatchedDecodeGraphBuffers(
            input_ids=torch.zeros((bucket, 1), dtype=torch.long, device=device),
            positions=torch.zeros((bucket, 1), dtype=torch.long, device=device),
            block_tables=torch.zeros(
                (bucket, self.max_blocks_per_sequence),
                dtype=torch.int32,
                device=device,
            ),
            sequence_lengths=torch.ones((bucket,), dtype=torch.int32, device=device),
            slot_mapping=torch.zeros((bucket,), dtype=torch.int64, device=device),
        )

    def _capture(
        self,
        model,
        attention_backend,
        kv_backend,
        buffers: BatchedDecodeGraphBuffers,
    ) -> _BatchedDecodeGraphEntry:
        device = buffers.input_ids.device
        warmup_stream = torch.cuda.Stream(device=device)
        warmup_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(warmup_stream), torch.inference_mode():
            for _ in range(2):
                self._forward(model, attention_backend, kv_backend, buffers)
        torch.cuda.current_stream(device).wait_stream(warmup_stream)
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph):
            buffers.output_logits = self._forward(
                model,
                attention_backend,
                kv_backend,
                buffers,
            )
        if buffers.output_logits is None:
            raise RuntimeError("Batched Graph capture produced no output logits.")
        return _BatchedDecodeGraphEntry(
            graph,
            buffers,
            kv_backend.key_pool.data_ptr(),
            kv_backend.value_pool.data_ptr(),
        )

    def _forward(
        self,
        model,
        attention_backend,
        kv_backend,
        buffers: BatchedDecodeGraphBuffers,
    ) -> torch.Tensor:
        return model.forward_with_kv_batch(
            buffers.input_ids,
            buffers.positions,
            kv_backend=kv_backend,
            decode_attention=attention_backend,
            block_tables=buffers.block_tables,
            sequence_lengths=buffers.sequence_lengths,
            slot_mapping=buffers.slot_mapping,
        )

    def run(self, batch, kv_backend) -> tuple[torch.Tensor | None, str | None]:
        bucket = int(batch.batch_size)
        if bucket not in self._entries:
            self._fallbacks += 1
            self._last_fallback_reason = (
                f"batch={bucket} is not an exact cached Graph bucket"
            )
            return None, self._last_fallback_reason
        entry = self._entries[bucket]
        if (
            entry.key_pool_data_ptr != kv_backend.key_pool.data_ptr()
            or entry.value_pool_data_ptr != kv_backend.value_pool.data_ptr()
        ):
            raise RuntimeError("Batched Graph K/V pool pointer changed after capture.")
        if batch.block_tables.shape[1] > self.max_blocks_per_sequence:
            raise RuntimeError("Batched Graph block table exceeds static capacity.")
        buffers = entry.buffers
        buffers.block_tables.zero_()
        buffers.block_tables[:, : batch.block_tables.shape[1]].copy_(
            batch.block_tables
        )
        buffers.input_ids.copy_(batch.input_ids.reshape(bucket, 1))
        buffers.positions.copy_(batch.positions.reshape(bucket, 1))
        buffers.sequence_lengths.copy_(batch.context_lens)
        buffers.slot_mapping.copy_(batch.slot_mapping)
        entry.graph.replay()
        self._replays += 1
        assert buffers.output_logits is not None
        return buffers.output_logits, None
