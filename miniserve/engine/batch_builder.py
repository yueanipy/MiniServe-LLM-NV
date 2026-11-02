"""把纯描述 ExecutionPlan 转成设备 Tensor。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .flat_batch import FlatDecodeBatch, flatten_decode_batch
from .plan import ExecutionPlan
from .request import SamplingParams
from .state import RequestStatus, SequenceStore
from miniserve.kv.base import KVSequenceExecutionMetadata, KVStepContext


@dataclass(frozen=True)
class PreparedBatch:
    mode: str
    input_ids: torch.Tensor
    positions: torch.Tensor
    request_ids: tuple[str, ...]
    sampling_params: tuple[SamplingParams, ...]
    seq_lens: torch.Tensor
    step_id: int
    # 这些值与设备 tensor 同源，由 SequenceState/KVManager 在 CPU 上生成。
    # CUDA serving 可用它们做地址安全校验，避免再从 GPU 读回 tiny metadata。
    host_input_token_ids: tuple[int, ...]
    kv_metadata: KVSequenceExecutionMetadata | None = None
    sample_after_execution: bool = True
    block_tables: torch.Tensor | None = None
    slot_mapping: torch.Tensor | None = None


@dataclass(frozen=True)
class PreparedStep:
    """同一事务内固定 decode 后 prefill；decode 可为 serial 或真实 flat batch。"""

    decode: tuple[PreparedBatch, ...] = ()
    prefill: tuple[PreparedBatch, ...] = ()
    flat_decode: "PreparedFlatDecode | None" = None

    @property
    def empty(self) -> bool:
        return not self.decode and not self.prefill and self.flat_decode is None


@dataclass(frozen=True)
class PreparedFlatDecode:
    batch: FlatDecodeBatch
    sampling_params: tuple[SamplingParams, ...]
    metadata: tuple[KVSequenceExecutionMetadata, ...]
    host_input_token_ids: tuple[int, ...]
    step_id: int

    def __post_init__(self) -> None:
        if not isinstance(self.batch, FlatDecodeBatch):
            raise TypeError("batch must be a FlatDecodeBatch.")
        if len(self.sampling_params) != self.batch.batch_size:
            raise ValueError("sampling params must match the flat batch size.")
        if len(self.metadata) != self.batch.batch_size:
            raise ValueError("metadata must match the flat batch size.")
        if len(self.host_input_token_ids) != self.batch.batch_size:
            raise ValueError("host input tokens must match the flat batch size.")
        if any(
            metadata.request_id != request_id
            for request_id, metadata in zip(self.batch.request_ids, self.metadata)
        ):
            raise ValueError("metadata order must match flat request order.")


class BatchBuilder:
    """构造 serial 或 flat decode/逐请求 prefill 的 PreparedStep。"""

    def __init__(
        self,
        store: SequenceStore,
        *,
        device: torch.device | str,
        decode_batching: str = "serial",
        block_size: int | None = None,
        physical_block_count: int | None = None,
    ) -> None:
        if decode_batching not in {"serial", "batched"}:
            raise ValueError("decode_batching must be serial or batched.")
        self._store = store
        self._device = torch.device(device)
        self.decode_batching = decode_batching
        self._block_size = block_size
        self._physical_block_count = physical_block_count

    def prepare_recompute(self, plan: ExecutionPlan) -> PreparedBatch:
        item_count = len(plan.prefill_items) + len(plan.decode_items)
        if plan.empty or item_count != 1:
            raise ValueError("Phase 1 BatchBuilder requires exactly one scheduled item.")

        if plan.prefill_items:
            item = plan.prefill_items[0]
            sequence = self._store.get(item.request_id)
            if sequence.status not in {RequestStatus.WAITING, RequestStatus.PREFILLING}:
                raise RuntimeError("Prefill item does not match sequence state.")
            if (item.token_start, item.token_end) != (0, len(sequence.prompt_token_ids)):
                raise RuntimeError("Phase 1 only supports one full-prompt prefill.")
            mode = "prefill"
        else:
            item = plan.decode_items[0]
            sequence = self._store.get(item.request_id)
            if sequence.status is not RequestStatus.DECODING:
                raise RuntimeError("Decode item does not match sequence state.")
            if item.position != sequence.computed_tokens or item.input_token_id != sequence.all_tokens[item.position]:
                raise RuntimeError("Decode item is not the canonical pending token.")
            mode = "decode"

        # Recompute 故意在每轮复制 prompt + 已生成 token；Phase 2 才由 KV 消除重复计算。
        tokens = torch.tensor([sequence.all_tokens], dtype=torch.long, device=self._device)
        positions = torch.arange(tokens.shape[1], dtype=torch.long, device=self._device).unsqueeze(0)
        lengths = torch.tensor([tokens.shape[1]], dtype=torch.int32, device=self._device)
        return PreparedBatch(
            mode=mode,
            input_ids=tokens,
            positions=positions,
            request_ids=(sequence.request_id,),
            sampling_params=(sequence.sampling_params,),
            seq_lens=lengths,
            step_id=plan.step_id,
            host_input_token_ids=tuple(sequence.all_tokens),
        )

    def prepare_contiguous(self, plan: ExecutionPlan, kv_step: KVStepContext) -> PreparedBatch:
        """只打包本 step 新 token；历史 token 已经存在 contiguous KV 中。"""

        item_count = len(plan.prefill_items) + len(plan.decode_items)
        if plan.empty or item_count != 1 or plan.step_id != kv_step.step_id:
            raise ValueError("Phase 2 requires one plan item and matching KV context.")
        request_id = (plan.prefill_items or plan.decode_items)[0].request_id
        sequence = self._store.get(request_id)
        metadata = kv_step.sequence_metadata[request_id]
        if plan.prefill_items:
            item = plan.prefill_items[0]
            if sequence.status not in {RequestStatus.WAITING, RequestStatus.PREFILLING}:
                raise RuntimeError("Prefill item does not match sequence state.")
            token_ids = sequence.prompt_token_ids[item.token_start:item.token_end]
            mode = "prefill"
        else:
            item = plan.decode_items[0]
            if sequence.status is not RequestStatus.DECODING:
                raise RuntimeError("Decode item does not match sequence state.")
            if item.position != metadata.context_len_before_step:
                raise RuntimeError("Decode position does not match committed KV context.")
            token_ids = (item.input_token_id,)
            mode = "decode"
        expected_tokens = metadata.context_len_after_step - metadata.context_len_before_step
        if len(token_ids) != expected_tokens:
            raise RuntimeError("Prepared token count does not match KV reservation.")
        tokens = torch.tensor([token_ids], dtype=torch.long, device=self._device)
        positions = torch.arange(
            metadata.context_len_before_step,
            metadata.context_len_after_step,
            dtype=torch.long,
            device=self._device,
        ).unsqueeze(0)
        lengths = torch.tensor(
            [metadata.context_len_after_step], dtype=torch.int32, device=self._device
        )
        return PreparedBatch(
            mode=mode,
            input_ids=tokens,
            positions=positions,
            request_ids=(request_id,),
            sampling_params=(sequence.sampling_params,),
            seq_lens=lengths,
            step_id=plan.step_id,
            host_input_token_ids=tuple(token_ids),
            kv_metadata=metadata,
            sample_after_execution=(
                mode == "decode" or metadata.context_len_after_step == len(sequence.prompt_token_ids)
            ),
            block_tables=(
                torch.tensor([metadata.block_table], dtype=torch.int32, device=self._device)
                if metadata.block_table
                else None
            ),
            slot_mapping=(
                torch.tensor(metadata.slot_mapping, dtype=torch.int64, device=self._device)
                if metadata.slot_mapping
                else None
            ),
        )

    def prepare_contiguous_step(
        self, plan: ExecutionPlan, kv_step: KVStepContext
    ) -> PreparedStep:
        if plan.empty or plan.step_id != kv_step.step_id:
            raise ValueError("PreparedStep requires a non-empty matching plan/KV context.")
        flat_decode: PreparedFlatDecode | None = None
        if self.decode_batching == "batched" and plan.decode_items:
            if self._block_size is None or self._physical_block_count is None:
                raise RuntimeError("Batched decode requires KV block layout metadata.")
            metadata = tuple(
                kv_step.sequence_metadata[item.request_id] for item in plan.decode_items
            )
            flat = flatten_decode_batch(
                metadata,
                host_input_token_ids=tuple(item.input_token_id for item in plan.decode_items),
                block_size=self._block_size,
                device=self._device,
                physical_block_count=self._physical_block_count,
            )
            flat_decode = PreparedFlatDecode(
                batch=flat,
                sampling_params=tuple(
                    self._store.get(item.request_id).sampling_params
                    for item in plan.decode_items
                ),
                metadata=metadata,
                host_input_token_ids=tuple(
                    item.input_token_id for item in plan.decode_items
                ),
                step_id=plan.step_id,
            )
            decode_batches: tuple[PreparedBatch, ...] = ()
        else:
            decode_batches = tuple(
                self.prepare_contiguous(
                    ExecutionPlan(
                        step_id=plan.step_id,
                        decode_items=(item,),
                        scheduled_tokens=1,
                    ),
                    kv_step,
                )
                for item in plan.decode_items
            )
        prefill_batches = tuple(
            self.prepare_contiguous(
                ExecutionPlan(
                    step_id=plan.step_id,
                    prefill_items=(item,),
                    scheduled_tokens=item.token_end - item.token_start,
                ),
                kv_step,
            )
            for item in plan.prefill_items
        )
        return PreparedStep(
            decode=decode_batches,
            prefill=prefill_batches,
            flat_decode=flat_decode,
        )
