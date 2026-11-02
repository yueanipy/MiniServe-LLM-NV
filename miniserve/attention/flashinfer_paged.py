"""Opt-in FlashInfer FP16 paged-GQA decode adapter.

This backend intentionally owns only the per-layer attention call.  Paged KV
allocation, reservation/commit/abort, prefix sharing, chunking, sampling and
publication remain in MiniServe's existing execution plan.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from miniserve.engine.flat_batch import (
    FlatDecodeBatch,
    validate_flat_decode_provenance,
)
from miniserve.model.qwen3 import Qwen3ForCausalLM

from .cuda_paged import CudaPagedDecodeAttention

FLASHINFER_VERSION = "0.7.2.post1"
FLASHINFER_COMMIT = "beabf5004585b03da60e927bbb803acbaa62fe36"
DEFAULT_WORKSPACE_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True)
class FlashInferCSRPlan:
    """One host-derived CSR page plan shared by all layers in a decode step."""

    indptr: torch.Tensor
    indices: torch.Tensor
    last_page_len: torch.Tensor
    page_size: int
    batch_size: int
    page_count: int


def build_flashinfer_csr_plan(
    batch: FlatDecodeBatch,
    *,
    device: torch.device | str = "cpu",
) -> FlashInferCSRPlan:
    """Build FlashInfer's CSR page metadata from the trusted batch proof.

    The dense batch table is intentionally not passed to the wrapper: padding
    columns are an implementation detail of MiniServe's V1 kernel.  Taking the
    exact active prefix here keeps FlashInfer's page indices bounded by the
    request's context length.
    """

    if not isinstance(batch, FlatDecodeBatch):
        raise TypeError("batch must be a FlatDecodeBatch.")
    batch.assert_invariants()
    host = batch.host_metadata
    if host is None:
        raise ValueError("FlashInfer planning requires a trusted batch proof.")

    indptr_values = [0]
    index_values: list[int] = []
    last_page_len_values: list[int] = []
    for row_index, (length, table) in enumerate(
        zip(host.context_lens, host.block_tables)
    ):
        if length <= 0:
            raise ValueError("FlashInfer context lengths must be positive.")
        required = (length + batch.block_size - 1) // batch.block_size
        active = table[:required]
        if len(active) != required:
            raise ValueError(
                f"FlashInfer block table row {row_index} is shorter than its context."
            )
        if any(
            not isinstance(block, int) or isinstance(block, bool) or block < 0
            for block in active
        ):
            raise ValueError("FlashInfer block table contains an invalid block ID.")
        index_values.extend(active)
        indptr_values.append(len(index_values))
        last_page_len_values.append((length - 1) % batch.block_size + 1)

    if not index_values:
        raise ValueError("FlashInfer planning produced no pages.")
    if len(indptr_values) != batch.batch_size + 1:
        raise ValueError("FlashInfer CSR indptr length does not match the batch.")
    if len(last_page_len_values) != batch.batch_size:
        raise ValueError("FlashInfer last-page length does not match the batch.")

    return FlashInferCSRPlan(
        indptr=torch.tensor(indptr_values, dtype=torch.int32, device=device),
        indices=torch.tensor(index_values, dtype=torch.int32, device=device),
        last_page_len=torch.tensor(
            last_page_len_values, dtype=torch.int32, device=device
        ),
        page_size=batch.block_size,
        batch_size=batch.batch_size,
        page_count=len(index_values),
    )


class FlashInferPagedDecodeAttention(CudaPagedDecodeAttention):
    """FlashInfer FA2 adapter for eager flat decode.

    The class deliberately rejects Graph execution.  The current graph manager
    captures MiniServe's V1 kernel; silently mixing a different attention
    implementation into that graph would invalidate its metadata and lifetime
    proof.
    """

    name = "flashinfer_paged"

    def __init__(
        self,
        kv_backend,
        *,
        workspace_bytes: int = DEFAULT_WORKSPACE_BYTES,
        disable_split_kv: bool = False,
        use_tensor_cores: bool = False,
    ) -> None:
        if (
            not isinstance(workspace_bytes, int)
            or isinstance(workspace_bytes, bool)
            or workspace_bytes <= 0
        ):
            raise ValueError("workspace_bytes must be a positive integer.")
        if not isinstance(disable_split_kv, bool):
            raise ValueError("disable_split_kv must be a boolean.")
        if not isinstance(use_tensor_cores, bool):
            raise ValueError("use_tensor_cores must be a boolean.")
        super().__init__(kv_backend)
        self._workspace_bytes = workspace_bytes
        self._disable_split_kv = disable_split_kv
        self._use_tensor_cores = use_tensor_cores
        self._wrapper = None
        self._shape: tuple[int, int, int, torch.dtype] | None = None
        self._active_plan: FlashInferCSRPlan | None = None
        self._planning_count = 0
        self._run_count = 0
        self._last_plan_ms: float | None = None
        self._resolved_backend: str | None = None

    @property
    def stats(self) -> dict[str, object]:
        return {
            "backend": self.name,
            "flashinfer_version": FLASHINFER_VERSION,
            "flashinfer_commit": FLASHINFER_COMMIT,
            "resolved_backend": self._resolved_backend,
            "workspace_bytes": self._workspace_bytes,
            "disable_split_kv": self._disable_split_kv,
            "use_tensor_cores": self._use_tensor_cores,
            "planning_count": self._planning_count,
            "run_count": self._run_count,
            "last_plan_ms": self._last_plan_ms,
            "shape": None
            if self._shape is None
            else {
                "num_qo_heads": self._shape[0],
                "num_kv_heads": self._shape[1],
                "head_dim": self._shape[2],
                "dtype": str(self._shape[3]),
            },
        }

    def bind_model(self, model: Qwen3ForCausalLM) -> None:
        """Bind model dimensions without hardcoding a Qwen3 checkpoint shape."""

        if self._wrapper is not None:
            if self._shape is None:
                raise RuntimeError("FlashInfer wrapper is not fully bound.")
            expected = (
                int(model.config.num_attention_heads),
                int(model.config.num_key_value_heads),
                int(model.config.head_dim),
                self.kv_backend.layout.dtype,
            )
            if expected != self._shape:
                raise RuntimeError(
                    "FlashInfer attention backend cannot be rebound to a different "
                    "model shape or K/V dtype."
                )
            return

        try:
            import flashinfer
        except Exception as exc:  # pragma: no cover - exercised by dependency failure
            raise RuntimeError(
                "FLASHINFER_UNAVAILABLE: install the pinned flashinfer-python "
                f"{FLASHINFER_VERSION} distribution before selecting this backend."
            ) from exc
        version = getattr(flashinfer, "__version__", None)
        commit = getattr(flashinfer, "__git_commit__", None)
        if version != FLASHINFER_VERSION or commit != FLASHINFER_COMMIT:
            raise RuntimeError(
                "FLASHINFER_VERSION_MISMATCH: expected "
                f"{FLASHINFER_VERSION} @ {FLASHINFER_COMMIT}, got "
                f"{version!r} @ {commit!r}."
            )

        num_qo_heads = int(model.config.num_attention_heads)
        num_kv_heads = int(model.config.num_key_value_heads)
        head_dim = int(model.config.head_dim)
        dtype = self.kv_backend.layout.dtype
        if num_qo_heads <= 0 or num_kv_heads <= 0 or head_dim <= 0:
            raise ValueError("FlashInfer requires positive attention dimensions.")
        if num_qo_heads % num_kv_heads != 0:
            raise ValueError("FlashInfer GQA requires q heads divisible by kv heads.")
        if dtype not in {torch.float16, torch.bfloat16}:
            raise ValueError("FlashInfer paged decode supports float16 or bfloat16 K/V.")

        device = self.kv_backend.layout.device
        workspace = torch.zeros(
            self._workspace_bytes, dtype=torch.uint8, device=device
        )
        self._wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
            workspace,
            kv_layout="HND",
            backend="fa2",
            use_tensor_cores=self._use_tensor_cores,
        )
        self._shape = (num_qo_heads, num_kv_heads, head_dim, dtype)

    def _require_bound(self) -> tuple[int, int, int, torch.dtype]:
        if self._wrapper is None or self._shape is None:
            raise RuntimeError(
                "FlashInfer attention backend must be bound to a loaded model before decode."
            )
        return self._shape

    def enable_cuda_graph(self, *, max_model_len: int) -> None:
        raise RuntimeError(
            "flashinfer_paged is an eager-only backend; the legacy CUDA Graph "
            "path remains on cuda_paged."
        )

    def enable_batched_graph(self, *, model, max_model_len: int) -> None:
        raise RuntimeError(
            "flashinfer_paged is an eager-only backend and cannot be combined "
            "with batched_graph."
        )

    def decode(
        self,
        query: torch.Tensor,
        *,
        layer_index: int,
        block_tables: torch.Tensor,
        sequence_lengths: torch.Tensor,
    ) -> torch.Tensor:
        raise RuntimeError(
            "flashinfer_paged only executes through forward_flat_decode; "
            "the serial decode path remains on cuda_paged."
        )

    def decode_graph_safe(
        self,
        query: torch.Tensor,
        *,
        layer_index: int,
        block_tables: torch.Tensor,
        sequence_lengths: torch.Tensor,
    ) -> torch.Tensor:
        del block_tables, sequence_lengths
        num_qo_heads, num_kv_heads, head_dim, dtype = self._require_bound()
        plan = self._active_plan
        if plan is None:
            raise RuntimeError(
                "FlashInfer plan must be prepared once at the flat-decode boundary."
            )
        if query.ndim != 3 or tuple(query.shape) != (
            plan.batch_size,
            num_qo_heads,
            head_dim,
        ):
            raise ValueError(
                "FlashInfer query shape must be [batch, num_qo_heads, head_dim]."
            )
        if query.dtype != dtype or not query.is_cuda:
            raise ValueError("FlashInfer query dtype/device does not match the K/V pool.")
        if not isinstance(layer_index, int) or isinstance(layer_index, bool):
            raise TypeError("layer_index must be an integer.")
        if layer_index < 0 or layer_index >= self.kv_backend.layout.num_layers:
            raise ValueError("layer_index is outside the K/V pool.")

        key = self.kv_backend.key_pool[layer_index]
        value = self.kv_backend.value_pool[layer_index]
        output = self._wrapper.run(query, (key, value))
        self._run_count += 1
        if tuple(output.shape) != tuple(query.shape) or output.dtype != query.dtype:
            raise RuntimeError("FlashInfer returned an invalid attention output.")
        if num_kv_heads != int(key.shape[1]):
            raise RuntimeError("FlashInfer K/V head count changed after binding.")
        return output

    def forward_flat_decode(
        self,
        model: Qwen3ForCausalLM,
        batch: FlatDecodeBatch,
        *,
        metadata,
        host_input_token_ids: tuple[int, ...],
    ) -> torch.Tensor:
        self.bind_model(model)
        num_qo_heads, num_kv_heads, head_dim, dtype = self._require_bound()
        if self.graph_manager is not None or self.batched_graph_manager is not None:
            raise RuntimeError("FlashInfer eager decode cannot run with a graph manager.")
        if self.kv_backend.layout.block_size != batch.block_size:
            raise ValueError("FlashInfer page size does not match the batch block size.")

        validate_flat_decode_provenance(
            batch,
            metadata,
            host_input_token_ids,
            physical_block_count=self.kv_backend.allocator.num_blocks,
            vocab_size=model.config.vocab_size,
            max_model_len=model.config.max_model_len,
        )
        plan = build_flashinfer_csr_plan(batch)
        started = time.perf_counter()
        self._wrapper.plan(
            plan.indptr,
            plan.indices,
            plan.last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            batch.block_size,
            pos_encoding_mode="NONE",
            q_data_type=dtype,
            kv_data_type=dtype,
            o_data_type=dtype,
            sm_scale=head_dim**-0.5,
            disable_split_kv=self._disable_split_kv,
        )
        self._planning_count += 1
        self._last_plan_ms = (time.perf_counter() - started) * 1000.0
        self._resolved_backend = getattr(self._wrapper, "resolved_backend", None)
        self._active_plan = plan
        try:
            return model.forward_with_kv_batch(
                batch.input_ids.reshape(batch.batch_size, 1),
                batch.positions.reshape(batch.batch_size, 1),
                kv_backend=self.kv_backend,
                decode_attention=self,
                block_tables=batch.block_tables,
                sequence_lengths=batch.context_lens,
                slot_mapping=batch.slot_mapping,
            )
        finally:
            self._active_plan = None
