"""PreparedBatch 到 next token 的唯一模型执行入口。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from miniserve.attention.base import AttentionBackend
from miniserve.model.qwen3 import Qwen3ForCausalLM
from miniserve.model.sampling import PerRequestSampler, StagedSample

from .batch_builder import PreparedBatch, PreparedFlatDecode, PreparedStep


ForwardObserver = Callable[[str, int, int], None]


def _sampling_path(
    *,
    execution: str,
    logits: torch.Tensor,
    params: tuple,
) -> str:
    """Return the concrete flat-decode sampling path for this step."""

    if execution == "reference":
        return "reference"
    if not logits.is_cuda:
        return "non_cuda"
    if any(item.temperature != 0.0 for item in params):
        return "mixed_sampling_params"
    return "batched_greedy"


@dataclass(frozen=True)
class ExecutorOutput:
    request_ids: tuple[str, ...]
    next_token_ids: tuple[int, ...]
    # 只保留最后位置的 CPU FP32 logits，便于 correctness/evidence；约 0.6 MB，
    # 不会让整个 [batch, sequence, vocab] 长期占用显存。
    last_logits: torch.Tensor | None
    staged_samples: tuple[StagedSample, ...] = ()


@dataclass(frozen=True)
class StepExecutorOutput:
    outputs: tuple[ExecutorOutput, ...]
    execution_order: tuple[str, ...]


class ModelExecutor:
    def __init__(
        self,
        model: Qwen3ForCausalLM,
        attention_backend: AttentionBackend,
        *,
        retain_last_logits: bool = False,
        validate_logits_finite: bool = True,
        sampling_execution: str = "reference",
    ) -> None:
        if not isinstance(retain_last_logits, bool) or not isinstance(
            validate_logits_finite, bool
        ):
            raise TypeError("ModelExecutor validation/retention flags must be bools.")
        self.model = model.eval()
        self.attention_backend = attention_backend
        self.retain_last_logits = retain_last_logits
        self.sampler = PerRequestSampler(validate_finite=validate_logits_finite)
        if sampling_execution not in {"reference", "batched_greedy"}:
            raise ValueError("sampling_execution must be reference or batched_greedy.")
        self.sampling_execution = sampling_execution
        self._sampling_batched_steps = 0
        self._sampling_batched_rows = 0
        self._sampling_fallback_steps = 0
        self._sampling_fallback_rows = 0
        self._sampling_fallback_reasons: dict[str, int] = {}
        self._forward_observer: ForwardObserver | None = None

    @property
    def sampling_stats(self) -> dict[str, object]:
        return {
            "execution": self.sampling_execution,
            "batched_greedy_steps": self._sampling_batched_steps,
            "batched_greedy_rows": self._sampling_batched_rows,
            "fallback_steps": self._sampling_fallback_steps,
            "fallback_rows": self._sampling_fallback_rows,
            "fallback_reasons": dict(sorted(self._sampling_fallback_reasons.items())),
        }

    def set_forward_observer(self, observer: ForwardObserver | None) -> None:
        """Observe each successfully completed model forward at its execution boundary."""

        if observer is not None and not callable(observer):
            raise TypeError("forward observer must be callable or None.")
        self._forward_observer = observer

    def _notify_forward_completed(
        self,
        mode: str,
        *,
        decode_rows: int = 0,
        prefill_input_tokens: int = 0,
    ) -> None:
        if self._forward_observer is not None:
            self._forward_observer(mode, decode_rows, prefill_input_tokens)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def execute(self, batch: PreparedBatch) -> ExecutorOutput:
        if (
            batch.input_ids.shape[0] != 1
            or len(batch.request_ids) != 1
            or len(batch.sampling_params) != 1
        ):
            raise RuntimeError("Phase 1 executor supports batch size 1 only.")
        with torch.inference_mode():
            if batch.kv_metadata is None:
                logits = self.attention_backend.forward_recompute(
                    self.model, batch.input_ids, batch.positions
                )
            else:
                logits = self.attention_backend.forward_with_kv(
                    self.model,
                    batch.input_ids,
                    batch.positions,
                    request_id=batch.kv_metadata.request_id,
                    context_len_before=batch.kv_metadata.context_len_before_step,
                    mode=batch.mode,
                    block_tables=batch.block_tables,
                    seq_lens=batch.seq_lens,
                    slot_mapping=batch.slot_mapping,
                    host_input_token_ids=batch.host_input_token_ids,
                    kv_metadata=batch.kv_metadata,
                )
            if batch.mode == "decode":
                self._notify_forward_completed(
                    "decode", decode_rows=int(batch.input_ids.shape[0])
                )
            elif batch.mode == "prefill":
                self._notify_forward_completed(
                    "prefill", prefill_input_tokens=int(batch.input_ids.numel())
                )
            else:
                raise RuntimeError(f"Unsupported PreparedBatch mode: {batch.mode!r}")
            if batch.sample_after_execution:
                last_logits = logits[:, -1, :]
                staged = (
                    self.sampler.stage(
                        batch.request_ids[0], last_logits, batch.sampling_params[0]
                    ),
                )
                cpu_logits = (
                    last_logits.detach().to(device="cpu", dtype=torch.float32)
                    if self.retain_last_logits
                    else None
                )
            else:
                staged = ()
                cpu_logits = None
        return ExecutorOutput(
            request_ids=batch.request_ids,
            next_token_ids=tuple(sample.token_id for sample in staged),
            last_logits=cpu_logits,
            staged_samples=staged,
        )

    def commit_staged_samples(self, outputs: tuple[ExecutorOutput, ...]) -> None:
        """KV/sequence 提交后一次性推进本轮的 RNG 快照。"""

        staged = tuple(sample for output in outputs for sample in output.staged_samples)
        self.sampler.commit(staged)

    def _execute_flat_decode(self, prepared: PreparedFlatDecode) -> ExecutorOutput:
        with torch.inference_mode():
            forward = getattr(self.attention_backend, "forward_flat_decode", None)
            if forward is None:
                raise RuntimeError("Attention backend does not implement flat decode.")
            logits = forward(
                self.model,
                prepared.batch,
                metadata=prepared.metadata,
                host_input_token_ids=prepared.host_input_token_ids,
            )
            self._notify_forward_completed(
                "decode", decode_rows=prepared.batch.batch_size
            )
            if logits.ndim != 3 or logits.shape[0] != prepared.batch.batch_size or logits.shape[1] != 1:
                raise RuntimeError("Batched decode model returned an invalid logits shape.")
            last_logits = logits[:, -1, :]
            sampling_path = _sampling_path(
                execution=self.sampling_execution,
                logits=last_logits,
                params=prepared.sampling_params,
            )
            if sampling_path == "batched_greedy":
                staged = self.sampler.stage_greedy_batch(
                    prepared.batch.request_ids,
                    last_logits,
                    prepared.sampling_params,
                )
                self._sampling_batched_steps += 1
                self._sampling_batched_rows += prepared.batch.batch_size
            else:
                if sampling_path != "reference":
                    self._sampling_fallback_steps += 1
                    self._sampling_fallback_rows += prepared.batch.batch_size
                    self._sampling_fallback_reasons[sampling_path] = (
                        self._sampling_fallback_reasons.get(sampling_path, 0) + 1
                    )
                staged = tuple(
                    self.sampler.stage(
                        request_id,
                        last_logits[index : index + 1],
                        prepared.sampling_params[index],
                    )
                    for index, request_id in enumerate(prepared.batch.request_ids)
                )
            cpu_logits = (
                last_logits.detach().to(device="cpu", dtype=torch.float32)
                if self.retain_last_logits
                else None
            )
        return ExecutorOutput(
            request_ids=prepared.batch.request_ids,
            next_token_ids=tuple(sample.token_id for sample in staged),
            last_logits=cpu_logits,
            staged_samples=staged,
        )

    def release_request(self, request_id: str) -> None:
        """与 Engine terminal cleanup 同步释放该请求的随机流。"""

        self.sampler.release(request_id)

    def execute_step(self, prepared: PreparedStep) -> StepExecutorOutput:
        """Reference executor 固定 decode→prefill；forward 观测按实际完成点保留。"""

        if prepared.empty:
            raise ValueError("Cannot execute an empty PreparedStep.")
        outputs: list[ExecutorOutput] = []
        order: list[str] = []
        if prepared.flat_decode is not None:
            outputs.append(self._execute_flat_decode(prepared.flat_decode))
            order.extend(prepared.flat_decode.batch.request_ids)
        for batch in prepared.decode:
            outputs.append(self.execute(batch))
            order.extend(batch.request_ids)
        for batch in prepared.prefill:
            outputs.append(self.execute(batch))
            order.extend(batch.request_ids)
        return StepExecutorOutput(tuple(outputs), tuple(order))
