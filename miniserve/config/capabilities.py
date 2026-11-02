"""M0 语义能力开关与独立合法性校验（解耦历史 Phase 强绑定）。

EngineConfig 的 ``validate_phaseN`` 保留为兼容回归入口；EngineCore 与 bootstrap 改用
本模块的 capability validation，使独立语义开关（kv backend / attention / rmsnorm /
decode execution / prefix / chunk）可以按**真实实现能力**组合。

M0 冻结的真实能力（正是这些组合，不多不少）：
- recompute：单请求、torch_contiguous attention、torch rmsnorm、eager、无 prefix、无 chunk。
- contiguous：torch_contiguous attention、torch rmsnorm、eager、无 prefix；chunk 任选；
  max_num_seqs==1 为单请求链，>1 为 continuous batching。
- paged：attention ∈ {torch_paged_ref, cuda_paged}；rmsnorm ∈ {torch, cuda}；
  decode ∈ {eager, cuda_graph}；num_kv_blocks 必填；prefix 与 chunk 任选。
  cuda_graph 需要 cuda_paged attention + cuda rmsnorm（native 库在运行时加载）。
- 显式拒绝（不是静默 fallback）：真实 batched（B>1）执行（属 M1）、graph+非 cuda 路径、
  prefix+非 paged、contiguous/recompute 上的 CUDA decode、缺少 num_kv_blocks 的 paged。
"""

from __future__ import annotations

from dataclasses import dataclass

from .engine_config import EngineConfig


@dataclass(frozen=True)
class CapabilitySummary:
    """规范化后的能力面；写入启动记录与证据，避免只靠文件名猜配置。"""

    kv_backend: str
    attention_backend: str
    rmsnorm_backend: str
    decode_execution: str
    decode_batching: str
    batched_graph: bool
    sampling_execution: str
    prefix_cache: bool
    chunked_prefill: bool
    prefill_chunk_size: int | None
    continuous_batching: bool
    # M0 恒为 False：一次模型 forward 处理多个请求（真实 batched）属于 M1。
    batched_execution: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "kv_backend": self.kv_backend,
            "attention_backend": self.attention_backend,
            "rmsnorm_backend": self.rmsnorm_backend,
            "decode_execution": self.decode_execution,
            "decode_batching": self.decode_batching,
            "batched_graph": self.batched_graph,
            "sampling_execution": self.sampling_execution,
            "prefix_cache": self.prefix_cache,
            "chunked_prefill": self.chunked_prefill,
            "prefill_chunk_size": self.prefill_chunk_size,
            "continuous_batching": self.continuous_batching,
            "batched_execution": self.batched_execution,
        }


def describe_capabilities(engine: EngineConfig) -> CapabilitySummary:
    """只做规范化描述，不判断合法性（合法性见 validate_capabilities）。"""

    if not isinstance(engine, EngineConfig):
        raise TypeError("describe_capabilities expects an EngineConfig.")
    return CapabilitySummary(
        kv_backend=engine.kv_backend,
        attention_backend=engine.attention_backend,
        rmsnorm_backend=engine.rmsnorm_backend,
        decode_execution=engine.decode_execution,
        decode_batching=engine.decode_batching,
        batched_graph=engine.batched_graph,
        sampling_execution=engine.sampling_execution,
        prefix_cache=engine.enable_prefix_cache,
        chunked_prefill=engine.enable_chunked_prefill,
        prefill_chunk_size=engine.prefill_chunk_size,
        continuous_batching=(
            engine.kv_backend == "paged"
            or (engine.kv_backend == "contiguous" and engine.max_num_seqs > 1)
        ),
        batched_execution=engine.decode_batching == "batched",
    )


def validate_capabilities(engine: EngineConfig) -> CapabilitySummary:
    """按独立开关校验合法组合；未实现组合给 ``CAPABILITY_UNSUPPORTED`` 前缀错误。"""

    summary = describe_capabilities(engine)
    engine.validate()
    if engine.kv_backend == "recompute":
        if engine.max_num_seqs != 1:
            raise ValueError(
                "CAPABILITY_UNSUPPORTED: recompute KV supports exactly one active request."
            )
        if engine.attention_backend != "torch_contiguous":
            raise ValueError(
                "CAPABILITY_UNSUPPORTED: recompute KV requires torch_contiguous attention."
            )
        if engine.rmsnorm_backend != "torch":
            raise ValueError(
                "CAPABILITY_UNSUPPORTED: recompute KV requires the torch RMSNorm backend."
            )
        if engine.decode_execution != "eager":
            raise ValueError(
                "CAPABILITY_UNSUPPORTED: recompute KV requires eager decode execution."
            )
        if engine.enable_prefix_cache or engine.enable_chunked_prefill:
            raise ValueError(
                "CAPABILITY_UNSUPPORTED: recompute KV has no prefix cache or chunked prefill."
            )
    elif engine.kv_backend == "contiguous":
        if engine.attention_backend != "torch_contiguous":
            raise ValueError(
                "CAPABILITY_UNSUPPORTED: contiguous KV requires torch_contiguous attention."
            )
        if engine.rmsnorm_backend != "torch":
            raise ValueError(
                "CAPABILITY_UNSUPPORTED: contiguous KV currently supports the torch RMSNorm backend only."
            )
        if engine.decode_execution != "eager":
            raise ValueError(
                "CAPABILITY_UNSUPPORTED: contiguous KV requires eager decode execution."
            )
        if engine.enable_prefix_cache:
            raise ValueError(
                "CAPABILITY_UNSUPPORTED: prefix cache requires the paged KV backend."
            )
    else:  # paged
        if engine.num_kv_blocks is None:
            raise ValueError(
                "CAPABILITY_UNSUPPORTED: the paged KV backend requires num_kv_blocks."
            )
        if engine.decode_execution == "cuda_graph":
            if engine.attention_backend != "cuda_paged":
                raise ValueError(
                    "CAPABILITY_UNSUPPORTED: CUDA Graph decode requires the cuda_paged attention backend."
                )
            if engine.rmsnorm_backend != "cuda":
                raise ValueError(
                    "CAPABILITY_UNSUPPORTED: CUDA Graph decode requires the cuda RMSNorm backend."
                )
        if engine.decode_batching == "batched":
            if engine.attention_backend not in {"cuda_paged", "flashinfer_paged"}:
                raise ValueError(
                    "CAPABILITY_UNSUPPORTED: batched decode requires cuda_paged "
                    "or flashinfer_paged attention."
                )
            if engine.rmsnorm_backend != "cuda":
                raise ValueError(
                    "CAPABILITY_UNSUPPORTED: batched decode requires CUDA RMSNorm."
                )
            if engine.batched_graph:
                if engine.decode_execution != "cuda_graph":
                    raise ValueError(
                        "CAPABILITY_UNSUPPORTED: batched_graph requires cuda_graph execution."
                    )
            elif engine.decode_execution != "eager":
                raise ValueError(
                    "CAPABILITY_UNSUPPORTED: batched decode requires eager execution."
                )
            if engine.attention_backend == "flashinfer_paged" and (
                engine.decode_execution != "eager" or engine.batched_graph
            ):
                raise ValueError(
                    "CAPABILITY_UNSUPPORTED: flashinfer_paged is eager-only and "
                    "cannot be combined with Graph execution."
                )
    return summary
