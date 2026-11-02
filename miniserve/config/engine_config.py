"""Engine 能力开关及 Phase 1 的启动时校验。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EngineConfig:
    """逐阶段演进时保持字段稳定的 Engine 配置。

    公共默认值面向后续最终引擎；Phase 1 必须使用 ``for_phase1`` 或显式传入
    recompute。这样不会把尚未实现的 contiguous backend 悄悄当作可用。
    """

    max_num_seqs: int = 8
    max_num_batched_tokens: int = 4096
    kv_block_size: int = 16
    num_kv_blocks: int | None = None
    enable_prefix_cache: bool = False
    enable_chunked_prefill: bool = False
    prefill_chunk_size: int | None = None
    attention_backend: str = "torch_contiguous"
    kv_backend: str = "contiguous"
    rmsnorm_backend: str = "torch"
    decode_execution: str = "eager"
    decode_batching: str = "serial"
    batched_graph: bool = False
    sampling_execution: str = "reference"
    max_pending_requests: int = 64
    prefix_cache_max_entries: int = 64

    @classmethod
    def for_phase1(cls, *, max_num_batched_tokens: int = 128) -> "EngineConfig":
        """构造低负载、batch=1 的 Phase 1 correctness 配置。"""

        return cls(
            max_num_seqs=1,
            max_num_batched_tokens=max_num_batched_tokens,
            kv_backend="recompute",
            attention_backend="torch_contiguous",
            rmsnorm_backend="torch",
            decode_execution="eager",
            max_pending_requests=1,
        )

    @classmethod
    def for_phase2(cls, *, max_num_batched_tokens: int = 128) -> "EngineConfig":
        """构造单请求 contiguous KV correctness 配置。"""

        return cls(
            max_num_seqs=1,
            max_num_batched_tokens=max_num_batched_tokens,
            kv_backend="contiguous",
            attention_backend="torch_contiguous",
            rmsnorm_backend="torch",
            decode_execution="eager",
            max_pending_requests=1,
        )

    @classmethod
    def for_phase3(
        cls,
        *,
        max_num_seqs: int = 3,
        max_num_batched_tokens: int = 32,
        max_pending_requests: int = 8,
    ) -> "EngineConfig":
        return cls(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            kv_backend="contiguous",
            attention_backend="torch_contiguous",
            rmsnorm_backend="torch",
            decode_execution="eager",
            max_pending_requests=max_pending_requests,
        )

    @classmethod
    def for_phase4(
        cls,
        *,
        max_num_seqs: int = 3,
        max_num_batched_tokens: int = 16,
        max_pending_requests: int = 8,
    ) -> "EngineConfig":
        return cls(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            kv_backend="contiguous",
            attention_backend="torch_contiguous",
            rmsnorm_backend="torch",
            decode_execution="eager",
            enable_chunked_prefill=True,
            max_pending_requests=max_pending_requests,
        )

    @classmethod
    def for_phase5(
        cls,
        *,
        max_num_seqs: int = 3,
        max_num_batched_tokens: int = 16,
        num_kv_blocks: int = 12,
        kv_block_size: int = 16,
        max_pending_requests: int = 8,
    ) -> "EngineConfig":
        return cls(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            kv_block_size=kv_block_size,
            num_kv_blocks=num_kv_blocks,
            kv_backend="paged",
            attention_backend="torch_paged_ref",
            rmsnorm_backend="torch",
            decode_execution="eager",
            enable_chunked_prefill=True,
            max_pending_requests=max_pending_requests,
        )

    @classmethod
    def for_phase6(
        cls,
        *,
        max_num_seqs: int = 3,
        max_num_batched_tokens: int = 16,
        num_kv_blocks: int = 12,
        kv_block_size: int = 16,
        prefix_cache_max_entries: int = 8,
        max_pending_requests: int = 8,
    ) -> "EngineConfig":
        return cls(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            kv_block_size=kv_block_size,
            num_kv_blocks=num_kv_blocks,
            kv_backend="paged",
            attention_backend="torch_paged_ref",
            rmsnorm_backend="torch",
            decode_execution="eager",
            enable_chunked_prefill=True,
            enable_prefix_cache=True,
            prefix_cache_max_entries=prefix_cache_max_entries,
            max_pending_requests=max_pending_requests,
        )

    @classmethod
    def for_phase7(
        cls,
        *,
        max_num_seqs: int = 2,
        max_num_batched_tokens: int = 16,
        num_kv_blocks: int = 8,
        kv_block_size: int = 16,
        prefix_cache_max_entries: int = 8,
        max_pending_requests: int = 4,
    ) -> "EngineConfig":
        """在完整 Phase 6 主链上只把 RMSNorm backend 切换为 CUDA。"""

        return cls(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            kv_block_size=kv_block_size,
            num_kv_blocks=num_kv_blocks,
            kv_backend="paged",
            attention_backend="torch_paged_ref",
            rmsnorm_backend="cuda",
            decode_execution="eager",
            enable_chunked_prefill=True,
            enable_prefix_cache=True,
            prefix_cache_max_entries=prefix_cache_max_entries,
            max_pending_requests=max_pending_requests,
        )

    @classmethod
    def for_phase8(
        cls,
        *,
        max_num_seqs: int = 2,
        max_num_batched_tokens: int = 16,
        num_kv_blocks: int = 8,
        kv_block_size: int = 16,
        prefix_cache_max_entries: int = 8,
        max_pending_requests: int = 4,
    ) -> "EngineConfig":
        """在 Phase 7 上只把 decode attention backend 切换为 CUDA paged。"""

        return cls(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            kv_block_size=kv_block_size,
            num_kv_blocks=num_kv_blocks,
            kv_backend="paged",
            attention_backend="cuda_paged",
            rmsnorm_backend="cuda",
            decode_execution="eager",
            enable_chunked_prefill=True,
            enable_prefix_cache=True,
            prefix_cache_max_entries=prefix_cache_max_entries,
            max_pending_requests=max_pending_requests,
        )

    @classmethod
    def for_phase9(
        cls,
        *,
        max_num_seqs: int = 2,
        max_num_batched_tokens: int = 16,
        num_kv_blocks: int = 8,
        kv_block_size: int = 16,
        prefix_cache_max_entries: int = 8,
        max_pending_requests: int = 4,
    ) -> "EngineConfig":
        """Phase 8 eager path 保留，只把 decode execution 切换为 CUDA Graph。"""

        return cls(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            kv_block_size=kv_block_size,
            num_kv_blocks=num_kv_blocks,
            kv_backend="paged",
            attention_backend="cuda_paged",
            rmsnorm_backend="cuda",
            decode_execution="cuda_graph",
            enable_chunked_prefill=True,
            enable_prefix_cache=True,
            prefix_cache_max_entries=prefix_cache_max_entries,
            max_pending_requests=max_pending_requests,
        )

    @classmethod
    def for_phase10(
        cls,
        *,
        max_num_seqs: int = 2,
        max_num_batched_tokens: int = 16,
        num_kv_blocks: int = 8,
        kv_block_size: int = 16,
        prefix_cache_max_entries: int = 8,
        max_pending_requests: int = 4,
    ) -> "EngineConfig":
        """Profile L 服务配置：复用 Phase 9 主链，只加入 API/采样外围能力。"""

        return cls.for_phase9(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            num_kv_blocks=num_kv_blocks,
            kv_block_size=kv_block_size,
            prefix_cache_max_entries=prefix_cache_max_entries,
            max_pending_requests=max_pending_requests,
        )

    def validate(self) -> None:
        """校验不依赖当前阶段的通用字段关系。"""

        for name in (
            "max_num_seqs",
            "max_num_batched_tokens",
            "kv_block_size",
            "max_pending_requests",
            "prefix_cache_max_entries",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.num_kv_blocks is not None and (
            not isinstance(self.num_kv_blocks, int)
            or isinstance(self.num_kv_blocks, bool)
            or self.num_kv_blocks <= 0
        ):
            raise ValueError("num_kv_blocks must be a positive integer when set.")
        for name in ("enable_prefix_cache", "enable_chunked_prefill"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean.")
        if not isinstance(self.batched_graph, bool):
            raise ValueError("batched_graph must be a boolean.")
        if self.prefill_chunk_size is not None:
            if (
                not isinstance(self.prefill_chunk_size, int)
                or isinstance(self.prefill_chunk_size, bool)
                or self.prefill_chunk_size <= 0
            ):
                raise ValueError("prefill_chunk_size must be a positive integer or None.")
            if not self.enable_chunked_prefill:
                raise ValueError(
                    "prefill_chunk_size requires enable_chunked_prefill=True."
                )
            if self.prefill_chunk_size > self.max_num_batched_tokens:
                raise ValueError(
                    "prefill_chunk_size cannot exceed max_num_batched_tokens."
                )
        if self.kv_backend not in {"recompute", "contiguous", "paged"}:
            raise ValueError("Unknown kv_backend.")
        if self.attention_backend not in {
            "torch_contiguous",
            "torch_paged_ref",
            "cuda_paged",
            "flashinfer_paged",
        }:
            raise ValueError("Unknown attention_backend.")
        if self.rmsnorm_backend not in {"torch", "cuda"}:
            raise ValueError("Unknown rmsnorm_backend.")
        if self.decode_execution not in {"eager", "cuda_graph"}:
            raise ValueError("Unknown decode_execution.")
        if self.decode_batching not in {"serial", "batched"}:
            raise ValueError("Unknown decode_batching.")
        if self.batched_graph and (
            self.decode_batching != "batched"
            or self.decode_execution != "cuda_graph"
            or self.attention_backend != "cuda_paged"
            or self.rmsnorm_backend != "cuda"
            or self.kv_backend != "paged"
        ):
            raise ValueError(
                "batched_graph requires paged cuda_paged decode, "
                "batched batching, cuda RMSNorm, and cuda_graph execution."
            )
        if self.attention_backend == "flashinfer_paged":
            if (
                self.decode_batching != "batched"
                or self.decode_execution != "eager"
                or self.batched_graph
                or self.kv_backend != "paged"
            ):
                raise ValueError(
                    "flashinfer_paged requires eager batched decode with paged KV "
                    "and does not support Graph."
                )
        if self.sampling_execution not in {"reference", "batched_greedy"}:
            raise ValueError("Unknown sampling_execution.")
        if (
            self.sampling_execution == "batched_greedy"
            and self.decode_batching != "batched"
        ):
            raise ValueError(
                "sampling_execution=batched_greedy requires batched decode."
            )
        if self.batched_graph and self.sampling_execution != "reference":
            raise ValueError(
                "batched_graph is only validated with sampling_execution=reference."
            )
        if self.enable_prefix_cache and self.kv_backend != "paged":
            raise ValueError("Prefix cache requires the paged KV backend.")
        if self.attention_backend in {
            "torch_paged_ref",
            "cuda_paged",
            "flashinfer_paged",
        } and self.kv_backend != "paged":
            raise ValueError("Paged attention requires the paged KV backend.")
        if self.decode_execution == "cuda_graph" and self.kv_backend == "recompute":
            raise ValueError("CUDA Graph decode cannot be combined with recompute.")
        if (
            self.decode_batching == "batched"
            and self.decode_execution == "cuda_graph"
            and not self.batched_graph
        ):
            raise ValueError(
                "Batched decode cannot be combined with the legacy CUDA Graph path."
            )

    def validate_phase1(self) -> None:
        """拒绝所有 Phase 1 尚未实现的 backend/并发能力。"""

        self.validate()
        expected = {
            "max_num_seqs": (self.max_num_seqs, 1),
            "kv_backend": (self.kv_backend, "recompute"),
            "attention_backend": (self.attention_backend, "torch_contiguous"),
            "rmsnorm_backend": (self.rmsnorm_backend, "torch"),
            "decode_execution": (self.decode_execution, "eager"),
        }
        mismatches = [name for name, (actual, wanted) in expected.items() if actual != wanted]
        if mismatches or self.enable_prefix_cache or self.enable_chunked_prefill:
            detail = ", ".join(mismatches) or "future feature flag"
            raise ValueError(f"Phase 1 only implements single-request recompute: {detail}.")

    def validate_phase2(self) -> None:
        """Phase 2 只开放单请求 contiguous KV，其他优化仍启动失败。"""

        self.validate()
        expected = {
            "max_num_seqs": (self.max_num_seqs, 1),
            "kv_backend": (self.kv_backend, "contiguous"),
            "attention_backend": (self.attention_backend, "torch_contiguous"),
            "rmsnorm_backend": (self.rmsnorm_backend, "torch"),
            "decode_execution": (self.decode_execution, "eager"),
        }
        mismatches = [name for name, (actual, wanted) in expected.items() if actual != wanted]
        if mismatches or self.enable_prefix_cache or self.enable_chunked_prefill:
            detail = ", ".join(mismatches) or "future feature flag"
            raise ValueError(f"Phase 2 only implements single-request contiguous KV: {detail}.")

    def validate_phase3(self) -> None:
        self.validate()
        expected = {
            "kv_backend": (self.kv_backend, "contiguous"),
            "attention_backend": (self.attention_backend, "torch_contiguous"),
            "rmsnorm_backend": (self.rmsnorm_backend, "torch"),
            "decode_execution": (self.decode_execution, "eager"),
        }
        mismatches = [name for name, (actual, wanted) in expected.items() if actual != wanted]
        if self.max_num_seqs < 2:
            mismatches.append("max_num_seqs")
        if self.max_pending_requests < self.max_num_seqs:
            mismatches.append("max_pending_requests")
        if mismatches or self.enable_prefix_cache or self.enable_chunked_prefill:
            detail = ", ".join(mismatches) or "future feature flag"
            raise ValueError(f"Phase 3 only implements contiguous-KV continuous batching: {detail}.")

    def validate_phase4(self) -> None:
        self.validate()
        expected = {
            "kv_backend": (self.kv_backend, "contiguous"),
            "attention_backend": (self.attention_backend, "torch_contiguous"),
            "rmsnorm_backend": (self.rmsnorm_backend, "torch"),
            "decode_execution": (self.decode_execution, "eager"),
            "enable_chunked_prefill": (self.enable_chunked_prefill, True),
        }
        mismatches = [name for name, (actual, wanted) in expected.items() if actual != wanted]
        if self.max_num_seqs < 2 or self.max_pending_requests < self.max_num_seqs:
            mismatches.append("multi_request_limits")
        if mismatches or self.enable_prefix_cache:
            detail = ", ".join(mismatches) or "future feature flag"
            raise ValueError(f"Phase 4 only implements contiguous-KV chunked prefill: {detail}.")

    def validate_phase5(self) -> None:
        self.validate()
        expected = {
            "kv_backend": (self.kv_backend, "paged"),
            "attention_backend": (self.attention_backend, "torch_paged_ref"),
            "rmsnorm_backend": (self.rmsnorm_backend, "torch"),
            "decode_execution": (self.decode_execution, "eager"),
            "enable_chunked_prefill": (self.enable_chunked_prefill, True),
        }
        mismatches = [name for name, (actual, wanted) in expected.items() if actual != wanted]
        if self.num_kv_blocks is None or self.num_kv_blocks <= 0:
            mismatches.append("num_kv_blocks")
        if self.max_num_seqs < 2 or self.max_pending_requests < self.max_num_seqs:
            mismatches.append("multi_request_limits")
        if mismatches or self.enable_prefix_cache:
            detail = ", ".join(mismatches) or "future feature flag"
            raise ValueError(f"Phase 5 only implements paged-KV torch reference: {detail}.")

    def validate_phase6(self) -> None:
        self.validate()
        expected = {
            "kv_backend": (self.kv_backend, "paged"),
            "attention_backend": (self.attention_backend, "torch_paged_ref"),
            "enable_chunked_prefill": (self.enable_chunked_prefill, True),
            "enable_prefix_cache": (self.enable_prefix_cache, True),
        }
        mismatches = [name for name, (actual, wanted) in expected.items() if actual != wanted]
        if self.num_kv_blocks is None or self.max_num_seqs < 2:
            mismatches.append("paged_capacity")
        if mismatches:
            raise ValueError(f"Phase 6 requires exact full-block prefix cache: {', '.join(mismatches)}.")

    def validate_phase7(self) -> None:
        """Phase 7 只允许在已验收的 Phase 6 主链上启用 CUDA RMSNorm。"""

        self.validate()
        expected = {
            "kv_backend": (self.kv_backend, "paged"),
            "attention_backend": (self.attention_backend, "torch_paged_ref"),
            "rmsnorm_backend": (self.rmsnorm_backend, "cuda"),
            "decode_execution": (self.decode_execution, "eager"),
            "enable_chunked_prefill": (self.enable_chunked_prefill, True),
            "enable_prefix_cache": (self.enable_prefix_cache, True),
        }
        mismatches = [name for name, (actual, wanted) in expected.items() if actual != wanted]
        if self.num_kv_blocks is None or self.max_num_seqs < 2:
            mismatches.append("paged_capacity")
        if self.max_pending_requests < self.max_num_seqs:
            mismatches.append("max_pending_requests")
        if mismatches:
            raise ValueError(
                "Phase 7 requires the Phase 6 paged/prefix path with CUDA RMSNorm: "
                + ", ".join(mismatches)
                + "."
            )

    def validate_phase8(self) -> None:
        """Phase 8 仅允许替换 decode attention，不改变其他已验收能力。"""

        self.validate()
        expected = {
            "kv_backend": (self.kv_backend, "paged"),
            "attention_backend": (self.attention_backend, "cuda_paged"),
            "rmsnorm_backend": (self.rmsnorm_backend, "cuda"),
            "decode_execution": (self.decode_execution, "eager"),
            "enable_chunked_prefill": (self.enable_chunked_prefill, True),
            "enable_prefix_cache": (self.enable_prefix_cache, True),
        }
        mismatches = [name for name, (actual, wanted) in expected.items() if actual != wanted]
        if self.num_kv_blocks is None or self.max_num_seqs < 2:
            mismatches.append("paged_capacity")
        if self.max_pending_requests < self.max_num_seqs:
            mismatches.append("max_pending_requests")
        if mismatches:
            raise ValueError(
                "Phase 8 requires CUDA paged decode on the Phase 7 path: "
                + ", ".join(mismatches)
                + "."
            )

    def validate_phase9(self) -> None:
        self.validate()
        expected = {
            "kv_backend": (self.kv_backend, "paged"),
            "attention_backend": (self.attention_backend, "cuda_paged"),
            "rmsnorm_backend": (self.rmsnorm_backend, "cuda"),
            "decode_execution": (self.decode_execution, "cuda_graph"),
            "enable_chunked_prefill": (self.enable_chunked_prefill, True),
            "enable_prefix_cache": (self.enable_prefix_cache, True),
        }
        mismatches = [name for name, (actual, wanted) in expected.items() if actual != wanted]
        if self.num_kv_blocks is None or self.max_num_seqs < 2:
            mismatches.append("paged_capacity")
        if self.max_pending_requests < self.max_num_seqs:
            mismatches.append("max_pending_requests")
        if mismatches:
            raise ValueError(
                "Phase 9 requires CUDA Graph on the Phase 8 eager path: "
                + ", ".join(mismatches)
                + "."
            )

    def validate_phase10(self) -> None:
        """Phase 10 不改变 GPU backend 合同，必须继续满足 Phase 9 Gate。"""

        self.validate_phase9()
