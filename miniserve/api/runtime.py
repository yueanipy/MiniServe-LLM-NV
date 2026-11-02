"""FastAPI 可见的服务 facade；EngineCore 仍不知道 HTTP/Pydantic。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from miniserve.engine.async_engine import AsyncEngine
from miniserve.observability.metrics import ServiceMetrics

from .streaming import AsyncResultBridge


@dataclass
class ServiceRuntime:
    engine: AsyncEngine
    tokenizer: object
    model_id: str
    max_model_len: int
    max_pending_requests: int
    vocab_size: int | None = None
    result_queue_size: int = 16
    stop_token_ids: tuple[int, ...] = ()
    default_temperature: float = 0.0
    default_top_p: float = 1.0
    default_top_k: int = 0
    whole_prefill_token_budget: int | None = field(default=None, kw_only=True)
    metrics: ServiceMetrics = field(default_factory=ServiceMetrics)
    _started: bool = field(default=False, init=False)
    _ready: bool = field(default=False, init=False)
    _pending: int = field(default=0, init=False)
    _admission_lock: asyncio.Lock | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not self.model_id or self.max_model_len <= 0 or self.max_pending_requests <= 0:
            raise ValueError("Service runtime limits and model_id must be valid.")
        if self.vocab_size is not None and (
            not isinstance(self.vocab_size, int)
            or isinstance(self.vocab_size, bool)
            or self.vocab_size <= 0
        ):
            raise ValueError("vocab_size must be a positive integer when provided.")
        if any(token < 0 for token in self.stop_token_ids):
            raise ValueError("stop_token_ids must be non-negative.")
        if self.whole_prefill_token_budget is not None and (
            not isinstance(self.whole_prefill_token_budget, int)
            or isinstance(self.whole_prefill_token_budget, bool)
            or not 0 < self.whole_prefill_token_budget <= self.max_model_len
        ):
            raise ValueError("whole_prefill_token_budget must fit the positive context limit.")
        # 复用 Engine 层的权威边界验证，避免 API/Engine 两套不同规则。
        from miniserve.engine.request import SamplingParams

        SamplingParams(
            temperature=self.default_temperature,
            top_p=self.default_top_p,
            top_k=self.default_top_k,
            max_new_tokens=1,
        )
        self.bridge = AsyncResultBridge(self.engine, max_queue_size=self.result_queue_size)

    def start(self) -> None:
        if self._started:
            return
        self.engine.start()
        self._started = True
        self._ready = True

    def validate_sampling_scope(self, request: object) -> None:
        """Check public admission limits before registering a result channel."""

        if self.whole_prefill_token_budget is not None:
            prompt_length = len(getattr(request, "prompt_token_ids"))
            if prompt_length > self.whole_prefill_token_budget:
                raise ValueError(
                    "CHUNKED_PREFILL_DISABLED: prompt length "
                    f"{prompt_length} exceeds max_num_batched_tokens "
                    f"{self.whole_prefill_token_budget}; enable chunked prefill "
                    "or provide a larger explicit budget."
                )
        validator = getattr(self.engine, "validate_sampling_scope", None)
        if callable(validator):
            validator(getattr(request, "sampling_params"))

    def shutdown(self) -> None:
        self._ready = False
        if self._started:
            self.engine.shutdown()
            self._started = False

    @property
    def fatal_error(self) -> BaseException | None:
        """Engine fatal 出口的原始异常；非 fatal、未触发或 engine 无该出口时为 None。"""

        worker = getattr(self.engine, "worker", None)
        return getattr(worker, "fatal_error", None)

    @property
    def ready(self) -> bool:
        if self.fatal_error is not None:
            # 坏 CUDA context 上永不重新就绪；只有干净进程可以重新服务。
            return False
        engine_alive = getattr(self.engine, "is_alive", True)
        if callable(engine_alive):
            engine_alive = engine_alive()
        return self._ready and bool(engine_alive)

    async def try_admit(self) -> bool:
        if self._admission_lock is None:
            self._admission_lock = asyncio.Lock()
        async with self._admission_lock:
            if self._pending >= self.max_pending_requests:
                return False
            self._pending += 1
            return True

    async def release_admission(self) -> None:
        assert self._admission_lock is not None
        async with self._admission_lock:
            if self._pending <= 0:
                raise RuntimeError("Service admission counter underflow.")
            self._pending -= 1

    @property
    def pending_requests(self) -> int:
        return self._pending
