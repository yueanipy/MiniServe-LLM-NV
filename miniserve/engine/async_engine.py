"""Phase 3 的有界 command/result facade；不向外暴露 mutable EngineCore。"""

from __future__ import annotations

from .commands import CancelCommand, ShutdownCommand, SubmitCommand
from .events import EngineEvent
from .request import EngineRequest, SamplingParams
from .worker import EngineWorker


class AsyncEngine:
    def __init__(self, worker: EngineWorker) -> None:
        self.worker = worker

    def start(self) -> None:
        self.worker.start()

    def submit(self, request: EngineRequest) -> None:
        self.worker.submit_command(SubmitCommand(request))

    def validate_sampling_scope(self, sampling_params: SamplingParams) -> None:
        """Use the core's public request-scope check before enqueueing."""

        self.worker.core.validate_sampling_scope(sampling_params)

    def cancel(self, request_id: str, *, reason: str = "user_cancel") -> None:
        self.worker.submit_command(CancelCommand(request_id, reason))

    def subscribe(self, request_id: str, callback) -> int:
        """注册 Worker→API 事件回调；API 无需接触 mutable EngineCore。"""

        return self.worker.core.publisher.subscribe(request_id, callback)

    def unsubscribe(self, request_id: str, subscription_id: int) -> None:
        self.worker.core.publisher.unsubscribe(request_id, subscription_id)

    def wait(self, request_id: str, *, timeout: float = 30.0) -> tuple[EngineEvent, ...]:
        return self.worker.core.publisher.wait_terminal(request_id, timeout=timeout)

    def shutdown(self, *, timeout: float = 10.0) -> None:
        if self.worker.is_alive:
            self.worker.submit_command(ShutdownCommand())
        self.worker.join(timeout)

    @property
    def is_alive(self) -> bool:
        """供 readiness 查询；不暴露 mutable worker/core。"""

        return self.worker.is_alive

    @property
    def fatal_error(self) -> BaseException | None:
        return self.worker.fatal_error
