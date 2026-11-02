"""API/AsyncEngine 到唯一 Worker 的不可变 FIFO commands。"""

from __future__ import annotations

from dataclasses import dataclass

from .request import EngineRequest


@dataclass(frozen=True)
class SubmitCommand:
    request: EngineRequest


@dataclass(frozen=True)
class CancelCommand:
    request_id: str
    reason: str = "user_cancel"

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty.")
        if not self.reason:
            raise ValueError("cancel reason must not be empty.")


@dataclass(frozen=True)
class ShutdownCommand:
    reason: str = "requested"


EngineCommand = SubmitCommand | CancelCommand | ShutdownCommand
