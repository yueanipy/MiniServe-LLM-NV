"""线程安全、terminal exactly-once 的 Engine 结果发布器。"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable

from .state import RequestStatus


class EngineEventKind(Enum):
    TOKEN = auto()
    TERMINAL = auto()


@dataclass(frozen=True)
class EngineEvent:
    request_id: str
    kind: EngineEventKind
    token_id: int | None = None
    status: RequestStatus | None = None
    reason: str | None = None
    timestamp_ns: int = 0

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty.")
        if self.kind is EngineEventKind.TOKEN:
            if self.token_id is None or self.status is not None:
                raise ValueError("TOKEN event requires token_id and no terminal status.")
        elif self.status not in {RequestStatus.FINISHED, RequestStatus.FAILED, RequestStatus.CANCELLED}:
            raise ValueError("TERMINAL event requires a terminal request status.")


class ResultPublisher:
    """发布事件并只保留最近的终态请求，避免服务历史无界增长。

    活动请求及仍有 listener 的终态请求不会被淘汰。``max_retained_requests``
    是已完成且已取消订阅请求的查询/去重窗口，不是最大并发数。
    """

    def __init__(
        self,
        *,
        max_retained_requests: int = 1024,
        max_listener_failures: int = 128,
    ) -> None:
        if max_retained_requests <= 0 or max_listener_failures <= 0:
            raise ValueError("ResultPublisher retention limits must be positive.")
        self._condition = threading.Condition()
        self._events: dict[str, list[EngineEvent]] = {}
        self._terminal_ids: set[str] = set()
        self._terminal_order: deque[str] = deque()
        self._listeners: dict[str, dict[int, Callable[[EngineEvent], None]]] = {}
        self._next_listener_id = 1
        self._max_retained_requests = max_retained_requests
        self._listener_failures: deque[str] = deque(maxlen=max_listener_failures)

    def publish(self, event: EngineEvent) -> None:
        with self._condition:
            if event.request_id in self._terminal_ids:
                raise RuntimeError(f"Cannot publish after terminal event: {event.request_id}")
            if event.kind is EngineEventKind.TERMINAL:
                self._terminal_ids.add(event.request_id)
                self._terminal_order.append(event.request_id)
            self._events.setdefault(event.request_id, []).append(event)
            listeners = tuple(self._listeners.get(event.request_id, {}).values())
            self._trim_terminal_history_locked()
            self._condition.notify_all()
        # 回调可能使用 loop.call_soon_threadsafe；绝不能在 Condition 锁内调用。
        for listener in listeners:
            try:
                listener(event)
            except Exception as exc:
                # API loop 关闭竞态不能杀死唯一 Engine Worker，但必须保留可见诊断。
                with self._condition:
                    self._listener_failures.append(f"{type(exc).__name__}: {exc}")

    def subscribe(self, request_id: str, listener: Callable[[EngineEvent], None]) -> int:
        if not request_id:
            raise ValueError("request_id must not be empty.")
        if not callable(listener):
            raise TypeError("listener must be callable.")
        with self._condition:
            if request_id in self._terminal_ids:
                raise RuntimeError("Cannot subscribe after terminal event.")
            subscription_id = self._next_listener_id
            self._next_listener_id += 1
            self._listeners.setdefault(request_id, {})[subscription_id] = listener
            return subscription_id

    def unsubscribe(self, request_id: str, subscription_id: int) -> None:
        with self._condition:
            listeners = self._listeners.get(request_id)
            if listeners is None:
                return
            listeners.pop(subscription_id, None)
            if not listeners:
                self._listeners.pop(request_id, None)
                self._trim_terminal_history_locked()

    def listener_count(self, request_id: str) -> int:
        with self._condition:
            return len(self._listeners.get(request_id, ()))

    @property
    def listener_failures(self) -> tuple[str, ...]:
        with self._condition:
            return tuple(self._listener_failures)

    def events(self, request_id: str) -> tuple[EngineEvent, ...]:
        with self._condition:
            return tuple(self._events.get(request_id, ()))

    def wait_terminal(self, request_id: str, timeout: float | None = None) -> tuple[EngineEvent, ...]:
        with self._condition:
            reached = self._condition.wait_for(lambda: request_id in self._terminal_ids, timeout=timeout)
            if not reached:
                raise TimeoutError(f"Timed out waiting for terminal event: {request_id}")
            return tuple(self._events[request_id])

    def is_terminal(self, request_id: str) -> bool:
        with self._condition:
            return request_id in self._terminal_ids

    @property
    def retained_request_count(self) -> int:
        """返回当前保留事件的 request 数，供长稳测试和诊断使用。"""

        with self._condition:
            return len(self._events)

    def _trim_terminal_history_locked(self) -> None:
        """在 Condition 锁内淘汰最旧且已无 listener 的终态历史。

        若最旧请求的 API consumer 尚未取消订阅，允许暂时超过历史上限；这部分
        额外占用受服务最大并发约束。不能越过它去删除更新的 terminal，否则刚完成
        请求可能在调用 ``wait_terminal`` 前就丢失结果。
        """

        while len(self._terminal_order) > self._max_retained_requests:
            request_id = self._terminal_order[0]
            if self._listeners.get(request_id):
                break
            self._terminal_order.popleft()
            self._terminal_ids.remove(request_id)
            self._events.pop(request_id, None)
