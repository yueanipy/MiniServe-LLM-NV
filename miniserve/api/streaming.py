"""Worker 线程事件到 bounded asyncio channel，以及增量反分词。"""

from __future__ import annotations

import asyncio
import queue
from dataclasses import dataclass

from miniserve.engine.async_engine import AsyncEngine
from miniserve.engine.events import EngineEvent, EngineEventKind


class SlowConsumerError(RuntimeError):
    """客户端没有及时消费，result channel 已按有界策略关闭。"""


_OVERFLOW = object()


@dataclass
class _ChannelState:
    queue: asyncio.Queue
    subscription_id: int
    overflowed: bool = False


class ResultChannel:
    """单请求 async consumer；关闭操作幂等并移除 publisher listener。"""

    def __init__(self, bridge: "AsyncResultBridge", request_id: str) -> None:
        self._bridge = bridge
        self.request_id = request_id
        self._closed = False

    async def get(self) -> EngineEvent:
        state = self._bridge._state(self.request_id)
        value = await state.queue.get()
        if value is _OVERFLOW:
            raise SlowConsumerError(f"result queue overflowed for {self.request_id}")
        return value

    async def __aiter__(self):
        while True:
            event = await self.get()
            yield event
            if event.kind is EngineEventKind.TERMINAL:
                return

    def close(self, *, cancel_reason: str | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        self._bridge.unregister(self.request_id, cancel_reason=cancel_reason)


class AsyncResultBridge:
    """先注册 channel 后 submit；Worker 回调只安排 event-loop 内的 deliver。"""

    def __init__(self, engine: AsyncEngine, *, max_queue_size: int = 16) -> None:
        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive.")
        self.engine = engine
        self.max_queue_size = max_queue_size
        self._channels: dict[str, _ChannelState] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    def register(self, request_id: str) -> ResultChannel:
        if request_id in self._channels:
            raise KeyError(f"Duplicate result channel: {request_id}")
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise RuntimeError("AsyncResultBridge cannot span multiple event loops.")

        def publish_from_worker(event: EngineEvent) -> None:
            loop.call_soon_threadsafe(self._deliver, event)

        subscription_id = self.engine.subscribe(request_id, publish_from_worker)
        self._channels[request_id] = _ChannelState(
            asyncio.Queue(maxsize=self.max_queue_size), subscription_id
        )
        return ResultChannel(self, request_id)

    def _deliver(self, event: EngineEvent) -> None:
        state = self._channels.get(event.request_id)
        if state is None or state.overflowed:
            return
        if state.queue.full():
            state.overflowed = True
            try:
                self.engine.cancel(event.request_id, reason="slow_consumer")
            except (RuntimeError, queue.Full):
                pass
            while not state.queue.empty():
                state.queue.get_nowait()
            state.queue.put_nowait(_OVERFLOW)
            return
        state.queue.put_nowait(event)

    def unregister(self, request_id: str, *, cancel_reason: str | None = None) -> None:
        state = self._channels.pop(request_id, None)
        if state is None:
            return
        self.engine.unsubscribe(request_id, state.subscription_id)
        if cancel_reason is not None:
            try:
                self.engine.cancel(request_id, reason=cancel_reason)
            except (RuntimeError, queue.Full):
                # 请求可能与 disconnect 同时正常终止；取消和 finalize 都是幂等边界。
                pass

    def _state(self, request_id: str) -> _ChannelState:
        try:
            return self._channels[request_id]
        except KeyError as exc:
            raise RuntimeError(f"Result channel is closed: {request_id}") from exc

    @property
    def active_request_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._channels))


class IncrementalDetokenizer:
    """每次 decode 完整 token 前缀，只发布相对上次的稳定文本增量。"""

    def __init__(self, tokenizer: object) -> None:
        self.tokenizer = tokenizer
        self._token_ids: list[int] = []
        self._emitted = ""

    def _decode(self) -> str:
        return self.tokenizer.decode(
            self._token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def push(self, token_id: int) -> str:
        self._token_ids.append(token_id)
        decoded = self._decode()
        if not decoded.startswith(self._emitted):
            raise RuntimeError("Tokenizer decode is non-monotonic; emitted text cannot be retracted.")
        delta = decoded[len(self._emitted) :]
        # Byte-level token 可能暂时解出 replacement character；等后续 token 补全。
        stable = delta.split("\ufffd", 1)[0]
        self._emitted += stable
        return stable

    def finish(self) -> str:
        decoded = self._decode()
        if not decoded.startswith(self._emitted):
            raise RuntimeError("Tokenizer decode is non-monotonic at stream finish.")
        delta = decoded[len(self._emitted) :]
        self._emitted = decoded
        return delta

    @property
    def token_ids(self) -> tuple[int, ...]:
        return tuple(self._token_ids)
