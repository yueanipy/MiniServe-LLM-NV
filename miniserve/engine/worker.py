"""唯一 EngineCore 执行线程与有界 FIFO command bridge。"""

from __future__ import annotations

import inspect
import os
import queue
import threading
import time
from typing import TYPE_CHECKING

from .commands import CancelCommand, EngineCommand, ShutdownCommand, SubmitCommand
from .events import EngineEvent, EngineEventKind
from .state import RequestStatus

if TYPE_CHECKING:
    from .core import EngineCore


# 进程级 fail-stop 退出码：区分于普通异常退出，便于服务管理器与证据核对。
FATAL_EXIT_CODE = 70


class EngineWorker:
    """Phase 3 reference worker；GPU forward 只在 ``run`` 线程调用。"""

    def __init__(
        self,
        core: EngineCore,
        *,
        max_commands: int,
        idle_wait_seconds: float = 0.002,
        exit_on_fatal: bool = False,
    ) -> None:
        if max_commands <= 0 or idle_wait_seconds <= 0:
            raise ValueError("Worker queue size and idle wait must be positive.")
        if not isinstance(exit_on_fatal, bool):
            raise TypeError("exit_on_fatal must be a bool.")
        # 仅正式服务进程启用：坏 CUDA context 必须由进程退出收敛，不能继续服务。
        self._exit_on_fatal = exit_on_fatal
        self.core = core
        self._commands: queue.Queue[EngineCommand] = queue.Queue(maxsize=max_commands)
        self._wake = threading.Event()
        self._started = threading.Event()
        self._stopped = threading.Event()
        self._shutdown_requested = False
        # 命令接收状态与入队动作必须由同一把锁保护。否则 ShutdownCommand 已经
        # 入队、worker 即将退出时，另一个线程仍可能把永远不会处理的请求放进队列。
        self._lifecycle_lock = threading.Lock()
        self._accepting_commands = True
        self._fatal_error: BaseException | None = None
        cancel_method = getattr(core, "cancel", None)
        self._cancel_supports_reason = callable(cancel_method) and (
            "reason" in inspect.signature(cancel_method).parameters
        )
        self._idle_wait_seconds = idle_wait_seconds
        self._thread = threading.Thread(target=self.run, name="miniserve-engine-worker", daemon=False)

    def start(self) -> None:
        if self._thread.is_alive() or self._started.is_set():
            raise RuntimeError("EngineWorker can only be started once.")
        self._thread.start()
        if not self._started.wait(timeout=5):
            raise TimeoutError("EngineWorker did not start.")

    def submit_command(self, command: EngineCommand, *, block: bool = False) -> None:
        while True:
            with self._lifecycle_lock:
                if not self._accepting_commands or self._stopped.is_set():
                    raise RuntimeError("EngineWorker is shutting down or stopped.")
                try:
                    # 所有入队都在生命周期锁内线性化；队列满时绝不拿着该锁等待，
                    # 否则 worker fatal 出口也可能被一个阻塞提交卡住。
                    self._commands.put_nowait(command)
                except queue.Full:
                    if not block:
                        raise
                else:
                    if isinstance(command, ShutdownCommand):
                        self._accepting_commands = False
                    break
            self._wake.set()
            self._stopped.wait(self._idle_wait_seconds)
        self._wake.set()

    def run(self) -> None:
        self._started.set()
        try:
            while not self._shutdown_requested:
                drained = self._drain_commands()
                if self._shutdown_requested:
                    break
                progressed = bool(self.core.step_multi())
                if not drained and not progressed:
                    self._wake.wait(self._idle_wait_seconds)
                    self._wake.clear()
            # Shutdown 统一取消并释放所有尚未完成请求。
            for sequence in tuple(self.core.store.active()):
                self._cancel_core(sequence.request_id, "shutdown")
        except BaseException as exc:
            self._fatal_error = exc
            # 先阻止新 command 入队，再让 active/尚未 admission 的请求都得到明确终态。
            # fatal CUDA 后不尝试继续释放/复用 GPU 状态；进程级重启负责物理恢复。
            with self._lifecycle_lock:
                self._accepting_commands = False
            self._stopped.set()
            self._publish_fatal_terminals(exc)
            if self._exit_on_fatal:
                # fail-stop：readiness 已随线程停止为 false；直接终止进程，绝不在坏
                # context 上继续 GPU 清理、推理或重用。测试默认不启用该出口。
                os._exit(FATAL_EXIT_CODE)
        finally:
            with self._lifecycle_lock:
                self._accepting_commands = False
            self._stopped.set()

    def _publish_fatal_terminals(self, exc: BaseException) -> None:
        reason = f"fatal_worker: {type(exc).__name__}: {exc}"
        # NV-ERR-03（M0 真实链探针发现）：EngineCore 在 fatal 路径上已把触发请求标为
        # FAILED，因此不能只遍历 store.active()——否则该请求永远收不到终态事件，API
        # 侧会一直等到超时。这里覆盖所有尚未发布终态的序列（活动 + 已终态未发布）。
        store = self.core.store
        sequences = tuple(store.active()) + tuple(store.terminal())
        request_ids = {sequence.request_id for sequence in sequences}
        # _drain_commands 可能在一条坏 command 中途失败；剩余 SubmitCommand 也必须收敛。
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                break
            try:
                if isinstance(command, SubmitCommand):
                    request_ids.add(command.request.request_id)
            finally:
                self._commands.task_done()
        for request_id in sorted(request_ids):
            if self.core.publisher.is_terminal(request_id):
                continue
            self.core.publisher.publish(
                EngineEvent(
                    request_id,
                    EngineEventKind.TERMINAL,
                    status=RequestStatus.FAILED,
                    reason=reason,
                    timestamp_ns=time.monotonic_ns(),
                )
            )

    def _drain_commands(self) -> bool:
        drained = False
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                break
            drained = True
            try:
                if isinstance(command, SubmitCommand):
                    try:
                        self.core.submit(command.request)
                    except Exception as exc:
                        self.core.publisher.publish(
                            EngineEvent(
                                command.request.request_id,
                                EngineEventKind.TERMINAL,
                                status=RequestStatus.FAILED,
                                reason=f"admission: {type(exc).__name__}: {exc}",
                                timestamp_ns=time.monotonic_ns(),
                            )
                        )
                elif isinstance(command, CancelCommand):
                    # Phase 3～9 的学习合同只有 request_id；Phase 10 才增加 reason。
                    # Worker 兼容旧命令，但仍把 Phase 10 的显式原因原样向下传递。
                    cancel_reason = getattr(command, "reason", "cancelled")
                    if self.core.store.contains(command.request_id):
                        self._cancel_core(command.request_id, cancel_reason)
                    elif not self.core.publisher.is_terminal(command.request_id):
                        self.core.publisher.publish(
                            EngineEvent(
                                command.request_id,
                                EngineEventKind.TERMINAL,
                                status=RequestStatus.CANCELLED,
                                reason=cancel_reason,
                                timestamp_ns=time.monotonic_ns(),
                            )
                        )
                elif isinstance(command, ShutdownCommand):
                    self._shutdown_requested = True
                else:
                    raise TypeError(f"Unknown EngineCommand: {type(command).__name__}")
            finally:
                self._commands.task_done()
        return drained

    def _cancel_core(self, request_id: str, reason: str) -> None:
        if self._cancel_supports_reason:
            self.core.cancel(request_id, reason=reason)
        else:
            self.core.cancel(request_id)

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise TimeoutError("EngineWorker did not stop.")
        if self._fatal_error is not None:
            raise RuntimeError("EngineWorker failed.") from self._fatal_error

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    @property
    def fatal_error(self) -> BaseException | None:
        """fatal 出口保存的原始异常；供 API readiness/诊断读取。"""

        return self._fatal_error

    @property
    def is_fatal(self) -> bool:
        return self._fatal_error is not None
