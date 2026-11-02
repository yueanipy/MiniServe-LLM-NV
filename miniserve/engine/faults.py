"""普通可恢复失败与 CUDA context 级 fatal 的分类（M0 冻结）。

契约（R3 3.2）：
- 可恢复失败：KV 预算拒绝、队列满、断开、超时、普通 backend 失败。先中止未提交
  reservation、收敛请求终态、释放该请求资源，后续请求可继续。
- fatal context：CUDA illegal access / device-side assert 等**污染 CUDA context** 的错误。
- fatal state：rollback/commit 不变量失败等引擎状态已不能安全复用的错误。
- 两类 fatal 都必须 fail-stop：停止 admission、readiness=false、CPU 侧终止活动请求、
  退出进程；禁止在不可信状态上继续清理、推理或重用。重新服务只能来自干净进程。
- CUDA OOM 的可恢复性取决于失败位置与上下文状态，M0 不按字符串一律重试：OOM 归为
  可恢复（进程继续、该请求失败），但不会自动重试，并保留原始诊断。

分类只依据异常类型与规范化消息文本；它决定控制流，不替代真实设备证据。
"""

from __future__ import annotations

from enum import Enum


class FaultKind(str, Enum):
    RECOVERABLE = "recoverable"
    FATAL_CONTEXT = "fatal_context"
    FATAL_STATE = "fatal_state"


# 这些子串来自 CUDA runtime/driver 的真实错误文本；命中即视为 context 不可继续使用。
FATAL_MARKERS: tuple[str, ...] = (
    "an illegal memory access was encountered",
    "illegal memory access",
    "device-side assert triggered",
    "device side assert triggered",
    "cuda error: an illegal instruction was encountered",
    "an illegal instruction was encountered",
    "unspecified launch failure",
    "cuda error: unspecified launch failure",
    "context is destroyed",
    "cudaerrorcontextisdestroyed",
    "cuda error: context is destroyed",
    "misaligned address",
    "cuda error: misaligned address",
)

# OOM 单独识别：可恢复但绝不自动重试。
_OOM_MARKERS: tuple[str, ...] = (
    "cuda out of memory",
    "outofmemoryerror",
)


class EngineFatalError(RuntimeError):
    """由 EngineCore 在检测到 context 级错误后重新抛出，触发 Worker fail-stop。"""

    def __init__(self, message: str, *, original: BaseException | None = None) -> None:
        super().__init__(message)
        self.original = original


class EngineStateError(EngineFatalError):
    """不可恢复的引擎状态不变量错误；不是 CUDA context 故障。"""


def _message_chain(exc: BaseException) -> str:
    """把异常及其 __cause__/__context__ 的文本合并，避免只看到包装层。"""

    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__
    return " | ".join(parts).lower()


def classify_fault(exc: BaseException) -> FaultKind:
    """返回 FATAL_CONTEXT、FATAL_STATE 或 RECOVERABLE；无设备交互。"""

    if isinstance(exc, EngineStateError):
        return FaultKind.FATAL_STATE
    if isinstance(exc, EngineFatalError):
        return FaultKind.FATAL_CONTEXT
    text = _message_chain(exc)
    if any(marker in text for marker in FATAL_MARKERS):
        return FaultKind.FATAL_CONTEXT
    return FaultKind.RECOVERABLE


def is_out_of_memory(exc: BaseException) -> bool:
    """识别 CUDA OOM（可恢复但不自动重试）。"""

    text = _message_chain(exc)
    return any(marker in text for marker in _OOM_MARKERS)


def describe_fault(exc: BaseException) -> dict[str, object]:
    """写入证据的稳定描述；不包含设备指针或用户数据。"""

    return {
        "kind": classify_fault(exc).value,
        "out_of_memory": is_out_of_memory(exc),
        "exception_type": type(exc).__name__,
        "message": str(exc)[:2000],
    }
