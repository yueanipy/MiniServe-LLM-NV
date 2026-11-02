"""从 HTTP/SSE 客户端视角计算 TTFT、TPOT、ITL 与吞吐。

M0 冻结语义（R3）：
- TTFT 与延迟均为 client 视角：client send 到首个可见内容 / 协议完成（含尾部开销）。
- ``client_tpot_ms`` 是请求级单值 ``(protocol_done - first_content) / (N - 1)``；
  N <= 1 时为 None，绝不用间隔中位数冒充。``client_tpot_last_content_ms`` 是去掉
  协议尾部开销的对照版本；两个客户端版本不能与 engine TPOT 混画。
- ``inter_chunk_ms`` 是 SSE content chunk 到达间隔，不是 token 级 ITL：chunk 数不是
  token 数，空内容 / role-only / usage 帧不计入。engine 侧 TTFT / TPOT / ITL 由服务
  自身时间线提供（``miniserve.observability.metrics``），两端分开报告。
- 无可信 usage 时 token 计数为 None 且 ``usage_source="unavailable"``；聚合必须把该
  请求记为不可比较，而不是按 0 计入或静默丢弃。截断流（无 ``[DONE]``）显式标记。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Mapping

_USAGE_SOURCES = ("usage", "tokenizer_recount", "unavailable")


@dataclass(frozen=True)
class ServiceObservation:
    """一条真实 HTTP/SSE 请求的客户端观测。

    前 11 个位置参数保持既有顺序以兼容历史调用；时间线字段为关键字默认值。
    """

    workload: str
    request_id: str
    status_code: int
    ttft_ms: float | None
    latency_ms: float
    inter_chunk_ms: tuple[float, ...]
    prompt_tokens: int | None
    completion_tokens: int | None
    finish_reason: str
    text: str
    # SSE 传输字节数；并发账本用它计算 wire throughput，不参与单请求延迟。
    download_bytes: int = 0
    # client 时间线（相对 client send 的毫秒值）。
    last_content_ms: float | None = None
    server_terminal_ms: float | None = None
    protocol_done_ms: float | None = None
    truncated: bool = False
    usage_source: str = "usage"
    content_chunk_count: int = 0
    terminal_chunk_count: int = 0

    def __post_init__(self) -> None:
        if not self.workload or not self.request_id:
            raise ValueError("Observation workload and request_id must not be empty.")
        if not isinstance(self.status_code, int) or isinstance(self.status_code, bool) or not 100 <= self.status_code <= 599:
            raise ValueError("Observation status_code must be a valid integer HTTP status.")
        if not math.isfinite(self.latency_ms) or self.latency_ms < 0:
            raise ValueError("Observation latency_ms must be finite and non-negative.")
        if self.ttft_ms is not None and (
            not math.isfinite(self.ttft_ms) or self.ttft_ms < 0 or self.ttft_ms > self.latency_ms
        ):
            raise ValueError("Observation ttft_ms must be finite, non-negative and within latency.")
        if any(not math.isfinite(value) or value < 0 for value in self.inter_chunk_ms):
            raise ValueError("Observation inter-chunk intervals must be finite and non-negative.")
        for name in ("prompt_tokens", "completion_tokens"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
                raise ValueError(f"Observation {name} must be a non-negative integer or None.")
        if (self.prompt_tokens is None) != (self.completion_tokens is None):
            raise ValueError("Observation token counts must be both known or both unknown.")
        if not isinstance(self.finish_reason, str) or not self.finish_reason:
            raise ValueError("Observation finish_reason must be a non-empty string.")
        if not isinstance(self.text, str):
            raise TypeError("Observation text must be a string.")
        if (
            not isinstance(self.download_bytes, int)
            or isinstance(self.download_bytes, bool)
            or self.download_bytes < 0
        ):
            raise ValueError("Observation download_bytes must be a non-negative integer.")
        for name in ("last_content_ms", "server_terminal_ms", "protocol_done_ms"):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Observation {name} must be finite and non-negative or None.")
            if value > self.latency_ms:
                raise ValueError(f"Observation {name} cannot exceed measured latency.")
        if self.ttft_ms is not None and self.last_content_ms is not None and self.last_content_ms < self.ttft_ms:
            raise ValueError("Observation last_content_ms cannot precede first content.")
        if not isinstance(self.truncated, bool):
            raise TypeError("Observation truncated must be a bool.")
        if self.truncated:
            if self.protocol_done_ms is not None:
                raise ValueError("A truncated stream must not claim a protocol-done timestamp.")
        elif self.protocol_done_ms is None:
            raise ValueError("A non-truncated observation requires protocol_done_ms.")
        if self.usage_source not in _USAGE_SOURCES:
            raise ValueError("Observation usage_source must be usage, tokenizer_recount or unavailable.")
        if self.usage_source == "unavailable" and self.completion_tokens is not None:
            raise ValueError("Unavailable usage must not carry token counts.")
        for name in ("content_chunk_count", "terminal_chunk_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"Observation {name} must be a non-negative integer.")
        if self.content_chunk_count == 0 and self.ttft_ms is not None:
            raise ValueError("Observation without content chunks must not report a first-content time.")
        if self.content_chunk_count > 0 and self.ttft_ms is None:
            raise ValueError("Observation with content chunks must report a first-content time.")

    @property
    def client_tpot_ms(self) -> float | None:
        """请求级 client TPOT：``(protocol_done - first_content) / (N - 1)``。"""

        if self.completion_tokens is None or self.completion_tokens < 2:
            return None
        if self.ttft_ms is None or self.protocol_done_ms is None:
            return None
        return (self.protocol_done_ms - self.ttft_ms) / (self.completion_tokens - 1)

    @property
    def client_tpot_last_content_ms(self) -> float | None:
        """去协议尾部的对照版本：``(last_content - first_content) / (N - 1)``。"""

        if self.completion_tokens is None or self.completion_tokens < 2:
            return None
        if self.ttft_ms is None or self.last_content_ms is None:
            return None
        return (self.last_content_ms - self.ttft_ms) / (self.completion_tokens - 1)

    @property
    def tpot_ms(self) -> float | None:
        """历史名称，现在唯一指向请求级 ``client_tpot_ms``（不再返回间隔中位数）。"""

        return self.client_tpot_ms

    @property
    def protocol_tail_ms(self) -> float | None:
        """协议完成到测量结束之间的尾部开销（[DONE]/连接关闭）。"""

        if self.protocol_done_ms is None:
            return None
        return self.latency_ms - self.protocol_done_ms

    @property
    def output_tokens_per_second(self) -> float | None:
        """client 视角生成速率；token 数不可信时为 None。"""

        if self.completion_tokens is None or self.ttft_ms is None:
            return None
        generation_ms = max(self.latency_ms - self.ttft_ms, 0.001)
        return self.completion_tokens / (generation_ms / 1000)

    @property
    def comparable(self) -> bool:
        """usage 可信且非截断时才适合进入跨服务口径比较。"""

        return not self.truncated and self.usage_source != "unavailable"

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ServiceObservation":
        """从持久化的 ``as_dict`` 结构重建观测；推导字段被忽略。"""

        names = {field.name for field in fields(cls)}
        kwargs = {key: value for key, value in payload.items() if key in names}
        if isinstance(kwargs.get("inter_chunk_ms"), list):
            kwargs["inter_chunk_ms"] = tuple(kwargs["inter_chunk_ms"])
        return cls(**kwargs)

    def as_dict(self) -> dict[str, object]:
        return {
            "workload": self.workload,
            "request_id": self.request_id,
            "status_code": self.status_code,
            "ttft_ms": self.ttft_ms,
            "latency_ms": self.latency_ms,
            "inter_chunk_ms": list(self.inter_chunk_ms),
            "last_content_ms": self.last_content_ms,
            "server_terminal_ms": self.server_terminal_ms,
            "protocol_done_ms": self.protocol_done_ms,
            "truncated": self.truncated,
            "usage_source": self.usage_source,
            "content_chunk_count": self.content_chunk_count,
            "terminal_chunk_count": self.terminal_chunk_count,
            "client_tpot_ms": self.client_tpot_ms,
            "client_tpot_last_content_ms": self.client_tpot_last_content_ms,
            "tpot_ms": self.tpot_ms,
            "protocol_tail_ms": self.protocol_tail_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "output_tokens_per_second": self.output_tokens_per_second,
            "finish_reason": self.finish_reason,
            "text": self.text,
            "download_bytes": self.download_bytes,
        }