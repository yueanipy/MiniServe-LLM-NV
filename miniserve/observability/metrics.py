"""低基数、线程安全的服务时间线指标。"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field

from miniserve.engine.events import EngineEvent, EngineEventKind


@dataclass
class _RequestTimeline:
    arrival_ns: int
    prompt_tokens: int = 0
    first_token_ns: int | None = None
    finish_ns: int | None = None
    output_tokens: int = 0
    finish_reason: str | None = None
    token_timestamps_ns: list[int] = field(default_factory=list)


class ServiceMetrics:
    """保存活动请求和最近终态时间线；不在每 token 做 CUDA synchronize。"""

    def __init__(self, *, max_completed_requests: int = 1024) -> None:
        if max_completed_requests <= 0:
            raise ValueError("max_completed_requests must be positive.")
        self._lock = threading.Lock()
        self._max_completed_requests = max_completed_requests
        self._requests: OrderedDict[str, _RequestTimeline] = OrderedDict()
        self._completed_ids: OrderedDict[str, None] = OrderedDict()
        # 下面都是常数空间累计值。即使最近请求时间线被淘汰，服务级计数仍可观察。
        self._requests_started_total = 0
        self._terminal_totals = {
            "finished": 0,
            "failed": 0,
            "cancelled": 0,
            "aborted": 0,
        }
        self._prompt_tokens_total = 0
        self._output_tokens_total = 0
        self._ttft_ms_sum = 0.0
        self._ttft_count = 0
        self._latency_ms_sum = 0.0
        self._latency_count = 0
        self._itl_ms_sum = 0.0
        self._itl_count = 0
        # M0：请求级 engine TPOT = (last_engine_token - first_engine_token)/(N-1)，
        # 每请求一个值；N<=1 不产生样本（绝不以 0 填充）。
        self._engine_tpot_ms_sum = 0.0
        self._engine_tpot_count = 0
        self._invalid_timestamp_total = 0

    def request_started(
        self,
        request_id: str,
        arrival_ns: int,
        *,
        prompt_tokens: int = 0,
    ) -> None:
        with self._lock:
            if not request_id:
                raise ValueError("request_id must not be empty.")
            if request_id in self._requests:
                raise KeyError(f"Duplicate metrics request_id: {request_id}")
            if not isinstance(arrival_ns, int) or isinstance(arrival_ns, bool) or arrival_ns < 0:
                raise ValueError("arrival_ns must be a non-negative integer.")
            if (
                not isinstance(prompt_tokens, int)
                or isinstance(prompt_tokens, bool)
                or prompt_tokens < 0
            ):
                raise ValueError("prompt_tokens must be a non-negative integer.")
            self._requests[request_id] = _RequestTimeline(
                arrival_ns, prompt_tokens=prompt_tokens
            )
            self._requests_started_total += 1
            self._prompt_tokens_total += prompt_tokens

    def observe_event(self, event: EngineEvent) -> None:
        with self._lock:
            timeline = self._requests.get(event.request_id)
            if timeline is None:
                return
            if event.kind is EngineEventKind.TOKEN:
                if timeline.finish_ns is not None:
                    return
                previous_ns = (
                    timeline.token_timestamps_ns[-1]
                    if timeline.token_timestamps_ns
                    else timeline.arrival_ns
                )
                if event.timestamp_ns < previous_ns:
                    # 指标异常不能杀死服务，也不能用负耗时污染汇总。
                    self._invalid_timestamp_total += 1
                    return
                # 只有通过时间单调性校验的事件才属于可观察输出，避免坏事件污染
                # request snapshot 与服务级 counter。
                timeline.output_tokens += 1
                self._output_tokens_total += 1
                timeline.token_timestamps_ns.append(event.timestamp_ns)
                if timeline.first_token_ns is None:
                    timeline.first_token_ns = event.timestamp_ns
                    self._ttft_ms_sum += (
                        event.timestamp_ns - timeline.arrival_ns
                    ) / 1_000_000
                    self._ttft_count += 1
                elif len(timeline.token_timestamps_ns) >= 2:
                    self._itl_ms_sum += (
                        timeline.token_timestamps_ns[-1]
                        - timeline.token_timestamps_ns[-2]
                    ) / 1_000_000
                    self._itl_count += 1
            else:
                if timeline.finish_ns is not None:
                    return
                timeline.finish_ns = event.timestamp_ns
                timeline.finish_reason = event.reason
                self._record_terminal_locked(timeline, event.status)
                self._completed_ids[event.request_id] = None
                self._completed_ids.move_to_end(event.request_id)
                self._trim_completed_locked()

    def request_aborted(self, request_id: str, timestamp_ns: int, reason: str) -> None:
        """在 listener 已移除、无法再观察 terminal 时收敛本地时间线。"""

        if not isinstance(timestamp_ns, int) or isinstance(timestamp_ns, bool) or timestamp_ns < 0:
            raise ValueError("timestamp_ns must be a non-negative integer.")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("abort reason must not be empty.")
        with self._lock:
            timeline = self._requests.get(request_id)
            if timeline is None or timeline.finish_ns is not None:
                return
            timeline.finish_ns = timestamp_ns
            timeline.finish_reason = reason
            self._record_terminal_locked(timeline, None)
            self._completed_ids[request_id] = None
            self._completed_ids.move_to_end(request_id)
            self._trim_completed_locked()

    def snapshot(self, request_id: str) -> dict[str, object]:
        with self._lock:
            timeline = self._requests[request_id]
            timestamps = tuple(timeline.token_timestamps_ns)
            itl = [
                (right - left) / 1_000_000
                for left, right in zip(timestamps, timestamps[1:])
            ]
            return {
                "request_id": request_id,
                "arrival_ns": timeline.arrival_ns,
                "prompt_tokens": timeline.prompt_tokens,
                "first_token_ns": timeline.first_token_ns,
                "finish_ns": timeline.finish_ns,
                "ttft_ms": (
                    None
                    if timeline.first_token_ns is None
                    else (timeline.first_token_ns - timeline.arrival_ns) / 1_000_000
                ),
                "latency_ms": (
                    None
                    if timeline.finish_ns is None
                    else (timeline.finish_ns - timeline.arrival_ns) / 1_000_000
                ),
                "itl_ms": itl,
                "engine_tpot_ms": (
                    None
                    if len(timestamps) < 2
                    else (timestamps[-1] - timestamps[0]) / 1_000_000 / (len(timestamps) - 1)
                ),
                "output_tokens": timeline.output_tokens,
                "finish_reason": timeline.finish_reason,
            }

    def summary(self) -> dict[str, object]:
        """返回常数空间服务汇总；不会暴露 request_id 等高基数标签。"""

        with self._lock:
            active = sum(
                timeline.finish_ns is None for timeline in self._requests.values()
            )
            return {
                "requests_started_total": self._requests_started_total,
                "requests_terminal_total": dict(self._terminal_totals),
                "requests_active": active,
                "prompt_tokens_total": self._prompt_tokens_total,
                "output_tokens_total": self._output_tokens_total,
                "ttft_ms_sum": self._ttft_ms_sum,
                "ttft_count": self._ttft_count,
                "latency_ms_sum": self._latency_ms_sum,
                "latency_count": self._latency_count,
                "itl_ms_sum": self._itl_ms_sum,
                "itl_count": self._itl_count,
                "engine_tpot_ms_sum": self._engine_tpot_ms_sum,
                "engine_tpot_count": self._engine_tpot_count,
                "invalid_timestamp_total": self._invalid_timestamp_total,
                "retained_requests": len(self._requests),
            }

    def render_prometheus(self, *, pending_requests: int, ready: bool) -> str:
        """生成无需额外依赖的 Prometheus 0.0.4 文本。"""

        if (
            not isinstance(pending_requests, int)
            or isinstance(pending_requests, bool)
            or pending_requests < 0
        ):
            raise ValueError("pending_requests must be a non-negative integer.")
        if not isinstance(ready, bool):
            raise TypeError("ready must be a bool.")
        values = self.summary()
        terminal = values["requests_terminal_total"]
        assert isinstance(terminal, dict)
        lines = [
            "# HELP miniserve_ready Whether the serving runtime is ready.",
            "# TYPE miniserve_ready gauge",
            f"miniserve_ready {1 if ready else 0}",
            "# HELP miniserve_requests_active Requests admitted without a terminal outcome.",
            "# TYPE miniserve_requests_active gauge",
            f"miniserve_requests_active {values['requests_active']}",
            "# HELP miniserve_requests_pending Requests holding API admission capacity.",
            "# TYPE miniserve_requests_pending gauge",
            f"miniserve_requests_pending {pending_requests}",
            "# HELP miniserve_requests_started_total Requests admitted since process start.",
            "# TYPE miniserve_requests_started_total counter",
            f"miniserve_requests_started_total {values['requests_started_total']}",
            "# HELP miniserve_requests_terminal_total Terminal outcomes since process start.",
            "# TYPE miniserve_requests_terminal_total counter",
        ]
        lines.extend(
            f'miniserve_requests_terminal_total{{outcome="{outcome}"}} {terminal[outcome]}'
            for outcome in ("finished", "failed", "cancelled", "aborted")
        )
        lines.extend(
            [
                "# TYPE miniserve_prompt_tokens_total counter",
                f"miniserve_prompt_tokens_total {values['prompt_tokens_total']}",
                "# TYPE miniserve_output_tokens_total counter",
                f"miniserve_output_tokens_total {values['output_tokens_total']}",
                "# TYPE miniserve_request_ttft_milliseconds summary",
                f"miniserve_request_ttft_milliseconds_sum {values['ttft_ms_sum']}",
                f"miniserve_request_ttft_milliseconds_count {values['ttft_count']}",
                "# TYPE miniserve_request_latency_milliseconds summary",
                f"miniserve_request_latency_milliseconds_sum {values['latency_ms_sum']}",
                f"miniserve_request_latency_milliseconds_count {values['latency_count']}",
                "# TYPE miniserve_inter_token_latency_milliseconds summary",
                f"miniserve_inter_token_latency_milliseconds_sum {values['itl_ms_sum']}",
                f"miniserve_inter_token_latency_milliseconds_count {values['itl_count']}",
                "# HELP miniserve_engine_request_tpot_milliseconds Request-level engine TPOT (N>1 only).",
                "# TYPE miniserve_engine_request_tpot_milliseconds summary",
                f"miniserve_engine_request_tpot_milliseconds_sum {values['engine_tpot_ms_sum']}",
                f"miniserve_engine_request_tpot_milliseconds_count {values['engine_tpot_count']}",
                "# TYPE miniserve_metrics_retained_requests gauge",
                f"miniserve_metrics_retained_requests {values['retained_requests']}",
                "# TYPE miniserve_metrics_invalid_timestamp_total counter",
                f"miniserve_metrics_invalid_timestamp_total {values['invalid_timestamp_total']}",
            ]
        )
        return "\n".join(lines) + "\n"

    @property
    def retained_request_count(self) -> int:
        with self._lock:
            return len(self._requests)

    def _trim_completed_locked(self) -> None:
        while len(self._completed_ids) > self._max_completed_requests:
            request_id, _ = self._completed_ids.popitem(last=False)
            self._requests.pop(request_id, None)

    def _record_terminal_locked(
        self,
        timeline: _RequestTimeline,
        status,
    ) -> None:
        # 请求级 engine TPOT：只在 N>=2 时产生一个样本（N<=1 为 null 不计入）。
        timestamps = timeline.token_timestamps_ns
        if len(timestamps) >= 2:
            self._engine_tpot_ms_sum += (
                (timestamps[-1] - timestamps[0]) / 1_000_000 / (len(timestamps) - 1)
            )
            self._engine_tpot_count += 1
        if status is None:
            outcome = "aborted"
        elif status.name == "FINISHED":
            outcome = "finished"
        elif status.name == "FAILED":
            outcome = "failed"
        elif status.name == "CANCELLED":
            outcome = "cancelled"
        else:  # EngineEvent 已拒绝非终态；这里保留防御式约束。
            raise ValueError("Metrics received a non-terminal request status.")
        self._terminal_totals[outcome] += 1
        if timeline.finish_ns is not None and timeline.finish_ns >= timeline.arrival_ns:
            self._latency_ms_sum += (
                timeline.finish_ns - timeline.arrival_ns
            ) / 1_000_000
            self._latency_count += 1
        else:
            self._invalid_timestamp_total += 1
