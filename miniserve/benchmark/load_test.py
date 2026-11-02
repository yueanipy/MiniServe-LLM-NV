"""并发负载：burst、bounded closed-loop 与 open-loop 到达曲线（M0 revision 2）。

关键语义（owner D2）：
- 总请求期限与 HTTP inactivity timeout 分开：``request_deadline_s`` 默认 120s；
  open-loop 从 scheduled arrival 起算，closed-loop 从实际 send 起算。未显式提供时
  回退到 ``timeout_seconds``（保持历史调用/探针兼容）。
- 固定窗口 ``[start,end)``；窗口结束后不再创建新请求。drain 最多 ``drain_timeout_s``
  （默认等于 120s），且不超过各请求自身 deadline。到界取消并 await 所有任务。
- 每个请求恰好一个互斥终态：success / truncated / failed / timeout / cancelled /
  never_sent / unresolved；``incomparable``（缺 usage）是正交属性，不重复计数。
- CLI 保存 manifest + 聚合 + Goodput + 逐请求 sidecar，可离线重算同一结果。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import random
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import httpx

from .aggregate import (
    AggregateResult,
    LatencySummary,
    VramSample,
    VramTrace,
    build_aggregate,
    format_nvidia_smi_query,
    format_table,
    parse_nvidia_smi_query,
    percentiles,
)
from .goodput import DEFAULT_SLO_GRID, GoodputReport, SloGrid, evaluate_goodput, format_goodput_table
from .metrics import ServiceObservation
from .runner import run_one
from .workloads import ServiceWorkload

DEFAULT_CONCURRENCY: tuple[int, ...] = (1, 4, 16, 64)
DEFAULT_ALLOWED_SEND_LAG_MS = 50.0
DEFAULT_REQUEST_DEADLINE_S = 120.0
DEFAULT_DRAIN_TIMEOUT_S = 120.0
ARTIFACT_SCHEMA_VERSION = 2
TERMINAL_KINDS = ("success", "truncated", "failed", "timeout", "cancelled", "never_sent", "unresolved")


def parse_concurrency(text: str) -> tuple[int, ...]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("concurrency specification must not be empty.")
    levels: list[int] = []
    for token in text.split(","):
        stripped = token.strip()
        if not stripped:
            raise ValueError("concurrency specification contains an empty entry.")
        try:
            level = int(stripped)
        except ValueError as exc:
            raise ValueError(f"concurrency entry is not an integer: {stripped!r}") from exc
        if level <= 0:
            raise ValueError("concurrency levels must be positive.")
        levels.append(level)
    if len(set(levels)) != len(levels):
        raise ValueError("concurrency levels must be unique.")
    return tuple(levels)


def build_request_plan(workloads: Sequence[ServiceWorkload], *, concurrency: int) -> tuple[ServiceWorkload, ...]:
    if not workloads:
        raise ValueError("At least one workload is required.")
    if concurrency <= 0:
        raise ValueError("concurrency must be positive.")
    return tuple(workloads[index % len(workloads)] for index in range(concurrency))


def build_arrival_schedule(*, arrival_rate_per_s: float, duration_s: float, seed: int) -> tuple[float, ...]:
    if not isinstance(arrival_rate_per_s, (int, float)) or isinstance(arrival_rate_per_s, bool):
        raise TypeError("arrival_rate_per_s must be a number.")
    if not 0.0 < float(arrival_rate_per_s) <= 1e6:
        raise ValueError("arrival_rate_per_s must be within (0, 1e6].")
    if not isinstance(duration_s, (int, float)) or isinstance(duration_s, bool) or duration_s <= 0:
        raise ValueError("duration_s must be positive.")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an int.")
    rng = random.Random(seed)
    offsets: list[float] = []
    current = 0.0
    while True:
        current += rng.expovariate(float(arrival_rate_per_s))
        if current >= duration_s:
            break
        offsets.append(current)
    return tuple(offsets)


def classify_failure(exc: BaseException) -> str:
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError, httpx.TimeoutException)):
        return "timeout"
    return "error"


class VramSampler:
    def __init__(self, *, interval_seconds: float = 0.1, gpu_index: int | None = None) -> None:
        if not 0.0 < interval_seconds <= 60.0:
            raise ValueError("interval_seconds must be within (0, 60].")
        self.interval_seconds = interval_seconds
        self.gpu_index = gpu_index
        self._samples: list[VramSample] = []
        self._reason: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def unavailable_reason(self) -> str | None:
        return self._reason

    def _query(self) -> tuple[VramSample, ...]:
        if shutil.which("nvidia-smi") is None:
            raise FileNotFoundError("nvidia-smi is not on PATH")
        completed = subprocess.run(format_nvidia_smi_query(self.gpu_index), capture_output=True, text=True, timeout=10, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"nvidia-smi exited with {completed.returncode}")
        return parse_nvidia_smi_query(completed.stdout)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                fresh = self._query()
            except Exception as exc:
                with self._lock:
                    if self._reason is None:
                        self._reason = f"{type(exc).__name__}: {exc}"
                return
            if not fresh:
                with self._lock:
                    self._reason = "nvidia-smi returned no parseable GPU rows"
                return
            with self._lock:
                self._samples.append(fresh[0])
            self._stop.wait(self.interval_seconds)

    def __enter__(self) -> "VramSampler":
        self._thread = threading.Thread(target=self._loop, name="miniserve-vram-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.stop()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def trace(self) -> VramTrace:
        with self._lock:
            samples = tuple(self._samples)
            reason = self._reason
        if samples:
            return VramTrace(samples)
        return VramTrace(reason=reason or "sampler produced no samples")


def _merge_samplers(samplers: Sequence[VramSampler]) -> VramTrace:
    traces = [sampler.trace() for sampler in samplers]
    merged = tuple(sample for trace in traces for sample in trace.samples)
    if merged:
        return VramTrace(merged)
    reasons = [trace.reason for trace in traces if trace.reason]
    return VramTrace(reason="; ".join(reasons) if reasons else "no sampler configured")


@dataclass(frozen=True)
class RequestRecord:
    """一个客户端请求的互斥终态与时间线（相对窗口起点的纳秒偏移）。"""

    client_index: int
    workload: str
    terminal: str
    scheduled_offset_ns: int | None = None
    sent_offset_ns: int | None = None
    deadline_offset_ns: int | None = None
    completed_offset_ns: int | None = None
    queue_wait_ms: float | None = None
    error: str | None = None
    observation: ServiceObservation | None = None

    def __post_init__(self) -> None:
        if self.terminal not in TERMINAL_KINDS:
            raise ValueError(f"unknown terminal kind: {self.terminal}")

    def as_dict(self) -> dict[str, object]:
        def secs(value: int | None) -> float | None:
            return None if value is None else value / 1e9

        observation = self.observation
        return {
            "client_index": self.client_index,
            "workload": self.workload,
            "terminal": self.terminal,
            "scheduled_offset_s": secs(self.scheduled_offset_ns),
            "sent_offset_s": secs(self.sent_offset_ns),
            "deadline_offset_s": secs(self.deadline_offset_ns),
            "completed_offset_s": secs(self.completed_offset_ns),
            "queue_wait_ms": self.queue_wait_ms,
            "error": (self.error or "")[:1000] or None,
            "server_request_id": None if observation is None else observation.request_id,
            "usage_source": None if observation is None else observation.usage_source,
            "completion_tokens": None if observation is None else observation.completion_tokens,
            "observation": None if observation is None else observation.as_dict(),
        }


def _terminal_counts(records: Sequence[RequestRecord], *, issued: int) -> dict[str, int]:
    counts = {kind: 0 for kind in TERMINAL_KINDS}
    for record in records:
        counts[record.terminal] += 1
    recorded = sum(counts[kind] for kind in TERMINAL_KINDS if kind != "unresolved")
    unresolved = issued - recorded
    if unresolved < 0:
        raise ValueError("terminal records exceed the issued cohort.")
    counts["unresolved"] = unresolved
    return counts


def _window_aggregate(
    records: Sequence[RequestRecord],
    *,
    concurrency: int,
    window_s: float,
    started_ns: int,
    finished_ns: int,
    vram: VramTrace,
) -> AggregateResult:
    observations: list[ServiceObservation] = []
    failed = timeouts = cancelled = 0
    for record in records:
        if record.terminal in {"success", "truncated"}:
            if record.observation is None:
                raise RuntimeError("successful terminal without an observation.")
            observations.append(record.observation)
        elif record.terminal == "failed":
            failed += 1
        elif record.terminal == "timeout":
            timeouts += 1
        else:
            cancelled += 1
    return build_aggregate(
        concurrency=concurrency,
        issued=len(records),
        observations=observations,
        failures=failed + timeouts,
        timeouts=timeouts,
        incomplete=cancelled,
        started_ns=started_ns,
        finished_ns=finished_ns,
        window_s=window_s,
        vram=vram,
    )
@dataclass(frozen=True)
class LoadLevel:
    aggregate: AggregateResult
    failures: tuple[str, ...]
    observations: tuple[ServiceObservation, ...] = ()

    def as_dict(self) -> dict[str, object]:
        payload = self.aggregate.as_dict()
        payload["failure_reasons"] = list(self.failures)
        return payload


@dataclass(frozen=True)
class LoadWindowResult:
    mode: str
    concurrency: int
    window_s: float
    drain_s: float
    issued: int
    cohort_size: int
    in_flight_at_window_end: int
    drain_completed: int
    drain_failed: int
    drain_timeouts: int
    unfinished_after_drain: int
    scheduling_lag: LatencySummary | None
    load_generator_limited: bool
    load_generator_limited_reason: str | None
    aggregate: AggregateResult
    failures: tuple[str, ...]
    window_observations: tuple[ServiceObservation, ...] = ()
    cohort_observations: tuple[ServiceObservation, ...] = ()
    drain_truncated: int = 0
    drain_cancelled_in_flight: int = 0
    never_sent_after_drain: int = 0
    requests: tuple[RequestRecord, ...] = ()
    terminal_counts: dict[str, int] = field(default_factory=dict)
    accounting_closed: bool = True
    config: dict[str, object] = field(default_factory=dict)

    def effective_terminal_counts(self) -> dict[str, int]:
        """返回互斥终态计数；旧式构造（无 terminal_counts）按聚合账本回退。"""

        if self.terminal_counts:
            return dict(self.terminal_counts)
        counts = {kind: 0 for kind in TERMINAL_KINDS}
        counts["success"] = self.aggregate.completed
        counts["truncated"] = self.aggregate.truncated_requests
        counts["timeout"] = self.aggregate.failed_timeouts
        counts["failed"] = max(0, self.aggregate.failed - self.aggregate.failed_timeouts)
        counts["cancelled"] = self.aggregate.incomplete
        recorded = sum(counts[kind] for kind in TERMINAL_KINDS if kind != "unresolved")
        counts["unresolved"] = max(0, self.issued - recorded)
        return counts

    def effective_requests(self) -> tuple[RequestRecord, ...]:
        """返回逐请求记录；旧式构造（无 requests）从 cohort 观测合成。"""

        if self.requests:
            return tuple(self.requests)
        return tuple(
            RequestRecord(
                index,
                observation.workload,
                "truncated" if observation.truncated else "success",
                observation=observation,
            )
            for index, observation in enumerate(self.cohort_observations)
        )

    def as_dict(self) -> dict[str, object]:
        payload = self.aggregate.as_dict()
        payload.update(
            {
                "record_type": "window",
                "mode": self.mode,
                "window_s": self.window_s,
                "drain_s": self.drain_s,
                "issued": self.issued,
                "cohort_size": self.cohort_size,
                "window_resolved": self.aggregate.issued,
                "window_completed": self.aggregate.completed,
                "window_failed": self.aggregate.failed,
                "window_truncated": self.aggregate.truncated_requests,
                "completed_total": self.aggregate.completed + self.drain_completed,
                "failed_total": self.aggregate.failed + self.drain_failed,
                "truncated_total": self.aggregate.truncated_requests + self.drain_truncated,
                "in_flight_at_window_end": self.in_flight_at_window_end,
                "drain_completed": self.drain_completed,
                "drain_truncated": self.drain_truncated,
                "drain_failed": self.drain_failed,
                "drain_timeouts": self.drain_timeouts,
                "drain_cancelled_in_flight": self.drain_cancelled_in_flight,
                "never_sent_after_drain": self.never_sent_after_drain,
                "unfinished_after_drain": self.unfinished_after_drain,
                "terminal_counts": self.effective_terminal_counts(),
                "accounting_closed": self.accounting_closed,
                "scheduling_lag_ms": None if self.scheduling_lag is None else self.scheduling_lag.as_dict(),
                "load_generator_limited": self.load_generator_limited,
                "load_generator_limited_reason": self.load_generator_limited_reason,
                "failure_reasons": list(self.failures),
                "config": dict(self.config),
                "requests": [record.as_dict() for record in self.effective_requests()],
            }
        )
        return payload


async def run_level(
    base_url: str,
    workloads: Sequence[ServiceWorkload],
    *,
    model: str,
    concurrency: int,
    timeout_seconds: float = 120,
    max_connections: int | None = None,
    samplers: Sequence[VramSampler] = (),
) -> LoadLevel:
    plan = build_request_plan(workloads, concurrency=concurrency)
    limit = max_connections if max_connections is not None else max(concurrency, 8)
    limits = httpx.Limits(max_connections=limit, max_keepalive_connections=limit)
    observations: list[ServiceObservation] = []
    failures: list[str] = []
    timeouts = 0
    started_ns = time.monotonic_ns()
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout_seconds, limits=limits) as client:
        results = await asyncio.gather(*(run_one(client, workload, model=model) for workload in plan), return_exceptions=True)
    finished_ns = time.monotonic_ns()
    for result in results:
        if isinstance(result, BaseException):
            if classify_failure(result) == "timeout":
                timeouts += 1
            failures.append(f"{type(result).__name__}: {result}")
        else:
            observations.append(result)
    aggregate = build_aggregate(
        concurrency=concurrency,
        issued=len(plan),
        observations=observations,
        failures=len(failures),
        timeouts=timeouts,
        started_ns=started_ns,
        finished_ns=finished_ns,
        vram=_merge_samplers(samplers),
    )
    return LoadLevel(aggregate=aggregate, failures=tuple(failures), observations=tuple(observations))


def _effective_deadline(request_deadline_s: float | None, timeout_seconds: float) -> float:
    value = float(timeout_seconds if request_deadline_s is None else request_deadline_s)
    if not (value > 0.0):
        raise ValueError("request deadline must be positive.")
    return value


async def _execute_with_deadline(
    client: httpx.AsyncClient, workload: ServiceWorkload, *, model: str, deadline_ns: int
) -> ServiceObservation:
    remaining = (deadline_ns - time.monotonic_ns()) / 1e9
    if remaining <= 0:
        raise TimeoutError("request deadline exceeded before send")
    return await asyncio.wait_for(run_one(client, workload, model=model), timeout=remaining)


async def run_closed_loop(
    base_url: str,
    workloads: Sequence[ServiceWorkload],
    *,
    model: str,
    concurrency: int,
    duration_s: float,
    timeout_seconds: float = 120,
    request_deadline_s: float | None = None,
    drain_timeout_s: float | None = None,
    max_connections: int | None = None,
    samplers: Sequence[VramSampler] = (),
) -> LoadWindowResult:
    if concurrency <= 0:
        raise ValueError("concurrency must be positive.")
    if not isinstance(duration_s, (int, float)) or isinstance(duration_s, bool) or duration_s <= 0:
        raise ValueError("duration_s must be positive.")
    if not workloads:
        raise ValueError("At least one workload is required.")
    deadline_s = _effective_deadline(request_deadline_s, timeout_seconds)
    drain_limit_s = float(DEFAULT_DRAIN_TIMEOUT_S if drain_timeout_s is None else drain_timeout_s)
    if drain_limit_s <= 0:
        raise ValueError("drain_timeout_s must be positive.")
    limit = max_connections if max_connections is not None else max(concurrency, 8)
    limits = httpx.Limits(max_connections=limit, max_keepalive_connections=limit)
    records: list[RequestRecord] = []
    counter = {"next": 0}

    async with httpx.AsyncClient(base_url=base_url, timeout=timeout_seconds, limits=limits) as client:
        window_start_ns = time.monotonic_ns()
        window_end_ns = window_start_ns + int(duration_s * 1e9)

        async def worker() -> None:
            while time.monotonic_ns() < window_end_ns:
                index = counter["next"]
                counter["next"] += 1
                workload = workloads[index % len(workloads)]
                sent_ns = time.monotonic_ns()
                sent_offset = sent_ns - window_start_ns
                deadline_ns = sent_ns + int(deadline_s * 1e9)
                deadline_offset = deadline_ns - window_start_ns
                try:
                    observation = await _execute_with_deadline(client, workload, model=model, deadline_ns=deadline_ns)
                except asyncio.CancelledError:
                    records.append(RequestRecord(index, workload.name, "cancelled", scheduled_offset_ns=sent_offset, sent_offset_ns=sent_offset, deadline_offset_ns=deadline_offset, completed_offset_ns=time.monotonic_ns() - window_start_ns, error="cancelled by bounded drain"))
                    raise
                except TimeoutError as exc:
                    records.append(RequestRecord(index, workload.name, "timeout", scheduled_offset_ns=sent_offset, sent_offset_ns=sent_offset, deadline_offset_ns=deadline_offset, completed_offset_ns=time.monotonic_ns() - window_start_ns, error=str(exc)))
                except Exception as exc:
                    records.append(RequestRecord(index, workload.name, "failed", scheduled_offset_ns=sent_offset, sent_offset_ns=sent_offset, deadline_offset_ns=deadline_offset, completed_offset_ns=time.monotonic_ns() - window_start_ns, error=f"{type(exc).__name__}: {exc}"))
                else:
                    terminal = "truncated" if observation.truncated else "success"
                    records.append(RequestRecord(index, workload.name, terminal, scheduled_offset_ns=sent_offset, sent_offset_ns=sent_offset, deadline_offset_ns=deadline_offset, completed_offset_ns=time.monotonic_ns() - window_start_ns, observation=observation))

        tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
        # 等待上限 = 窗口 + drain；只用 drain 会在长窗口提前截断 worker。
        done, pending = await asyncio.wait(tasks, timeout=float(duration_s) + drain_limit_s)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        finished_ns = time.monotonic_ns()

    # 固定窗口是 [start,end)：恰好 end 完成属于窗口外（drain）。
    window_end_offset = window_end_ns - window_start_ns
    window_records = [r for r in records if r.completed_offset_ns is not None and r.completed_offset_ns < window_end_offset]
    drain_records = [r for r in records if r.completed_offset_ns is not None and r.completed_offset_ns >= window_end_offset]
    unresolved = len([r for r in records if r.completed_offset_ns is None])
    terminal_counts = _terminal_counts(records, issued=len(records) + unresolved)
    accounting_closed = sum(terminal_counts.values()) == len(records) + unresolved
    window_aggregate = _window_aggregate(
        window_records,
        concurrency=concurrency,
        window_s=float(duration_s),
        started_ns=window_start_ns,
        finished_ns=window_end_ns,
        vram=_merge_samplers(samplers),
    )
    in_flight = sum(
        1
        for r in records
        if r.sent_offset_ns is not None
        and r.sent_offset_ns < window_end_offset
        and (r.completed_offset_ns is None or r.completed_offset_ns >= window_end_offset)
    )
    failures = tuple(
        f"{r.terminal}: {r.error}"
        for r in window_records
        if r.terminal in {"failed", "timeout", "cancelled", "never_sent"}
    )
    return LoadWindowResult(
        mode="closed_loop",
        concurrency=concurrency,
        window_s=float(duration_s),
        drain_s=max(0.0, (finished_ns - window_end_ns) / 1e9),
        issued=len(records) + unresolved,
        cohort_size=len(records) + unresolved,
        in_flight_at_window_end=in_flight,
        drain_completed=sum(r.terminal == "success" for r in drain_records),
        drain_failed=sum(r.terminal == "failed" for r in drain_records),
        drain_timeouts=sum(r.terminal == "timeout" for r in drain_records),
        unfinished_after_drain=unresolved,
        scheduling_lag=None,
        load_generator_limited=False,
        load_generator_limited_reason=None,
        aggregate=window_aggregate,
        failures=failures[:32],
        window_observations=tuple(r.observation for r in window_records if r.observation is not None),
        cohort_observations=tuple(r.observation for r in records if r.observation is not None),
        drain_truncated=sum(r.terminal == "truncated" for r in drain_records),
        drain_cancelled_in_flight=sum(r.terminal == "cancelled" for r in drain_records),
        never_sent_after_drain=0,
        requests=tuple(records),
        terminal_counts=terminal_counts,
        accounting_closed=accounting_closed,
        config={
            "mode": "closed_loop",
            "concurrency": concurrency,
            "duration_s": float(duration_s),
            "request_deadline_s": deadline_s,
            "drain_timeout_s": drain_limit_s,
            "http_inactivity_timeout_s": float(timeout_seconds),
        },
    )
async def run_open_loop(
    base_url: str,
    workloads: Sequence[ServiceWorkload],
    *,
    model: str,
    arrival_rate_per_s: float,
    duration_s: float,
    seed: int = 0,
    max_inflight: int = 64,
    timeout_seconds: float = 120,
    request_deadline_s: float | None = None,
    drain_timeout_s: float | None = None,
    allowed_send_lag_ms: float = DEFAULT_ALLOWED_SEND_LAG_MS,
    max_connections: int | None = None,
    samplers: Sequence[VramSampler] = (),
) -> LoadWindowResult:
    if not workloads:
        raise ValueError("At least one workload is required.")
    if not isinstance(max_inflight, int) or isinstance(max_inflight, bool) or max_inflight <= 0:
        raise ValueError("max_inflight must be a positive integer.")
    if not isinstance(allowed_send_lag_ms, (int, float)) or isinstance(allowed_send_lag_ms, bool) or allowed_send_lag_ms < 0:
        raise ValueError("allowed_send_lag_ms must be non-negative.")
    deadline_s = _effective_deadline(request_deadline_s, timeout_seconds)
    drain_limit_s = float(DEFAULT_DRAIN_TIMEOUT_S if drain_timeout_s is None else drain_timeout_s)
    if drain_limit_s <= 0:
        raise ValueError("drain_timeout_s must be positive.")
    schedule = build_arrival_schedule(arrival_rate_per_s=arrival_rate_per_s, duration_s=duration_s, seed=seed)
    limit = max_connections if max_connections is not None else max(max_inflight, 8)
    limits = httpx.Limits(max_connections=limit, max_keepalive_connections=limit)
    records: list[RequestRecord] = []
    lag_ms: list[float] = []
    limited_reasons: list[str] = []
    sent_tasks: dict[asyncio.Task, int] = {}

    async with httpx.AsyncClient(base_url=base_url, timeout=timeout_seconds, limits=limits) as client:
        window_start_ns = time.monotonic_ns()
        window_end_ns = window_start_ns + int(duration_s * 1e9)
        semaphore = asyncio.Semaphore(max_inflight)

        async def launch(index: int, offset_s: float) -> None:
            workload = workloads[index % len(workloads)]
            target_offset = int(offset_s * 1e9)
            target_ns = window_start_ns + target_offset
            deadline_ns = target_ns + int(deadline_s * 1e9)
            deadline_offset = deadline_ns - window_start_ns
            sleep_s = (target_ns - time.monotonic_ns()) / 1e9
            if sleep_s > 0:
                await asyncio.sleep(sleep_s)
            acquired = False
            try:
                remaining = (deadline_ns - time.monotonic_ns()) / 1e9
                if remaining <= 0:
                    records.append(RequestRecord(index, workload.name, "never_sent", scheduled_offset_ns=target_offset, deadline_offset_ns=deadline_offset, completed_offset_ns=time.monotonic_ns() - window_start_ns, error="deadline exceeded before send"))
                    return
                try:
                    # 排队等待本身也受该请求的绝对期限约束。
                    await asyncio.wait_for(semaphore.acquire(), timeout=remaining)
                    acquired = True
                except TimeoutError as exc:
                    records.append(RequestRecord(index, workload.name, "never_sent", scheduled_offset_ns=target_offset, deadline_offset_ns=deadline_offset, completed_offset_ns=time.monotonic_ns() - window_start_ns, error=f"deadline exceeded while queued: {exc}"))
                    return
                remaining = (deadline_ns - time.monotonic_ns()) / 1e9
                if remaining <= 0:
                    records.append(RequestRecord(index, workload.name, "never_sent", scheduled_offset_ns=target_offset, deadline_offset_ns=deadline_offset, completed_offset_ns=time.monotonic_ns() - window_start_ns, error="deadline exceeded before send"))
                    return
                # 只有真正调用 run_one 才记录实际 send。
                send_ns = time.monotonic_ns()
                send_offset = send_ns - window_start_ns
                current = asyncio.current_task()
                if current is not None:
                    sent_tasks[current] = send_offset
                lag = (send_ns - target_ns) / 1e6
                lag_ms.append(lag)
                if lag > float(allowed_send_lag_ms):
                    limited_reasons.append(
                        f"client send lag {lag:.3f} ms exceeds allowed {float(allowed_send_lag_ms):.3f} ms"
                    )
                try:
                    observation = await asyncio.wait_for(run_one(client, workload, model=model), timeout=remaining)
                except TimeoutError as exc:
                    records.append(RequestRecord(index, workload.name, "timeout", scheduled_offset_ns=target_offset, sent_offset_ns=send_offset, deadline_offset_ns=deadline_offset, completed_offset_ns=time.monotonic_ns() - window_start_ns, queue_wait_ms=lag, error=str(exc)))
                except Exception as exc:
                    records.append(RequestRecord(index, workload.name, "failed", scheduled_offset_ns=target_offset, sent_offset_ns=send_offset, deadline_offset_ns=deadline_offset, completed_offset_ns=time.monotonic_ns() - window_start_ns, queue_wait_ms=lag, error=f"{type(exc).__name__}: {exc}"))
                else:
                    terminal = "truncated" if observation.truncated else "success"
                    records.append(RequestRecord(index, workload.name, terminal, scheduled_offset_ns=target_offset, sent_offset_ns=send_offset, deadline_offset_ns=deadline_offset, completed_offset_ns=time.monotonic_ns() - window_start_ns, queue_wait_ms=lag, observation=observation))
            except asyncio.CancelledError:
                current = asyncio.current_task()
                if acquired and current is not None and current in sent_tasks:
                    records.append(RequestRecord(index, workload.name, "cancelled", scheduled_offset_ns=target_offset, sent_offset_ns=sent_tasks[current], deadline_offset_ns=deadline_offset, completed_offset_ns=time.monotonic_ns() - window_start_ns, error="cancelled by bounded drain"))
                else:
                    records.append(RequestRecord(index, workload.name, "never_sent", scheduled_offset_ns=target_offset, deadline_offset_ns=deadline_offset, completed_offset_ns=time.monotonic_ns() - window_start_ns, error="drain deadline reached before send"))
                raise
            finally:
                if acquired:
                    semaphore.release()

        tasks = [asyncio.create_task(launch(index, offset)) for index, offset in enumerate(schedule)]
        unfinished = 0
        if tasks:
            drain_deadline_ns = window_end_ns + int(drain_limit_s * 1e9)
            _, pending = await asyncio.wait(tasks, timeout=max((drain_deadline_ns - time.monotonic_ns()) / 1e9, 0.0))
            pending_list = list(pending)
            for task in pending_list:
                task.cancel()
            if pending_list:
                await asyncio.gather(*pending_list, return_exceptions=True)
            unfinished = len(pending_list)
        finished_ns = time.monotonic_ns()

    window_end_offset = window_end_ns - window_start_ns
    terminalized = {record.client_index for record in records}
    for index, _offset in enumerate(schedule):
        if index not in terminalized:
            records.append(RequestRecord(index, workloads[index % len(workloads)].name, "unresolved", error="no terminal recorded"))
    window_records = [
        r for r in records if r.completed_offset_ns is not None and r.completed_offset_ns < window_end_offset
    ]
    drain_records = [
        r for r in records if r.completed_offset_ns is not None and r.completed_offset_ns >= window_end_offset
    ]
    terminal_counts = _terminal_counts(records, issued=len(schedule))
    accounting_closed = sum(terminal_counts.values()) == len(schedule)
    window_aggregate = _window_aggregate(
        window_records,
        concurrency=int(max_inflight),
        window_s=float(duration_s),
        started_ns=window_start_ns,
        finished_ns=max(finished_ns, window_end_ns),
        vram=_merge_samplers(samplers),
    )
    in_flight = sum(
        1
        for r in records
        if r.sent_offset_ns is not None
        and r.sent_offset_ns < window_end_offset
        and (r.completed_offset_ns is None or r.completed_offset_ns >= window_end_offset)
    )
    lag_bands = percentiles(lag_ms) if lag_ms else None
    scheduling_lag = None
    if lag_bands is not None:
        scheduling_lag = LatencySummary(
            count=len(lag_ms),
            p50_ms=lag_bands["p50"],
            p90_ms=lag_bands["p90"],
            p95_ms=lag_bands["p95"],
            p99_ms=lag_bands["p99"],
            mean_ms=sum(lag_ms) / len(lag_ms),
            max_ms=max(lag_ms),
        )
    limited_reason: str | None = None
    if unfinished:
        limited_reasons.append(f"{unfinished} request(s) unfinished after drain deadline")
    if limited_reasons:
        limited_reason = "; ".join(sorted(set(limited_reasons))[:3])
    failures = tuple(
        f"{r.terminal}: {r.error}"
        for r in records
        if r.terminal in {"failed", "timeout", "cancelled", "never_sent", "unresolved"}
    )
    return LoadWindowResult(
        mode="open_loop",
        concurrency=int(max_inflight),
        window_s=float(duration_s),
        drain_s=max(0.0, (finished_ns - window_end_ns) / 1e9),
        issued=len(schedule),
        cohort_size=len(schedule),
        in_flight_at_window_end=in_flight,
        drain_completed=sum(r.terminal == "success" for r in drain_records),
        drain_failed=sum(r.terminal == "failed" for r in drain_records),
        drain_timeouts=sum(r.terminal == "timeout" for r in drain_records),
        unfinished_after_drain=unfinished,
        scheduling_lag=scheduling_lag,
        load_generator_limited=bool(limited_reasons),
        load_generator_limited_reason=limited_reason,
        aggregate=window_aggregate,
        failures=failures[:32],
        window_observations=tuple(r.observation for r in window_records if r.observation is not None),
        cohort_observations=tuple(r.observation for r in records if r.observation is not None),
        drain_truncated=sum(r.terminal == "truncated" for r in drain_records),
        drain_cancelled_in_flight=sum(r.terminal == "cancelled" for r in drain_records),
        never_sent_after_drain=sum(r.terminal == "never_sent" for r in records),
        requests=tuple(records),
        terminal_counts=terminal_counts,
        accounting_closed=accounting_closed,
        config={
            "mode": "open_loop",
            "arrival_rate_per_s": float(arrival_rate_per_s),
            "duration_s": float(duration_s),
            "seed": seed,
            "max_inflight": int(max_inflight),
            "request_deadline_s": deadline_s,
            "drain_timeout_s": drain_limit_s,
            "http_inactivity_timeout_s": float(timeout_seconds),
            "allowed_send_lag_ms": float(allowed_send_lag_ms),
        },
    )


async def run_sweep(
    base_url: str,
    workloads: Sequence[ServiceWorkload],
    *,
    model: str,
    concurrency_levels: Sequence[int] = DEFAULT_CONCURRENCY,
    timeout_seconds: float = 120,
    sampler_interval_seconds: float | None = 0.1,
    gpu_index: int | None = None,
) -> tuple[LoadLevel, ...]:
    levels: list[LoadLevel] = []
    for concurrency in sorted(concurrency_levels):
        if sampler_interval_seconds is None:
            levels.append(await run_level(base_url, workloads, model=model, concurrency=concurrency, timeout_seconds=timeout_seconds, samplers=()))
            continue
        with VramSampler(interval_seconds=sampler_interval_seconds, gpu_index=gpu_index) as sampler:
            levels.append(await run_level(base_url, workloads, model=model, concurrency=concurrency, timeout_seconds=timeout_seconds, samplers=(sampler,)))
    return tuple(levels)


def _candidate_identity() -> dict[str, object]:
    identity: dict[str, object] = {"python": platform.python_version(), "platform": platform.platform()}
    repo = Path(__file__).resolve().parents[2]
    try:
        identity["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, timeout=10, check=False).stdout.strip() or None
        identity["git_dirty"] = bool(subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, timeout=10, check=False).stdout.strip())
    except Exception:
        identity["git_commit"] = None
    try:
        from miniserve._native import find_extension

        path = find_extension()
        identity["so_path"] = str(path)
        identity["so_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    except Exception as exc:
        identity["so_path"] = None
        identity["so_sha256"] = None
        identity["so_error"] = f"{type(exc).__name__}: {exc}"
    try:
        wheels = sorted(repo.glob("evidence/final/profile-d/generated/wheel/*.whl"))
        if wheels:
            identity["wheel"] = wheels[-1].name
            identity["wheel_sha256"] = hashlib.sha256(wheels[-1].read_bytes()).hexdigest()
    except Exception as exc:
        identity["wheel_sha256"] = None
        identity["wheel_error"] = f"{type(exc).__name__}: {exc}"
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        identity["gpu"] = completed.stdout.strip() or None
    except Exception:
        identity["gpu"] = None
    return identity


def _write_artifacts(
    output: Path,
    *,
    manifest: dict[str, object],
    rows: Sequence[dict[str, object]],
    goodput_reports: Sequence[GoodputReport] | None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    observations_path = output.with_suffix(output.suffix + ".observations.jsonl")
    goodput_path = output.with_suffix(output.suffix + ".goodput.json")
    manifest = dict(manifest)
    manifest["files"] = {
        "aggregate": str(output),
        "observations": str(observations_path),
        "goodput": str(goodput_path) if goodput_reports is not None else None,
    }
    lines: list[dict[str, object]] = [manifest]
    for index, row in enumerate(rows):
        enriched = dict(row)
        enriched["window_index"] = index
        if goodput_reports is not None:
            enriched["goodput"] = goodput_reports[index].as_dict()
        lines.append(enriched)
    output.write_text("".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines), encoding="utf-8")
    with observations_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            for record in row.get("requests", []):
                handle.write(json.dumps({"window_index": index, **record}, ensure_ascii=False) + "\n")
    if goodput_reports is not None:
        goodput_path.write_text(json.dumps([report.as_dict() for report in goodput_reports], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _artifact_load_mode(
    manifest: dict[str, object], row: dict[str, object]
) -> object:
    """Resolve load semantics independently from decode_batching/manifest mode."""

    kind = manifest.get("kind")
    if kind in {"closed", "open"}:
        return "closed-loop" if kind == "closed" else "open-loop"
    mode = manifest.get("mode")
    if mode in {"closed-loop", "open-loop", "burst"}:
        return mode
    row_mode = row.get("mode")
    if row_mode == "closed_loop":
        return "closed-loop"
    if row_mode == "open_loop":
        return "open-loop"
    return mode


def recompute_goodput_from_artifacts(output: Path) -> list[dict[str, object]]:
    """离线从 manifest + 逐请求记录重算 Goodput；用于证明在线/离线一致。"""

    lines = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines or lines[0].get("record_type") != "manifest":
        raise ValueError("artifact has no manifest line")
    manifest = lines[0]
    slo_payload = manifest.get("slo")
    if slo_payload:
        slo = SloGrid(
            ttft_ms=tuple(slo_payload["ttft_ms"]),
            tpot_ms=tuple(slo_payload["tpot_ms"]),
            e2e_ms=slo_payload.get("e2e_ms"),
        )
    else:
        slo = DEFAULT_SLO_GRID
    reports: list[dict[str, object]] = []
    for row in lines[1:]:
        # 兼容两套语义轴：manifest.mode 表示 load 模式，decode_batching 才是服务策略；
        # 新 runner 同时写 kind=closed/open。
        mode = _artifact_load_mode(manifest, row)
        requests = row.get("requests", [])
        issued_override = int(row["issued"])
        if mode == "closed-loop":
            window_s = float(row["window_s"])
            requests = [
                record
                for record in requests
                if record.get("completed_offset_s") is not None
                and float(record["completed_offset_s"]) < window_s
            ]
            # closed-loop 主口径：issued 用窗口内解析数，而不是 cohort 分母。
            issued_override = len(requests)
        observations = [
            ServiceObservation.from_dict(record["observation"])
            for record in requests
            if record.get("observation") is not None
        ]
        counts = {kind: 0 for kind in TERMINAL_KINDS}
        for record in requests:
            kind = record.get("terminal")
            if kind not in counts:
                raise ValueError(f"unknown terminal in artifact: {kind}")
            counts[kind] += 1
        report = evaluate_goodput(
            observations,
            window_s=float(row["window_s"]),
            slo=slo,
            issued=issued_override,
            completed=counts["success"],
            failed=counts["failed"] + counts["timeout"],
            incomplete=counts["cancelled"] + counts["never_sent"] + counts["unresolved"],
            truncated=counts["truncated"],
        )
        reports.append(report.as_dict())
    return reports

def main() -> None:
    parser = argparse.ArgumentParser(description="Load the MiniServe HTTP/SSE endpoint (burst/closed-loop/open-loop).")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("burst", "closed-loop", "open-loop"), default="burst")
    parser.add_argument("--concurrency", default=",".join(str(item) for item in DEFAULT_CONCURRENCY))
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--rate", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-inflight", type=int, default=64)
    parser.add_argument("--request-deadline", type=float, default=DEFAULT_REQUEST_DEADLINE_S)
    parser.add_argument("--drain-timeout", type=float, default=None)
    parser.add_argument("--allowed-send-lag-ms", type=float, default=DEFAULT_ALLOWED_SEND_LAG_MS)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--prompt", default="Explain paged attention in one sentence.")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--gpu-index", type=int, default=None)
    parser.add_argument("--sample-interval", type=float, default=0.1)
    parser.add_argument("--goodput", action="store_true")
    parser.add_argument("--slo-ttft", default=None)
    parser.add_argument("--slo-tpot", default=None)
    args = parser.parse_args()

    levels = parse_concurrency(args.concurrency)
    workloads = (ServiceWorkload("load", args.prompt, args.max_tokens),)
    sample_interval = None if args.sample_interval <= 0 else args.sample_interval
    slo = SloGrid.from_strings(ttft_csv=args.slo_ttft, tpot_csv=args.slo_tpot)
    rows: list[dict[str, object]] = []
    reports: list[GoodputReport] = []
    if args.mode == "burst":
        results = asyncio.run(
            run_sweep(
                args.url,
                workloads,
                model=args.model,
                concurrency_levels=levels,
                timeout_seconds=args.timeout,
                sampler_interval_seconds=sample_interval,
                gpu_index=args.gpu_index,
            )
        )
        rows = [item.as_dict() for item in results]
        print(format_table([item.aggregate for item in results]))
    else:
        windows: list[LoadWindowResult] = []
        for level in sorted(levels):
            sampler = (
                VramSampler(interval_seconds=sample_interval, gpu_index=args.gpu_index)
                if sample_interval is not None
                else None
            )
            samplers: Sequence[VramSampler] = ()
            if sampler is not None:
                sampler.__enter__()
                samplers = (sampler,)
            try:
                if args.mode == "closed-loop":
                    window = asyncio.run(
                        run_closed_loop(
                            args.url,
                            workloads,
                            model=args.model,
                            concurrency=level,
                            duration_s=args.duration,
                            timeout_seconds=args.timeout,
                            request_deadline_s=args.request_deadline,
                            drain_timeout_s=args.drain_timeout,
                            samplers=samplers,
                        )
                    )
                else:
                    window = asyncio.run(
                        run_open_loop(
                            args.url,
                            workloads,
                            model=args.model,
                            arrival_rate_per_s=args.rate,
                            duration_s=args.duration,
                            seed=args.seed,
                            max_inflight=max(args.max_inflight, level),
                            timeout_seconds=args.timeout,
                            request_deadline_s=args.request_deadline,
                            drain_timeout_s=args.drain_timeout,
                            allowed_send_lag_ms=args.allowed_send_lag_ms,
                            samplers=samplers,
                        )
                    )
            finally:
                if sampler is not None:
                    sampler.__exit__()
            windows.append(window)
            rows.append(window.as_dict())
            if window.load_generator_limited:
                print(f"LOAD_GENERATOR_LIMITED: {window.load_generator_limited_reason}")
            if args.goodput:
                counts = window.effective_terminal_counts()
                if args.mode == "closed-loop":
                    # closed-loop 主 Goodput 只计测量窗口内成功/达标；cohort 版本单列命名。
                    report = evaluate_goodput(
                        window.window_observations,
                        window_s=window.window_s,
                        slo=slo,
                        issued=window.aggregate.issued,
                        completed=window.aggregate.completed,
                        failed=window.aggregate.failed,
                        incomplete=window.aggregate.incomplete,
                        truncated=window.aggregate.truncated_requests,
                    )
                    cohort_report = evaluate_goodput(
                        window.cohort_observations,
                        window_s=window.window_s,
                        slo=slo,
                        issued=window.issued,
                        completed=counts.get("success", 0),
                        failed=counts.get("failed", 0) + counts.get("timeout", 0),
                        incomplete=counts.get("cancelled", 0)
                        + counts.get("never_sent", 0)
                        + counts.get("unresolved", 0),
                        truncated=counts.get("truncated", 0),
                    )
                    rows[windows.index(window)]["cohort_goodput"] = cohort_report.as_dict()
                else:
                    # open-loop 主 Goodput 使用 scheduled cohort + 固定发流窗口分母。
                    report = evaluate_goodput(
                        window.cohort_observations,
                        window_s=window.window_s,
                        slo=slo,
                        issued=window.issued,
                        completed=counts.get("success", 0),
                        failed=counts.get("failed", 0) + counts.get("timeout", 0),
                        incomplete=counts.get("cancelled", 0)
                        + counts.get("never_sent", 0)
                        + counts.get("unresolved", 0),
                        truncated=counts.get("truncated", 0),
                    )
                reports.append(report)
        print(format_table([window.aggregate for window in windows]))
        if args.goodput:
            for report in reports:
                print(format_goodput_table(report))

    manifest: dict[str, object] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "record_type": "manifest",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "args": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "workloads": [workload.metadata() for workload in workloads],
        "slo": slo.as_dict() if args.goodput else None,
        "statistics": {
            "percentile_method": "linear-interpolation",
            "sample_unit": "run",
            "terminal_kinds": list(TERMINAL_KINDS),
        },
        "candidate": _candidate_identity(),
    }
    _write_artifacts(
        args.output,
        manifest=manifest,
        rows=rows,
        goodput_reports=reports if args.goodput else None,
    )


if __name__ == "__main__":
    main()
