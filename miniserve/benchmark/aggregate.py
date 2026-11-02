"""并发负载的统计聚合：分位数、聚合吞吐、失败分母与显存账本。

M0 冻结语义（含 Sol-REWORK 修正）：
- ``completed`` 只统计**正常完成**的观测；截断流（无 [DONE]/终态）单独计入
  ``truncated_requests``，不得混入成功完成数、成功率或成功吞吐。
- ``tpot`` 只汇总**请求级** ``client_tpot_ms``（N<=1 为 None 不计入）。
- ``itl`` 是 **engine token 事件间隔**；客户端 harness 没有 token 级事件，因此这里
  保持 count=0，绝不把 SSE chunk gap 冒充 ITL。SSE content chunk 间隔单列为
  ``client_inter_chunk``。
- 吞吐分母固定为 ``window_s``（固定测量窗口）；``duration_s`` 是实测墙钟跨度。
- 分母守恒：completed + truncated + failed + incomplete <= issued；允许 issued=0
  的合法零 cohort（吞吐为 0，不虚构样本）。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from statistics import fmean
from typing import Iterable, Sequence

from .metrics import ServiceObservation

DEFAULT_PERCENTILES: tuple[float, ...] = (50.0, 90.0, 95.0, 99.0)
_HEADERS = ("p50", "p90", "p95", "p99")


def percentile(values: Sequence[float], percent: float) -> float | None:
    """线性插值分位数；空输入返回 None。percent 落在 [0, 100]。"""

    if not isinstance(percent, (int, float)) or isinstance(percent, bool):
        raise TypeError("percent must be a number.")
    if not math.isfinite(percent) or not 0.0 <= percent <= 100.0:
        raise ValueError("percent must be finite and within [0, 100].")
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (percent / 100.0) * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(ordered[lower])
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def percentiles(
    values: Sequence[float],
    *,
    percentiles_to_compute: Sequence[float] = DEFAULT_PERCENTILES,
) -> dict[str, float | None]:
    """返回 ``{"p50": ..., "p90": ..., ...}``；名称由固定表映射，维持低基数。"""

    if len(percentiles_to_compute) != len(_HEADERS):
        raise ValueError("Only the four canonical percentile columns are supported.")
    return {
        name: percentile(values, percent)
        for name, percent in zip(_HEADERS, percentiles_to_compute)
    }


@dataclass(frozen=True)
class VramSample:
    """一次显存采样；均以 MiB 记录，便于跨卡比较。"""

    used_mib: int
    total_mib: int


@dataclass(frozen=True)
class VramTrace:
    samples: tuple[VramSample, ...] = ()
    reason: str | None = None

    @property
    def available(self) -> bool:
        return bool(self.samples)

    @property
    def peak_used_mib(self) -> int | None:
        return max((item.used_mib for item in self.samples), default=None)

    @property
    def mean_used_mib(self) -> float | None:
        if not self.samples:
            return None
        return fmean(item.used_mib for item in self.samples)

    @property
    def total_mib(self) -> int | None:
        return self.samples[0].total_mib if self.samples else None

    def as_dict(self) -> dict[str, object]:
        return {
            "available": self.available,
            "reason": self.reason,
            "sample_count": len(self.samples),
            "total_mib": self.total_mib,
            "peak_used_mib": self.peak_used_mib,
            "mean_used_mib": self.mean_used_mib,
        }


_QUERY = "index,name,memory.used,memory.total"
_ROW = re.compile(r"^\s*(?P<index>\d+)\s*,\s*(?P<name>.*?)\s*,\s*(?P<used>\d+)\s*,\s*(?P<total>\d+)\s*$")


def parse_nvidia_smi_query(text: str) -> tuple[VramSample, ...]:
    """解析 ``nvidia-smi --query-gpu=... --format=csv,noheader,nounits`` 输出。"""

    if not isinstance(text, str):
        raise TypeError("nvidia-smi output must be a string.")
    samples: list[VramSample] = []
    for line in text.splitlines():
        match = _ROW.match(line)
        if match is None:
            continue
        used = int(match.group("used"))
        total = int(match.group("total"))
        if total <= 0 or used < 0:
            continue
        samples.append(VramSample(min(used, total), total))
    return tuple(samples)


def format_nvidia_smi_query(gpu_index: int | None = None) -> tuple[str, ...]:
    """构造固定的 nvidia-smi 命令；指定 index 时只查询该卡。"""

    if gpu_index is not None and (
        not isinstance(gpu_index, int) or isinstance(gpu_index, bool) or gpu_index < 0
    ):
        raise ValueError("gpu_index must be a non-negative integer when provided.")
    selector = f"--id={gpu_index}" if gpu_index is not None else None
    command = ["nvidia-smi"]
    if selector is not None:
        command.append(selector)
    command.extend([f"--query-gpu={_QUERY}", "--format=csv,noheader,nounits"])
    return tuple(command)


@dataclass(frozen=True)
class LatencySummary:
    count: int
    p50_ms: float | None
    p90_ms: float | None
    p95_ms: float | None
    p99_ms: float | None
    mean_ms: float | None
    max_ms: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "count": self.count,
            "p50_ms": self.p50_ms,
            "p90_ms": self.p90_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "mean_ms": self.mean_ms,
            "max_ms": self.max_ms,
        }


def _summarize(values: Sequence[float]) -> LatencySummary:
    bands = percentiles(values)
    return LatencySummary(
        count=len(values),
        p50_ms=bands["p50"],
        p90_ms=bands["p90"],
        p95_ms=bands["p95"],
        p99_ms=bands["p99"],
        mean_ms=fmean(values) if values else None,
        max_ms=max(values) if values else None,
    )


@dataclass(frozen=True)
class AggregateResult:
    """一次负载档位的完整账本（成功/截断/失败/未完成分账）。"""

    concurrency: int
    issued: int
    completed: int
    failed: int
    incomplete: int
    duration_s: float
    window_s: float
    request_throughput: float
    output_token_throughput: float
    total_token_throughput: float
    wire_mib_per_s: float
    ttft: LatencySummary
    tpot: LatencySummary
    tpot_request_count: int
    itl: LatencySummary
    client_inter_chunk: LatencySummary
    latency: LatencySummary
    prompt_tokens: int
    completion_tokens: int
    total_input_tokens: int
    failed_timeouts: int
    no_content_requests: int
    truncated_requests: int
    incomparable_requests: int
    vram: VramTrace

    @property
    def success_rate(self) -> float:
        if self.issued == 0:
            return 0.0
        return self.completed / self.issued

    def as_dict(self) -> dict[str, object]:
        return {
            "concurrency": self.concurrency,
            "issued": self.issued,
            "completed": self.completed,
            "failed": self.failed,
            "incomplete": self.incomplete,
            "failed_timeouts": self.failed_timeouts,
            "success_rate": self.success_rate,
            "duration_s": self.duration_s,
            "window_s": self.window_s,
            "request_throughput_per_s": self.request_throughput,
            "output_token_throughput_per_s": self.output_token_throughput,
            "total_token_throughput_per_s": self.total_token_throughput,
            "wire_mib_per_s": self.wire_mib_per_s,
            "ttft": self.ttft.as_dict(),
            "tpot": self.tpot.as_dict(),
            "tpot_request_count": self.tpot_request_count,
            "itl": self.itl.as_dict(),
            "client_inter_chunk": self.client_inter_chunk.as_dict(),
            "latency": self.latency.as_dict(),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_input_tokens": self.total_input_tokens,
            "no_content_requests": self.no_content_requests,
            "truncated_requests": self.truncated_requests,
            "incomparable_requests": self.incomparable_requests,
            "vram": self.vram.as_dict(),
        }


def _iter_values(observations: Iterable[ServiceObservation], attribute: str) -> list[float]:
    values: list[float] = []
    for item in observations:
        value = getattr(item, attribute)
        if value is not None:
            values.append(float(value))
    return values


def _client_inter_chunk(observations: Sequence[ServiceObservation]) -> list[float]:
    """池化 SSE content chunk 间隔（客户端观测）；不是 token ITL。"""

    intervals: list[float] = []
    for item in observations:
        intervals.extend(value for value in item.inter_chunk_ms if value > 0.0)
    return intervals


def build_aggregate(
    *,
    concurrency: int,
    issued: int,
    observations: Iterable[ServiceObservation],
    failures: int,
    started_ns: int,
    finished_ns: int,
    vram: VramTrace | None = None,
    timeouts: int = 0,
    incomplete: int = 0,
    window_s: float | None = None,
) -> AggregateResult:
    """把一次负载的观测聚合成可持久化账本。零除与空样本都收敛为 0 而不是异常。"""

    if concurrency <= 0:
        raise ValueError("concurrency must be positive.")
    if issued < 0:
        raise ValueError("issued must be non-negative.")
    if failures < 0 or incomplete < 0 or timeouts < 0 or timeouts > failures:
        raise ValueError("failures/incomplete must be non-negative and timeouts must not exceed failures.")
    if window_s is not None and (not math.isfinite(window_s) or window_s <= 0.0):
        raise ValueError("window_s must be finite and positive when provided.")
    items = tuple(observations)
    if finished_ns < started_ns:
        raise ValueError("finished_ns must not precede started_ns.")

    completed_items = tuple(item for item in items if not item.truncated)
    truncated_requests = len(items) - len(completed_items)
    if len(completed_items) + truncated_requests + failures + incomplete > issued:
        raise ValueError(
            "issued cannot be smaller than completed + truncated + failed + incomplete."
        )
    duration_s = (finished_ns - started_ns) / 1e9
    effective_window_s = window_s if window_s is not None else max(duration_s, 0.001)
    completion_tokens = sum(
        item.completion_tokens for item in completed_items if item.completion_tokens is not None
    )
    prompt_tokens = sum(
        item.prompt_tokens for item in completed_items if item.prompt_tokens is not None
    )
    download_bytes = sum(item.download_bytes for item in completed_items)
    tpot_values = [
        value for value in (item.client_tpot_ms for item in completed_items) if value is not None
    ]
    return AggregateResult(
        concurrency=concurrency,
        issued=issued,
        completed=len(completed_items),
        failed=failures,
        incomplete=incomplete,
        duration_s=duration_s,
        window_s=effective_window_s,
        request_throughput=len(completed_items) / effective_window_s,
        output_token_throughput=completion_tokens / effective_window_s,
        total_token_throughput=(completion_tokens + prompt_tokens) / effective_window_s,
        wire_mib_per_s=download_bytes / (1024.0 * 1024.0) / effective_window_s,
        ttft=_summarize(_iter_values(completed_items, "ttft_ms")),
        tpot=_summarize(tpot_values),
        tpot_request_count=len(tpot_values),
        # engine token ITL 需要 token 级事件时序；客户端 harness 没有，保持空样本。
        itl=LatencySummary(0, None, None, None, None, None, None),
        client_inter_chunk=_summarize(_client_inter_chunk(completed_items)),
        latency=_summarize([item.latency_ms for item in completed_items]),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_input_tokens=prompt_tokens + completion_tokens,
        failed_timeouts=timeouts,
        no_content_requests=sum(item.content_chunk_count == 0 for item in completed_items),
        truncated_requests=truncated_requests,
        incomparable_requests=sum(
            item.usage_source == "unavailable" for item in completed_items
        ),
        vram=vram or VramTrace(reason="not sampled"),
    )


COLUMN_HEADERS = (
    "concurrency",
    "issued",
    "ok",
    "trunc",
    "fail",
    "duration_s",
    "req/s",
    "out_tok/s",
    "total_tok/s",
    "ttft_p50",
    "ttft_p95",
    "ttft_p99",
    "tpot_p50",
    "tpot_p95",
    "tpot_n",
    "chunk_p50",
    "lat_p50",
    "lat_p95",
    "lat_p99",
)


def format_table(results: Sequence[AggregateResult]) -> str:
    """渲染固定宽度对照表；缺失分位数显示为 ``-``。"""

    def cell(value: float | None, digits: int = 2) -> str:
        return "-" if value is None else f"{value:.{digits}f}"

    lines = ["  ".join(f"{name:>10}" for name in COLUMN_HEADERS)]
    for item in results:
        row = (
            f"{item.concurrency:>10}",
            f"{item.issued:>10}",
            f"{item.completed:>10}",
            f"{item.truncated_requests:>10}",
            f"{item.failed:>10}",
            f"{item.duration_s:>10.3f}",
            f"{item.request_throughput:>10.2f}",
            f"{item.output_token_throughput:>10.2f}",
            f"{item.total_token_throughput:>10.2f}",
            f"{cell(item.ttft.p50_ms):>10}",
            f"{cell(item.ttft.p95_ms):>10}",
            f"{cell(item.ttft.p99_ms):>10}",
            f"{cell(item.tpot.p50_ms):>10}",
            f"{cell(item.tpot.p95_ms):>10}",
            f"{item.tpot_request_count:>10}",
            f"{cell(item.client_inter_chunk.p50_ms):>10}",
            f"{cell(item.latency.p50_ms):>10}",
            f"{cell(item.latency.p95_ms):>10}",
            f"{cell(item.latency.p99_ms):>10}",
        )
        lines.append("  ".join(row))
    return "\n".join(lines)