"""只通过真实 HTTP SSE endpoint 的服务 benchmark runner（client 时间线）。

解析并区分：content delta、finish_reason 终态 chunk、``choices=[]`` 的 usage 帧、
OpenAI 风格 error 帧、以及缺少 ``[DONE]`` 的截断流。绝不把 usage/空/role-only 帧
当成输出 token。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

from .metrics import ServiceObservation
from .workloads import ServiceWorkload


class SseErrorPayloadError(RuntimeError):
    """服务在 SSE 流中返回了 OpenAI 风格 error 帧。"""


def _parse_usage(payload: dict) -> tuple[int, int] | None:
    """只在 usage 同时给出可信 prompt/completion 整数时返回 token 计数。"""

    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    for value in (prompt, completion):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            return None
    return int(prompt), int(completion)


async def run_one(
    client: httpx.AsyncClient,
    workload: ServiceWorkload,
    *,
    model: str,
) -> ServiceObservation:
    started_ns = time.monotonic_ns()
    content_times: list[int] = []
    content: list[str] = []
    request_id = "unknown"
    finish_reason = "unknown"
    usage_tokens: tuple[int, int] | None = None
    saw_done = False
    protocol_done_ns: int | None = None
    server_terminal_ns: int | None = None
    terminal_chunk_count = 0
    download_bytes = 0
    async with client.stream(
        "POST", "/v1/chat/completions", json=workload.request_body(model)
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            download_bytes += len(line.encode("utf-8"))
            data = line[5:].lstrip(" ")
            if data == "[DONE]":
                saw_done = True
                protocol_done_ns = time.monotonic_ns()
                break
            payload = json.loads(data)
            if "error" in payload:
                message = payload["error"]
                if isinstance(message, dict):
                    message = message.get("message", "SSE error frame")
                raise SseErrorPayloadError(str(message))
            if payload.get("id"):
                request_id = str(payload["id"])
            parsed = _parse_usage(payload)
            if parsed is not None:
                usage_tokens = parsed
            # choices 允许为空列表（usage-only 帧）或缺失；空内容 delta 不产生内容事件。
            for choice in payload.get("choices") or []:
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta") or {}
                text = delta.get("content")
                if text:
                    content.append(str(text))
                    content_times.append(time.monotonic_ns())
                if choice.get("finish_reason") is not None:
                    finish_reason = str(choice["finish_reason"])
                    terminal_chunk_count += 1
                    if server_terminal_ns is None:
                        server_terminal_ns = time.monotonic_ns()
        status_code = response.status_code
    finished_ns = time.monotonic_ns()

    def relative(ns: int | None) -> float | None:
        return None if ns is None else (ns - started_ns) / 1_000_000

    prompt_tokens = usage_tokens[0] if usage_tokens is not None else None
    completion_tokens = usage_tokens[1] if usage_tokens is not None else None
    inter_chunk_ms = tuple(
        (right - left) / 1_000_000 for left, right in zip(content_times, content_times[1:])
    )
    return ServiceObservation(
        workload=workload.name,
        request_id=request_id,
        status_code=status_code,
        ttft_ms=relative(content_times[0]) if content_times else None,
        latency_ms=(finished_ns - started_ns) / 1_000_000,
        inter_chunk_ms=inter_chunk_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        finish_reason=finish_reason,
        text="".join(content),
        download_bytes=download_bytes,
        last_content_ms=relative(content_times[-1]) if content_times else None,
        server_terminal_ms=relative(server_terminal_ns),
        protocol_done_ms=relative(protocol_done_ns),
        truncated=not saw_done,
        usage_source="usage" if usage_tokens is not None else "unavailable",
        content_chunk_count=len(content_times),
        terminal_chunk_count=terminal_chunk_count,
    )


async def run_workloads(
    base_url: str,
    workloads: tuple[ServiceWorkload, ...],
    *,
    model: str,
    timeout_seconds: float = 60,
) -> tuple[ServiceObservation, ...]:
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout_seconds) as client:
        return tuple([await run_one(client, workload, model=model) for workload in workloads])


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the unique MiniServe HTTP path.")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    workloads = (
        ServiceWorkload("greedy-short", "你好", 2),
        ServiceWorkload("sampled-short", "MiniServe", 2, 0.6, 20, 0.95, 2026),
    )
    observations = asyncio.run(run_workloads(args.url, workloads, model=args.model))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(item.as_dict(), ensure_ascii=False) + "\n" for item in observations),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()