"""自包含并发扫描：启动真实 Uvicorn + 真实权重服务并测量吞吐/分位数。

与 profile_l.py 一样自行拉起服务，因此可以单独执行，也可以被 benchmark 脚本调用。
输出 JSONL 第一行是环境与方法元数据，其余每行是一个并发档位的聚合账本。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import socket
import threading
import time
from pathlib import Path

import torch
import uvicorn

from miniserve.api.bootstrap import build_profile_l_runtime
from miniserve.api.server import create_app

from .aggregate import format_table
from .load_test import DEFAULT_CONCURRENCY, parse_concurrency, run_sweep
from .runner import run_workloads
from .workloads import ServiceWorkload

SCHEMA_VERSION = 1


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_ready(server: uvicorn.Server, thread: threading.Thread, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while not server.started and time.monotonic() < deadline:
        if not thread.is_alive():
            raise RuntimeError("Uvicorn exited before becoming ready.")
        time.sleep(0.01)
    if not server.started:
        raise RuntimeError("Uvicorn did not become ready.")


def run(
    model_lock: Path,
    output: Path,
    *,
    concurrency_levels: tuple[int, ...] = DEFAULT_CONCURRENCY,
    max_tokens: int = 32,
    prompt: str = "Explain paged attention in one sentence.",
    gpu_index: int | None = None,
    sample_interval: float | None = 0.1,
    timeout_seconds: float = 120.0,
) -> tuple[LoadLevel, ...]:
    """执行一次并发扫描并写入 JSONL；返回各并发档位的聚合结果。"""

    service = build_profile_l_runtime(model_lock)
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(create_app(service), host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        _wait_ready(server, thread, timeout=30)
        base_url = f"http://127.0.0.1:{port}"
        workload = ServiceWorkload("concurrency-sweep", prompt, max_tokens)
        # 预热不进入正式行，避免首个档位承担 CUDA/Graph 首次 capture 开销。
        asyncio.run(run_workloads(base_url, (ServiceWorkload("warmup", prompt, 1),), model=service.model_id))
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        levels = asyncio.run(
            run_sweep(
                base_url,
                (workload,),
                model=service.model_id,
                concurrency_levels=concurrency_levels,
                timeout_seconds=timeout_seconds,
                sampler_interval_seconds=sample_interval,
                gpu_index=gpu_index,
            )
        )
        peak_allocated = (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        )
    finally:
        server.should_exit = True
        thread.join(timeout=30)

    metadata: dict[str, object] = {
        "record_type": "metadata",
        "schema_version": SCHEMA_VERSION,
        "profile": "L",
        "scope": "learning-only-low-load-concurrency-sweep",
        "production_claim": False,
        "python": platform.python_version(),
        "concurrency_levels": list(concurrency_levels),
        "prompt_characters": len(prompt),
        "max_tokens": max_tokens,
        "sampling": "greedy",
        "vram_sampler_interval_seconds": sample_interval,
        "gpu_index": gpu_index,
        "torch_peak_allocated_bytes": peak_allocated,
        # 分位数用线性插值；样本数小于 100 时 p99 等于最大值，解读时必须连同样本数。
        "percentile_method": "linear-interpolation",
        "note": "Throughput window starts at the first request launch, so ramp-up dilutes small levels.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in (metadata, *(item.as_dict() for item in levels))
        ),
        encoding="utf-8",
    )
    return levels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep concurrency against a self-hosted real MiniServe service."
    )
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--concurrency",
        default=",".join(str(item) for item in DEFAULT_CONCURRENCY),
    )
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--prompt", default="Explain paged attention in one sentence.")
    parser.add_argument("--gpu-index", type=int, default=None)
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.1,
        help="nvidia-smi sampling interval in seconds; 0 disables sampling.",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    levels = run(
        args.model_lock,
        args.output,
        concurrency_levels=parse_concurrency(args.concurrency),
        max_tokens=args.max_tokens,
        prompt=args.prompt,
        gpu_index=args.gpu_index,
        sample_interval=None if args.sample_interval <= 0 else args.sample_interval,
        timeout_seconds=args.timeout,
    )
    print(format_table([item.aggregate for item in levels]))
    for item in levels:
        aggregate = item.aggregate
        if not aggregate.vram.available:
            print(
                f"note: VRAM sampling unavailable at concurrency={aggregate.concurrency} "
                f"({aggregate.vram.reason})"
            )
        if item.failures:
            print(
                f"warn: {len(item.failures)} failed request(s) at "
                f"concurrency={aggregate.concurrency}: {item.failures[0]}"
            )


if __name__ == "__main__":
    main()
