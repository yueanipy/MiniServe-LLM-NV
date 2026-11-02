"""自包含真实权重/真实 Uvicorn/低负载 Profile L benchmark。"""

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

from .compare_hf import comparison_status as hf_status
from .compare_vllm import comparison_status as vllm_status
from .runner import run_workloads
from .workloads import ServiceWorkload


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def run(model_lock: Path, output: Path) -> list[dict[str, object]]:
    service = build_profile_l_runtime(model_lock)
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(create_app(service), host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 30
        while not server.started and time.monotonic() < deadline:
            if not thread.is_alive():
                raise RuntimeError("Uvicorn exited before becoming ready.")
            time.sleep(0.01)
        if not server.started:
            raise RuntimeError("Uvicorn did not become ready.")
        base_url = f"http://127.0.0.1:{port}"
        # 一次短预热不进入正式 rows；正式仍只有两个 batch=1、2-token 请求。
        asyncio.run(
            run_workloads(
                base_url,
                (ServiceWorkload("warmup", "你好", 1),),
                model=service.model_id,
            )
        )
        torch.cuda.reset_peak_memory_stats()
        workloads = (
            ServiceWorkload("greedy-short", "你好", 2),
            ServiceWorkload("sampled-short", "MiniServe", 2, 0.6, 20, 0.95, 2026),
        )
        observations = asyncio.run(
            run_workloads(base_url, workloads, model=service.model_id)
        )
        rows = [item.as_dict() for item in observations]
        metadata = {
            "record_type": "metadata",
            "profile": "L",
            "scope": "learning-only-low-load-real-http",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "peak_allocated_bytes_during_measured_requests": int(torch.cuda.max_memory_allocated()),
            "warmup_requests": 1,
            "measured_requests": len(rows),
            "max_concurrency": 1,
            "hf_comparison": hf_status(),
            "vllm_comparison": vllm_status(),
            "production_claim": False,
        }
        rows.insert(0, metadata)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        return rows
    finally:
        server.should_exit = True
        thread.join(timeout=15)
        if thread.is_alive():
            raise RuntimeError("Uvicorn did not stop cleanly.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-lock",
        type=Path,
        default=Path("model-locks/qwen3-0.6b-profile-l.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = run(args.model_lock, args.output)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
