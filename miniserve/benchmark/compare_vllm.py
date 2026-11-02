"""顺序启动真实 MiniServe/vLLM HTTP 服务的同权重低负载比较。

vLLM 故意安装在独立虚拟环境中；本模块不 import vllm，也不会把“能
import”冒充成 benchmark。两个服务依次占用 GPU，适合 Profile D。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import httpx

from miniserve.benchmark.compare_hf import compare_latency, summarize_samples
from miniserve.benchmark.metrics import ServiceObservation
from miniserve.benchmark.workloads import ServiceWorkload


DEFAULT_VLLM_PYTHON = Path(
    "~/.local/share/miniserve-llm-nv/vllm-baseline/venv/bin/python"
).expanduser()
# vLLM 0.27 的 V2 runner 依赖 UVA；WSL 默认禁用 pinned/UVA host buffer。
# 官方 issue 给出的兼容路径是退回 V1 model runner，而不是开启不安全的伪 UVA。
VLLM_ENV_OVERRIDES = {"VLLM_USE_V2_MODEL_RUNNER": "0"}


def comparison_status(vllm_python: str | Path | None = None) -> dict[str, object]:
    """报告真实外部服务入口是否存在，不检查当前环境能否 import vllm。"""

    configured = Path(
        vllm_python
        or os.environ.get("MINISERVE_VLLM_PYTHON", str(DEFAULT_VLLM_PYTHON))
    ).expanduser()
    available = configured.is_file() and os.access(configured, os.X_OK)
    return {
        "backend": "vllm",
        "available": available,
        "status": (
            "real_external_service_benchmark_available"
            if available
            else "isolated_vllm_python_missing"
        ),
        "formal_service_benchmark": False,
        "command": "python -m miniserve.benchmark.compare_vllm --output <path>",
        "vllm_python": str(configured),
        "scope": "same locked weights, sequential localhost services, Profile D",
    }


def build_vllm_command(
    vllm_python: str | Path,
    *,
    snapshot: str | Path,
    model_id: str,
    port: int,
    gpu_memory_utilization: float = 0.70,
) -> tuple[str, ...]:
    """构造固定的低负载 vLLM 服务命令，便于单测审查所有边界。"""

    python_path = Path(vllm_python).expanduser()
    snapshot_path = Path(snapshot).expanduser()
    if not str(model_id).strip():
        raise ValueError("model_id must not be empty.")
    if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
        raise ValueError("port must be a valid TCP port.")
    if not 0.1 <= gpu_memory_utilization <= 0.9:
        raise ValueError("gpu_memory_utilization must be between 0.1 and 0.9.")
    return (
        str(python_path),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(snapshot_path),
        "--served-model-name",
        model_id,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--dtype",
        "half",
        "--max-model-len",
        "64",
        "--max-num-seqs",
        "2",
        "--max-num-batched-tokens",
        "64",
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--generation-config",
        "vllm",
        "--default-chat-template-kwargs",
        '{"enable_thinking":false}',
    )


def _build_miniserve_command(
    python_path: str | Path,
    *,
    model_lock: str | Path,
    port: int,
) -> tuple[str, ...]:
    return (
        str(Path(python_path)),
        "-m",
        "miniserve.api.server",
        "--model-lock",
        str(Path(model_lock)),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--max-model-len",
        "64",
        "--num-kv-blocks",
        "8",
        "--max-num-seqs",
        "2",
        "--max-pending-requests",
        "4",
    )


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass
class _RunningServer:
    process: subprocess.Popen[str]
    log_lines: deque[str]
    reader: threading.Thread

    def log_tail(self) -> str:
        return "\n".join(self.log_lines)

    def stop(self) -> None:
        if os.name == "posix":
            # vLLM 会派生 EngineCore 子进程；即使 API 父进程先失败，也要清理整组。
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        elif self.process.poll() is None:
            self.process.terminate()
        if self.process.poll() is None:
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    try:
                        os.killpg(self.process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    self.process.kill()
                self.process.wait(timeout=10)
        self.reader.join(timeout=5)


@contextmanager
def _serve(
    command: Sequence[str],
    *,
    cwd: Path,
    base_url: str,
    ready_path: str,
    startup_timeout_seconds: float,
    env_overrides: dict[str, str] | None = None,
) -> Iterator[_RunningServer]:
    if startup_timeout_seconds <= 0:
        raise ValueError("startup_timeout_seconds must be positive.")
    child_env = os.environ.copy()
    if env_overrides:
        child_env.update(env_overrides)
    process = subprocess.Popen(
        tuple(command),
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=child_env,
        start_new_session=(os.name == "posix"),
    )
    assert process.stdout is not None
    lines: deque[str] = deque(maxlen=240)

    def drain() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            lines.append(line.rstrip())

    reader = threading.Thread(target=drain, name="benchmark-server-log", daemon=True)
    reader.start()
    running = _RunningServer(process, lines, reader)
    try:
        deadline = time.monotonic() + startup_timeout_seconds
        last_error = "not contacted"
        while time.monotonic() < deadline:
            return_code = process.poll()
            if return_code is not None:
                raise RuntimeError(
                    f"Server exited during startup with code {return_code}.\n{running.log_tail()}"
                )
            try:
                response = httpx.get(
                    f"{base_url}{ready_path}", timeout=1.0, follow_redirects=False
                )
                if response.status_code == 200:
                    break
                last_error = f"HTTP {response.status_code}"
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(0.25)
        else:
            raise RuntimeError(
                f"Server readiness timed out ({last_error}).\n{running.log_tail()}"
            )
        yield running
    finally:
        running.stop()


def _request_body(
    workload: ServiceWorkload,
    *,
    model_id: str,
    backend: str,
) -> dict[str, object]:
    body: dict[str, object] = {
        "model": model_id,
        "messages": [{"role": "user", "content": workload.prompt}],
        "max_tokens": workload.max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "stream": True,
    }
    if backend == "vllm":
        # vLLM 只在显式请求时把 usage 放进最后一个 choices=[] 的流式 chunk。
        body["stream_options"] = {"include_usage": True}
        body["chat_template_kwargs"] = {"enable_thinking": False}
    elif backend != "miniserve":
        raise ValueError(f"Unknown benchmark backend: {backend}")
    return body


async def _run_one(
    client: httpx.AsyncClient,
    workload: ServiceWorkload,
    *,
    model_id: str,
    backend: str,
) -> ServiceObservation:
    """解析 MiniServe 与 vLLM 两种合法 SSE 终止布局。"""

    started_ns = time.monotonic_ns()
    content_times: list[int] = []
    content: list[str] = []
    request_id = "unknown"
    finish_reason = "unknown"
    usage = None
    saw_done = False
    saw_final_choice = False
    server_terminal_ns: int | None = None
    protocol_done_ns: int | None = None
    terminal_chunk_count = 0
    async with client.stream(
        "POST",
        "/v1/chat/completions",
        json=_request_body(workload, model_id=model_id, backend=backend),
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                saw_done = True
                protocol_done_ns = time.monotonic_ns()
                break
            payload = json.loads(data)
            if "error" in payload:
                raise RuntimeError(payload["error"].get("message", "backend error"))
            if payload.get("id"):
                request_id = str(payload["id"])
            if payload.get("usage") is not None:
                usage = payload["usage"]
            choices = payload.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta", {}).get("content")
            if delta:
                content.append(str(delta))
                content_times.append(time.monotonic_ns())
            if choice.get("finish_reason") is not None:
                finish_reason = str(choice["finish_reason"])
                saw_final_choice = True
                terminal_chunk_count += 1
                if server_terminal_ns is None:
                    server_terminal_ns = time.monotonic_ns()
        status_code = response.status_code
    finished_ns = time.monotonic_ns()
    if usage is None or not content_times or not saw_final_choice or not saw_done:
        raise RuntimeError(
            f"{backend} SSE omitted content timing, final choice/usage, or [DONE]."
        )
    inter_chunk_ms = tuple(
        (right - left) / 1_000_000
        for left, right in zip(content_times, content_times[1:])
    )
    def relative(value: int | None) -> float | None:
        return None if value is None else (value - started_ns) / 1_000_000

    return ServiceObservation(
        workload=workload.name,
        request_id=request_id,
        status_code=status_code,
        ttft_ms=relative(content_times[0]) if content_times else None,
        latency_ms=(finished_ns - started_ns) / 1_000_000,
        inter_chunk_ms=inter_chunk_ms,
        prompt_tokens=int(usage["prompt_tokens"]),
        completion_tokens=int(usage["completion_tokens"]),
        finish_reason=finish_reason,
        text="".join(content),
        last_content_ms=relative(content_times[-1]) if content_times else None,
        server_terminal_ms=relative(server_terminal_ns),
        protocol_done_ms=relative(protocol_done_ns),
        truncated=not saw_done,
        usage_source="usage",
        content_chunk_count=len(content_times),
        terminal_chunk_count=terminal_chunk_count,
    )


async def _capture_endpoint(
    base_url: str,
    workloads: Sequence[ServiceWorkload],
    *,
    model_id: str,
    backend: str,
    warmup: int,
    iterations: int,
) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    async with httpx.AsyncClient(base_url=base_url, timeout=120.0) as client:
        for workload in workloads:
            for warmup_index in range(warmup):
                await _run_one(
                    client,
                    ServiceWorkload(
                        f"{workload.name}-warmup-{warmup_index}",
                        workload.prompt,
                        workload.max_tokens,
                    ),
                    model_id=model_id,
                    backend=backend,
                )
            observations = tuple(
                [
                    await _run_one(
                        client,
                        ServiceWorkload(
                            f"{workload.name}-{index}",
                            workload.prompt,
                            workload.max_tokens,
                        ),
                        model_id=model_id,
                        backend=backend,
                    )
                    for index in range(iterations)
                ]
            )
            tpot_values = [
                item.tpot_ms for item in observations if item.tpot_ms is not None
            ]
            rows[workload.name] = {
                "warmup_iterations": warmup,
                "measured_iterations": iterations,
                "ttft_ms": summarize_samples([item.ttft_ms for item in observations]),
                "latency_ms": summarize_samples(
                    [item.latency_ms for item in observations]
                ),
                "tpot_ms": summarize_samples(tpot_values) if tpot_values else None,
                "completion_tokens": [item.completion_tokens for item in observations],
                "prompt_tokens": [item.prompt_tokens for item in observations],
                "texts": [item.text for item in observations],
                "finish_reasons": [item.finish_reason for item in observations],
                "output_tokens_per_second": summarize_samples(
                    [item.output_tokens_per_second for item in observations if item.output_tokens_per_second is not None]
                ),
            }
    return rows


def compare_captures(
    miniserve: dict[str, dict[str, object]],
    vllm: dict[str, dict[str, object]],
    workloads: Sequence[ServiceWorkload],
) -> tuple[list[dict[str, object]], bool, bool]:
    reports: list[dict[str, object]] = []
    all_protocol_valid = True
    all_exact_match = True
    for workload in workloads:
        own = miniserve[workload.name]
        reference = vllm[workload.name]
        own_counts = tuple(int(value) for value in own["completion_tokens"])
        reference_counts = tuple(int(value) for value in reference["completion_tokens"])
        own_texts = tuple(str(value) for value in own["texts"])
        reference_texts = tuple(str(value) for value in reference["texts"])
        own_prompt_tokens = tuple(int(value) for value in own["prompt_tokens"])
        reference_prompt_tokens = tuple(
            int(value) for value in reference["prompt_tokens"]
        )
        own_reasons = tuple(str(value) for value in own["finish_reasons"])
        reference_reasons = tuple(str(value) for value in reference["finish_reasons"])
        stable = (
            bool(own_texts)
            and bool(reference_texts)
            and len(set(own_texts)) == 1
            and len(set(reference_texts)) == 1
        )
        fixed_counts = (
            all(value == workload.max_tokens for value in own_counts)
            and all(value == workload.max_tokens for value in reference_counts)
        )
        prompt_contract_match = (
            bool(own_prompt_tokens)
            and bool(reference_prompt_tokens)
            and len(set(own_prompt_tokens)) == 1
            and len(set(reference_prompt_tokens)) == 1
            and own_prompt_tokens[0] == reference_prompt_tokens[0]
        )
        finish_contract_valid = (
            bool(own_reasons)
            and bool(reference_reasons)
            and len(set(own_reasons)) == 1
            and len(set(reference_reasons)) == 1
            and all(value in {"stop", "length"} for value in own_reasons)
            and all(value in {"stop", "length"} for value in reference_reasons)
        )
        protocol_valid = (
            stable and fixed_counts and prompt_contract_match and finish_contract_valid
        )
        exact_match = stable and own_texts[0] == reference_texts[0]
        all_protocol_valid &= protocol_valid
        all_exact_match &= exact_match
        own_latency = float(own["latency_ms"]["median"])
        reference_latency = float(reference["latency_ms"]["median"])
        own_ttft = float(own["ttft_ms"]["median"])
        reference_ttft = float(reference["ttft_ms"]["median"])
        reports.append(
            {
                "workload": workload.name,
                "prompt": workload.prompt,
                "max_new_tokens": workload.max_tokens,
                "miniserve": own,
                "vllm": reference,
                "correctness": {
                    "both_backends_deterministic": stable,
                    "exact_text_match": exact_match,
                    "completion_counts_match_fixed_length": fixed_counts,
                    "prompt_token_counts_match": prompt_contract_match,
                    "finish_reasons_valid_and_stable": finish_contract_valid,
                    "finish_reasons_exact_match": (
                        finish_contract_valid and own_reasons[0] == reference_reasons[0]
                    ),
                    "performance_workload_contract_valid": protocol_valid,
                },
                "comparisons": {
                    "miniserve_vs_vllm_latency": compare_latency(
                        reference_latency, own_latency
                    ),
                    "miniserve_vs_vllm_ttft": compare_latency(
                        reference_ttft, own_ttft
                    ),
                },
            }
        )
    return reports, all_protocol_valid, all_exact_match


def _package_version(python_path: Path, package: str) -> str:
    completed = subprocess.run(
        (
            str(python_path),
            "-c",
            f"import {package}; print({package}.__version__)",
        ),
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return completed.stdout.strip()


def run(
    model_lock: Path,
    output: Path,
    *,
    vllm_python: Path = DEFAULT_VLLM_PYTHON,
    warmup: int = 1,
    iterations: int = 5,
    max_new_tokens: int = 8,
    startup_timeout_seconds: float = 600,
) -> dict[str, object]:
    if warmup < 0 or iterations < 3 or max_new_tokens < 2:
        raise ValueError("Use warmup>=0, iterations>=3 and max_new_tokens>=2.")
    # venv/bin/python 通常是指向 /usr/bin/python 的符号链接；resolve() 会丢失
    # pyvenv.cfg 上下文，使子进程退回系统 Python。absolute() 只补全路径，不跟随链接。
    vllm_python = vllm_python.expanduser().absolute()
    if not vllm_python.is_file():
        raise RuntimeError(f"Isolated vLLM Python does not exist: {vllm_python}")

    from miniserve.model.loader import resolve_model_revision

    snapshot, revision = resolve_model_revision(model_lock)
    lock = json.loads(model_lock.read_text(encoding="utf-8"))
    model_id = str(lock.get("model_id", ""))
    if not model_id:
        raise ValueError("model lock has no model_id.")
    workloads = (
        ServiceWorkload("short-chat", "你好", max_new_tokens),
        ServiceWorkload(
            "repeated-prefix-chat", "请简短解释KV缓存的作用。", max_new_tokens
        ),
    )
    root = Path(__file__).resolve().parents[2]

    own_port = _free_port()
    own_url = f"http://127.0.0.1:{own_port}"
    own_command = _build_miniserve_command(
        sys.executable, model_lock=model_lock, port=own_port
    )
    with _serve(
        own_command,
        cwd=root,
        base_url=own_url,
        ready_path="/health/ready",
        startup_timeout_seconds=startup_timeout_seconds,
    ):
        own_rows = asyncio.run(
            _capture_endpoint(
                own_url,
                workloads,
                model_id=model_id,
                backend="miniserve",
                warmup=warmup,
                iterations=iterations,
            )
        )

    # 等待被终止的 CUDA context 完全退出，再启动另一个独占 GPU 的服务。
    time.sleep(2.0)
    vllm_port = _free_port()
    vllm_url = f"http://127.0.0.1:{vllm_port}"
    vllm_command = build_vllm_command(
        vllm_python,
        snapshot=snapshot,
        model_id=model_id,
        port=vllm_port,
    )
    with _serve(
        vllm_command,
        cwd=root,
        base_url=vllm_url,
        ready_path="/health",
        startup_timeout_seconds=startup_timeout_seconds,
        env_overrides=VLLM_ENV_OVERRIDES,
    ):
        vllm_rows = asyncio.run(
            _capture_endpoint(
                vllm_url,
                workloads,
                model_id=model_id,
                backend="vllm",
                warmup=warmup,
                iterations=iterations,
            )
        )

    workload_reports, all_protocol_valid, all_exact_match = compare_captures(
        own_rows, vllm_rows, workloads
    )
    faster_latency = all(
        row["comparisons"]["miniserve_vs_vllm_latency"]["verdict"] == "faster"
        for row in workload_reports
    )
    report = {
        "schema_version": 1,
        "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "Profile D low-load same-weight sequential HTTP service comparison",
        "production_claim": False,
        "model_lock": str(model_lock),
        "model_snapshot": str(snapshot),
        "resolved_revision": revision,
        "environment": {
            "python": platform.python_version(),
            "vllm": _package_version(vllm_python, "vllm"),
        },
        "methodology": {
            "services_resident_sequentially": True,
            "same_locked_snapshot": True,
            "same_chat_template_kwargs": {"enable_thinking": False},
            "batch_size": 1,
            "max_model_len": 64,
            "max_num_seqs": 2,
            "greedy": True,
            "warmup_iterations": warmup,
            "measured_iterations": iterations,
            "model_load_and_server_start_excluded": True,
            "protocol": "localhost OpenAI-compatible HTTP/SSE",
            "noise_band_percent": 3.0,
            "miniserve_command": list(own_command),
            "vllm_command": list(vllm_command),
            "vllm_environment_overrides": dict(VLLM_ENV_OVERRIDES),
        },
        "workloads": workload_reports,
        "feasibility": {
            "both_real_http_services_completed": True,
            "all_performance_workload_contracts_valid": all_protocol_valid,
            "all_outputs_exact_match": all_exact_match,
            "performance_is_reported_not_assumed": True,
        },
        "observed_performance": {
            "miniserve_faster_than_vllm_latency_on_all_workloads": faster_latency,
        },
        "limitations": [
            "RTX 5060 Ti Profile D sm_120; low-load comparison only",
            "batch=1, max_model_len=64, short deterministic outputs",
            "no sustained concurrency, long soak, energy or production SLO claim",
            "HTTP wall time includes each server's tokenizer, scheduler and SSE overhead",
            "Exact greedy text can diverge across optimized FP16 kernels at a near-tied logit; the report never hides this",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not all_protocol_valid:
        raise RuntimeError("MiniServe and vLLM workload/protocol contracts did not match.")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare sequential MiniServe and vLLM HTTP services on one locked model."
    )
    parser.add_argument(
        "--model-lock",
        type=Path,
        default=Path("model-locks/qwen3-0.6b-profile-l.json"),
    )
    parser.add_argument("--vllm-python", type=Path, default=DEFAULT_VLLM_PYTHON)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--startup-timeout-seconds", type=float, default=600)
    args = parser.parse_args()
    report = run(
        args.model_lock,
        args.output,
        vllm_python=args.vllm_python,
        warmup=args.warmup,
        iterations=args.iterations,
        max_new_tokens=args.max_new_tokens,
        startup_timeout_seconds=args.startup_timeout_seconds,
    )
    print(
        json.dumps(
            {
                "feasibility": report["feasibility"],
                "observed_performance": report["observed_performance"],
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
