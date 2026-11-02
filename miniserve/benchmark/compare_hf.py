"""同权重、同 token、低负载的 HF 与 MiniServe 真实对比基准。

这个模块不把“能启动 HTTP”误写成性能提升。它顺序加载 HF 与 MiniServe，避免
两份 0.6B 权重同时驻留 6 GiB 显存，并分别测量：

1. Hugging Face 每 token 完整重算；
2. Hugging Face 原生 KV cache；
3. MiniServe 自建模型完整重算；
4. MiniServe Paged KV + CUDA RMSNorm + CUDA paged decode + full decode Graph；
5. MiniServe 真实 localhost HTTP/SSE。

所有 direct 路径使用相同 chat-template token IDs、batch=1、greedy 和固定输出长度。
基准只用于 Profile L 学习结论，不是生产吞吐声明。
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import math
import platform
import socket
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


def comparison_status() -> dict[str, object]:
    """供服务 benchmark metadata 使用；明确实际比较入口与边界。"""

    return {
        "backend": "hugging-face-transformers",
        "status": "explicit_real_weight_benchmark_available",
        "formal_service_benchmark": False,
        "command": "python -m miniserve.benchmark.compare_hf --output <path>",
        "scope": "same locked weights, batch=1, short-context Profile L",
    }


@dataclass(frozen=True)
class ComparisonWorkload:
    name: str
    prompt: str
    max_new_tokens: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Comparison workload name must not be empty.")
        if not isinstance(self.prompt, str) or not self.prompt:
            raise ValueError("Comparison workload prompt must not be empty.")
        if (
            not isinstance(self.max_new_tokens, int)
            or isinstance(self.max_new_tokens, bool)
            or self.max_new_tokens <= 1
        ):
            raise ValueError("Comparison requires at least two generated tokens.")


def _nearest_rank(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def summarize_samples(samples: Sequence[float]) -> dict[str, float]:
    """汇总正的有限耗时；公开为纯函数，便于 CPU 单元测试。"""

    values = tuple(float(value) for value in samples)
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Benchmark samples must be finite positive values.")
    return {
        "min": min(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": _nearest_rank(values, 0.95),
        "max": max(values),
    }


def compare_latency(reference_ms: float, candidate_ms: float) -> dict[str, object]:
    """正值表示 candidate 更快；3% 内只记为测量噪声区间。"""

    if (
        not math.isfinite(reference_ms)
        or not math.isfinite(candidate_ms)
        or reference_ms <= 0
        or candidate_ms <= 0
    ):
        raise ValueError("Latency comparison requires finite positive values.")
    delta_percent = (reference_ms - candidate_ms) / reference_ms * 100.0
    if delta_percent > 3.0:
        verdict = "faster"
    elif delta_percent < -3.0:
        verdict = "slower"
    else:
        verdict = "within_3_percent_noise_band"
    return {
        "reference_median_ms": reference_ms,
        "candidate_median_ms": candidate_ms,
        "speedup_ratio": reference_ms / candidate_ms,
        "latency_reduction_percent": delta_percent,
        "verdict": verdict,
    }


def compare_memory(reference_bytes: int, candidate_bytes: int) -> dict[str, object]:
    """比较稳态 allocated bytes；负 reduction 明确表示候选占用更多。"""

    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in (reference_bytes, candidate_bytes)
    ):
        raise ValueError("Memory comparison requires positive integer byte counts.")
    reduction_percent = (reference_bytes - candidate_bytes) / reference_bytes * 100.0
    return {
        "reference_bytes": reference_bytes,
        "candidate_bytes": candidate_bytes,
        "candidate_minus_reference_bytes": candidate_bytes - reference_bytes,
        "memory_reduction_percent": reduction_percent,
        "verdict": (
            "lower" if candidate_bytes < reference_bytes else
            "higher" if candidate_bytes > reference_bytes else
            "equal"
        ),
    }


def reference_path_matches(
    hf_recompute: tuple[int, ...],
    hf_cache: tuple[int, ...],
    custom_recompute: tuple[int, ...],
    custom_final: tuple[int, ...],
) -> dict[str, bool]:
    """Compare each candidate path to its corresponding HF execution mode."""

    return {
        "hf_recompute_matches_custom_recompute": hf_recompute == custom_recompute,
        "hf_cache_matches_custom_final": hf_cache == custom_final,
        "hf_cross_mode_equal": hf_recompute == hf_cache,
    }


def _chat_token_ids(tokenizer: object, prompt: str) -> tuple[int, ...]:
    values = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if hasattr(values, "tolist"):
        values = values.tolist()
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in values)
    ):
        raise ValueError("Tokenizer returned invalid comparison token IDs.")
    return tuple(values)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _measure_mode(
    torch_module,
    run_once: Callable[[], tuple[tuple[int, ...], float, float]],
    *,
    warmup: int,
    iterations: int,
    expected_tokens: tuple[int, ...] | None = None,
) -> tuple[dict[str, object], tuple[int, ...]]:
    """预热不计入样本；每次输出 token 必须稳定且长度相同。"""

    observed: tuple[int, ...] | None = None
    for _ in range(warmup):
        tokens, _, _ = run_once()
        if observed is None:
            observed = tokens
        elif tokens != observed:
            raise RuntimeError("Benchmark warmup produced unstable greedy tokens.")

    gc.collect()
    torch_module.cuda.synchronize()
    resident_before = int(torch_module.cuda.memory_allocated())
    torch_module.cuda.reset_peak_memory_stats()
    ttft_samples: list[float] = []
    latency_samples: list[float] = []
    for _ in range(iterations):
        tokens, ttft_ms, latency_ms = run_once()
        if observed is None:
            observed = tokens
        elif tokens != observed:
            raise RuntimeError("Benchmark measurement produced unstable greedy tokens.")
        if len(tokens) < 2 or not 0 < ttft_ms <= latency_ms:
            raise RuntimeError("Benchmark returned an invalid timing/token observation.")
        ttft_samples.append(ttft_ms)
        latency_samples.append(latency_ms)
    assert observed is not None
    peak = int(torch_module.cuda.max_memory_allocated())
    ttft = summarize_samples(ttft_samples)
    latency = summarize_samples(latency_samples)
    generated = len(observed)
    decode_seconds = max((latency["median"] - ttft["median"]) / 1000.0, 1e-9)
    return (
        {
            "tokens": list(observed),
            "matches_expected_tokens": expected_tokens is None or observed == expected_tokens,
            "generated_tokens": generated,
            "warmup_iterations": warmup,
            "measured_iterations": iterations,
            "ttft_ms": ttft,
            "latency_ms": latency,
            "request_output_tokens_per_second": generated / (latency["median"] / 1000.0),
            "decode_tokens_per_second_excluding_first": (generated - 1) / decode_seconds,
            "resident_allocated_bytes_before_measurement": resident_before,
            "peak_allocated_bytes": peak,
            "incremental_peak_bytes": max(0, peak - resident_before),
        },
        observed,
    )


def _hf_once(torch_module, model, token_ids: tuple[int, ...], max_new_tokens: int, *, use_cache: bool):
    """只调用 forward + argmax；TTFT 和总耗时均含 CPU→GPU prompt copy。"""

    torch_module.cuda.synchronize()
    started_ns = time.perf_counter_ns()
    generated: list[int] = []
    first_token_ns: int | None = None
    with torch_module.inference_mode():
        if use_cache:
            current = torch_module.tensor([token_ids], dtype=torch_module.long, device="cuda:0")
            past = None
            for _ in range(max_new_tokens):
                output = model(
                    input_ids=current,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
                next_token = torch_module.argmax(output.logits[:, -1, :], dim=-1)
                token_id = int(next_token.item())
                generated.append(token_id)
                if first_token_ns is None:
                    torch_module.cuda.synchronize()
                    first_token_ns = time.perf_counter_ns()
                past = output.past_key_values
                current = next_token.reshape(1, 1)
        else:
            history = torch_module.tensor([token_ids], dtype=torch_module.long, device="cuda:0")
            for _ in range(max_new_tokens):
                output = model(input_ids=history, use_cache=False, return_dict=True)
                next_token = torch_module.argmax(output.logits[:, -1, :], dim=-1)
                token_id = int(next_token.item())
                generated.append(token_id)
                if first_token_ns is None:
                    torch_module.cuda.synchronize()
                    first_token_ns = time.perf_counter_ns()
                history = torch_module.cat((history, next_token.reshape(1, 1)), dim=1)
    torch_module.cuda.synchronize()
    finished_ns = time.perf_counter_ns()
    assert first_token_ns is not None
    return (
        tuple(generated),
        (first_token_ns - started_ns) / 1_000_000.0,
        (finished_ns - started_ns) / 1_000_000.0,
    )


def _custom_recompute_once(torch_module, model, token_ids: tuple[int, ...], max_new_tokens: int):
    """MiniServe 自建 dense model 的完整历史重算基线。"""

    torch_module.cuda.synchronize()
    started_ns = time.perf_counter_ns()
    history = torch_module.tensor([token_ids], dtype=torch_module.long, device="cuda:0")
    generated: list[int] = []
    first_token_ns: int | None = None
    with torch_module.inference_mode():
        for _ in range(max_new_tokens):
            logits, _ = model.forward_recompute(history)
            next_token = torch_module.argmax(logits[:, -1, :], dim=-1)
            token_id = int(next_token.item())
            generated.append(token_id)
            if first_token_ns is None:
                torch_module.cuda.synchronize()
                first_token_ns = time.perf_counter_ns()
            history = torch_module.cat((history, next_token.reshape(1, 1)), dim=1)
    torch_module.cuda.synchronize()
    finished_ns = time.perf_counter_ns()
    assert first_token_ns is not None
    return (
        tuple(generated),
        (first_token_ns - started_ns) / 1_000_000.0,
        (finished_ns - started_ns) / 1_000_000.0,
    )


def _hf_teacher_logits(torch_module, model, token_ids, expected_tokens):
    """沿 HF greedy 历史保存每一步 CPU FP32 logits，用于分歧归因，不计时。"""

    history = torch_module.tensor([token_ids], dtype=torch_module.long, device="cuda:0")
    rows = []
    with torch_module.inference_mode():
        for token_id in expected_tokens:
            output = model(input_ids=history, use_cache=False, return_dict=True)
            rows.append(output.logits[:, -1, :].detach().to(device="cpu", dtype=torch_module.float32))
            history = torch_module.cat(
                (
                    history,
                    torch_module.tensor([[token_id]], dtype=torch_module.long, device="cuda:0"),
                ),
                dim=1,
            )
    return tuple(rows)


def _custom_teacher_diagnostics(
    torch_module, model, token_ids, expected_tokens, hf_logits_by_step
):
    """同一 teacher-forced 历史比较 logits、top margin 与实际数值误差。"""

    history = torch_module.tensor([token_ids], dtype=torch_module.long, device="cuda:0")
    diagnostics = []
    with torch_module.inference_mode():
        for step, (expected_token, hf_logits) in enumerate(
            zip(expected_tokens, hf_logits_by_step, strict=True)
        ):
            custom_logits_gpu, _ = model.forward_recompute(history)
            custom_logits = custom_logits_gpu[:, -1, :].detach().to(
                device="cpu", dtype=torch_module.float32
            )
            difference = (custom_logits - hf_logits).abs()
            # topk 对完全相等值的返回顺序不保证等同 argmax；greedy 使用 argmax，
            # 因此 runner-up 必须先屏蔽真实 argmax 再求一次，而不能直接信 topk[0]。
            hf_top1 = int(torch_module.argmax(hf_logits, dim=-1).item())
            custom_top1 = int(torch_module.argmax(custom_logits, dim=-1).item())
            hf_without_top1 = hf_logits.clone()
            custom_without_top1 = custom_logits.clone()
            hf_without_top1[0, hf_top1] = -torch_module.inf
            custom_without_top1[0, custom_top1] = -torch_module.inf
            hf_top2 = int(torch_module.argmax(hf_without_top1, dim=-1).item())
            custom_top2 = int(torch_module.argmax(custom_without_top1, dim=-1).item())
            hf_margin = float((hf_logits[0, hf_top1] - hf_logits[0, hf_top2]).item())
            custom_margin = float(
                (custom_logits[0, custom_top1] - custom_logits[0, custom_top2]).item()
            )
            max_abs_error = float(difference.max().item())
            diagnostics.append(
                {
                    "step": step,
                    "teacher_forced_expected_token": int(expected_token),
                    "hf_top1": hf_top1,
                    "hf_top2": hf_top2,
                    "custom_top1": custom_top1,
                    "custom_top2": custom_top2,
                    "teacher_forced_top1_match": hf_top1 == custom_top1,
                    "hf_top1_margin": hf_margin,
                    "custom_top1_margin": custom_margin,
                    "max_abs_logit_error": max_abs_error,
                    "mean_abs_logit_error": float(difference.mean().item()),
                    "near_tie_relative_to_error": hf_margin <= 2.0 * max_abs_error,
                }
            )
            history = torch_module.cat(
                (
                    history,
                    torch_module.tensor(
                        [[expected_token]], dtype=torch_module.long, device="cuda:0"
                    ),
                ),
                dim=1,
            )
    return diagnostics


def _core_once(torch_module, core, token_ids: tuple[int, ...], max_new_tokens: int, request_id: str):
    """最终 EngineCore 主链；耗时包含 admission、scheduler、KV 与事件发布。"""

    from miniserve.engine.events import EngineEventKind
    from miniserve.engine.request import EngineRequest, SamplingParams
    from miniserve.engine.state import RequestStatus

    request = EngineRequest(
        request_id,
        token_ids,
        SamplingParams(max_new_tokens=max_new_tokens, stop_token_ids=()),
        time.monotonic_ns(),
    )
    torch_module.cuda.synchronize()
    started_ns = time.perf_counter_ns()
    core.submit(request)
    first_token_ns: int | None = None
    result = None
    max_steps = math.ceil(len(token_ids) / 16) + max_new_tokens + 4
    for _ in range(max_steps):
        core.step_multi()
        torch_module.cuda.synchronize()
        if first_token_ns is None and any(
            event.kind is EngineEventKind.TOKEN for event in core.publisher.events(request_id)
        ):
            first_token_ns = time.perf_counter_ns()
        result = core.finalized_result(request_id)
        if result is not None:
            break
    finished_ns = time.perf_counter_ns()
    if result is None or result.status is not RequestStatus.FINISHED:
        raise RuntimeError("MiniServe direct benchmark request did not finish successfully.")
    if result.finish_reason != "length" or len(result.generated_token_ids) != max_new_tokens:
        raise RuntimeError("MiniServe direct benchmark did not generate the fixed token count.")
    if first_token_ns is None:
        raise RuntimeError("MiniServe direct benchmark published no token event.")
    return (
        result.generated_token_ids,
        (first_token_ns - started_ns) / 1_000_000.0,
        (finished_ns - started_ns) / 1_000_000.0,
    )


def _http_modes(torch_module, service, workloads, expected_by_name, *, warmup: int, iterations: int):
    """在同一个事件循环和 HTTP client 生命周期中执行真实 localhost SSE。"""

    import uvicorn

    from miniserve.api.server import create_app
    from miniserve.benchmark.runner import run_workloads
    from miniserve.benchmark.workloads import ServiceWorkload

    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(create_app(service), host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 30.0
        while not server.started and time.monotonic() < deadline:
            if not thread.is_alive():
                raise RuntimeError("Comparison Uvicorn exited before startup.")
            time.sleep(0.01)
        if not server.started:
            raise RuntimeError("Comparison Uvicorn did not become ready.")

        async def collect():
            rows = {}
            for workload in workloads:
                warm = tuple(
                    ServiceWorkload(
                        f"{workload.name}-http-warm-{index}",
                        workload.prompt,
                        workload.max_new_tokens,
                    )
                    for index in range(warmup)
                )
                if warm:
                    await run_workloads(
                        f"http://127.0.0.1:{port}", warm, model=service.model_id
                    )
                measured = tuple(
                    ServiceWorkload(
                        f"{workload.name}-http-{index}",
                        workload.prompt,
                        workload.max_new_tokens,
                    )
                    for index in range(iterations)
                )
                observations = await run_workloads(
                    f"http://127.0.0.1:{port}", measured, model=service.model_id
                )
                rows[workload.name] = observations
            return rows

        gc.collect()
        torch_module.cuda.synchronize()
        resident = int(torch_module.cuda.memory_allocated())
        torch_module.cuda.reset_peak_memory_stats()
        observations_by_name = asyncio.run(collect())
        peak = int(torch_module.cuda.max_memory_allocated())
        result = {}
        for workload in workloads:
            observations = observations_by_name[workload.name]
            expected_text = service.tokenizer.decode(
                expected_by_name[workload.name], skip_special_tokens=True
            )
            texts = tuple(item.text for item in observations)
            completion_counts = tuple(item.completion_tokens for item in observations)
            result[workload.name] = {
                "generated_tokens": workload.max_new_tokens,
                "warmup_iterations": warmup,
                "measured_iterations": iterations,
                "ttft_ms": summarize_samples([item.ttft_ms for item in observations]),
                "latency_ms": summarize_samples([item.latency_ms for item in observations]),
                "completion_token_counts": list(completion_counts),
                "texts": list(texts),
                "expected_direct_decode": expected_text,
                "text_match": all(text == expected_text for text in texts),
                "resident_allocated_bytes_before_measurement": resident,
                "peak_allocated_bytes": peak,
                "incremental_peak_bytes": max(0, peak - resident),
            }
        return result
    finally:
        server.should_exit = True
        thread.join(timeout=15)
        if thread.is_alive():
            raise RuntimeError("Comparison Uvicorn did not stop cleanly.")


def run(
    model_lock: Path,
    output: Path,
    *,
    warmup: int = 1,
    iterations: int = 5,
    max_new_tokens: int = 8,
) -> dict[str, object]:
    """顺序加载同一锁定 checkpoint，写出可复查 JSON 报告。"""

    if warmup < 0 or iterations < 3 or max_new_tokens < 2:
        raise ValueError("Use warmup>=0, iterations>=3 and max_new_tokens>=2.")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from miniserve.api.bootstrap import build_profile_l_runtime
    from miniserve.model.loader import resolve_model_revision

    if not torch.cuda.is_available():
        raise RuntimeError("Real HF comparison requires a CUDA GPU.")
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    snapshot, revision = resolve_model_revision(model_lock)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False
    )
    workloads = (
        ComparisonWorkload("short-chat", "你好", max_new_tokens),
        ComparisonWorkload(
            "repeated-prefix-chat", "请简短解释KV缓存的作用。", max_new_tokens
        ),
    )
    tokens_by_name = {
        workload.name: _chat_token_ids(tokenizer, workload.prompt)
        for workload in workloads
    }
    if len(tokens_by_name["short-chat"]) >= 16:
        raise RuntimeError("Short comparison unexpectedly contains a full prefix block.")
    if len(tokens_by_name["repeated-prefix-chat"]) < 16:
        raise RuntimeError("Prefix comparison does not contain one full 16-token block.")
    if any(len(tokens) + max_new_tokens > 64 for tokens in tokens_by_name.values()):
        raise RuntimeError("Comparison workload exceeds the frozen max_model_len=64.")

    torch.cuda.empty_cache()
    hf_model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to("cuda:0").eval()
    hf_attention = str(getattr(hf_model.config, "_attn_implementation", "unknown"))
    hf_rows: dict[str, dict[str, object]] = {}
    hf_expected: dict[str, tuple[int, ...]] = {}
    hf_cache_expected: dict[str, tuple[int, ...]] = {}
    hf_cross_mode_equal: dict[str, bool] = {}
    hf_teacher_logits: dict[str, tuple[object, ...]] = {}
    for workload in workloads:
        token_ids = tokens_by_name[workload.name]
        recompute, expected = _measure_mode(
            torch,
            lambda ids=token_ids, count=workload.max_new_tokens: _hf_once(
                torch, hf_model, ids, count, use_cache=False
            ),
            warmup=warmup,
            iterations=iterations,
        )
        kv_cache, kv_tokens = _measure_mode(
            torch,
            lambda ids=token_ids, count=workload.max_new_tokens: _hf_once(
                torch, hf_model, ids, count, use_cache=True
            ),
            warmup=warmup,
            iterations=iterations,
            expected_tokens=expected,
        )
        hf_expected[workload.name] = expected
        hf_cache_expected[workload.name] = kv_tokens
        hf_cross_mode_equal[workload.name] = kv_tokens == expected
        hf_teacher_logits[workload.name] = _hf_teacher_logits(
            torch, hf_model, token_ids, expected
        )
        hf_rows[workload.name] = {
            "hf_sdpa_recompute": recompute,
            "hf_sdpa_kv_cache": kv_cache,
        }
    del hf_model
    gc.collect()
    torch.cuda.empty_cache()
    hf_residual_after_release = int(torch.cuda.memory_allocated())
    # CUDA context/allocator 或第三方 kernel 可以保留少量 live buffer；要求精确为 0
    # 会把已释放模型误判为仍驻留。超过 256 MiB 才足以怀疑权重/大 cache 仍被引用。
    if hf_residual_after_release > 256 * 1024 * 1024:
        raise RuntimeError(
            "HF model was not sufficiently released before loading MiniServe: "
            f"{hf_residual_after_release} bytes remain allocated."
        )

    service = build_profile_l_runtime(
        model_lock,
        max_model_len=64,
        num_kv_blocks=8,
        max_num_seqs=2,
        max_pending_requests=4,
    )
    core = service.engine.worker.core
    custom_model = core.executor.model
    custom_rows: dict[str, dict[str, object]] = {}
    teacher_diagnostics: dict[str, list[dict[str, object]]] = {}
    direct_expected: dict[str, tuple[int, ...]] = {}
    request_counter = 0
    for workload in workloads:
        token_ids = tokens_by_name[workload.name]
        recompute, custom_tokens = _measure_mode(
            torch,
            lambda ids=token_ids, count=workload.max_new_tokens: _custom_recompute_once(
                torch, custom_model, ids, count
            ),
            warmup=warmup,
            iterations=iterations,
            expected_tokens=hf_expected[workload.name],
        )
        teacher_diagnostics[workload.name] = _custom_teacher_diagnostics(
            torch,
            custom_model,
            token_ids,
            hf_expected[workload.name],
            hf_teacher_logits[workload.name],
        )

        def direct_once(ids=token_ids, count=workload.max_new_tokens):
            nonlocal request_counter
            request_counter += 1
            return _core_once(
                torch, core, ids, count, f"compare-{workload.name}-{request_counter}"
            )

        cold_tokens, cold_ttft, cold_latency = direct_once()
        final_direct, final_tokens = _measure_mode(
            torch,
            direct_once,
            warmup=warmup,
            iterations=iterations,
            expected_tokens=hf_cache_expected[workload.name],
        )
        direct_expected[workload.name] = final_tokens
        custom_rows[workload.name] = {
            "miniserve_cuda_rmsnorm_recompute": recompute,
            "miniserve_final_direct": final_direct,
            "miniserve_final_cold_request": {
                "tokens": list(cold_tokens),
                "ttft_ms": cold_ttft,
                "latency_ms": cold_latency,
            },
        }

    http_rows = _http_modes(
        torch,
        service,
        workloads,
        direct_expected,
        warmup=warmup,
        iterations=iterations,
    )

    workload_reports = []
    all_direct_match = True
    all_miniserve_internal_match = True
    all_final_hf_match = True
    all_http_match = True
    for workload in workloads:
        name = workload.name
        modes = {**hf_rows[name], **custom_rows[name], "miniserve_http_sse": http_rows[name]}
        hf_recompute_tokens = tuple(hf_rows[name]["hf_sdpa_recompute"]["tokens"])
        hf_cache_tokens = tuple(hf_rows[name]["hf_sdpa_kv_cache"]["tokens"])
        custom_tokens = tuple(custom_rows[name]["miniserve_cuda_rmsnorm_recompute"]["tokens"])
        final_tokens = tuple(custom_rows[name]["miniserve_final_direct"]["tokens"])
        cold_tokens = tuple(custom_rows[name]["miniserve_final_cold_request"]["tokens"])
        path_matches = reference_path_matches(
            hf_recompute_tokens,
            hf_cache_tokens,
            custom_tokens,
            final_tokens,
        )
        recompute_match = path_matches["hf_recompute_matches_custom_recompute"]
        incremental_match = path_matches["hf_cache_matches_custom_final"]
        direct_match = recompute_match and incremental_match
        miniserve_internal_match = custom_tokens == final_tokens
        final_hf_match = hf_cache_tokens == final_tokens == cold_tokens
        http_match = bool(http_rows[name]["text_match"]) and all(
            count == workload.max_new_tokens
            for count in http_rows[name]["completion_token_counts"]
        )
        all_direct_match &= direct_match
        all_miniserve_internal_match &= miniserve_internal_match
        all_final_hf_match &= final_hf_match
        all_http_match &= http_match
        teacher_mismatches = [
            row
            for row in teacher_diagnostics[name]
            if not row["teacher_forced_top1_match"]
        ]
        final_median = custom_rows[name]["miniserve_final_direct"]["latency_ms"]["median"]
        workload_reports.append(
            {
                "workload": name,
                "prompt": workload.prompt,
                "prompt_tokens": len(tokens_by_name[name]),
                "full_prefix_blocks": len(tokens_by_name[name]) // 16,
                "max_new_tokens": workload.max_new_tokens,
                "modes": modes,
                "correctness": {
                    "hf_custom_recompute_and_final_all_match": direct_match,
                    "hf_recompute_matches_miniserve_recompute": recompute_match,
                    "hf_cache_matches_miniserve_incremental": incremental_match,
                    "hf_recompute_cache_cross_mode_equal": hf_cross_mode_equal[name],
                    "old_recompute_and_final_exact_match": miniserve_internal_match,
                    "hf_and_miniserve_final_exact_greedy_tokens_match": final_hf_match,
                    "http_text_and_token_count_match": http_match,
                    "teacher_forced_mismatch_steps": [
                        row["step"] for row in teacher_mismatches
                    ],
                    "all_teacher_forced_mismatches_are_near_ties": all(
                        row["near_tie_relative_to_error"] for row in teacher_mismatches
                    ),
                    "teacher_forced_custom_recompute_vs_hf": teacher_diagnostics[name],
                },
                "comparisons": {
                    "final_vs_miniserve_recompute": compare_latency(
                        custom_rows[name]["miniserve_cuda_rmsnorm_recompute"]["latency_ms"]["median"],
                        final_median,
                    ),
                    "final_vs_hf_recompute": compare_latency(
                        hf_rows[name]["hf_sdpa_recompute"]["latency_ms"]["median"],
                        final_median,
                    ),
                    "final_vs_hf_kv_cache": compare_latency(
                        hf_rows[name]["hf_sdpa_kv_cache"]["latency_ms"]["median"],
                        final_median,
                    ),
                    "http_vs_hf_kv_cache": compare_latency(
                        hf_rows[name]["hf_sdpa_kv_cache"]["latency_ms"]["median"],
                        http_rows[name]["latency_ms"]["median"],
                    ),
                    "final_resident_memory_vs_hf_kv_cache": compare_memory(
                        hf_rows[name]["hf_sdpa_kv_cache"][
                            "resident_allocated_bytes_before_measurement"
                        ],
                        custom_rows[name]["miniserve_final_direct"][
                            "resident_allocated_bytes_before_measurement"
                        ],
                    ),
                },
            }
        )

    latency_final_vs_own = all(
        row["comparisons"]["final_vs_miniserve_recompute"]["verdict"] == "faster"
        for row in workload_reports
    )
    latency_final_vs_hf_kv = all(
        row["comparisons"]["final_vs_hf_kv_cache"]["verdict"] == "faster"
        for row in workload_reports
    )
    latency_http_vs_hf_kv = all(
        row["comparisons"]["http_vs_hf_kv_cache"]["verdict"] == "faster"
        for row in workload_reports
    )
    resident_memory_lower_than_hf_kv = all(
        row["comparisons"]["final_resident_memory_vs_hf_kv_cache"]["verdict"]
        == "lower"
        for row in workload_reports
    )

    report = {
        "schema_version": 1,
        "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "Profile D low-load same-weight path-aligned feasibility/performance comparison",
        "production_claim": False,
        "model_lock": str(model_lock),
        "model_snapshot": str(snapshot),
        "resolved_revision": revision,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "hf_attention_implementation": hf_attention,
        },
        "methodology": {
            "models_resident_sequentially": True,
            "hf_allocated_bytes_after_release": hf_residual_after_release,
            "model_load_time_excluded": True,
            "same_chat_template_token_ids": True,
            "batch_size": 1,
            "greedy": True,
            "warmup_iterations": warmup,
            "measured_iterations": iterations,
            "timing": "host wall time with CUDA synchronization; TTFT includes prompt H2D",
            "noise_band_percent": 3.0,
        },
        "workloads": workload_reports,
        "feasibility": {
            "hf_custom_recompute_and_final_all_match": all_direct_match,
            "hf_recompute_matches_miniserve_recompute": all(
                row["correctness"]["hf_recompute_matches_miniserve_recompute"]
                for row in workload_reports
            ),
            "hf_cache_matches_miniserve_incremental": all(
                row["correctness"]["hf_cache_matches_miniserve_incremental"]
                for row in workload_reports
            ),
            "hf_recompute_cache_cross_mode_equal": all(
                hf_cross_mode_equal.values()
            ),
            "old_recompute_and_final_exact_match": all_miniserve_internal_match,
            "old_recompute_near_tie_divergence_observed": not all_miniserve_internal_match,
            "all_hf_and_miniserve_final_tokens_match": all_final_hf_match,
            "all_http_text_and_token_counts_match": all_http_match,
            "same_locked_weights_reach_real_http": all_final_hf_match and all_http_match,
            "performance_is_reported_not_assumed": True,
        },
        "observed_performance": {
            "final_faster_than_miniserve_recompute_on_all_workloads": latency_final_vs_own,
            "final_faster_than_hf_kv_on_all_workloads": latency_final_vs_hf_kv,
            "http_faster_than_hf_kv_on_all_workloads": latency_http_vs_hf_kv,
            "final_resident_memory_lower_than_hf_kv_on_all_workloads": (
                resident_memory_lower_than_hf_kv
            ),
            "memory_and_latency_are_separate_claims": True,
        },
        "limitations": [
            "RTX 5060 Ti Profile D sm_120; low-load comparison only",
            "batch=1, max_model_len=64, short deterministic outputs",
            "no long soak, sustained concurrency, energy or production SLO claim",
            "HF direct execution and MiniServe HTTP have different protocol overhead; direct comparison is primary",
            "HF recompute and HF cache may diverge; each MiniServe path is compared to its corresponding HF mode",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    del service, core, custom_model
    gc.collect()
    torch.cuda.empty_cache()
    if not all_final_hf_match or not all_http_match:
        raise RuntimeError("HF/final MiniServe/HTTP paths failed output equivalence.")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare HF recompute/KV with MiniServe recompute/final/HTTP on one locked model."
    )
    parser.add_argument(
        "--model-lock",
        type=Path,
        default=Path("model-locks/qwen3-0.6b-profile-l.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()
    report = run(
        args.model_lock,
        args.output,
        warmup=args.warmup,
        iterations=args.iterations,
        max_new_tokens=args.max_new_tokens,
    )
    summary = {
        "feasibility": report["feasibility"],
        "observed_performance": report["observed_performance"],
        "comparisons": {
            row["workload"]: row["comparisons"] for row in report["workloads"]
        },
        "output": str(args.output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
