"""Phase 9 eager vs CUDA Graph paged-decode 的 Event 与 host latency 样本。"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from .benchmark_paged_decode import _inputs, parse_workload
from .benchmark_vector_add import percentile, time_cuda


def _host_samples(torch, operation, warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        operation()
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1_000_000.0)
    return samples


def _summary(operation, workload, device_samples, host_samples, warmup):
    return {
        "schema_version": 1,
        "operation": operation,
        "dtype": "float16",
        "batch": workload[0],
        "context": workload[1],
        "warmup": warmup,
        "iterations": len(device_samples),
        "device_unit": "ms",
        "device_median_ms": statistics.median(device_samples),
        "device_p95_ms": percentile(device_samples, 0.95),
        "device_samples_ms": device_samples,
        "host_unit": "ms",
        "host_median_ms": statistics.median(host_samples),
        "host_p95_ms": percentile(host_samples, 0.95),
        "host_samples_ms": host_samples,
    }


def run_benchmark(workloads, warmup, iterations):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA Graph benchmark requires CUDA and has no fallback.")
    from miniserve._native import paged_decode_attention
    from miniserve.engine.graph import GraphManager

    rows = []
    for workload in workloads:
        query, key, value, tables, lengths = _inputs(torch, *workload, torch.float16)
        scale = 128**-0.5
        manager = GraphManager(
            buckets=(1, 2, 4, 8),
            max_blocks_per_sequence=5,
            model_revision="phase9-benchmark",
            validate_metadata_values=False,
        )

        def eager():
            return paged_decode_attention(query, key, value, tables, lengths, scale)

        def replay():
            output, reason = manager.run_paged_decode(
                query, key, value, tables, lengths, scale, layer_index=0
            )
            if output is None:
                raise RuntimeError(f"Unexpected graph fallback: {reason}")
            return output

        expected = eager()
        actual = replay()  # 首次 capture 在所有正式样本之外。
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, atol=8e-3, rtol=8e-3)

        label = f"{workload[0]}x{workload[1]}"
        torch.cuda.nvtx.range_push(f"decode_eager/float16/{label}")
        eager_device = time_cuda(torch, eager, warmup, iterations)
        torch.cuda.nvtx.range_pop()
        eager_host = _host_samples(torch, eager, warmup, iterations)
        rows.append(_summary("eager", workload, eager_device, eager_host, warmup))

        torch.cuda.nvtx.range_push(f"decode_graph_replay/float16/{label}")
        graph_device = time_cuda(torch, replay, warmup, iterations)
        torch.cuda.nvtx.range_pop()
        graph_host = _host_samples(torch, replay, warmup, iterations)
        rows.append(_summary("cuda_graph_replay", workload, graph_device, graph_host, warmup))
        rows[-1]["graph_captures"] = manager.stats.captures
        rows[-1]["graph_replays"] = manager.stats.replays
        rows[-1]["graph_fallbacks"] = manager.stats.fallbacks
        manager.clear()
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workloads", nargs="+", default=["1x17", "2x33", "4x65"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("warmup must be non-negative and iterations positive")
    try:
        workloads = [parse_workload(item) for item in args.workloads]
    except ValueError as error:
        parser.error(str(error))
    rows = run_benchmark(workloads, args.warmup, args.iterations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    for row in rows:
        print(
            f"{row['operation']} B={row['batch']} C={row['context']} "
            f"device_median={row['device_median_ms']:.6f} ms "
            f"host_median={row['host_median_ms']:.6f} ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
