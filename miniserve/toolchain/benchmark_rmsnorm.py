"""用 CUDA Event 比较自定义 RMSNorm 与永久保留的 Torch reference。"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from .benchmark_vector_add import percentile, time_cuda


def parse_shape(text: str) -> tuple[int, int]:
    """把 CLI 的 ``ROWSxHIDDEN`` 转成正整数 workload。"""

    try:
        rows_text, hidden_text = text.lower().split("x", maxsplit=1)
        rows, hidden = int(rows_text), int(hidden_text)
    except (TypeError, ValueError) as error:
        raise ValueError("RMSNorm shape must use ROWSxHIDDEN, for example 7x1024.") from error
    if rows <= 0 or hidden <= 0:
        raise ValueError("RMSNorm rows and hidden size must be positive.")
    return rows, hidden


def _summary(
    operation: str,
    dtype_name: str,
    shape: tuple[int, int],
    warmup: int,
    samples: list[float],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "operation": operation,
        "dtype": dtype_name,
        "shape": list(shape),
        "unit": "ms",
        "warmup": warmup,
        "iterations": len(samples),
        "median_ms": statistics.median(samples),
        "p95_ms": percentile(samples, 0.95),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def run_benchmark(
    shapes: list[tuple[int, int]],
    dtype_names: list[str],
    warmup: int,
    iterations: int,
) -> list[dict[str, Any]]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; RMSNorm benchmark has no CPU fallback.")
    from miniserve._native import load_native_ops, rmsnorm
    from miniserve.model.layers import rmsnorm_torch

    load_native_ops()
    rows: list[dict[str, Any]] = []
    for dtype_name in dtype_names:
        dtype = getattr(torch, dtype_name)
        for shape in shapes:
            generator = torch.Generator(device="cuda").manual_seed(7000 + shape[0] + shape[1])
            values = torch.randn(shape, generator=generator, device="cuda", dtype=dtype)
            weight = torch.linspace(0.5, 1.5, shape[1], device="cuda", dtype=dtype)
            expected = rmsnorm_torch(values, weight, 1e-6)
            actual = rmsnorm(values, weight, 1e-6)
            tolerance = 2e-5 if dtype == torch.float32 else 4e-3
            torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)

            label = f"{shape[0]}x{shape[1]}"
            torch.cuda.nvtx.range_push(f"rmsnorm_custom/{dtype_name}/{label}")
            custom_samples = time_cuda(torch, lambda: rmsnorm(values, weight, 1e-6), warmup, iterations)
            torch.cuda.nvtx.range_pop()
            rows.append(_summary("miniserve::rmsnorm", dtype_name, shape, warmup, custom_samples))

            torch.cuda.nvtx.range_push(f"rmsnorm_torch/{dtype_name}/{label}")
            torch_samples = time_cuda(
                torch,
                lambda: rmsnorm_torch(values, weight, 1e-6),
                warmup,
                iterations,
            )
            torch.cuda.nvtx.range_pop()
            rows.append(_summary("torch_rmsnorm_reference", dtype_name, shape, warmup, torch_samples))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+", default=["1x1024", "7x1024", "33x1024", "4x4096"])
    parser.add_argument("--dtypes", nargs="+", choices=("float32", "float16"), default=["float16"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("warmup must be non-negative and iterations must be positive")
    try:
        shapes = [parse_shape(item) for item in args.shapes]
    except ValueError as error:
        parser.error(str(error))
    rows = run_benchmark(shapes, args.dtypes, args.warmup, args.iterations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    for row in rows:
        print(
            f"{row['operation']} dtype={row['dtype']} shape={row['shape']} "
            f"median={row['median_ms']:.6f} ms p95={row['p95_ms']:.6f} ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
