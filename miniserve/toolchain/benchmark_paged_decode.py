"""Phase 8 paged decode 的低负载 CUDA Event 基准与原始样本记录。"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from .benchmark_vector_add import percentile, time_cuda


def parse_workload(text: str) -> tuple[int, int]:
    try:
        batch_text, context_text = text.lower().split("x", maxsplit=1)
        batch, context = int(batch_text), int(context_text)
    except (TypeError, ValueError) as error:
        raise ValueError("Paged workload must use BATCHxCONTEXT, for example 2x33.") from error
    if batch <= 0 or context <= 0:
        raise ValueError("Paged batch and context must be positive.")
    return batch, context


def _inputs(torch: Any, batch: int, context: int, dtype: Any):
    query_heads, kv_heads, block_size, head_dim = 16, 8, 16, 128
    max_blocks = (context + block_size - 1) // block_size
    num_blocks = batch * max_blocks + 1
    tables = torch.empty(batch, max_blocks, device="cuda", dtype=torch.int32)
    for row in range(batch):
        first = row * max_blocks
        tables[row] = torch.arange(
            first + max_blocks - 1,
            first - 1,
            -1,
            device="cuda",
            dtype=torch.int32,
        )
    lengths = torch.full((batch,), context, device="cuda", dtype=torch.int32)
    generator = torch.Generator(device="cuda").manual_seed(8800 + batch * 100 + context)
    query = torch.randn(batch, query_heads, head_dim, device="cuda", dtype=dtype, generator=generator)
    key = torch.randn(
        num_blocks,
        kv_heads,
        block_size,
        head_dim,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    value = torch.randn(
        num_blocks,
        kv_heads,
        block_size,
        head_dim,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    return query, key, value, tables, lengths


def _torch_materialized(torch: Any, query, key, value, tables, lengths, scale: float):
    batch, query_heads, _ = query.shape
    kv_heads, block_size = key.shape[1], key.shape[2]
    mapping = torch.arange(query_heads, device=query.device) // (query_heads // kv_heads)
    outputs = []
    for row in range(batch):
        length = int(lengths[row].item())
        positions = torch.arange(length, device=query.device)
        physical = tables[row, positions // block_size].long()
        offsets = positions % block_size
        keys = key[physical, :, offsets, :][:, mapping, :].permute(1, 0, 2)
        values = value[physical, :, offsets, :][:, mapping, :].permute(1, 0, 2)
        scores = torch.einsum("hd,htd->ht", query[row].float(), keys.float()) * scale
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
        outputs.append(torch.einsum("ht,htd->hd", probabilities, values.float()).to(query.dtype))
    return torch.stack(outputs)


def _summary(operation, dtype_name, workload, warmup, samples):
    return {
        "schema_version": 1,
        "operation": operation,
        "dtype": dtype_name,
        "batch": workload[0],
        "context": workload[1],
        "unit": "ms",
        "warmup": warmup,
        "iterations": len(samples),
        "median_ms": statistics.median(samples),
        "p95_ms": percentile(samples, 0.95),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def run_benchmark(workloads, dtype_names, warmup, iterations):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; paged decode benchmark has no CPU fallback.")
    from miniserve._native import load_native_ops, paged_decode_attention

    load_native_ops()
    rows = []
    for dtype_name in dtype_names:
        dtype = getattr(torch, dtype_name)
        for workload in workloads:
            query, key, value, tables, lengths = _inputs(torch, *workload, dtype)
            scale = 128**-0.5
            reference = _torch_materialized(torch, query, key, value, tables, lengths, scale)
            actual = paged_decode_attention(query, key, value, tables, lengths, scale)
            tolerance = 2e-4 if dtype == torch.float32 else 8e-3
            torch.testing.assert_close(actual, reference, atol=tolerance, rtol=tolerance)
            label = f"{workload[0]}x{workload[1]}"
            torch.cuda.nvtx.range_push(f"paged_decode_custom/{dtype_name}/{label}")
            custom_samples = time_cuda(
                torch,
                lambda: paged_decode_attention(query, key, value, tables, lengths, scale),
                warmup,
                iterations,
            )
            torch.cuda.nvtx.range_pop()
            rows.append(_summary("miniserve::paged_decode_attention", dtype_name, workload, warmup, custom_samples))
            torch.cuda.nvtx.range_push(f"paged_decode_torch/{dtype_name}/{label}")
            torch_samples = time_cuda(
                torch,
                lambda: _torch_materialized(torch, query, key, value, tables, lengths, scale),
                warmup,
                iterations,
            )
            torch.cuda.nvtx.range_pop()
            rows.append(_summary("torch_materialized_paged", dtype_name, workload, warmup, torch_samples))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workloads", nargs="+", default=["1x1", "1x17", "2x33", "4x65"])
    parser.add_argument("--dtypes", nargs="+", choices=("float32", "float16"), default=["float16"])
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
    rows = run_benchmark(workloads, args.dtypes, args.warmup, args.iterations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    for row in rows:
        print(
            f"{row['operation']} dtype={row['dtype']} B={row['batch']} context={row['context']} "
            f"median={row['median_ms']:.6f} ms p95={row['p95_ms']:.6f} ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
