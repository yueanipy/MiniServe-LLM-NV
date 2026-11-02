"""Benchmark the real CUDA operator with CUDA Events.

This command produces observations, not a speedup Gate.  A simple educational
kernel is allowed to be slower than torch.add; the important Phase 0 property
is that both paths are measured with the same synchronization boundaries.

这里测量的是单个 vector-add 微基准，不包含模型、调度、KV Cache 或 HTTP 开销。
原始样本会全部保存，避免只保留一个“好看”的平均值。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable


def percentile(values: list[float], probability: float) -> float:
    """用确定性的 nearest-rank 定义计算分位数，不做插值。"""

    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 < probability <= 1.0:
        raise ValueError("probability must be in (0, 1]")
    # nearest-rank 的位置从 1 开始；转成 Python 0-based index 后需要减一。
    ordered = sorted(values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def time_cuda(torch: Any, operation: Callable[[], Any], warmup: int, iterations: int) -> list[float]:
    """使用同一 CUDA stream 上的 Events 测量 GPU 时间，单位为毫秒。"""

    # warmup 排除首次 context 初始化、allocator 建立和 lazy module load 的主要影响。
    for _ in range(warmup):
        operation()
    # 保证正式采样开始前 warmup 已经在 GPU 上完成。
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(iterations):
        # CUDA launch 对 CPU 是异步的，不能用 time.time() 包围 operation 得到 kernel
        # 时间。Events 被记录在当前 stream 上，end.synchronize() 只等待本次终点。
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples


def summarize(name: str, dtype: str, size: int, warmup: int, samples: list[float]) -> dict[str, Any]:
    """把一次 workload 的统计量与全部原始样本封装成稳定 evidence schema。"""

    # schema_version 允许未来增加字段时让读取端显式处理版本差异。
    return {
        "schema_version": 1,
        "operation": name,
        "dtype": dtype,
        "size": size,
        "unit": "ms",
        "warmup": warmup,
        "iterations": len(samples),
        "median_ms": statistics.median(samples),
        "p95_ms": percentile(samples, 0.95),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def run_benchmark(sizes: list[int], dtype_names: list[str], warmup: int, iterations: int) -> list[dict[str, Any]]:
    """在相同输入与同步边界下依次测量自定义算子和 torch.add。"""

    # 延迟导入让 percentile/summarize 的 CPU unit tests 不依赖 CUDA 环境。
    import torch

    # 性能工具禁止 CPU fallback，否则报告会把完全不同的执行路径放在一起比较。
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; benchmark has no CPU fallback.")

    from miniserve._native import load_native_ops, vector_add

    load_native_ops()
    rows: list[dict[str, Any]] = []
    for dtype_name in dtype_names:
        dtype = getattr(torch, dtype_name)
        for size in sizes:
            # size 参与 seed，使每个 workload 可重复但输入不完全相同；generator 明确
            # 位于 CUDA，避免隐含的 CPU→GPU 数据传输污染测量。
            generator = torch.Generator(device="cuda").manual_seed(2026 + size)
            a = torch.randn(size, device="cuda", dtype=dtype, generator=generator)
            b = torch.randn(size, device="cuda", dtype=dtype, generator=generator)
            # 任何错误实现都不应进入计时。FP16 使用与 correctness Gate 一致的容差。
            torch.testing.assert_close(vector_add(a, b), a + b, atol=1e-3 if dtype == torch.float16 else 1e-6, rtol=1e-3 if dtype == torch.float16 else 1e-6)

            # NVTX 只标记整组采样，供 Nsight 将 custom 与 reference 路径投影到时间线。
            torch.cuda.nvtx.range_push(f"vector_add_custom/{dtype_name}/{size}")
            custom_samples = time_cuda(torch, lambda: vector_add(a, b), warmup, iterations)
            torch.cuda.nvtx.range_pop()
            rows.append(summarize("miniserve::vector_add", dtype_name, size, warmup, custom_samples))

            torch.cuda.nvtx.range_push(f"torch_add/{dtype_name}/{size}")
            torch_samples = time_cuda(torch, lambda: a + b, warmup, iterations)
            torch.cuda.nvtx.range_pop()
            rows.append(summarize("torch.add", dtype_name, size, warmup, torch_samples))
    return rows


def main(argv: list[str] | None = None) -> int:
    """解析 CLI、验证 workload 参数并写入 JSON Lines 证据。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1024, 4099, 1048576])
    parser.add_argument("--dtypes", nargs="+", choices=("float32", "float16"), default=["float32", "float16"])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if any(size <= 0 for size in args.sizes):
        parser.error("all sizes must be positive")
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("warmup must be non-negative and iterations must be positive")

    rows = run_benchmark(args.sizes, args.dtypes, args.warmup, args.iterations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # JSONL 支持流式读取，也能让每个 workload 保持一条独立记录。
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    for row in rows:
        print(
            f"{row['operation']} dtype={row['dtype']} size={row['size']} "
            f"median={row['median_ms']:.6f} ms p95={row['p95_ms']:.6f} ms"
        )
    return 0


if __name__ == "__main__":
    # 把返回值交给 shell，便于 Phase Gate 根据退出码停止流水线。
    raise SystemExit(main())
