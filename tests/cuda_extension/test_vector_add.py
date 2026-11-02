"""真实 Linux CUDA extension 的 Phase 0 正确性与集成 Gate。

本文件不接受 CPU fallback。固定边界、随机输入、Dispatcher schema/FakeTensor 和
clean process 分别覆盖不同故障，不能用单个 ``torch.allclose`` 代替。
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest

# 模块级 marker 让 CPU-only 与真实 CUDA 测试可以分层运行和统计。
pytestmark = pytest.mark.cuda_extension

# 31/32/33 覆盖 warp 边界，257 覆盖 256-thread block 尾部，1031 覆盖更大非对齐
# 输入。随机 seed 固定，保证失败可复现。
FIXED_SIZES = (1, 31, 32, 33, 257, 1024, 1031)
RANDOM_SEEDS = (7, 23, 101, 2026)


@pytest.fixture(scope="module")
def cuda_runtime():
    """一次性确认目标 CUDA 环境，并严格加载真实共享库。"""

    # 在 Windows 收集测试时允许显式 skip；这只表示测试未执行，不会计入 PASS。
    if platform.system() != "Linux":
        pytest.skip("Phase 0 CUDA extension Gate requires Linux/WSL2")
    torch = pytest.importorskip("torch", reason="PyTorch is not installed in the target Linux environment")
    if not torch.cuda.is_available():
        pytest.skip("torch.cuda.is_available() is false in the target Linux environment")

    from miniserve._native import load_native_ops

    try:
        library_path = load_native_ops()
    except RuntimeError as error:
        # 一旦 Linux CUDA 已可用，缺少/损坏 .so 就是项目错误，不能继续 skip。
        pytest.fail(f"CUDA is available but the real MiniServe extension is unavailable: {error}")
    return torch, library_path


def _tolerances(torch, dtype) -> tuple[float, float]:
    """返回与 dtype 数值精度匹配的绝对/相对容差。"""

    if dtype == torch.float32:
        return 1e-6, 1e-6
    if dtype == torch.float16:
        return 1e-3, 1e-3
    raise AssertionError(f"Unexpected Phase 0 dtype: {dtype}")


def _record_metric(record: dict) -> None:
    """在 Gate 脚本指定路径时追加 JSONL；普通 pytest 运行不产生文件。"""

    destination = os.environ.get("MINISERVE_PHASE0_METRICS")
    if not destination:
        return
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 每个 case 一行，便于流式绘图；由 run_phase0.sh 在测试前删除旧文件。
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


@pytest.mark.parametrize("dtype_name", ["float32", "float16"])
@pytest.mark.parametrize("size", FIXED_SIZES)
def test_vector_add_fixed_boundaries(cuda_runtime, dtype_name: str, size: int) -> None:
    """验证线程/warp/block 边界附近的 shape、dtype、device 与数值合同。"""

    torch, _ = cuda_runtime
    from miniserve._native import vector_add

    dtype = getattr(torch, dtype_name)
    a = torch.linspace(-2.0, 2.0, size, device="cuda", dtype=dtype)
    b = torch.linspace(3.0, -1.0, size, device="cuda", dtype=dtype)
    actual = vector_add(a, b)
    expected = a + b
    atol, rtol = _tolerances(torch, dtype)
    max_abs_error = float((actual - expected).abs().max().item())
    passed = bool(torch.allclose(actual, expected, atol=atol, rtol=rtol))
    # seed=-1 表示确定性构造输入，和后面的 seeded-random case 区分。
    _record_metric({"kind": "fixed", "dtype": dtype_name, "size": size, "seed": -1, "max_abs_error": max_abs_error, "atol": atol, "rtol": rtol, "passed": passed})

    # 属性断言防止“数值碰巧相同但输出被搬回 CPU/换 dtype”的错误实现。
    assert actual.is_cuda
    assert actual.shape == a.shape
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("seed", RANDOM_SEEDS)
@pytest.mark.parametrize("dtype_name", ["float32", "float16"])
def test_vector_add_seeded_random_cases(cuda_runtime, dtype_name: str, seed: int) -> None:
    """用可复现随机值覆盖符号、幅度组合与非 block 对齐长度 4099。"""

    torch, _ = cuda_runtime
    from miniserve._native import vector_add

    dtype = getattr(torch, dtype_name)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn(4099, generator=generator, device="cuda", dtype=dtype)
    b = torch.randn(4099, generator=generator, device="cuda", dtype=dtype)
    actual = vector_add(a, b)
    expected = a + b
    atol, rtol = _tolerances(torch, dtype)
    max_abs_error = float((actual - expected).abs().max().item())
    passed = bool(torch.allclose(actual, expected, atol=atol, rtol=rtol))
    _record_metric({"kind": "random", "dtype": dtype_name, "size": 4099, "seed": seed, "max_abs_error": max_abs_error, "atol": atol, "rtol": rtol, "passed": passed})

    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def test_operator_schema_and_fake_tensor_registration(cuda_runtime) -> None:
    """用 opcheck 验证 Dispatcher schema 和 FakeTensor metadata 规则一致。"""

    torch, _ = cuda_runtime
    a = torch.randn(33, device="cuda")
    b = torch.randn(33, device="cuda")

    # 这里只选择 Phase 0 已实现的 schema/fake 两项；它不替代真实 CUDA 数值测试。
    result = torch.library.opcheck(
        torch.ops.miniserve.vector_add.default,
        (a, b),
        test_utils=("test_schema", "test_faketensor"),
        raise_exception=True,
    )

    assert set(result) == {"test_schema", "test_faketensor"}
    assert all(item == "SUCCESS" for item in result.values())


def test_clean_process_loads_and_executes_real_operator(cuda_runtime) -> None:
    """在全新解释器验证 wheel 定位、RUNPATH、注册和真实 kernel 执行。"""

    _, library_path = cuda_runtime
    # 子进程不能复用当前 pytest 进程已经 dlopen 的 libtorch/_miniserve_ops，因而能
    # 暴露依赖仅靠 import 顺序才能解析的问题。
    code = """
import json
import torch
from miniserve._native import load_native_ops, vector_add
loaded = load_native_ops()
a = torch.tensor([1.0, 2.0, 3.0], device='cuda')
b = torch.tensor([4.0, 5.0, 6.0], device='cuda')
out = vector_add(a, b)
print(json.dumps({'library': str(loaded), 'device': out.device.type, 'values': out.cpu().tolist()}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    # PyTorch 可能输出警告，因此只把最后一行当作测试进程的 JSON 协议。
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert Path(payload["library"]).resolve() == library_path.resolve()
    assert payload["device"] == "cuda"
    assert payload["values"] == [5.0, 7.0, 9.0]
