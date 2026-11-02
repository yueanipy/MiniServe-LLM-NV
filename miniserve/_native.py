"""Strict loader for MiniServe native operators.

There is intentionally no Python or CPU fallback.  A missing or unusable shared
library is a Phase 0 failure, not a reason to execute a different code path.

加载器同时服务两种执行方式：从已安装 wheel 导入，以及 pytest 从源码目录导入。
无论哪种方式，都必须唯一定位同一个真实 Linux ``.so``。
"""

from __future__ import annotations

import platform
from importlib import metadata
from pathlib import Path
from typing import Any

# 进程内缓存避免重复 dlopen 和重复注册 FakeTensor kernel。它们不是跨进程状态；
# clean-process 测试仍会从零重新定位并加载动态库。
_LOADED_PATH: Path | None = None
_FAKE_REGISTERED = False


def find_extension() -> Path:
    """唯一定位真实 Linux 算子库；不存在或出现多个候选都立即失败。"""

    # Windows DLL 不是项目目标产物，禁止为了让测试“能跑”而切换平台路径。
    if platform.system() != "Linux":
        raise RuntimeError("MiniServe native operators require Linux/WSL2.")

    package_dir = Path(__file__).resolve().parent
    # 普通 wheel 导入时，.py 与 .so 都位于 site-packages/miniserve。
    candidates = {path.resolve() for path in package_dir.glob("_miniserve_ops*.so")}

    # A non-editable wheel can be installed while pytest is launched from the
    # source tree.  In that case Python imports this source file first, but the
    # compiled library lives in site-packages.  Distribution metadata gives us
    # the installed artifact without searching arbitrary filesystem paths.
    try:
        distribution = metadata.distribution("miniserve-llm-nv")
    except metadata.PackageNotFoundError:
        distribution = None
    if distribution is not None:
        for relative_path in distribution.files or ():
            relative = Path(str(relative_path))
            # 只接受本 distribution 的 miniserve/_miniserve_ops*.so；不递归搜索
            # 整台机器，避免加载旧 build 或其他虚拟环境中的同名文件。
            if relative.parent.name == "miniserve" and relative.name.startswith("_miniserve_ops") and relative.suffix == ".so":
                located = Path(distribution.locate_file(relative_path)).resolve()
                if located.is_file():
                    candidates.add(located)

    # set 去除源码路径和 distribution metadata 指向同一文件时的重复项；排序只为
    # 让错误输出稳定，便于测试与诊断复现。
    ordered_candidates = sorted(candidates)
    if len(ordered_candidates) != 1:
        raise RuntimeError(
            "Expected exactly one installed _miniserve_ops*.so, "
            f"found {len(ordered_candidates)} across the source package and installed distribution."
        )
    return ordered_candidates[0]


def load_native_ops() -> Path:
    """加载共享库，并为编译/追踪场景注册不执行真实 CUDA 的 FakeTensor 规则。"""

    global _FAKE_REGISTERED, _LOADED_PATH
    # 同一进程重复调用必须幂等；torch.library 不允许重复注册相同 fake kernel。
    if _LOADED_PATH is not None:
        return _LOADED_PATH

    import torch

    library_path = find_extension()
    # dlopen 会执行 TORCH_LIBRARY/TORCH_LIBRARY_IMPL 的静态注册代码。
    torch.ops.load_library(str(library_path))
    # 文件存在不代表注册成功，因此必须从 Dispatcher namespace 再验证一次。
    required = (
        "vector_add",
        "rmsnorm",
        "paged_decode_attention",
        "paged_decode_attention_graph",
        "paged_kv_append_graph",
    )
    if any(not hasattr(torch.ops.miniserve, name) for name in required):
        raise RuntimeError("Loaded library did not register all required MiniServe operators.")

    if not _FAKE_REGISTERED:

        @torch.library.register_fake("miniserve::vector_add")
        def _vector_add_fake(a: Any, b: Any) -> Any:
            # torch._check 能保留 symbolic shape 条件，适用于 FakeTensor/torch.compile；
            # 普通 Python assert 会过早把符号条件求值。
            torch._check(a.shape == b.shape)
            torch._check(a.dtype == b.dtype)
            torch._check(a.device == b.device)
            # empty_like 描述输出 metadata；FakeTensor 模式下不会启动 CUDA kernel。
            return torch.empty_like(a)

        @torch.library.register_fake("miniserve::rmsnorm")
        def _rmsnorm_fake(x: Any, weight: Any, eps: float) -> Any:
            torch._check(x.ndim >= 1)
            torch._check(weight.ndim == 1)
            torch._check(x.shape[-1] == weight.shape[0])
            torch._check(x.dtype == weight.dtype)
            torch._check(x.device == weight.device)
            torch._check(eps > 0.0)
            return torch.empty_like(x)

        @torch.library.register_fake("miniserve::paged_decode_attention")
        def _paged_decode_attention_fake(
            query: Any,
            key_cache: Any,
            value_cache: Any,
            block_tables: Any,
            sequence_lengths: Any,
            scale: float,
        ) -> Any:
            torch._check(query.ndim == 3)
            torch._check(key_cache.ndim == 4)
            torch._check(key_cache.shape == value_cache.shape)
            torch._check(query.shape[0] == block_tables.shape[0])
            torch._check(query.shape[0] == sequence_lengths.shape[0])
            torch._check(query.shape[-1] == key_cache.shape[-1])
            torch._check(query.dtype == key_cache.dtype)
            torch._check(query.dtype == value_cache.dtype)
            torch._check(query.device == key_cache.device)
            torch._check(query.device == value_cache.device)
            torch._check(scale > 0.0)
            return torch.empty_like(query)

        @torch.library.register_fake("miniserve::paged_decode_attention_graph")
        def _paged_decode_attention_graph_fake(
            query: Any,
            key_cache: Any,
            value_cache: Any,
            block_tables: Any,
            sequence_lengths: Any,
            scale: float,
        ) -> Any:
            return _paged_decode_attention_fake(
                query, key_cache, value_cache, block_tables, sequence_lengths, scale
            )

        @torch.library.register_fake("miniserve::paged_kv_append_graph")
        def _paged_kv_append_graph_fake(
            key_pool: Any,
            value_pool: Any,
            layer_index: int,
            key: Any,
            value: Any,
            slot_mapping: Any,
        ) -> Any:
            torch._check(key_pool.ndim == 5)
            torch._check(key_pool.shape == value_pool.shape)
            torch._check(key.ndim == 3)
            torch._check(key.shape == value.shape)
            torch._check(key.shape[1] == key_pool.shape[2])
            torch._check(key.shape[2] == key_pool.shape[4])
            torch._check(slot_mapping.ndim == 1)
            torch._check(slot_mapping.shape[0] == key.shape[0])
            torch._check(slot_mapping.dtype == torch.int64)
            torch._check(key_pool.dtype == value_pool.dtype)
            torch._check(key_pool.dtype == key.dtype)
            torch._check(key.dtype == value.dtype)
            torch._check(key_pool.device == value_pool.device)
            torch._check(key_pool.device == key.device)
            torch._check(key_pool.device == value.device)
            torch._check(key_pool.device == slot_mapping.device)
            torch._check(layer_index >= 0)
            torch._check(layer_index < key_pool.shape[0])
            # schema 声明输出与被原地写入的 key_pool 别名；fake 只描述 metadata。
            return key_pool

        _FAKE_REGISTERED = True

    _LOADED_PATH = library_path
    return library_path


def vector_add(a: Any, b: Any) -> Any:
    """通过 Dispatcher 调用真实 CUDA 默认 overload，不提供 Python 计算替代。"""

    import torch

    load_native_ops()
    # 显式选择 default overload，避免以后新增 overload 后调用语义变得含糊。
    return torch.ops.miniserve.vector_add.default(a, b)


def rmsnorm(x: Any, weight: Any, eps: float) -> Any:
    """调用真实 CUDA RMSNorm；没有 Python silent fallback。"""

    import torch

    load_native_ops()
    return torch.ops.miniserve.rmsnorm.default(x, weight, float(eps))


def paged_decode_attention(
    query: Any,
    key_cache: Any,
    value_cache: Any,
    block_tables: Any,
    sequence_lengths: Any,
    scale: float,
) -> Any:
    """调用真实 CUDA paged decode；metadata 错误直接失败，不回退到 Torch。"""

    import torch

    load_native_ops()
    return torch.ops.miniserve.paged_decode_attention.default(
        query,
        key_cache,
        value_cache,
        block_tables,
        sequence_lengths,
        float(scale),
    )


def paged_decode_attention_graph(
    query: Any,
    key_cache: Any,
    value_cache: Any,
    block_tables: Any,
    sequence_lengths: Any,
    scale: float,
) -> Any:
    """GraphManager 内部入口；调用方必须先校验 metadata value range。"""

    import torch

    load_native_ops()
    return torch.ops.miniserve.paged_decode_attention_graph.default(
        query,
        key_cache,
        value_cache,
        block_tables,
        sequence_lengths,
        float(scale),
    )


def paged_kv_append_graph(
    key_pool: Any,
    value_pool: Any,
    layer_index: int,
    key: Any,
    value: Any,
    slot_mapping: Any,
) -> Any:
    """Graph-safe in-place KV append；slot value 由 KV Manager 在 capture 外保证。"""

    import torch

    load_native_ops()
    return torch.ops.miniserve.paged_kv_append_graph.default(
        key_pool,
        value_pool,
        int(layer_index),
        key,
        value,
        slot_mapping,
    )
