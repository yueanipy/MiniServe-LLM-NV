"""按冻结 snapshot 严格、逐 Tensor 加载 safetensors。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .weight_map import expected_parameter_map

if TYPE_CHECKING:
    import torch

    from .qwen3 import Qwen3ForCausalLM


@dataclass(frozen=True)
class WeightLoadReport:
    snapshot_path: str
    loaded_tensors: int
    source_bytes: int
    source_dtype_counts: dict[str, int]
    target_dtype: str
    target_device: str


def _resolve_snapshot_member(snapshot: Path, relative_name: object) -> Path:
    """拒绝路径穿越；仅允许 HF snapshot 指向同仓库 ``blobs/`` 的链接。"""

    if not isinstance(relative_name, str) or not relative_name:
        raise ValueError("Model lock/index file names must be non-empty strings.")
    relative = Path(relative_name)
    if relative.is_absolute():
        raise ValueError(f"Model file path must be relative: {relative_name}")
    root = snapshot.resolve()
    # 先按词法归一化检查 ``..``；不能直接只看 resolve，因为 Hugging Face
    # snapshot 的普通成员本来就是指向同一 repo cache/blobs 的 symlink。
    lexical = Path(root, relative)
    normalized = lexical.parent.resolve() / lexical.name
    try:
        normalized.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Model file escapes the locked snapshot: {relative_name}") from exc
    resolved = lexical.resolve()
    if not resolved.is_relative_to(root):
        repo_cache_root = root.parent.parent if root.parent.name == "snapshots" else None
        blobs_root = None if repo_cache_root is None else (repo_cache_root / "blobs").resolve()
        if blobs_root is None or not resolved.is_relative_to(blobs_root):
            raise ValueError(f"Model symlink escapes the allowed cache: {relative_name}")
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_model_revision(model_lock_path: Path, *, verify_hashes: bool = True) -> tuple[Path, str]:
    """只接受 commit 锁，并默认重新验证本地 snapshot 的大小与 SHA-256。"""

    lock = json.loads(model_lock_path.read_text(encoding="utf-8"))
    revision = str(lock.get("resolved_revision", ""))
    snapshot = Path(str(lock.get("local_snapshot", "")))
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision.lower()):
        raise ValueError("model-lock.json has no immutable 40-character commit revision.")
    if not snapshot.is_dir() or not (snapshot / "config.json").is_file():
        raise FileNotFoundError(f"Locked model snapshot is unavailable: {snapshot}")
    if verify_hashes:
        locked_files = lock.get("files")
        if not isinstance(locked_files, dict) or not locked_files:
            raise ValueError("model-lock.json contains no file hashes.")
        for relative_name, expected in locked_files.items():
            if not isinstance(expected, dict):
                raise ValueError(f"Invalid model lock metadata for {relative_name}.")
            path = _resolve_snapshot_member(snapshot, relative_name)
            if not path.is_file():
                raise FileNotFoundError(f"Locked model file is unavailable: {relative_name}")
            actual_size = path.stat().st_size
            expected_size = expected.get("bytes")
            expected_hash = expected.get("sha256")
            if (
                not isinstance(expected_size, int)
                or isinstance(expected_size, bool)
                or expected_size < 0
                or not isinstance(expected_hash, str)
                or len(expected_hash) != 64
                or any(character not in "0123456789abcdef" for character in expected_hash.lower())
            ):
                raise ValueError(f"Invalid size/SHA-256 metadata for {relative_name}.")
            if actual_size != expected_size:
                raise RuntimeError(
                    f"Locked model file size mismatch for {relative_name}: {actual_size} != {expected_size}"
                )
            actual_hash = _sha256_file(path)
            if actual_hash != expected_hash.lower():
                raise RuntimeError(f"Locked model SHA-256 mismatch for {relative_name}.")
    return snapshot, revision


def _weight_files(snapshot: Path) -> tuple[Path, ...]:
    index_files = sorted(snapshot.glob("*.safetensors.index.json"))
    if index_files:
        if len(index_files) != 1:
            raise RuntimeError("Expected at most one safetensors index.")
        index = json.loads(index_files[0].read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("Safetensors index has no non-empty weight_map.")
        if any(not isinstance(key, str) or not isinstance(value, str) for key, value in weight_map.items()):
            raise ValueError("Safetensors index weight_map must contain string keys and values.")
        names = tuple(sorted(set(weight_map.values())))
        files = tuple(_resolve_snapshot_member(snapshot, name) for name in names)
    else:
        files = tuple(sorted(snapshot.glob("*.safetensors")))
    if not files or any(not path.is_file() for path in files):
        raise FileNotFoundError("Locked snapshot has missing safetensors shards.")
    return files


def validate_all_weights_consumed(expected: set[str], observed: set[str]) -> None:
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if missing or unexpected:
        raise RuntimeError(f"Strict weight schema mismatch; missing={missing}, unexpected={unexpected}")


def _validate_tied_source_weights(locations: dict[str, Path], *, rows_per_chunk: int = 2048) -> None:
    """低内存分块证明 checkpoint 中 embedding 与 lm_head 逐 bit 相等。"""

    import torch
    from safetensors import safe_open

    embedding_file = locations["model.embed_tokens.weight"]
    lm_head_file = locations["lm_head.weight"]
    with (
        safe_open(embedding_file, framework="pt", device="cpu") as embedding_handle,
        safe_open(lm_head_file, framework="pt", device="cpu") as lm_head_handle,
    ):
        embedding = embedding_handle.get_slice("model.embed_tokens.weight")
        lm_head = lm_head_handle.get_slice("lm_head.weight")
        if embedding.get_shape() != lm_head.get_shape():
            raise RuntimeError("Tied embedding and lm_head source shapes differ.")
        row_count = embedding.get_shape()[0]
        for start in range(0, row_count, rows_per_chunk):
            end = min(start + rows_per_chunk, row_count)
            if not torch.equal(embedding[start:end], lm_head[start:end]):
                raise RuntimeError(f"Tied embedding and lm_head values differ in rows [{start}, {end}).")


def load_safetensors_strict(
    model: Qwen3ForCausalLM,
    snapshot_path: Path,
    *,
    device: torch.device | str,
) -> WeightLoadReport:
    """先完整核对 key/shape，再逐项 copy；任何不一致都在执行 forward 前失败。"""

    import torch
    from safetensors import safe_open

    snapshot = snapshot_path.resolve()
    weight_files = _weight_files(snapshot)
    expected_map = expected_parameter_map(model.config)

    observed_locations: dict[str, Path] = {}
    source_bytes = 0
    for weight_file in weight_files:
        source_bytes += weight_file.stat().st_size
        with safe_open(weight_file, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in observed_locations:
                    raise RuntimeError(f"Duplicate tensor key across shards: {key}")
                observed_locations[key] = weight_file
    validate_all_weights_consumed(set(expected_map), set(observed_locations))
    if model.config.checkpoint_has_lm_head:
        _validate_tied_source_weights(observed_locations)

    # Meta 初始化避免先随机填充 0.6B 参数；to_empty 只分配目标 storage，不运行初始化 kernel。
    if any(parameter.is_meta for parameter in model.parameters()):
        model.to_empty(device=device)
    else:
        model.to(device=device)
    model.tie_weights()
    parameters = dict(model.named_parameters(remove_duplicate=False))

    source_dtype_counts: dict[str, int] = {}
    loaded: set[str] = set()
    with torch.no_grad():
        for source_key, target_name in expected_map.items():
            weight_file = observed_locations[source_key]
            with safe_open(weight_file, framework="pt", device="cpu") as handle:
                source = handle.get_tensor(source_key)
            target = parameters[target_name]
            if tuple(source.shape) != tuple(target.shape):
                raise RuntimeError(
                    f"Shape mismatch for {source_key}: source={tuple(source.shape)}, target={tuple(target.shape)}"
                )
            source_dtype_counts[str(source.dtype)] = source_dtype_counts.get(str(source.dtype), 0) + 1
            target.copy_(source.to(device=target.device, dtype=target.dtype, non_blocking=False))
            loaded.add(source_key)
            del source
    validate_all_weights_consumed(set(expected_map), loaded)
    model.tie_weights()
    first_parameter = next(model.parameters())
    return WeightLoadReport(
        snapshot_path=str(snapshot),
        loaded_tensors=len(loaded),
        source_bytes=source_bytes,
        source_dtype_counts=source_dtype_counts,
        target_dtype=str(first_parameter.dtype),
        target_device=str(first_parameter.device),
    )
