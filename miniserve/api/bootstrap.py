"""从固定模型锁构建唯一 Phase 10 Engine/Worker/ServiceRuntime。"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer, GenerationConfig

from miniserve.attention import (
    CudaPagedDecodeAttention,
    FlashInferPagedDecodeAttention,
    TorchPagedReferenceAttention,
)
from miniserve.config import EngineConfig, ModelConfig, RuntimeConfig
from miniserve.engine.async_engine import AsyncEngine
from miniserve.engine.core import EngineCore
from miniserve.engine.executor import ModelExecutor
from miniserve.engine.worker import EngineWorker
from miniserve.kv import KVCacheManager, KVLayout
from miniserve.kv.paged import PagedKVCacheBackend
from miniserve.kv.prefix_cache import PrefixCache, PrefixNamespace
from miniserve.model.loader import load_safetensors_strict, resolve_model_revision
from miniserve.model.qwen3 import Qwen3ForCausalLM

from .adapter import normalize_stop_token_ids
from .runtime import ServiceRuntime


def _resolve_max_num_batched_tokens(
    max_model_len: int,
    max_num_batched_tokens: int | None,
) -> int:
    """Resolve the public whole-prefill budget without expanding an explicit value."""

    return max_model_len if max_num_batched_tokens is None else max_num_batched_tokens


def build_profile_l_runtime(
    model_lock: str | Path,
    *,
    max_model_len: int = 64,
    num_kv_blocks: int = 8,
    max_num_seqs: int = 2,
    max_pending_requests: int = 4,
    max_num_batched_tokens: int | None = None,
    kv_block_size: int = 16,
    attention_backend: str = "cuda_paged",
    rmsnorm_backend: str = "cuda",
    decode_execution: str = "cuda_graph",
    decode_batching: str = "serial",
    batched_graph: bool = False,
    sampling_execution: str = "reference",
    enable_prefix_cache: bool = True,
    enable_chunked_prefill: bool = False,
    prefill_chunk_size: int | None = None,
    prefix_cache_max_entries: int = 8,
) -> ServiceRuntime:
    """低负载真实服务启动；权重只加载一次，所有请求复用同一 Engine。

    M0：语义开关独立传入并经 capability validation；公开默认使用整段 prefill，
    chunked prefill 必须显式开启。未指定预算时按 max_model_len 解析。
    """

    if not torch.cuda.is_available():
        raise RuntimeError("Profile L runtime requires an available CUDA GPU.")
    max_num_batched_tokens = _resolve_max_num_batched_tokens(
        max_model_len, max_num_batched_tokens
    )
    snapshot, revision = resolve_model_revision(Path(model_lock))
    device = torch.device("cuda:0")
    hf_config = AutoConfig.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    generation_config = GenerationConfig.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False
    )
    generation_stop_ids = normalize_stop_token_ids(generation_config.eos_token_id)
    if bool(getattr(generation_config, "do_sample", False)):
        default_temperature = float(getattr(generation_config, "temperature", None) or 1.0)
        default_top_p = float(getattr(generation_config, "top_p", None) or 1.0)
        default_top_k = int(getattr(generation_config, "top_k", None) or 0)
    else:
        default_temperature, default_top_p, default_top_k = 0.0, 1.0, 0
    model_config = ModelConfig.from_hf_config(
        hf_config,
        model_id_or_path=str(snapshot),
        model_revision=revision,
        dtype="float16",
        max_model_len=max_model_len,
    )
    engine_config = EngineConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        kv_block_size=kv_block_size,
        num_kv_blocks=num_kv_blocks,
        enable_prefix_cache=enable_prefix_cache,
        enable_chunked_prefill=enable_chunked_prefill,
        prefill_chunk_size=prefill_chunk_size,
        attention_backend=attention_backend,
        kv_backend="paged",
        rmsnorm_backend=rmsnorm_backend,
        decode_execution=decode_execution,
        decode_batching=decode_batching,
        batched_graph=batched_graph,
        sampling_execution=sampling_execution,
        max_pending_requests=max_pending_requests,
        prefix_cache_max_entries=prefix_cache_max_entries,
    )
    runtime_config = RuntimeConfig(model_config, engine_config)
    runtime_config.validate_capabilities()

    model = Qwen3ForCausalLM(model_config, device="meta", dtype=torch.float16)
    load_safetensors_strict(model, snapshot, device=device)
    model.eval()
    layout = KVLayout(
        model_config.num_hidden_layers,
        max_model_len,
        model_config.num_key_value_heads,
        model_config.head_dim,
        torch.float16,
        device,
        num_blocks=num_kv_blocks,
        block_size=engine_config.kv_block_size,
    )
    kv_backend = PagedKVCacheBackend(layout, max_sequences=max_num_seqs)
    prefix_cache = None
    if enable_prefix_cache:
        prefix_cache = PrefixCache(
            namespace=PrefixNamespace(
                revision,
                revision,
                model_config.dtype,
                f"rope={model_config.rope_theta};head={model_config.head_dim}",
                chat_template_revision=revision,
                thinking_mode=False,
            ),
            block_size=engine_config.kv_block_size,
            max_entries=engine_config.prefix_cache_max_entries,
        )
    kv_manager = KVCacheManager(kv_backend, prefix_cache=prefix_cache)
    if attention_backend == "cuda_paged":
        attention = CudaPagedDecodeAttention(kv_backend)
    elif attention_backend == "flashinfer_paged":
        attention = FlashInferPagedDecodeAttention(kv_backend)
        attention.bind_model(model)
    elif attention_backend == "torch_paged_ref":
        attention = TorchPagedReferenceAttention(kv_backend)
    else:
        raise ValueError(f"Unsupported paged attention backend: {attention_backend}")
    core = EngineCore(
        runtime_config,
        ModelExecutor(
            model,
            attention,
            retain_last_logits=False,
            # 正式模型在加载/正确性 Gate 中检查数值；服务热路径不为每个 token
            # 再执行一次全词表 isfinite reduction + D2H 同步。
            validate_logits_finite=False,
            sampling_execution=sampling_execution,
        ),
        kv_manager=kv_manager,
    )
    worker = EngineWorker(core, max_commands=max_pending_requests, exit_on_fatal=True)
    lock = json.loads(Path(model_lock).read_text(encoding="utf-8"))
    served_model_id = str(lock.get("model_id", ""))
    if not served_model_id:
        raise ValueError("model lock has no model_id.")
    return ServiceRuntime(
        AsyncEngine(worker),
        tokenizer,
        model_id=served_model_id,
        max_model_len=max_model_len,
        max_pending_requests=max_pending_requests,
        vocab_size=model_config.vocab_size,
        stop_token_ids=tuple(sorted(set(generation_stop_ids))),
        default_temperature=default_temperature,
        default_top_p=default_top_p,
        default_top_k=default_top_k,
        whole_prefill_token_budget=(
            None if enable_chunked_prefill else max_num_batched_tokens
        ),
    )
