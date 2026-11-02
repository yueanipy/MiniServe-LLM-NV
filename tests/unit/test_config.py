"""Phase 1 配置合同的纯 CPU 测试。"""

from __future__ import annotations

from dataclasses import replace

import pytest

from miniserve.config import EngineConfig, ModelConfig, RuntimeConfig

pytestmark = pytest.mark.unit_cpu


def _hf_config() -> dict:
    return {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000,
        "max_position_embeddings": 40_960,
        "vocab_size": 151_936,
        "hidden_act": "silu",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "tie_word_embeddings": True,
        "sliding_window": None,
        "use_sliding_window": False,
    }


def _model(**overrides) -> ModelConfig:
    data = _hf_config()
    data.update(overrides)
    return ModelConfig.from_hf_config(
        data,
        model_id_or_path="Qwen/Qwen3-0.6B",
        model_revision="fixed-commit",
        dtype="float16",
        max_model_len=128,
    )


def _llama_model(**overrides) -> ModelConfig:
    data = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 960,
        "intermediate_size": 2560,
        "num_hidden_layers": 32,
        "num_attention_heads": 15,
        "num_key_value_heads": 5,
        "head_dim": 64,
        "rms_norm_eps": 1e-5,
        "rope_theta": 100_000,
        "max_position_embeddings": 8192,
        "vocab_size": 49_152,
        "hidden_act": "silu",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "mlp_bias": False,
        "tie_word_embeddings": True,
        "rope_interleaved": False,
        "rope_scaling": None,
        "sliding_window": None,
        "use_sliding_window": False,
    }
    data.update(overrides)
    return ModelConfig.from_hf_config(
        data,
        model_id_or_path="HuggingFaceTB/SmolLM2-360M-Instruct",
        model_revision="fixed-commit",
        dtype="float16",
        max_model_len=64,
    )


def test_qwen3_head_dim_is_read_from_config_not_inferred() -> None:
    model = _model()
    assert model.head_dim == 128
    assert model.hidden_size // model.num_attention_heads == 64


def test_strict_smollm2_compatible_llama_contract_is_accepted_without_qk_norm() -> None:
    model = _llama_model()
    assert model.model_family == "llama"
    assert model.head_dim == 64
    assert model.qk_norm is False
    assert model.checkpoint_has_lm_head is False


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"attention_bias": True}, "bias-free"),
        ({"mlp_bias": True}, "bias-free"),
        ({"rope_interleaved": True}, "Interleaved"),
        ({"rope_scaling": {"rope_type": "linear", "factor": 2.0}}, "scaled RoPE"),
        ({"tie_word_embeddings": False}, "tied"),
        ({"head_dim": 62}, "cover hidden_size"),
    ],
)
def test_llama_whitelist_rejects_unimplemented_variants(override: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _llama_model(**override)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"model_type": "llama"}, "UNSUPPORTED_MODEL_CONFIGURATION"),
        ({"sliding_window": 4096}, "Sliding-window"),
        ({"use_sliding_window": True}, "Sliding-window"),
        ({"tie_word_embeddings": False}, "tied"),
    ],
)
def test_model_config_rejects_unsupported_variants(override: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _model(**override)


def test_model_config_rejects_remote_code_and_implicit_fp32() -> None:
    with pytest.raises(ValueError, match="trust_remote_code"):
        ModelConfig.from_hf_config(
            _hf_config(), model_id_or_path="model", dtype="float16", trust_remote_code=True
        )
    with pytest.raises(ValueError, match="float16 or bfloat16"):
        ModelConfig.from_hf_config(_hf_config(), model_id_or_path="model", dtype="float32")


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"hidden_size": 1024.5}, "hidden_size must be an integer"),
        ({"num_attention_heads": True}, "num_attention_heads must be an integer"),
        ({"head_dim": "128"}, "head_dim must be an integer"),
        ({"rms_norm_eps": float("nan")}, "finite and positive"),
        ({"rope_theta": float("inf")}, "finite and positive"),
        ({"tie_word_embeddings": 1}, "tie_word_embeddings must be boolean"),
        ({"attention_dropout": "0.0"}, "attention_dropout must be numeric"),
    ],
)
def test_model_config_rejects_coercible_and_nonfinite_numeric_values(
    override: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _model(**override)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_num_batched_tokens", 1.5),
        ("max_pending_requests", True),
        ("num_kv_blocks", 2.5),
        ("prefix_cache_max_entries", False),
        ("enable_prefix_cache", 1),
    ],
)
def test_engine_config_rejects_bool_fractional_and_coercible_fields(field, value) -> None:
    config = replace(EngineConfig.for_phase5(), **{field: value})
    with pytest.raises(ValueError):
        config.validate()


def test_phase1_runtime_accepts_only_low_load_recompute_path() -> None:
    runtime = RuntimeConfig(model=_model(), engine=EngineConfig.for_phase1(max_num_batched_tokens=64))
    runtime.validate_phase1()

    with pytest.raises(ValueError, match="single-request recompute"):
        EngineConfig().validate_phase1()
    with pytest.raises(ValueError, match="single-request recompute"):
        EngineConfig.for_phase1().__class__(
            **{**EngineConfig.for_phase1().__dict__, "max_num_seqs": 2}
        ).validate_phase1()


def test_engine_config_rejects_incoherent_backend_combinations() -> None:
    with pytest.raises(ValueError, match="Prefix cache"):
        EngineConfig(enable_prefix_cache=True, kv_backend="contiguous").validate()
    with pytest.raises(ValueError, match="Paged attention"):
        EngineConfig(attention_backend="cuda_paged", kv_backend="contiguous").validate()


def test_phase2_accepts_only_single_request_contiguous_kv() -> None:
    phase2 = EngineConfig.for_phase2(max_num_batched_tokens=64)
    RuntimeConfig(model=_model(), engine=phase2).validate_phase2()
    with pytest.raises(ValueError, match="single-request contiguous KV"):
        EngineConfig.for_phase1(max_num_batched_tokens=64).validate_phase2()


def test_phase3_requires_multi_sequence_contiguous_configuration() -> None:
    phase3 = EngineConfig.for_phase3(
        max_num_seqs=3, max_num_batched_tokens=64, max_pending_requests=6
    )
    RuntimeConfig(model=_model(), engine=phase3).validate_phase3()
    with pytest.raises(ValueError, match="continuous batching"):
        EngineConfig.for_phase2(max_num_batched_tokens=64).validate_phase3()


def test_phase4_requires_chunked_prefill_flag() -> None:
    phase4 = EngineConfig.for_phase4(
        max_num_seqs=2, max_num_batched_tokens=16, max_pending_requests=4
    )
    RuntimeConfig(model=_model(), engine=phase4).validate_phase4()
    with pytest.raises(ValueError, match="chunked prefill"):
        EngineConfig.for_phase3(
            max_num_seqs=2, max_num_batched_tokens=16, max_pending_requests=4
        ).validate_phase4()


def test_prefill_chunk_size_requires_chunking_and_total_budget() -> None:
    config = EngineConfig(
        max_num_seqs=2,
        max_num_batched_tokens=1024,
        num_kv_blocks=16,
        enable_chunked_prefill=True,
        prefill_chunk_size=256,
        kv_backend="paged",
        attention_backend="cuda_paged",
        rmsnorm_backend="cuda",
        decode_execution="eager",
        max_pending_requests=4,
    )
    config.validate()
    with pytest.raises(ValueError, match="enable_chunked_prefill"):
        replace(config, enable_chunked_prefill=False).validate()
    with pytest.raises(ValueError, match="positive integer"):
        replace(config, prefill_chunk_size=0).validate()
    with pytest.raises(ValueError, match="positive integer"):
        replace(config, prefill_chunk_size=True).validate()
    with pytest.raises(ValueError, match="cannot exceed"):
        replace(config, prefill_chunk_size=1025).validate()


def test_phase5_requires_paged_kv_and_explicit_block_count() -> None:
    phase5 = EngineConfig.for_phase5(
        max_num_seqs=2,
        max_num_batched_tokens=16,
        num_kv_blocks=8,
        max_pending_requests=4,
    )
    RuntimeConfig(model=_model(), engine=phase5).validate_phase5()
    with pytest.raises(ValueError, match="paged-KV"):
        EngineConfig.for_phase4(
            max_num_seqs=2, max_num_batched_tokens=16, max_pending_requests=4
        ).validate_phase5()


def test_phase6_requires_prefix_cache_on_paged_backend() -> None:
    phase6 = EngineConfig.for_phase6(
        max_num_seqs=2,
        max_num_batched_tokens=16,
        num_kv_blocks=8,
        max_pending_requests=4,
    )
    RuntimeConfig(model=_model(), engine=phase6).validate_phase6()
    with pytest.raises(ValueError, match="prefix cache"):
        EngineConfig.for_phase5(
            max_num_seqs=2,
            max_num_batched_tokens=16,
            num_kv_blocks=8,
            max_pending_requests=4,
        ).validate_phase6()


def test_phase7_only_changes_rmsnorm_backend_on_phase6_path() -> None:
    phase7 = EngineConfig.for_phase7(
        max_num_seqs=2,
        max_num_batched_tokens=16,
        num_kv_blocks=8,
        max_pending_requests=4,
    )
    RuntimeConfig(model=_model(), engine=phase7).validate_phase7()
    assert phase7.rmsnorm_backend == "cuda"
    assert phase7.attention_backend == "torch_paged_ref"
    with pytest.raises(ValueError, match="CUDA RMSNorm"):
        EngineConfig.for_phase6(
            max_num_seqs=2,
            max_num_batched_tokens=16,
            num_kv_blocks=8,
            max_pending_requests=4,
        ).validate_phase7()


def test_phase8_only_changes_decode_attention_backend_on_phase7_path() -> None:
    phase8 = EngineConfig.for_phase8(
        max_num_seqs=2,
        max_num_batched_tokens=16,
        num_kv_blocks=8,
        max_pending_requests=4,
    )
    RuntimeConfig(model=_model(), engine=phase8).validate_phase8()
    assert phase8.attention_backend == "cuda_paged"
    assert phase8.rmsnorm_backend == "cuda"
    with pytest.raises(ValueError, match="CUDA paged decode"):
        EngineConfig.for_phase7(
            max_num_seqs=2,
            max_num_batched_tokens=16,
            num_kv_blocks=8,
            max_pending_requests=4,
        ).validate_phase8()


def test_phase9_only_changes_decode_execution_on_phase8_path() -> None:
    phase9 = EngineConfig.for_phase9(
        max_num_seqs=2,
        max_num_batched_tokens=16,
        num_kv_blocks=8,
        max_pending_requests=4,
    )
    RuntimeConfig(model=_model(), engine=phase9).validate_phase9()
    assert phase9.decode_execution == "cuda_graph"
    assert not hasattr(phase9, "cuda_graph_buckets")
    assert not hasattr(phase9, "kv_cache_bytes")
    with pytest.raises(ValueError, match="CUDA Graph"):
        EngineConfig.for_phase8(
            max_num_seqs=2,
            max_num_batched_tokens=16,
            num_kv_blocks=8,
            max_pending_requests=4,
        ).validate_phase9()
