"""把受支持的 Hugging Face dense-decoder 配置收缩为严格运行合同。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


def _as_mapping(config: Any) -> Mapping[str, Any]:
    """同时接受 HF Config 对象和普通字典，便于无网络的 CPU 单测。"""

    if isinstance(config, Mapping):
        return config
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    raise TypeError("hf_config must be a mapping or provide to_dict().")


@dataclass(frozen=True)
class ModelConfig:
    """Qwen3 / Llama-family dense 模型的不可变运行配置。

    ``head_dim`` 必须来自模型 config，不能用 ``hidden_size / heads`` 猜。
    Qwen3-0.6B 的两个值分别是 128 和 64，这正是本项目要显式防住的坑。
    """

    model_id_or_path: str
    dtype: str
    max_model_len: int
    model_revision: str | None
    trust_remote_code: bool
    architecture: str
    model_type: str
    model_family: str
    qk_norm: bool
    checkpoint_has_lm_head: bool
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    vocab_size: int
    hidden_act: str
    attention_bias: bool
    attention_dropout: float
    mlp_bias: bool
    tie_word_embeddings: bool
    rope_interleaved: bool
    rope_scaling: object | None
    sliding_window: int | None
    use_sliding_window: bool

    @classmethod
    def from_hf_config(
        cls,
        hf_config: Any,
        *,
        model_id_or_path: str,
        dtype: str,
        max_model_len: int | None = None,
        model_revision: str | None = None,
        trust_remote_code: bool = False,
    ) -> "ModelConfig":
        """读取 HF 配置并立即校验，不把未知默认值带进模型执行。"""

        data = _as_mapping(hf_config)
        integer_fields = (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "max_position_embeddings",
            "vocab_size",
        )
        for name in integer_fields:
            value = data.get(name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer in the model config.")
        if "head_dim" in data and (
            not isinstance(data["head_dim"], int) or isinstance(data["head_dim"], bool)
        ):
            raise ValueError("head_dim must be an integer in the model config.")
        for name in ("rms_norm_eps", "rope_theta", "attention_dropout"):
            value = data.get(name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{name} must be numeric in the model config.")
        for name in (
            "attention_bias",
            "mlp_bias",
            "tie_word_embeddings",
            "rope_interleaved",
            "use_sliding_window",
        ):
            value = data.get(name, False)
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be boolean in the model config.")
        if max_model_len is not None and (
            not isinstance(max_model_len, int) or isinstance(max_model_len, bool)
        ):
            raise ValueError("max_model_len must be an integer when provided.")
        architectures = data.get("architectures") or ()
        architecture = architectures[0] if len(architectures) == 1 else ""
        model_type = str(data.get("model_type", ""))
        if architecture == "Qwen3ForCausalLM" and model_type == "qwen3":
            if "head_dim" not in data:
                raise ValueError("Qwen3 model config must define integer head_dim.")
            model_family = "qwen3"
            qk_norm = True
            checkpoint_has_lm_head = True
            head_dim = int(data["head_dim"])
        elif architecture == "LlamaForCausalLM" and model_type == "llama":
            # 当前 Llama 白名单只针对无 bias、无 rope scaling、tied embedding 的
            # SmolLM2-compatible dense decoder；不把所有 Llama 变体都假定兼容。
            model_family = "llama"
            qk_norm = False
            checkpoint_has_lm_head = False
            head_dim = int(data.get("head_dim", int(data["hidden_size"]) // int(data["num_attention_heads"])))
        else:
            model_family = "unsupported"
            qk_norm = False
            checkpoint_has_lm_head = False
            head_dim = int(data.get("head_dim", 0))
        maximum = int(data["max_position_embeddings"])
        result = cls(
            model_id_or_path=model_id_or_path,
            dtype=dtype,
            max_model_len=maximum if max_model_len is None else max_model_len,
            model_revision=model_revision,
            trust_remote_code=trust_remote_code,
            architecture=architecture,
            model_type=model_type,
            model_family=model_family,
            qk_norm=qk_norm,
            checkpoint_has_lm_head=checkpoint_has_lm_head,
            hidden_size=int(data["hidden_size"]),
            intermediate_size=int(data["intermediate_size"]),
            num_hidden_layers=int(data["num_hidden_layers"]),
            num_attention_heads=int(data["num_attention_heads"]),
            num_key_value_heads=int(data["num_key_value_heads"]),
            head_dim=head_dim,
            rms_norm_eps=float(data["rms_norm_eps"]),
            rope_theta=float(data["rope_theta"]),
            max_position_embeddings=maximum,
            vocab_size=int(data["vocab_size"]),
            hidden_act=str(data.get("hidden_act", "")),
            attention_bias=bool(data.get("attention_bias", False)),
            attention_dropout=float(data.get("attention_dropout", 0.0)),
            mlp_bias=bool(data.get("mlp_bias", False)),
            tie_word_embeddings=bool(data.get("tie_word_embeddings", False)),
            rope_interleaved=bool(data.get("rope_interleaved", False)),
            rope_scaling=data.get("rope_scaling"),
            sliding_window=data.get("sliding_window"),
            use_sliding_window=bool(data.get("use_sliding_window", False)),
        )
        result.validate()
        return result

    def validate(self) -> None:
        """拒绝 Phase 1 没有实现的模型变体和含糊配置。"""

        if not self.model_id_or_path:
            raise ValueError("model_id_or_path must not be empty.")
        if self.trust_remote_code:
            raise ValueError("trust_remote_code=True is not supported in MiniServe V1.")
        if self.model_family not in {"qwen3", "llama"}:
            raise ValueError(
                "UNSUPPORTED_MODEL_CONFIGURATION: expected Qwen3ForCausalLM or the "
                "strict SmolLM2-compatible LlamaForCausalLM subset."
            )
        if self.dtype not in {"float16", "bfloat16"}:
            raise ValueError("dtype must be explicitly float16 or bfloat16.")
        positive_ints = {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "vocab_size": self.vocab_size,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads.")
        if self.model_family == "llama" and self.num_attention_heads * self.head_dim != self.hidden_size:
            raise ValueError("Llama attention heads must exactly cover hidden_size.")
        if self.head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head_dim.")
        if (
            not math.isfinite(self.rms_norm_eps)
            or not math.isfinite(self.rope_theta)
            or self.rms_norm_eps <= 0
            or self.rope_theta <= 0
        ):
            raise ValueError("rms_norm_eps and rope_theta must be finite and positive.")
        if not 0 < self.max_model_len <= self.max_position_embeddings:
            raise ValueError("max_model_len exceeds the model position limit.")
        if self.hidden_act != "silu":
            raise ValueError("Only the Qwen3 SiLU MLP is implemented.")
        if self.attention_bias or self.attention_dropout != 0.0 or self.mlp_bias:
            raise ValueError("Only bias-free, dropout-free dense attention/MLP is implemented.")
        if self.rope_interleaved or self.rope_scaling is not None:
            raise ValueError("Interleaved or scaled RoPE is not implemented.")
        if self.sliding_window is not None or self.use_sliding_window:
            raise ValueError("Sliding-window attention is not implemented.")
        if not self.tie_word_embeddings:
            raise ValueError("MiniServe V1 requires tied input/output embeddings.")

    def torch_dtype(self):
        """延迟导入 torch，使配置单测不会初始化 CUDA。"""

        import torch

        return {"float16": torch.float16, "bfloat16": torch.bfloat16}[self.dtype]
