"""受支持 Qwen3/Llama safetensors key 到共享 dense model 的严格映射。"""

from __future__ import annotations

import re

from miniserve.config.model_config import ModelConfig


_LAYER_KEY = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\."
    r"(?P<suffix>self_attn\.(?:q_proj|k_proj|v_proj|o_proj)\.weight|"
    r"self_attn\.(?:q_norm|k_norm)\.weight|"
    r"mlp\.(?:gate_proj|up_proj|down_proj)\.weight|"
    r"(?:input_layernorm|post_attention_layernorm)\.weight)$"
)


def expected_parameter_map(config: ModelConfig) -> dict[str, str]:
    """枚举必须恰好被消费的官方权重，包括 snapshot 中显式保存的 lm_head。"""

    mapping: dict[str, str] = {
        "model.embed_tokens.weight": "model.embed_tokens.weight",
        "model.norm.weight": "model.norm.weight",
    }
    if config.checkpoint_has_lm_head:
        # 冻结的 Qwen3 snapshot 显式保存 tied lm_head；SmolLM2 checkpoint 省略它。
        mapping["lm_head.weight"] = "lm_head.weight"
    suffixes = [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
    ]
    if config.qk_norm:
        suffixes[4:4] = ["self_attn.q_norm.weight", "self_attn.k_norm.weight"]
    for layer in range(config.num_hidden_layers):
        for suffix in suffixes:
            key = f"model.layers.{layer}.{suffix}"
            mapping[key] = key
    return mapping


def map_hf_key(hf_key: str, config: ModelConfig) -> str:
    """验证 key 属于已声明 schema 后返回目标名，未知 key 绝不猜测。"""

    mapping = expected_parameter_map(config)
    try:
        return mapping[hf_key]
    except KeyError as exc:
        match = _LAYER_KEY.match(hf_key)
        if match and int(match.group("layer")) >= config.num_hidden_layers:
            raise KeyError(f"Layer index outside config: {hf_key}") from exc
        raise KeyError(f"Unexpected Hugging Face weight: {hf_key}") from exc
