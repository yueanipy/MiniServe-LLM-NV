"""外部 chat schema 到既有 EngineRequest 的唯一适配器。"""

from __future__ import annotations

from collections.abc import Sequence

from miniserve.engine.request import EngineRequest, SamplingParams

from .schemas import ChatCompletionRequest


def _as_token_tuple(value: object) -> tuple[int, ...]:
    """归一化 HF tokenizer 的 list/Tensor 返回值，并拒绝空或嵌套 batch。"""

    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("Tokenizer must return a sequence of token IDs.")
    if any(not isinstance(token, int) or isinstance(token, bool) for token in value):
        raise TypeError("Tokenizer token IDs must be integers, not coercible values.")
    tokens = tuple(value)
    if not tokens:
        raise ValueError("Chat template produced an empty prompt.")
    if any(token < 0 for token in tokens):
        raise ValueError("Tokenizer returned a negative token ID.")
    return tokens


def normalize_stop_token_ids(eos: object) -> tuple[int, ...]:
    """把配置中的单个/list/tuple EOS ID 严格归一化。"""

    if eos is None:
        return ()
    values = eos if isinstance(eos, (list, tuple)) else (eos,)
    if any(
        not isinstance(token, int) or isinstance(token, bool) or token < 0
        for token in values
    ):
        raise ValueError("EOS token IDs must be non-negative integers.")
    return tuple(sorted(set(values)))


def model_stop_token_ids(tokenizer: object) -> tuple[int, ...]:
    """把 tokenizer 的单个/多个 EOS ID 归一化、去重并稳定排序。"""

    return normalize_stop_token_ids(getattr(tokenizer, "eos_token_id", None))


def to_engine_request(
    request: ChatCompletionRequest,
    tokenizer: object,
    *,
    request_id: str,
    arrival_ns: int,
    max_model_len: int,
    vocab_size: int | None = None,
    additional_stop_token_ids: tuple[int, ...] = (),
    default_temperature: float = 0.0,
    default_top_p: float = 1.0,
    default_top_k: int = 0,
) -> EngineRequest:
    """应用固定 chat template 后创建框架无关请求，不直接调用 model.forward。"""

    if not isinstance(max_model_len, int) or isinstance(max_model_len, bool) or max_model_len <= 0:
        raise ValueError("max_model_len must be a positive integer.")
    if vocab_size is not None and (
        not isinstance(vocab_size, int) or isinstance(vocab_size, bool) or vocab_size <= 0
    ):
        raise ValueError("vocab_size must be a positive integer when provided.")
    messages = [message.model_dump() for message in request.messages]
    token_ids = _as_token_tuple(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )
    if len(token_ids) + request.max_tokens > max_model_len:
        raise ValueError(
            "prompt_tokens + max_tokens exceeds max_model_len "
            f"({len(token_ids) + request.max_tokens} > {max_model_len})."
        )
    if vocab_size is not None and any(token >= vocab_size for token in token_ids):
        raise ValueError("Tokenizer returned a token outside the model vocabulary.")
    stop_token_ids = (
        ()
        if request.ignore_eos
        else tuple(
            sorted(
                set(model_stop_token_ids(tokenizer))
                | set(additional_stop_token_ids)
            )
        )
    )
    params = SamplingParams(
        temperature=(default_temperature if request.temperature is None else request.temperature),
        top_p=(default_top_p if request.top_p is None else request.top_p),
        top_k=(default_top_k if request.top_k is None else request.top_k),
        max_new_tokens=request.max_tokens,
        stop_token_ids=stop_token_ids,
        seed=request.seed,
    )
    if vocab_size is not None:
        if params.top_k > vocab_size:
            raise ValueError("top_k must not exceed the model vocabulary size.")
        if any(token >= vocab_size for token in params.stop_token_ids):
            raise ValueError("stop token ID is outside the model vocabulary.")
    return EngineRequest(request_id, token_ids, params, arrival_ns)
