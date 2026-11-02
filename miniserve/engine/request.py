"""外部输入进入调度器前的纯 Python 请求合同。"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SamplingParams:
    """与 HTTP 框架无关的生成参数。

    ``temperature == 0`` 明确表示 greedy；大于零时固定按 temperature、
    top-k、top-p 的顺序采样。RNG 状态由 ModelExecutor 按 request_id 持有，
    这里仅保存可序列化的 seed，避免把 ``torch.Generator`` 放进请求状态。
    """

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    max_new_tokens: int = 4
    stop_token_ids: tuple[int, ...] = ()
    seed: int | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.temperature, (int, float))
            or isinstance(self.temperature, bool)
            or not math.isfinite(self.temperature)
            or not 0 <= self.temperature <= 2
        ):
            raise ValueError("temperature must be finite and in [0, 2].")
        if (
            not isinstance(self.top_p, (int, float))
            or isinstance(self.top_p, bool)
            or not math.isfinite(self.top_p)
            or not 0 < self.top_p <= 1
        ):
            raise ValueError("top_p must be in (0, 1].")
        if not isinstance(self.top_k, int) or isinstance(self.top_k, bool) or self.top_k < 0:
            raise ValueError("top_k must be non-negative.")
        if (
            not isinstance(self.max_new_tokens, int)
            or isinstance(self.max_new_tokens, bool)
            or self.max_new_tokens <= 0
        ):
            raise ValueError("max_new_tokens must be positive.")
        if not isinstance(self.stop_token_ids, tuple):
            raise ValueError("stop_token_ids must be an immutable tuple.")
        if any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in self.stop_token_ids):
            raise ValueError("stop_token_ids must contain non-negative integers.")
        if len(set(self.stop_token_ids)) != len(self.stop_token_ids):
            raise ValueError("stop_token_ids must not contain duplicates.")
        if self.seed is not None and (
            not isinstance(self.seed, int)
            or isinstance(self.seed, bool)
            or not 0 <= self.seed <= (2**63 - 1)
        ):
            raise ValueError("seed must be an integer in [0, 2^63-1] or None.")

    def validate_phase1(self) -> None:
        if self.temperature != 0.0 or self.top_p != 1.0 or self.top_k != 0:
            raise ValueError("Phase 1 supports greedy sampling only.")


@dataclass(frozen=True)
class EngineRequest:
    """已 token 化的不可变请求；HTTP/Pydantic 类型不会进入 EngineCore。"""

    request_id: str
    prompt_token_ids: tuple[int, ...]
    sampling_params: SamplingParams
    arrival_ns: int

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("request_id must not be empty.")
        if not isinstance(self.prompt_token_ids, tuple) or not self.prompt_token_ids:
            raise ValueError("prompt_token_ids must not be empty.")
        if any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in self.prompt_token_ids):
            raise ValueError("prompt_token_ids must contain non-negative integers.")
        if not isinstance(self.sampling_params, SamplingParams):
            raise ValueError("sampling_params must be a SamplingParams instance.")
        if (
            not isinstance(self.arrival_ns, int)
            or isinstance(self.arrival_ns, bool)
            or self.arrival_ns < 0
        ):
            raise ValueError("arrival_ns must be non-negative.")
