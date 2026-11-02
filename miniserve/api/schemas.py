"""冻结的最小 OpenAI-compatible chat completion Pydantic schema。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictExternalModel(BaseModel):
    """外部输入不接受未知字段，防止“参数收到但实际未生效”。"""

    model_config = ConfigDict(extra="forbid")


class ChatMessage(StrictExternalModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=65_536)


class ChatCompletionRequest(StrictExternalModel):
    model: str = Field(min_length=1, max_length=512)
    messages: list[ChatMessage] = Field(min_length=1, max_length=256)
    max_tokens: int = Field(default=16, ge=1, le=4096)
    # None 表示使用当前已锁定 checkpoint 的 generation config；API schema 不写死模型默认值。
    temperature: float | None = Field(default=None, ge=0.0, le=2.0, allow_inf_nan=False)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0, allow_inf_nan=False)
    top_k: int | None = Field(default=None, ge=0)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    # Explicit experiment control: suppress model EOS while preserving the
    # normal/default behavior when false.
    ignore_eos: bool = False
    stream: bool = False
    # V1 尚未实现跨 token 字符串 stop matcher；显式字段用于给出清楚错误。
    stop: str | list[str] | None = None

    @field_validator("stop")
    @classmethod
    def reject_string_stop_until_incremental_matcher_exists(cls, value):
        if value is not None:
            raise ValueError("string stop is not supported in V1; use model EOS termination")
        return value


class CompletionUsage(StrictExternalModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class AssistantMessage(StrictExternalModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatCompletionChoice(StrictExternalModel):
    index: int = 0
    message: AssistantMessage
    finish_reason: Literal["stop", "length"]


class ChatCompletionResponse(StrictExternalModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: CompletionUsage


class StreamDelta(StrictExternalModel):
    role: Literal["assistant"] | None = None
    content: str | None = None


class StreamChoice(StrictExternalModel):
    index: int = 0
    delta: StreamDelta
    finish_reason: Literal["stop", "length"] | None = None


class ChatCompletionChunk(StrictExternalModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[StreamChoice]
    usage: CompletionUsage | None = None
