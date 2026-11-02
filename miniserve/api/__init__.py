"""Phase 10 外部 HTTP 协议层；这里的类型禁止反向进入 EngineCore。"""

from .adapter import to_engine_request
from .schemas import ChatCompletionRequest

__all__ = ["ChatCompletionRequest", "to_engine_request"]
