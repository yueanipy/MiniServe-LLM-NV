"""MiniServe 自建 Qwen3 模型；Hugging Face 模型只在 tests 中作为 oracle。"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .qwen3 import Qwen3ForCausalLM

__all__ = ["Qwen3ForCausalLM"]


def __getattr__(name: str):
    """只有调用者真正请求模型类时才导入 torch-heavy 实现。"""

    if name == "Qwen3ForCausalLM":
        from .qwen3 import Qwen3ForCausalLM

        return Qwen3ForCausalLM
    raise AttributeError(name)
