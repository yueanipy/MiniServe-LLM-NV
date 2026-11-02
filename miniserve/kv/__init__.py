"""KV cache 的稳定合同与 Phase 2 contiguous reference 实现。"""

from .base import KVCapacitySnapshot, KVSequenceExecutionMetadata, KVStepContext
from .contiguous import ContiguousKVCacheBackend
from .layout import KVLayout
from .manager import KVCacheManager

__all__ = [
    "ContiguousKVCacheBackend",
    "KVCacheManager",
    "KVCapacitySnapshot",
    "KVLayout",
    "KVSequenceExecutionMetadata",
    "KVStepContext",
]
