"""Attention 执行 backend 合同。"""

from .torch_contiguous import TorchContiguousAttention
from .torch_paged_ref import TorchPagedReferenceAttention
from .cuda_paged import CudaPagedDecodeAttention
from .flashinfer_paged import FlashInferPagedDecodeAttention

__all__ = [
    "TorchContiguousAttention",
    "TorchPagedReferenceAttention",
    "CudaPagedDecodeAttention",
    "FlashInferPagedDecodeAttention",
]
