"""可选 NVTX range；CUDA/NVTX 不可用时保持无副作用。"""

from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def nvtx_range(message: str):
    pushed = False
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.nvtx.range_push(message)
            pushed = True
        yield
    finally:
        if pushed:
            torch.cuda.nvtx.range_pop()
