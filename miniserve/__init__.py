"""MiniServe-LLM-NV.

Only the Phase 0 toolchain is implemented at this point.

包顶层故意不自动加载 CUDA 共享库：仅导入 ``miniserve`` 不应产生 GPU 上下文、
磁盘探测或动态链接副作用。需要原生算子时由 ``miniserve._native`` 显式加载。
"""

# 版本属于 Python distribution；它不表示后续 Qwen/Engine Phase 已经完成。
__version__ = "0.0.1"
