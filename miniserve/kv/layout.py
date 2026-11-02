"""所有 KV backend 共用的 canonical 逻辑布局。"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class KVLayout:
    """Phase 2 每请求连续布局：``[layer, position, kv_head, head_dim]``。

    K 与 V 使用两个独立 Tensor，因此 bytes/token 要乘 2。后续 paged backend
    会改变物理 storage 的 position 轴，但仍从同一个布局对象读取模型维度。
    """

    num_layers: int
    max_model_len: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype
    device: torch.device | str
    num_blocks: int | None = None
    block_size: int = 16

    def __post_init__(self) -> None:
        object.__setattr__(self, "device", torch.device(self.device))
        self.validate()

    def validate(self) -> None:
        for name in ("num_layers", "max_model_len", "num_kv_heads", "head_dim"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise ValueError("KV dtype must be float16, bfloat16, or float32.")
        if not isinstance(self.block_size, int) or isinstance(self.block_size, bool) or self.block_size <= 0:
            raise ValueError("block_size must be a positive integer.")
        if self.num_blocks is not None and (
            not isinstance(self.num_blocks, int)
            or isinstance(self.num_blocks, bool)
            or self.num_blocks <= 0
        ):
            raise ValueError("num_blocks must be a positive integer when configured.")

    @property
    def axis_order(self) -> tuple[str, ...]:
        return ("layer", "position", "kv_head", "head_dim")

    @property
    def sequence_shape(self) -> tuple[int, int, int, int]:
        return (self.num_layers, self.max_model_len, self.num_kv_heads, self.head_dim)

    @property
    def dtype_bytes(self) -> int:
        return torch.empty((), dtype=self.dtype).element_size()

    def bytes_per_token(self) -> int:
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * self.dtype_bytes

    def bytes_per_sequence(self) -> int:
        return self.max_model_len * self.bytes_per_token()

    @property
    def paged_pool_shape(self) -> tuple[int, int, int, int, int]:
        if self.num_blocks is None:
            raise ValueError("Paged pool requires num_blocks.")
        return (
            self.num_layers,
            self.num_blocks,
            self.num_kv_heads,
            self.block_size,
            self.head_dim,
        )

    def bytes_per_block(self) -> int:
        return self.block_size * self.bytes_per_token()

    def metadata(self) -> dict[str, object]:
        """写入 evidence 的稳定布局描述，不包含设备 pointer。"""

        metadata = {
            "axis_order": list(self.axis_order),
            "sequence_shape_per_tensor": list(self.sequence_shape),
            "k_v_separate_pools": True,
            "dtype": str(self.dtype),
            "dtype_bytes": self.dtype_bytes,
            "bytes_per_token": self.bytes_per_token(),
            "bytes_per_sequence": self.bytes_per_sequence(),
            "device": str(self.device),
            "num_blocks": self.num_blocks,
            "block_size": self.block_size,
            "bytes_per_block": self.bytes_per_block(),
        }
        if self.num_blocks is not None:
            metadata["paged_pool_shape_per_tensor"] = list(self.paged_pool_shape)
            metadata["paged_pool_bytes"] = self.num_blocks * self.bytes_per_block()
        return metadata
