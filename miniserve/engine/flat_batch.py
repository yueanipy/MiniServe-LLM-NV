"""扁平 ``[num_tokens]`` 批执行所需的设备张量与 host 形状证明。

vLLM v1 与 SGLang 使用 packed/ragged 输入；本模块的 M1 路径不同：decode 每个
请求恰好一个 token，``context_lens`` 是每个请求的绝对长度，**不是** cumulative
``query_start_loc``/``cu_seqlens``。跨请求隔离由 paged block table、每行
sequence length 和每请求一个 slot 保证。

本模块只负责**把多个 decode 请求拼成扁平批并完成全部校验**，不触碰模型前向。
这样设计的理由：拼接、padding 与采样索引推导是这一步最容易出错的部分——把
它们隔离到纯 CPU 可验证的层，GPU 侧就只剩"照着已有 metadata 调用 kernel"。

decode 先行（而非 prefill）的原因：decode 每请求恰好 1 个 token，flat 输入
天然是 token-major，正好匹配 ``paged_kv_append`` 的
``slot_mapping.size(0) == key.size(0)`` 契约；变长 prefill 的拼接需要
request-major 算子或逐请求循环。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from miniserve.kv.base import KVSequenceExecutionMetadata

_HOST_PROOF_TOKEN = object()
_UNTRUSTED_PROOF_TOKEN = object()


@dataclass(frozen=True)
class _TensorBinding:
    object_id: int
    version: int
    data_ptr: int
    shape: tuple[int, ...]
    dtype: str
    device: str


def _tensor_binding(tensor: torch.Tensor) -> _TensorBinding:
    return _TensorBinding(
        object_id=id(tensor),
        version=int(tensor._version),
        data_ptr=int(tensor.data_ptr()),
        shape=tuple(tensor.shape),
        dtype=str(tensor.dtype),
        device=str(tensor.device),
    )


@dataclass(frozen=True)
class FlatDecodeHostMetadata:
    """与设备张量同源且绑定到具体张量版本的 host 证明。"""

    context_lens: tuple[int, ...]
    positions: tuple[int, ...]
    slots: tuple[int, ...]
    last_token_indices: tuple[int, ...]
    block_tables: tuple[tuple[int, ...], ...]
    max_blocks: int
    physical_block_count: int | None
    input_tokens: tuple[int, ...]
    vocab_size: int | None
    max_model_len: int | None
    tensor_bindings: tuple[_TensorBinding, ...]
    proof_token: object | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class FlatDecodeBatch:
    """一次 batched decode 的全部设备张量与 CPU 侧 provenance。

    ``input_ids`` / ``positions`` / ``slot_mapping`` 都按请求顺序排列，每个
    请求恰好一项。``context_lens[i]`` 是第 i 个请求写入后的绝对长度；
    采样索引 ``last_token_indices`` 指向每个请求的最后一个（也是唯一一个）token。
    """

    input_ids: torch.Tensor
    positions: torch.Tensor
    slot_mapping: torch.Tensor
    context_lens: torch.Tensor
    block_tables: torch.Tensor
    last_token_indices: torch.Tensor
    request_ids: tuple[str, ...]
    block_size: int
    num_tokens: int
    # 仅由 ``flatten_decode_batch`` 从已经校验过的不可变 host metadata 构造。
    # 公开直接构造必须提供内部 proof；非可信外部张量走 from_untrusted_tensors
    # 的一次 D2H 校验入口。
    host_metadata: FlatDecodeHostMetadata | None = None

    @classmethod
    def from_untrusted_tensors(
        cls,
        *,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        context_lens: torch.Tensor,
        block_tables: torch.Tensor,
        last_token_indices: torch.Tensor,
        request_ids: tuple[str, ...],
        block_size: int,
        num_tokens: int,
        physical_block_count: int,
        vocab_size: int,
        max_model_len: int,
    ) -> "FlatDecodeBatch":
        """公开非可信入口：一次 D2H 后绑定 proof，并立即校验 pool/model 边界。"""

        tensors = (
            input_ids,
            positions,
            slot_mapping,
            context_lens,
            block_tables,
            last_token_indices,
        )
        if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise TypeError("All flat decode inputs must be torch.Tensor instances.")
        if block_tables.ndim != 2:
            raise ValueError("block_tables must be a 2-D [batch, max_blocks] tensor.")
        for name, value in (
            ("physical_block_count", physical_block_count),
            ("vocab_size", vocab_size),
            ("max_model_len", max_model_len),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        host_metadata = FlatDecodeHostMetadata(
            context_lens=tuple(int(value) for value in context_lens.tolist()),
            positions=tuple(int(value) for value in positions.tolist()),
            slots=tuple(int(value) for value in slot_mapping.tolist()),
            last_token_indices=tuple(
                int(value) for value in last_token_indices.tolist()
            ),
            block_tables=tuple(
                tuple(int(value) for value in row)
                for row in block_tables.tolist()
            ),
            max_blocks=block_tables.shape[1],
            physical_block_count=physical_block_count,
            input_tokens=tuple(int(value) for value in input_ids.tolist()),
            vocab_size=vocab_size,
            max_model_len=max_model_len,
            tensor_bindings=tuple(_tensor_binding(tensor) for tensor in tensors),
            proof_token=_UNTRUSTED_PROOF_TOKEN,
        )
        return cls(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            last_token_indices=last_token_indices,
            request_ids=request_ids,
            block_size=block_size,
            num_tokens=num_tokens,
            host_metadata=host_metadata,
        )

    def __post_init__(self) -> None:
        for name in ("block_size", "num_tokens"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if not isinstance(self.request_ids, tuple) or not self.request_ids:
            raise ValueError("request_ids must be a non-empty tuple.")
        if any(not isinstance(item, str) or not item for item in self.request_ids):
            raise ValueError("request_ids must contain non-empty strings.")
        if len(set(self.request_ids)) != len(self.request_ids):
            raise ValueError("request_ids must be unique within one batched decode.")
        batch = len(self.request_ids)
        if self.num_tokens != batch:
            raise ValueError("A decode batch contributes exactly one token per request.")
        expected = {
            "input_ids": (self.num_tokens,),
            "positions": (self.num_tokens,),
            "slot_mapping": (self.num_tokens,),
            "context_lens": (batch,),
            "last_token_indices": (batch,),
        }
        for name, shape in expected.items():
            tensor = getattr(self, name)
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor.")
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}.")
        if not isinstance(self.block_tables, torch.Tensor) or self.block_tables.ndim != 2:
            raise ValueError("block_tables must be a 2-D [batch, max_blocks] tensor.")
        if self.block_tables.shape[0] != batch:
            raise ValueError("block_tables first dimension must equal the batch size.")
        self.assert_invariants()

    def assert_invariants(self) -> None:
        """验证张量间的形状与取值关系；任何不一致立即抛出。"""

        for name in ("input_ids", "positions", "slot_mapping"):
            if getattr(self, name).dtype != torch.long:
                raise ValueError(f"{name} must be int64.")
        if self.context_lens.dtype != torch.int32:
            raise ValueError("context_lens must be int32 to match the paged kernels.")
        if self.block_tables.dtype != torch.int32:
            raise ValueError("block_tables must be int32 to match the paged kernels.")
        if self.last_token_indices.dtype != torch.long:
            raise ValueError("last_token_indices must be int64 for downstream gather.")
        devices = {
            self.input_ids.device,
            self.positions.device,
            self.slot_mapping.device,
            self.context_lens.device,
            self.block_tables.device,
            self.last_token_indices.device,
        }
        if len(devices) != 1:
            raise ValueError("All FlatDecodeBatch tensors must be on one device.")

        if self.host_metadata is None:
            raise ValueError(
                "FlatDecodeBatch requires an internal proof; public untrusted "
                "construction must use from_untrusted_tensors(...) with explicit "
                "pool/vocab/max_model_len bounds."
            )
        if self.host_metadata.proof_token not in {
            _HOST_PROOF_TOKEN,
            _UNTRUSTED_PROOF_TOKEN,
        }:
            raise ValueError(
                "host_metadata lacks a recognized construction proof."
            )
        self._assert_tensor_bindings(self.host_metadata)
        host = self.host_metadata
        self._validate_host_values(host)

    def _assert_tensor_bindings(self, host: FlatDecodeHostMetadata) -> None:
        """证明只适用于 factory 构造时的那组确切张量及其当前版本。"""

        tensors = (
            self.input_ids,
            self.positions,
            self.slot_mapping,
            self.context_lens,
            self.block_tables,
            self.last_token_indices,
        )
        if len(host.tensor_bindings) != len(tensors):
            raise ValueError("host_metadata tensor binding count is invalid.")
        for tensor, binding in zip(tensors, host.tensor_bindings):
            if (
                id(tensor) != binding.object_id
                or int(tensor._version) != binding.version
                or int(tensor.data_ptr()) != binding.data_ptr
                or tuple(tensor.shape) != binding.shape
                or str(tensor.dtype) != binding.dtype
                or str(tensor.device) != binding.device
            ):
                raise ValueError(
                    "host_metadata proof is not bound to the current tensors "
                    "or has become stale."
                )

    def _validate_host_values(self, host: FlatDecodeHostMetadata) -> None:
        if not isinstance(host, FlatDecodeHostMetadata):
            raise TypeError("host_metadata must be FlatDecodeHostMetadata.")
        batch = len(self.request_ids)
        for name in ("context_lens", "positions", "slots", "last_token_indices"):
            values = getattr(host, name)
            if not isinstance(values, tuple) or len(values) != batch:
                raise ValueError(f"Host {name} must contain exactly one value per request.")
        if not isinstance(host.block_tables, tuple) or len(host.block_tables) != batch:
            raise ValueError("Host block_tables must contain exactly one row per request.")
        if not isinstance(host.input_tokens, tuple) or len(host.input_tokens) != batch:
            raise ValueError("Host input_tokens must contain one token per request.")
        if (
            not isinstance(host.max_blocks, int)
            or isinstance(host.max_blocks, bool)
            or host.max_blocks <= 0
            or host.max_blocks != self.block_tables.shape[1]
        ):
            raise ValueError("Host max_blocks must equal the padded block-table width.")

        if any(value <= 0 for value in host.context_lens):
            raise ValueError("Every context length must be positive.")
        if any(value < 0 for value in host.input_tokens):
            raise ValueError("Host input tokens must be non-negative.")
        if host.vocab_size is not None and any(
            value >= host.vocab_size for value in host.input_tokens
        ):
            raise ValueError("Host input token is outside the model vocabulary.")
        if host.max_model_len is not None and any(
            value > host.max_model_len for value in host.context_lens
        ):
            raise ValueError("Host context length exceeds max_model_len.")
        if host.last_token_indices != tuple(range(batch)):
            raise ValueError("last_token_indices must be 0..batch-1 for a decode batch.")
        for index, length in enumerate(host.context_lens):
            if host.positions[index] != length - 1:
                raise ValueError("Each position must equal its request's last context index.")

        if any(value < 0 for value in host.slots):
            raise ValueError("slot_mapping must not contain negative slots.")
        if len(set(host.slots)) != len(host.slots):
            raise ValueError("slot_mapping must contain unique write slots.")
        for index, (length, slot) in enumerate(zip(host.context_lens, host.slots)):
            required = (length + self.block_size - 1) // self.block_size
            if required > host.max_blocks:
                raise ValueError("Context length requires more blocks than the padded table holds.")
            row = host.block_tables[index]
            if not isinstance(row, tuple) or len(row) != host.max_blocks:
                raise ValueError("Each host block-table row must match max_blocks.")
            logical, offset = divmod(length - 1, self.block_size)
            physical = row[logical]
            if (
                not isinstance(physical, int)
                or isinstance(physical, bool)
                or physical < 0
            ):
                raise ValueError("Host block table contains an invalid physical block ID.")
            if (
                host.physical_block_count is not None
                and physical >= host.physical_block_count
            ):
                raise ValueError("Host block table contains an out-of-range physical block ID.")
            if slot != physical * self.block_size + offset:
                raise ValueError(
                    f"Slot for request {self.request_ids[index]!r} must equal "
                    "block_table[logical] * block_size + offset."
                )
            if (
                host.physical_block_count is not None
                and slot >= host.physical_block_count * self.block_size
            ):
                raise ValueError("Host slot is outside the physical K/V pool.")

    @property
    def batch_size(self) -> int:
        return len(self.request_ids)

    def as_dict(self) -> dict[str, object]:
        if self.host_metadata is None:
            raise RuntimeError("FlatDecodeBatch has no host proof.")
        context_lens = self.host_metadata.context_lens
        last_token_indices = self.host_metadata.last_token_indices
        return {
            "batch_size": self.batch_size,
            "num_tokens": self.num_tokens,
            "block_size": self.block_size,
            "request_ids": list(self.request_ids),
            "context_lens": list(context_lens),
            "last_token_indices": list(last_token_indices),
            "max_blocks": int(self.block_tables.shape[1]),
        }


def flatten_decode_batch(
    metadata: tuple[KVSequenceExecutionMetadata, ...],
    *,
    host_input_token_ids: tuple[int, ...],
    block_size: int,
    device: torch.device | str,
    physical_block_count: int | None = None,
) -> FlatDecodeBatch:
    """把 N 个 decode 请求拼成一个扁平批，并完成全部校验。

    ``metadata`` 必须按请求顺序提供，且每项恰好推进一个 token——这是 decode
    step 的合同。任何违反都会在构造张量之前失败，避免把非法 metadata 送进 kernel。
    """

    # 类型错误与取值错误必须分开：调用方传 list 是编程错误（TypeError），
    # 传空 tuple 是取值错误（ValueError），混在一起会让调用方无法区分处理。
    if not isinstance(metadata, tuple):
        raise TypeError("metadata must be an immutable tuple of KV execution metadata.")
    if not metadata:
        raise ValueError("metadata must contain at least one decode request.")
    if not isinstance(host_input_token_ids, tuple):
        raise TypeError("host_input_token_ids must be an immutable tuple.")
    if len(host_input_token_ids) != len(metadata):
        raise ValueError("One host input token is required per decode request.")
    if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size <= 0:
        raise ValueError("block_size must be a positive integer.")
    if physical_block_count is not None and (
        not isinstance(physical_block_count, int)
        or isinstance(physical_block_count, bool)
        or physical_block_count <= 0
    ):
        raise ValueError("physical_block_count must be a positive integer when set.")

    request_ids: list[str] = []
    context_lens: list[int] = []
    slots: list[int] = []
    block_tables: list[tuple[int, ...]] = []

    for index, item in enumerate(metadata):
        if not isinstance(item, KVSequenceExecutionMetadata):
            raise TypeError("metadata entries must be KVSequenceExecutionMetadata.")
        if item.context_len_after_step - item.context_len_before_step != 1:
            raise ValueError("A batched decode requires exactly one new token per request.")
        if len(item.slot_mapping) != 1:
            raise ValueError("A batched decode requires exactly one slot mapping per request.")
        token = host_input_token_ids[index]
        if not isinstance(token, int) or isinstance(token, bool) or token < 0:
            raise ValueError("Host input tokens must be non-negative integers.")
        context = item.context_len_after_step
        required_blocks = (context + block_size - 1) // block_size
        if len(item.block_table) < required_blocks:
            raise ValueError("Block table is shorter than the context length requires.")
        active_table = item.block_table[:required_blocks]
        if any(value < 0 for value in active_table):
            raise ValueError("Block table must not contain negative physical block IDs.")
        if physical_block_count is not None and any(
            value >= physical_block_count for value in active_table
        ):
            raise ValueError("Block table contains an out-of-range physical block ID.")

        request_ids.append(item.request_id)
        context_lens.append(context)
        slots.append(item.slot_mapping[0])
        block_tables.append(item.block_table)

    if len(set(request_ids)) != len(request_ids):
        raise ValueError("Batched decode requires unique request IDs.")

    if len(set(slots)) != len(slots):
        raise ValueError("Batched decode requires unique write slots.")

    max_blocks = max(
        (length + block_size - 1) // block_size for length in context_lens
    )
    # 未使用的列填 0：kernel 只按 context length 推导的逻辑块数量读取，
    # 而 context length 又受 padding 后的列数上限约束，因此 padding 不会被读取。
    padded_host_tables: list[tuple[int, ...]] = []
    padded_tables_cpu = torch.zeros(
        (len(request_ids), max_blocks), dtype=torch.int32
    )
    for row, table in enumerate(block_tables):
        required = (context_lens[row] + block_size - 1) // block_size
        active = tuple(table[:required])
        padded = active + (0,) * (max_blocks - required)
        padded_host_tables.append(padded)
        padded_tables_cpu[row, : len(padded)] = torch.tensor(padded, dtype=torch.int32)
    padded_tables = padded_tables_cpu.to(device=device)
    input_ids = torch.tensor(host_input_token_ids, dtype=torch.long, device=device)
    positions = torch.tensor(context_lens, dtype=torch.long, device=device) - 1
    slot_mapping = torch.tensor(slots, dtype=torch.long, device=device)
    context_lens_tensor = torch.tensor(context_lens, dtype=torch.int32, device=device)
    last_token_indices = torch.arange(
        len(request_ids), dtype=torch.long, device=device
    )
    tensor_bindings = tuple(
        _tensor_binding(tensor)
        for tensor in (
            input_ids,
            positions,
            slot_mapping,
            context_lens_tensor,
            padded_tables,
            last_token_indices,
        )
    )
    host_metadata = FlatDecodeHostMetadata(
        context_lens=tuple(context_lens),
        positions=tuple(length - 1 for length in context_lens),
        slots=tuple(slots),
        last_token_indices=tuple(range(len(request_ids))),
        block_tables=tuple(padded_host_tables),
        max_blocks=max_blocks,
        physical_block_count=physical_block_count,
        input_tokens=host_input_token_ids,
        vocab_size=None,
        max_model_len=None,
        tensor_bindings=tensor_bindings,
        proof_token=_HOST_PROOF_TOKEN,
    )

    return FlatDecodeBatch(
        input_ids=input_ids,
        # decode 每请求恰好一个 token，其绝对位置就是该请求最后一个 context 下标。
        positions=positions,
        slot_mapping=slot_mapping,
        context_lens=context_lens_tensor,
        block_tables=padded_tables,
        last_token_indices=last_token_indices,
        request_ids=tuple(request_ids),
        block_size=block_size,
        num_tokens=len(request_ids),
        host_metadata=host_metadata,
    )


def validate_flat_decode_provenance(
    batch: FlatDecodeBatch,
    metadata: tuple[KVSequenceExecutionMetadata, ...],
    host_input_token_ids: tuple[int, ...],
    *,
    physical_block_count: int,
    vocab_size: int,
    max_model_len: int,
) -> None:
    """在发 GPU 前校验一次 batched decode 的 host provenance。"""

    if not isinstance(batch, FlatDecodeBatch):
        raise TypeError("batch must be a FlatDecodeBatch.")
    # 每轮 launch 前重验 proof 与 tensor 身份/版本，in-place 失效也必须在 kernel 前拒绝。
    batch.assert_invariants()
    proof = batch.host_metadata
    if proof is None:
        raise ValueError("batch has no host proof.")
    if not isinstance(metadata, tuple) or len(metadata) != batch.batch_size:
        raise ValueError("metadata must contain one entry per batched request.")
    if not isinstance(host_input_token_ids, tuple) or len(host_input_token_ids) != batch.batch_size:
        raise ValueError("host_input_token_ids must contain one token per request.")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in host_input_token_ids):
        raise ValueError("host input tokens must be non-negative integers.")
    if any(value >= vocab_size for value in host_input_token_ids):
        raise ValueError("host input token is outside the model vocabulary.")
    if physical_block_count <= 0 or max_model_len <= 0:
        raise ValueError("pool and model length must be positive.")
    if host_input_token_ids != proof.input_tokens:
        raise ValueError(
            "host input tokens do not match the values bound into the batch proof."
        )
    if any(value > max_model_len for value in proof.context_lens):
        raise ValueError("batch proof context length exceeds max_model_len.")
    if any(slot >= physical_block_count * batch.block_size for slot in proof.slots):
        raise ValueError("batch proof slot is outside the physical K/V pool.")
    for row in proof.block_tables:
        if any(block < 0 or block >= physical_block_count for block in row):
            raise ValueError(
                "batch proof block table contains an out-of-range physical block ID."
            )

    seen_slots: set[int] = set()
    for index, (request_id, item) in enumerate(zip(batch.request_ids, metadata)):
        if item.request_id != request_id:
            raise ValueError("metadata request order does not match FlatDecodeBatch.")
        if item.context_len_after_step != proof.context_lens[index]:
            raise ValueError(
                "metadata context length does not match the exact batch proof."
            )
        if item.context_len_before_step + 1 != item.context_len_after_step:
            raise ValueError("batched decode metadata must advance exactly one token.")
        if item.context_len_after_step <= 0 or item.context_len_after_step > max_model_len:
            raise ValueError("batched decode context length is outside max_model_len.")
        if len(item.slot_mapping) != 1:
            raise ValueError("batched decode metadata requires exactly one slot.")
        slot = item.slot_mapping[0]
        if slot != proof.slots[index]:
            raise ValueError(
                "metadata slot does not match the exact batch proof."
            )
        if slot in seen_slots:
            raise ValueError("batched decode metadata contains duplicate write slots.")
        seen_slots.add(slot)
        required = (item.context_len_after_step + batch.block_size - 1) // batch.block_size
        if len(item.block_table) < required:
            raise ValueError("block table is shorter than the batched context length.")
        if tuple(item.block_table[:required]) != proof.block_tables[index][:required]:
            raise ValueError(
                "metadata block table does not match the exact batch proof."
            )
        active = item.block_table[:required]
        if any(block < 0 or block >= physical_block_count for block in active):
            raise ValueError("block table contains an out-of-range physical block ID.")
        logical, offset = divmod(item.context_len_before_step, batch.block_size)
        expected_slot = active[logical] * batch.block_size + offset
        if slot != expected_slot:
            raise ValueError("slot does not match block table and pending position.")
        if slot >= physical_block_count * batch.block_size:
            raise ValueError("slot is outside the physical K/V pool.")
