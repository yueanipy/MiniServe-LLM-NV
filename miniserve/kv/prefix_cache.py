"""Exact full-block prefix cache metadata；storage/refcount 仍由 KV manager 所有。"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class PrefixNamespace:
    model_revision: str
    tokenizer_revision: str
    dtype: str
    rope_fingerprint: str
    chat_template_revision: str = "none"
    thinking_mode: bool = False

    def canonical_bytes(self) -> bytes:
        return json.dumps(self.__dict__, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True)
class PrefixMatch:
    matched_tokens: int
    physical_block_ids: tuple[int, ...]


@dataclass(frozen=True)
class PrefixEntry:
    digest: str
    previous_digest: str
    token_block: tuple[int, ...]
    physical_block_id: int


def stable_block_hash(
    previous_digest: str,
    token_block: tuple[int, ...],
    namespace: PrefixNamespace,
) -> str:
    payload = {
        "namespace": json.loads(namespace.canonical_bytes()),
        "previous_digest": previous_digest,
        "token_block": list(token_block),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class PrefixCache:
    """LRU metadata cache；每个 chained entry 对应一个完整 physical block。"""

    def __init__(
        self,
        *,
        namespace: PrefixNamespace,
        block_size: int,
        max_entries: int,
        hash_fn: Callable[[str, tuple[int, ...], PrefixNamespace], str] = stable_block_hash,
    ) -> None:
        if block_size <= 0 or max_entries <= 0:
            raise ValueError("block_size and max_entries must be positive.")
        self.namespace = namespace
        self.block_size = block_size
        self.max_entries = max_entries
        self._hash_fn = hash_fn
        self._entries: OrderedDict[str, PrefixEntry] = OrderedDict()

    def lookup(self, token_ids: tuple[int, ...]) -> PrefixMatch:
        previous = ""
        blocks: list[int] = []
        full_blocks = len(token_ids) // self.block_size
        for logical in range(full_blocks):
            start = logical * self.block_size
            token_block = token_ids[start : start + self.block_size]
            digest = self._hash_fn(previous, token_block, self.namespace)
            entry = self._entries.get(digest)
            # 哈希命中仍校验完整 payload，防止碰撞导致错误 KV 复用。
            if (
                entry is None
                or entry.previous_digest != previous
                or entry.token_block != token_block
            ):
                break
            self._entries.move_to_end(digest)
            blocks.append(entry.physical_block_id)
            previous = digest

        # 不缓存 logits：完整 prompt 命中时退回最后一个 block 重新 prefill。
        if blocks and len(blocks) * self.block_size == len(token_ids):
            blocks.pop()
        return PrefixMatch(len(blocks) * self.block_size, tuple(blocks))

    def insert(
        self,
        token_ids: tuple[int, ...],
        physical_block_ids: tuple[int, ...],
    ) -> tuple[tuple[PrefixEntry, ...], tuple[PrefixEntry, ...]]:
        full_blocks = len(token_ids) // self.block_size
        if len(physical_block_ids) < full_blocks:
            raise ValueError("Not enough physical blocks for complete token blocks.")
        previous = ""
        added: list[PrefixEntry] = []
        for logical in range(full_blocks):
            start = logical * self.block_size
            token_block = token_ids[start : start + self.block_size]
            digest = self._hash_fn(previous, token_block, self.namespace)
            existing = self._entries.get(digest)
            if existing is not None:
                if existing.previous_digest != previous or existing.token_block != token_block:
                    raise RuntimeError("PREFIX_HASH_COLLISION")
                self._entries.move_to_end(digest)
            else:
                entry = PrefixEntry(digest, previous, token_block, physical_block_ids[logical])
                self._entries[digest] = entry
                added.append(entry)
            previous = digest
        evicted: list[PrefixEntry] = []
        while len(self._entries) > self.max_entries:
            _, entry = self._entries.popitem(last=False)
            evicted.append(entry)
        return tuple(added), tuple(evicted)

    def evict_one(self, predicate: Callable[[PrefixEntry], bool]) -> PrefixEntry | None:
        for digest, entry in tuple(self._entries.items()):
            if predicate(entry):
                del self._entries[digest]
                return entry
        return None

    def entries(self) -> tuple[PrefixEntry, ...]:
        return tuple(self._entries.values())

    def physical_block_ids(self) -> tuple[int, ...]:
        """当前被 cache 元数据引用（并 retain）的 physical block 集合。"""

        return tuple(sorted({entry.physical_block_id for entry in self._entries.values()}))
