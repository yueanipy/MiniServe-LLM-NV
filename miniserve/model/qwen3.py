"""共享 Qwen3/SmolLM2-compatible dense decoder 前向；保留旧模块名兼容阶段代码。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

from miniserve.config.model_config import ModelConfig

from .layers import Qwen3MLP, RMSNorm
from .rope import apply_rotary_pos_emb, build_rope_cache

if TYPE_CHECKING:
    from miniserve.kv.base import KVCacheBackend


def repeat_kv(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
    """把每个 KV head 映射给固定数量的 query heads（GQA）。"""

    batch, kv_heads, sequence, head_dim = hidden_states.shape
    if repeats == 1:
        return hidden_states
    expanded = hidden_states[:, :, None, :, :].expand(batch, kv_heads, repeats, sequence, head_dim)
    return expanded.reshape(batch, kv_heads * repeats, sequence, head_dim)


def causal_attention_mask(sequence: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """返回加到 attention scores 上的上三角 ``-inf`` mask。"""

    mask = torch.full((sequence, sequence), torch.finfo(dtype).min, device=device, dtype=dtype)
    return torch.triu(mask, diagonal=1)[None, None, :, :]


def absolute_causal_attention_mask(
    query_positions: torch.Tensor,
    key_length: int,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """按绝对位置构造允许 ``key_position <= query_position`` 的 mask。"""

    if query_positions.ndim != 2 or key_length <= 0:
        raise ValueError("Invalid query positions or key length.")
    key_positions = torch.arange(key_length, device=query_positions.device, dtype=torch.long)
    allowed = key_positions.view(1, 1, key_length) <= query_positions.unsqueeze(-1)
    zeros = torch.zeros((), device=query_positions.device, dtype=dtype)
    blocked = torch.full(
        (), torch.finfo(dtype).min, device=query_positions.device, dtype=dtype
    )
    return torch.where(allowed, zeros, blocked).unsqueeze(1)


class Qwen3Attention(nn.Module):
    """RoPE + GQA attention；Qwen3 启用 Q/K norm，Llama 分支使用 identity。"""

    def __init__(self, config: ModelConfig, layer_idx: int, *, device=None, dtype=None) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.rope_theta = config.rope_theta

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.q_norm = (
            RMSNorm(config.head_dim, config.rms_norm_eps, device=device, dtype=dtype)
            if config.qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            RMSNorm(config.head_dim, config.rms_norm_eps, device=device, dtype=dtype)
            if config.qk_norm
            else nn.Identity()
        )

    def forward_recompute(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        rope_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        batch, sequence, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(batch, sequence, self.num_attention_heads, self.head_dim)
        key = self.k_proj(hidden_states).view(batch, sequence, self.num_key_value_heads, self.head_dim)
        value = self.v_proj(hidden_states).view(batch, sequence, self.num_key_value_heads, self.head_dim)

        # Qwen3 在最后一个 head_dim 做 Q/K norm；Llama-family 的 Identity 保持相同数据流。
        query = self.q_norm(query).transpose(1, 2)
        key = self.k_norm(key).transpose(1, 2)
        value = value.transpose(1, 2)
        if rope_cache is None:
            rope_cache = build_rope_cache(
                positions,
                head_dim=self.head_dim,
                rope_theta=self.rope_theta,
                output_dtype=query.dtype,
            )
        query, key = apply_rotary_pos_emb(query, key, *rope_cache)

        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)
        scores = torch.matmul(query, key.transpose(2, 3)) * self.scaling
        scores = scores + causal_attention_mask(sequence, device=scores.device, dtype=scores.dtype)
        probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        attended = torch.matmul(probabilities, value)
        attended = attended.transpose(1, 2).contiguous().reshape(batch, sequence, -1)
        return self.o_proj(attended)

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.forward_recompute(hidden_states, positions)

    def forward_with_kv(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        *,
        kv_backend: "KVCacheBackend",
        request_id: str,
        context_len_before: int,
        rope_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        decode_attention=None,
        block_tables: torch.Tensor | None = None,
        sequence_lengths: torch.Tensor | None = None,
        graph_slot_mapping: torch.Tensor | None = None,
        graph_capture: bool = False,
    ) -> torch.Tensor:
        """投影本轮新 token，写入连续 KV，再读取已提交历史+本轮数据。"""

        batch, query_length, _ = hidden_states.shape
        if batch != 1:
            raise RuntimeError("Phase 2 contiguous KV supports batch size 1.")
        if not graph_capture:
            expected_positions = torch.arange(
                context_len_before,
                context_len_before + query_length,
                device=positions.device,
                dtype=torch.long,
            ).unsqueeze(0)
            if not torch.equal(positions, expected_positions):
                raise RuntimeError("KV positions must be contiguous absolute positions.")

        query = self.q_proj(hidden_states).view(batch, query_length, self.num_attention_heads, self.head_dim)
        key = self.k_proj(hidden_states).view(batch, query_length, self.num_key_value_heads, self.head_dim)
        value = self.v_proj(hidden_states).view(batch, query_length, self.num_key_value_heads, self.head_dim)
        query = self.q_norm(query).transpose(1, 2)
        key = self.k_norm(key).transpose(1, 2)
        value = value.transpose(1, 2)
        if rope_cache is None:
            rope_cache = build_rope_cache(
                positions,
                head_dim=self.head_dim,
                rope_theta=self.rope_theta,
                output_dtype=query.dtype,
            )
        query, key = apply_rotary_pos_emb(query, key, *rope_cache)

        # backend canonical append 形状是 [new_tokens, kv_heads, head_dim]。
        append_key = key[0].transpose(0, 1).contiguous()
        append_value = value[0].transpose(0, 1).contiguous()
        if graph_capture:
            if graph_slot_mapping is None or not hasattr(kv_backend, "key_pool"):
                raise RuntimeError("Full decode Graph requires paged pools and slot mapping.")
            from miniserve._native import paged_kv_append_graph

            paged_kv_append_graph(
                kv_backend.key_pool,
                kv_backend.value_pool,
                self.layer_idx,
                append_key,
                append_value,
                graph_slot_mapping,
            )
        else:
            kv_backend.append(
                request_id,
                self.layer_idx,
                context_len_before,
                append_key,
                append_value,
            )
        context_len_after = context_len_before + query_length
        if decode_attention is not None:
            if query_length != 1 or block_tables is None or sequence_lengths is None:
                raise RuntimeError("Custom paged decode requires one query token and metadata.")
            if graph_capture:
                attended_heads = decode_attention.decode_graph_safe(
                    query[:, :, 0, :].contiguous(),
                    layer_index=self.layer_idx,
                    block_tables=block_tables,
                    sequence_lengths=sequence_lengths,
                )
            else:
                attended_heads = decode_attention.decode(
                    query[:, :, 0, :].contiguous(),
                    layer_index=self.layer_idx,
                    block_tables=block_tables,
                    sequence_lengths=sequence_lengths,
                )
            if attended_heads.shape != (batch, self.num_attention_heads, self.head_dim):
                raise RuntimeError("Paged decode backend returned an invalid output shape.")
            attended = attended_heads.unsqueeze(2).transpose(1, 2).contiguous().reshape(
                batch, 1, -1
            )
            return self.o_proj(attended)
        cached_key, cached_value = kv_backend.view(request_id, self.layer_idx, context_len_after)
        key_states = cached_key.transpose(0, 1).unsqueeze(0)
        value_states = cached_value.transpose(0, 1).unsqueeze(0)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        scores = torch.matmul(query, key_states.transpose(2, 3)) * self.scaling
        scores = scores + absolute_causal_attention_mask(
            positions, context_len_after, dtype=scores.dtype
        )
        probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        attended = torch.matmul(probabilities, value_states)
        attended = attended.transpose(1, 2).contiguous().reshape(batch, query_length, -1)
        return self.o_proj(attended)

    def forward_with_kv_batch(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        *,
        kv_backend,
        decode_attention,
        block_tables: torch.Tensor,
        sequence_lengths: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> torch.Tensor:
        """B 行、每行一个 token 的 paged decode；只在可信 host metadata 校验后调用。"""

        batch, query_length, _ = hidden_states.shape
        if batch <= 0 or query_length != 1:
            raise RuntimeError("Batched decode requires [B,1,hidden] with B>0.")
        if positions.shape != (batch, 1) or positions.dtype != torch.long:
            raise ValueError("Batched decode positions must be int64 [B,1].")
        if (
            slot_mapping.dtype != torch.long
            or tuple(slot_mapping.shape) != (batch,)
            or block_tables.dtype != torch.int32
            or block_tables.ndim != 2
            or block_tables.shape[0] != batch
            or sequence_lengths.dtype != torch.int32
            or tuple(sequence_lengths.shape) != (batch,)
        ):
            raise ValueError("Batched decode paged metadata has an invalid shape/dtype.")

        query = self.q_proj(hidden_states).view(
            batch, query_length, self.num_attention_heads, self.head_dim
        )
        key = self.k_proj(hidden_states).view(
            batch, query_length, self.num_key_value_heads, self.head_dim
        )
        value = self.v_proj(hidden_states).view(
            batch, query_length, self.num_key_value_heads, self.head_dim
        )
        query = self.q_norm(query).transpose(1, 2)
        key = self.k_norm(key).transpose(1, 2)
        value = value.transpose(1, 2)
        rope_cache = build_rope_cache(
            positions,
            head_dim=self.head_dim,
            rope_theta=self.rope_theta,
            output_dtype=query.dtype,
        )
        query, key = apply_rotary_pos_emb(query, key, *rope_cache)
        append_key = key[:, :, 0, :].contiguous()
        append_value = value[:, :, 0, :].contiguous()
        from miniserve._native import paged_kv_append_graph

        paged_kv_append_graph(
            kv_backend.key_pool,
            kv_backend.value_pool,
            self.layer_idx,
            append_key,
            append_value,
            slot_mapping,
        )
        attended_heads = decode_attention.decode_graph_safe(
            query[:, :, 0, :].contiguous(),
            layer_index=self.layer_idx,
            block_tables=block_tables,
            sequence_lengths=sequence_lengths,
        )
        if attended_heads.shape != (batch, self.num_attention_heads, self.head_dim):
            raise RuntimeError("Batched paged decode returned an invalid output shape.")
        attended = attended_heads.unsqueeze(2).transpose(1, 2).contiguous().reshape(
            batch, query_length, -1
        )
        return self.o_proj(attended)


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int, *, device=None, dtype=None) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(config, layer_idx, device=device, dtype=dtype)
        self.mlp = Qwen3MLP(config, device=device, dtype=dtype)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, device=device, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps, device=device, dtype=dtype
        )

    def forward_recompute(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        rope_cache: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = residual + self.self_attn.forward_recompute(
            self.input_layernorm(hidden_states), positions, rope_cache
        )
        residual = hidden_states
        return residual + self.mlp(self.post_attention_layernorm(hidden_states))

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor, rope_cache) -> torch.Tensor:
        return self.forward_recompute(hidden_states, positions, rope_cache)

    def forward_with_kv(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        *,
        kv_backend: "KVCacheBackend",
        request_id: str,
        context_len_before: int,
        rope_cache: tuple[torch.Tensor, torch.Tensor],
        decode_attention=None,
        block_tables: torch.Tensor | None = None,
        sequence_lengths: torch.Tensor | None = None,
        graph_slot_mapping: torch.Tensor | None = None,
        graph_capture: bool = False,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = residual + self.self_attn.forward_with_kv(
            self.input_layernorm(hidden_states),
            positions,
            kv_backend=kv_backend,
            request_id=request_id,
            context_len_before=context_len_before,
            rope_cache=rope_cache,
            decode_attention=decode_attention,
            block_tables=block_tables,
            sequence_lengths=sequence_lengths,
            graph_slot_mapping=graph_slot_mapping,
            graph_capture=graph_capture,
        )
        residual = hidden_states
        return residual + self.mlp(self.post_attention_layernorm(hidden_states))

    def forward_with_kv_batch(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        *,
        kv_backend,
        decode_attention,
        block_tables: torch.Tensor,
        sequence_lengths: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = residual + self.self_attn.forward_with_kv_batch(
            self.input_layernorm(hidden_states),
            positions,
            kv_backend=kv_backend,
            decode_attention=decode_attention,
            block_tables=block_tables,
            sequence_lengths=sequence_lengths,
            slot_mapping=slot_mapping,
        )
        residual = hidden_states
        return residual + self.mlp(self.post_attention_layernorm(hidden_states))


class Qwen3Model(nn.Module):
    def __init__(self, config: ModelConfig, *, device=None, dtype=None) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, index, device=device, dtype=dtype) for index in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, device=device, dtype=dtype)

    def forward_recompute(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
        *,
        output_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]:
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be int64 [batch, sequence].")
        batch, sequence = input_ids.shape
        if sequence == 0 or sequence > self.config.max_model_len:
            raise ValueError("Sequence length is empty or exceeds max_model_len.")
        if positions is None:
            positions = torch.arange(sequence, device=input_ids.device, dtype=torch.long).expand(batch, -1)
        if positions.shape != input_ids.shape or positions.dtype != torch.long:
            raise ValueError("positions must be int64 and match input_ids.")

        hidden_states = self.embed_tokens(input_ids)
        collected: list[torch.Tensor] | None = [hidden_states] if output_hidden_states else None
        rope_cache = build_rope_cache(
            positions,
            head_dim=self.config.head_dim,
            rope_theta=self.config.rope_theta,
            output_dtype=hidden_states.dtype,
        )
        for layer in self.layers:
            hidden_states = layer.forward_recompute(hidden_states, positions, rope_cache)
            if collected is not None:
                collected.append(hidden_states)
        hidden_states = self.norm(hidden_states)
        if collected is not None:
            # 与 HF output_hidden_states 对齐：最后一个元素是 final norm 后结果。
            collected[-1] = hidden_states
        return hidden_states, None if collected is None else tuple(collected)

    def forward_with_kv(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        *,
        kv_backend: "KVCacheBackend",
        request_id: str,
        context_len_before: int,
        decode_attention=None,
        block_tables: torch.Tensor | None = None,
        sequence_lengths: torch.Tensor | None = None,
        graph_slot_mapping: torch.Tensor | None = None,
        graph_capture: bool = False,
    ) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.dtype != torch.long or input_ids.shape[0] != 1:
            raise ValueError("Phase 2 input_ids must be int64 [1, new_tokens].")
        if positions.shape != input_ids.shape or positions.dtype != torch.long:
            raise ValueError("positions must be int64 and match input_ids.")
        query_length = input_ids.shape[1]
        if query_length == 0 or context_len_before + query_length > self.config.max_model_len:
            raise ValueError("KV forward exceeds max_model_len.")
        hidden_states = self.embed_tokens(input_ids)
        rope_cache = build_rope_cache(
            positions,
            head_dim=self.config.head_dim,
            rope_theta=self.config.rope_theta,
            output_dtype=hidden_states.dtype,
        )
        for layer in self.layers:
            hidden_states = layer.forward_with_kv(
                hidden_states,
                positions,
                kv_backend=kv_backend,
                request_id=request_id,
                context_len_before=context_len_before,
                rope_cache=rope_cache,
                decode_attention=decode_attention,
                block_tables=block_tables,
                sequence_lengths=sequence_lengths,
                graph_slot_mapping=graph_slot_mapping,
                graph_capture=graph_capture,
            )
        return self.norm(hidden_states)

    def forward_with_kv_batch(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        *,
        kv_backend,
        decode_attention,
        block_tables: torch.Tensor,
        sequence_lengths: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.dtype != torch.long or input_ids.shape[1] != 1:
            raise ValueError("Batched decode input_ids must be int64 [B,1].")
        if input_ids.shape[0] <= 0 or positions.shape != input_ids.shape:
            raise ValueError("Batched decode requires positive B and matching positions.")
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer.forward_with_kv_batch(
                hidden_states,
                positions,
                kv_backend=kv_backend,
                decode_attention=decode_attention,
                block_tables=block_tables,
                sequence_lengths=sequence_lengths,
                slot_mapping=slot_mapping,
            )
        return self.norm(hidden_states)


class Qwen3ForCausalLM(nn.Module):
    """共享 dense causal LM；sampling 位于独立模块。"""

    def __init__(self, config: ModelConfig, *, device=None, dtype=None) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config, device=device, dtype=dtype)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, device=device, dtype=dtype)
        self.tie_weights()

    def tie_weights(self) -> None:
        if not self.config.tie_word_embeddings:
            raise RuntimeError("This MiniServe dense model requires tied embeddings.")
        self.lm_head.weight = self.model.embed_tokens.weight

    def set_rmsnorm_backend(self, backend: str) -> None:
        for module in self.modules():
            if isinstance(module, RMSNorm):
                module.set_backend(backend)

    def forward_recompute(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor | None = None,
        *,
        output_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]:
        hidden_states, layers = self.model.forward_recompute(
            input_ids, positions, output_hidden_states=output_hidden_states
        )
        # HF 4.57.1 的 lm_head 直接沿用模型 dtype；测试 oracle 也固定相同版本和 eager backend。
        logits = self.lm_head(hidden_states)
        return logits, layers

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        logits, _ = self.forward_recompute(input_ids, positions)
        return logits

    def forward_with_kv(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        *,
        kv_backend: "KVCacheBackend",
        request_id: str,
        context_len_before: int,
        decode_attention=None,
        block_tables: torch.Tensor | None = None,
        sequence_lengths: torch.Tensor | None = None,
        graph_slot_mapping: torch.Tensor | None = None,
        graph_capture: bool = False,
    ) -> torch.Tensor:
        hidden_states = self.model.forward_with_kv(
            input_ids,
            positions,
            kv_backend=kv_backend,
            request_id=request_id,
            context_len_before=context_len_before,
            decode_attention=decode_attention,
            block_tables=block_tables,
            sequence_lengths=sequence_lengths,
            graph_slot_mapping=graph_slot_mapping,
            graph_capture=graph_capture,
        )
        return self.lm_head(hidden_states)

    def forward_with_kv_batch(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        *,
        kv_backend,
        decode_attention,
        block_tables: torch.Tensor,
        sequence_lengths: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.model.forward_with_kv_batch(
            input_ids,
            positions,
            kv_backend=kv_backend,
            decode_attention=decode_attention,
            block_tables=block_tables,
            sequence_lengths=sequence_lengths,
            slot_mapping=slot_mapping,
        )
        return self.lm_head(hidden_states)
