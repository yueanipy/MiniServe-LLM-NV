"""Greedy 与带每请求 RNG 的温度/top-k/top-p sampling。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from miniserve.engine.request import SamplingParams


def greedy_sample(
    logits: torch.Tensor, *, validate_finite: bool = True
) -> torch.Tensor:
    """对最后一维做 argmax；不偷偷应用 temperature/top-k/top-p。"""

    _validate_logits(logits, validate_finite=validate_finite)
    return torch.argmax(logits, dim=-1)


def _validate_logits(logits: torch.Tensor, *, validate_finite: bool = True) -> None:
    if not isinstance(validate_finite, bool):
        raise TypeError("validate_finite must be a bool.")
    if logits.ndim < 1 or logits.shape[-1] == 0:
        raise ValueError("logits must have a non-empty vocabulary dimension.")
    if not logits.is_floating_point():
        raise TypeError("logits must use a floating-point dtype.")
    # CUDA 上的 bool(tensor) 会同步设备。correctness/debug 默认保留完整检查；
    # 已通过权重与数值 Gate 的 serving 路径可显式关闭，避免每 token 多一次 D2H。
    if validate_finite and not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must contain only finite values.")


def _filter_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """保留每行最大的 k 个 logit；shape 不变，其他位置设为负无穷。"""

    if top_k == 0:
        return logits
    if top_k > logits.shape[-1]:
        raise ValueError("top_k must not exceed the vocabulary size.")
    values, indices = torch.topk(logits, top_k, dim=-1)
    # 按 indices scatter，确保并列的第 k 个值不会把额外候选一起保留。
    return torch.full_like(logits, float("-inf")).scatter(-1, indices, values)


def _filter_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """在 top-k 之后执行 nucleus filter，并保证每行至少保留一个 token。"""

    if top_p == 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1, dtype=torch.float32)
    remove = torch.cumsum(sorted_probs, dim=-1) > top_p
    # 让第一个越过阈值的 token 仍保留，避免概率质量不足或整行被过滤。
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    return torch.empty_like(sorted_logits).scatter(-1, sorted_indices, sorted_logits)


def sample_logits(
    logits: torch.Tensor,
    params: SamplingParams,
    *,
    generator: torch.Generator | None = None,
    validate_finite: bool = True,
) -> torch.Tensor:
    """按冻结顺序采样最后一维；greedy 路径不会创建或消耗 RNG。"""

    _validate_logits(logits, validate_finite=validate_finite)
    if params.temperature == 0:
        return torch.argmax(logits, dim=-1)
    if generator is None:
        raise ValueError("Stochastic sampling requires a per-request generator.")
    filtered = logits.to(dtype=torch.float32) / params.temperature
    filtered = _filter_top_k(filtered, params.top_k)
    filtered = _filter_top_p(filtered, params.top_p)
    probabilities = torch.softmax(filtered, dim=-1, dtype=torch.float32)
    return torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)


@dataclass(frozen=True)
class StagedSample:
    """采样结果的暂存快照；RNG 尚未推进到永久注册表。"""

    request_id: str
    token_id: int
    generator_state: torch.Tensor | None
    generator_device: torch.device

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty.")
        if not isinstance(self.token_id, int) or isinstance(self.token_id, bool):
            raise ValueError("token_id must be an integer.")
        if self.generator_state is not None and not isinstance(
            self.generator_state, torch.Tensor
        ):
            raise TypeError("generator_state must be a tensor when present.")
        if not isinstance(self.generator_device, torch.device):
            raise TypeError("generator_device must be a torch.device.")


class PerRequestSampler:
    """ModelExecutor 拥有的 RNG 注册表；batch 到达顺序不会串扰其他请求。"""

    def __init__(self, *, validate_finite: bool = True) -> None:
        if not isinstance(validate_finite, bool):
            raise TypeError("validate_finite must be a bool.")
        self.validate_finite = validate_finite
        self._generators: dict[str, torch.Generator] = {}

    def sample(
        self,
        request_id: str,
        logits: torch.Tensor,
        params: SamplingParams,
    ) -> torch.Tensor:
        if params.temperature == 0:
            return sample_logits(
                logits, params, validate_finite=self.validate_finite
            )
        generator = self._generators.get(request_id)
        if generator is None:
            generator = torch.Generator(device=logits.device)
            # seed=None 表示本次请求随机；仍然只初始化自己的 RNG 一次。
            seed = params.seed if params.seed is not None else generator.seed()
            generator.manual_seed(seed)
            self._generators[request_id] = generator
        return sample_logits(
            logits,
            params,
            generator=generator,
            validate_finite=self.validate_finite,
        )

    def stage(
        self,
        request_id: str,
        logits: torch.Tensor,
        params: SamplingParams,
    ) -> StagedSample:
        """采样但不改变永久 RNG；调用方须在事务 commit 后调用 commit。"""

        if params.temperature == 0:
            # Route through the public method so existing diagnostics/monkeypatches
            # observe the exact sampling boundary; greedy never creates RNG state.
            token = self.sample(request_id, logits, params)
            return StagedSample(
                request_id=request_id,
                token_id=int(token.reshape(-1)[0].item()),
                generator_state=None,
                generator_device=logits.device,
            )

        existing = self._generators.get(request_id)
        before_state = existing.get_state() if existing is not None else None
        try:
            token = self.sample(request_id, logits, params)
            generator = self._generators[request_id]
            after_state = generator.get_state()
        except Exception:
            if existing is None:
                self._generators.pop(request_id, None)
            else:
                existing.set_state(before_state)
            raise
        if existing is None:
            del self._generators[request_id]
        else:
            existing.set_state(before_state)
        return StagedSample(
            request_id=request_id,
            token_id=int(token.reshape(-1)[0].item()),
            generator_state=after_state,
            generator_device=generator.device,
        )

    def stage_greedy_batch(
        self,
        request_ids: tuple[str, ...],
        logits: torch.Tensor,
        params: tuple[SamplingParams, ...],
    ) -> tuple[StagedSample, ...]:
        """Materialize a full greedy batch with one bounded D2H transfer.

        This is an explicit experiment path. Only a homogeneous greedy batch is
        eligible; stochastic, mixed, malformed, and non-CUDA callers stay on the
        reference per-row path.
        """

        if not isinstance(request_ids, tuple) or not request_ids:
            raise ValueError("request_ids must be a non-empty tuple.")
        if (
            not isinstance(params, tuple)
            or len(params) != len(request_ids)
            or any(not isinstance(item, SamplingParams) for item in params)
        ):
            raise ValueError("sampling params must be a matching tuple.")
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("request_ids must be unique.")
        if logits.ndim != 2 or logits.shape[0] != len(request_ids) or logits.shape[1] == 0:
            raise ValueError("logits must have shape [batch, vocab] matching request_ids.")
        if any(item.temperature != 0.0 for item in params):
            raise ValueError("batched greedy sampling requires temperature=0 for every row.")
        _validate_logits(logits, validate_finite=self.validate_finite)
        token_ids = torch.argmax(logits, dim=-1)
        cpu_tokens = token_ids.to(device="cpu", dtype=torch.long).tolist()
        if len(cpu_tokens) != len(request_ids):
            raise RuntimeError("batched greedy returned the wrong token count.")
        return tuple(
            StagedSample(
                request_id=request_id,
                token_id=int(token_id),
                generator_state=None,
                generator_device=logits.device,
            )
            for request_id, token_id in zip(request_ids, cpu_tokens)
        )

    def commit(self, samples: tuple[StagedSample, ...]) -> None:
        """提交已经采样成功的 RNG 状态；只调用一次且不重采样。"""

        if not isinstance(samples, tuple) or any(
            not isinstance(sample, StagedSample) for sample in samples
        ):
            raise TypeError("samples must be a tuple of StagedSample objects.")
        for sample in samples:
            if sample.generator_state is None:
                continue
            generator = self._generators.get(sample.request_id)
            if generator is None:
                generator = torch.Generator(device=sample.generator_device)
                self._generators[sample.request_id] = generator
            generator.set_state(sample.generator_state)

    def release(self, request_id: str) -> None:
        self._generators.pop(request_id, None)

    @property
    def active_request_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._generators))
