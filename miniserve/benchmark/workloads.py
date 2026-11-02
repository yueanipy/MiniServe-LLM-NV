"""固定、可序列化的服务 workload 合同。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from miniserve.engine.request import SamplingParams


@dataclass(frozen=True)
class ServiceWorkload:
    name: str
    prompt: str
    max_tokens: int
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    seed: int | None = None
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name.strip()
            or not isinstance(self.prompt, str)
            or not self.prompt
            or not isinstance(self.max_tokens, int)
            or isinstance(self.max_tokens, bool)
            or self.max_tokens <= 0
        ):
            raise ValueError("Workload name, prompt and max_tokens must be valid.")
        if not isinstance(self.ignore_eos, bool):
            raise ValueError("ignore_eos must be a boolean.")
        SamplingParams(
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            max_new_tokens=self.max_tokens,
            seed=self.seed,
        )

    def request_body(self, model: str) -> dict[str, object]:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Benchmark model must not be empty.")
        return {
            "model": model,
            "messages": [{"role": "user", "content": self.prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "seed": self.seed,
            "ignore_eos": self.ignore_eos,
            "stream": True,
        }

    def metadata(self) -> dict[str, object]:
        return asdict(self)
