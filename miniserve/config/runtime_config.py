"""模型配置与引擎配置的单一组合入口。"""

from __future__ import annotations

from dataclasses import dataclass

from .engine_config import EngineConfig
from .model_config import ModelConfig


@dataclass(frozen=True)
class RuntimeConfig:
    """启动后传给 EngineCore 的完整、不可变配置。"""

    model: ModelConfig
    engine: EngineConfig

    def validate_phase1(self) -> None:
        self.model.validate()
        self.engine.validate_phase1()
        if self.engine.max_num_batched_tokens > self.model.max_model_len:
            raise ValueError("max_num_batched_tokens exceeds max_model_len.")

    def validate_phase2(self) -> None:
        self.model.validate()
        self.engine.validate_phase2()
        if self.engine.max_num_batched_tokens > self.model.max_model_len:
            raise ValueError("max_num_batched_tokens exceeds max_model_len.")

    def validate_phase3(self) -> None:
        self.model.validate()
        self.engine.validate_phase3()
        if self.engine.max_num_batched_tokens > self.model.max_model_len:
            raise ValueError("max_num_batched_tokens exceeds max_model_len.")

    def validate_phase4(self) -> None:
        self.model.validate()
        self.engine.validate_phase4()
        if self.engine.max_num_batched_tokens > self.model.max_model_len:
            raise ValueError("max_num_batched_tokens exceeds max_model_len.")

    def validate_phase5(self) -> None:
        self.model.validate()
        self.engine.validate_phase5()
        if self.engine.max_num_batched_tokens > self.model.max_model_len:
            raise ValueError("max_num_batched_tokens exceeds max_model_len.")

    def validate_phase6(self) -> None:
        self.model.validate()
        self.engine.validate_phase6()
        if self.engine.max_num_batched_tokens > self.model.max_model_len:
            raise ValueError("max_num_batched_tokens exceeds max_model_len.")

    def validate_phase7(self) -> None:
        self.model.validate()
        self.engine.validate_phase7()
        if self.engine.max_num_batched_tokens > self.model.max_model_len:
            raise ValueError("max_num_batched_tokens exceeds max_model_len.")

    def validate_phase8(self) -> None:
        self.model.validate()
        self.engine.validate_phase8()
        if self.engine.max_num_batched_tokens > self.model.max_model_len:
            raise ValueError("max_num_batched_tokens exceeds max_model_len.")

    def validate_phase9(self) -> None:
        self.model.validate()
        self.engine.validate_phase9()
        if self.engine.max_num_batched_tokens > self.model.max_model_len:
            raise ValueError("max_num_batched_tokens exceeds max_model_len.")

    def validate_phase10(self) -> None:
        self.model.validate()
        self.engine.validate_phase10()
        if self.engine.max_num_batched_tokens > self.model.max_model_len:
            raise ValueError("max_num_batched_tokens exceeds max_model_len.")

    def validate_capabilities(self) -> None:
        """M0 语义能力校验：独立开关组合取代 Phase 强绑定（Phase 入口保留兼容回归）。"""

        self.model.validate()
        from .capabilities import validate_capabilities

        validate_capabilities(self.engine)
        if self.engine.max_num_batched_tokens > self.model.max_model_len:
            raise ValueError("max_num_batched_tokens exceeds max_model_len.")
