"""声明式 SLO 网格与 Goodput 评估（M0 revision 2）。

语义（owner D2 6）：
- 终态互斥分区：success / truncated / failed / timeout / cancelled / never_sent /
  unresolved；``incomparable``（缺 usage）是正交质量属性，不重复计入终态数。
- ``cohort_attainment = good_in_cohort / cohort_size``（零 cohort 为 null）；
  ``conditional_attainment = good / eligible_successes``（零分母为 null）。
  历史字段 ``attainment`` 保留为 conditional 的别名，不再冒充总体达标率。
- 默认 ``issued`` 推断包含截断观测；显式计数必须与唯一请求账本一致，不能双算。
- N<=1 请求 TPOT 为 null，按 TTFT（可选 E2E）单列分组。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .metrics import ServiceObservation


@dataclass(frozen=True)
class SloGrid:
    ttft_ms: tuple[float, ...] = (250.0, 500.0, 1000.0, 2000.0)
    tpot_ms: tuple[float, ...] = (10.0, 25.0, 50.0, 100.0)
    # owner D1.9 预登记：N<=1 分组采用 TTFT 与 E2E<=2000ms。
    e2e_ms: float | None = 2000.0

    def __post_init__(self) -> None:
        for name, values in (("ttft_ms", self.ttft_ms), ("tpot_ms", self.tpot_ms)):
            if not isinstance(values, tuple) or not values:
                raise ValueError(f"SloGrid {name} must be a non-empty tuple.")
            for value in values:
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise TypeError(f"SloGrid {name} entries must be numbers.")
                if not math.isfinite(float(value)) or float(value) <= 0.0:
                    raise ValueError(f"SloGrid {name} entries must be finite and positive.")
        if self.e2e_ms is not None:
            if not isinstance(self.e2e_ms, (int, float)) or isinstance(self.e2e_ms, bool):
                raise TypeError("SloGrid e2e_ms must be a number or None.")
            if not math.isfinite(float(self.e2e_ms)) or float(self.e2e_ms) <= 0.0:
                raise ValueError("SloGrid e2e_ms must be finite and positive.")
        if tuple(sorted(set(self.ttft_ms))) != self.ttft_ms:
            raise ValueError("SloGrid ttft_ms must be sorted and unique.")
        if tuple(sorted(set(self.tpot_ms))) != self.tpot_ms:
            raise ValueError("SloGrid tpot_ms must be sorted and unique.")

    def as_dict(self) -> dict[str, object]:
        return {"ttft_ms": list(self.ttft_ms), "tpot_ms": list(self.tpot_ms), "e2e_ms": self.e2e_ms}

    @classmethod
    def from_strings(
        cls,
        *,
        ttft_csv: str | None = None,
        tpot_csv: str | None = None,
        e2e_ms: float | None = 2000.0,
    ) -> "SloGrid":
        def parse(text: str, name: str) -> tuple[float, ...]:
            entries: list[float] = []
            for token in text.split(","):
                token = token.strip()
                if not token:
                    raise ValueError(f"{name} contains an empty entry.")
                entries.append(float(token))
            return tuple(sorted(set(entries)))

        default = cls()
        ttft = default.ttft_ms if ttft_csv is None else parse(ttft_csv, "ttft_csv")
        tpot = default.tpot_ms if tpot_csv is None else parse(tpot_csv, "tpot_csv")
        return cls(ttft_ms=ttft, tpot_ms=tpot, e2e_ms=e2e_ms)


DEFAULT_SLO_GRID = SloGrid()


@dataclass(frozen=True)
class SloCell:
    ttft_ms: float
    tpot_ms: float
    good: int
    eligible: int
    cohort_size: int
    goodput_per_s: float
    conditional_attainment: float | None
    cohort_attainment: float | None

    @property
    def attainment(self) -> float | None:
        """历史名：条件达标率（成功且可比较样本中的达标比例）。"""

        return self.conditional_attainment

    def as_dict(self) -> dict[str, object]:
        return {
            "ttft_ms": self.ttft_ms,
            "tpot_ms": self.tpot_ms,
            "good": self.good,
            "eligible": self.eligible,
            "cohort_size": self.cohort_size,
            "attainment": self.attainment,
            "conditional_attainment": self.conditional_attainment,
            "cohort_attainment": self.cohort_attainment,
            "goodput_per_s": self.goodput_per_s,
        }


@dataclass(frozen=True)
class ShortOutputCell:
    ttft_ms: float
    good: int
    eligible: int
    cohort_size: int
    goodput_per_s: float
    conditional_attainment: float | None
    cohort_attainment: float | None

    @property
    def attainment(self) -> float | None:
        return self.conditional_attainment

    def as_dict(self) -> dict[str, object]:
        return {
            "ttft_ms": self.ttft_ms,
            "good": self.good,
            "eligible": self.eligible,
            "cohort_size": self.cohort_size,
            "attainment": self.attainment,
            "conditional_attainment": self.conditional_attainment,
            "cohort_attainment": self.cohort_attainment,
            "goodput_per_s": self.goodput_per_s,
        }


@dataclass(frozen=True)
class GoodputReport:
    window_s: float
    issued: int
    completed: int
    failed: int
    incomplete: int
    truncated_requests: int
    unresolved: int
    accounting_closed: bool
    completion_rate: float
    cells: tuple[SloCell, ...]
    short_output_requests: int
    short_output_cells: tuple[ShortOutputCell, ...]
    short_output_e2e_required: bool
    incomparable_requests: int

    def as_dict(self) -> dict[str, object]:
        return {
            "window_s": self.window_s,
            "issued": self.issued,
            "completed": self.completed,
            "failed": self.failed,
            "incomplete": self.incomplete,
            "truncated_requests": self.truncated_requests,
            "unresolved": self.unresolved,
            "accounting_closed": self.accounting_closed,
            "completion_rate": self.completion_rate,
            "short_output_requests": self.short_output_requests,
            "short_output_e2e_required": self.short_output_e2e_required,
            "incomparable_requests": self.incomparable_requests,
            "cells": [cell.as_dict() for cell in self.cells],
            "short_output_cells": [cell.as_dict() for cell in self.short_output_cells],
        }


def evaluate_goodput(
    observations: Iterable[ServiceObservation],
    *,
    window_s: float,
    slo: SloGrid = DEFAULT_SLO_GRID,
    issued: int | None = None,
    completed: int | None = None,
    failed: int = 0,
    incomplete: int = 0,
    truncated: int | None = None,
) -> GoodputReport:
    """对固定窗口内的观测做整张 SLO 网格评估（人工可复核的纯函数）。

    ``issued`` 默认 = 成功 + 截断 + 失败 + 未完成，保证默认路径不吞截断请求。
    显式传入时校验 issued >= 各终态之和；差额记为 unresolved，并使
    ``accounting_closed=false``（不静默消失）。
    """

    if not math.isfinite(window_s) or window_s <= 0.0:
        raise ValueError("window_s must be finite and positive.")
    if failed < 0 or incomplete < 0:
        raise ValueError("failed/incomplete must be non-negative.")
    items = tuple(observations)
    ids = [item.request_id for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate request_id in observation ledger.")
    observed_success = sum(not item.truncated for item in items)
    observed_truncated = sum(item.truncated for item in items)
    if completed is not None and int(completed) != observed_success:
        raise ValueError(
            f"explicit completed={completed} disagrees with {observed_success} unique successful observations"
        )
    if truncated is not None and int(truncated) != observed_truncated:
        raise ValueError(
            f"explicit truncated={truncated} disagrees with {observed_truncated} truncated observations"
        )
    completed_count = observed_success if completed is None else int(completed)
    truncated_count = observed_truncated if truncated is None else int(truncated)
    if completed_count < 0 or truncated_count < 0:
        raise ValueError("completed/truncated must be non-negative.")
    terminal_sum = completed_count + truncated_count + failed + incomplete
    if issued is None:
        issued = terminal_sum
    if issued < terminal_sum:
        raise ValueError(
            "issued cannot be smaller than completed + truncated + failed + incomplete."
        )
    unresolved = issued - terminal_sum

    comparable = [item for item in items if item.comparable]
    incomparable = len(items) - len(comparable)
    long_output = [
        item
        for item in comparable
        if (item.completion_tokens or 0) >= 2 and item.client_tpot_ms is not None
    ]
    short_output = [
        item
        for item in comparable
        if (item.completion_tokens or 0) < 2 or item.client_tpot_ms is None
    ]

    cells: list[SloCell] = []
    for ttft_limit in slo.ttft_ms:
        for tpot_limit in slo.tpot_ms:
            good = 0
            eligible = 0
            for item in long_output:
                if item.ttft_ms is None or item.client_tpot_ms is None:
                    continue
                eligible += 1
                if item.ttft_ms <= ttft_limit and item.client_tpot_ms <= tpot_limit:
                    good += 1
            cells.append(
                SloCell(
                    ttft_ms=float(ttft_limit),
                    tpot_ms=float(tpot_limit),
                    good=good,
                    eligible=eligible,
                    cohort_size=issued,
                    goodput_per_s=good / window_s,
                    conditional_attainment=(good / eligible) if eligible else None,
                    cohort_attainment=(good / issued) if issued else None,
                )
            )

    short_cells: list[ShortOutputCell] = []
    for ttft_limit in slo.ttft_ms:
        eligible = len(short_output)
        good = 0
        for item in short_output:
            if item.ttft_ms is None or item.ttft_ms > ttft_limit:
                continue
            if slo.e2e_ms is not None and item.latency_ms > slo.e2e_ms:
                continue
            good += 1
        short_cells.append(
            ShortOutputCell(
                ttft_ms=float(ttft_limit),
                good=good,
                eligible=eligible,
                cohort_size=issued,
                goodput_per_s=good / window_s,
                conditional_attainment=(good / eligible) if eligible else None,
                cohort_attainment=(good / issued) if issued else None,
            )
        )

    return GoodputReport(
        window_s=window_s,
        issued=issued,
        completed=completed_count,
        failed=failed,
        incomplete=incomplete,
        truncated_requests=truncated_count,
        unresolved=unresolved,
        accounting_closed=unresolved == 0,
        completion_rate=(completed_count / issued) if issued else 0.0,
        cells=tuple(cells),
        short_output_requests=len(short_output),
        short_output_cells=tuple(short_cells),
        short_output_e2e_required=slo.e2e_ms is not None,
        incomparable_requests=incomparable,
    )


def format_goodput_table(report: GoodputReport) -> str:
    def cell(value: float | None, digits: int) -> str:
        return "-" if value is None else f"{value:.{digits}f}"

    lines = [
        f"window_s={report.window_s:.3f} completed={report.completed}/{report.issued} "
        f"truncated={report.truncated_requests} failed={report.failed} "
        f"incomplete={report.incomplete} unresolved={report.unresolved} "
        f"completion_rate={report.completion_rate:.4f} closed={report.accounting_closed}",
        f"short_output_requests={report.short_output_requests} "
        f"e2e_required={report.short_output_e2e_required} "
        f"incomparable={report.incomparable_requests}",
        "  ".join(
            f"{name:>12}"
            for name in ("ttft_ms", "tpot_ms", "good", "eligible", "cohort", "cond", "cohort%", "goodput/s")
        ),
    ]
    for entry in report.cells:
        lines.append(
            "  ".join(
                (
                    f"{entry.ttft_ms:>12.1f}",
                    f"{entry.tpot_ms:>12.1f}",
                    f"{entry.good:>12}",
                    f"{entry.eligible:>12}",
                    f"{entry.cohort_size:>12}",
                    f"{cell(entry.conditional_attainment, 4):>12}",
                    f"{cell(entry.cohort_attainment, 4):>12}",
                    f"{entry.goodput_per_s:>12.4f}",
                )
            )
        )
    if report.short_output_requests:
        lines.append("short-output group (N<=1): ttft-only attainment")
        for entry in report.short_output_cells:
            lines.append(
                "  ".join(
                    (
                        f"{entry.ttft_ms:>12.1f}",
                        f"{'n/a':>12}",
                        f"{entry.good:>12}",
                        f"{entry.eligible:>12}",
                        f"{entry.cohort_size:>12}",
                        f"{cell(entry.conditional_attainment, 4):>12}",
                        f"{cell(entry.cohort_attainment, 4):>12}",
                        f"{entry.goodput_per_s:>12.4f}",
                    )
                )
            )
    return "\n".join(lines)