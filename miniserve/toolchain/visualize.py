"""从原始 Phase 0 证据生成环境表和 vector-add 误差 SVG。

可视化永远是原始 JSON/JSONL 的派生物；它帮助人工发现异常，但不能替代 pytest、
二进制检查或 profiler Gate。
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def read_metrics(path: Path) -> list[dict[str, Any]]:
    """逐行读取 metrics JSONL；文件不存在时返回空列表并让 SVG 明确标 BLOCKED。"""

    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def environment_markdown(report: dict[str, Any]) -> str:
    """选择最关键环境字段，生成便于 code review 的 Markdown 表格。"""

    # 使用 .get 保留“不存在”这一事实，报告生成器不会替缺失数据编造默认版本。
    torch_data = report.get("torch", {})
    commands = report.get("commands", {})
    rows = [
        ("Captured UTC", report.get("captured_at_utc")),
        ("Profile", report.get("profile")),
        ("OS", f"{report.get('host', {}).get('system')} {report.get('host', {}).get('release')}"),
        ("glibc", report.get("host", {}).get("glibc")),
        ("Python", report.get("python", {}).get("version")),
        ("PyTorch", torch_data.get("version")),
        ("PyTorch CUDA", torch_data.get("cuda_runtime")),
        ("GPU", torch_data.get("device_name")),
        ("Compute capability", torch_data.get("compute_capability")),
        ("NVCC", commands.get("nvcc", {}).get("output")),
        ("CMake", commands.get("cmake", {}).get("output")),
        ("Ninja", commands.get("ninja", {}).get("output")),
        ("Environment Gate", report.get("environment_gate", {}).get("status")),
    ]
    lines = ["# Phase 0 environment matrix", "", "| Field | Observed |", "|---|---|"]
    for key, value in rows:
        # list/dict 先序列化为 JSON；多行命令输出换成 <br>，避免破坏 Markdown 表格。
        rendered = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else str(value)
        lines.append(f"| {key} | {rendered.replace(chr(10), '<br>')} |")
    lines.append("")
    return "\n".join(lines)


def error_svg(metrics: list[dict[str, Any]]) -> str:
    # Keep dense Phase 0 case labels readable instead of squeezing every run
    # into a fixed 1000 px canvas.  The extra right/bottom margins prevent the
    # rotated final label from being clipped by the SVG viewport.
    width = max(1000, 220 + len(metrics) * 64)
    height = 460
    margin_left, margin_right, margin_top, margin_bottom = 90, 120, 45, 120
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    # y 轴同时覆盖实际误差和允许阈值；即使二者全为 0 也保留最小尺度。
    values = [float(item["max_abs_error"]) for item in metrics]
    thresholds = [float(item["atol"]) for item in metrics]
    y_max = max(values + thresholds + [1e-8]) * 1.15

    def x_at(index: int) -> float:
        # 单点时分母固定为 1，避免除零；多点均匀分布在绘图区。
        return margin_left + (plot_width / max(1, len(metrics) - 1)) * index

    def y_at(value: float) -> float:
        # SVG y 轴向下增长，因此用 1-ratio 将数值 0 放在底部。
        return margin_top + plot_height * (1.0 - min(value / y_max, 1.0))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.2f}" y="25" text-anchor="middle" font-family="sans-serif" font-size="18">Phase 0 vector-add max absolute error</text>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="black"/>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="black"/>',
        f'<text x="20" y="{margin_top + plot_height / 2}" transform="rotate(-90 20 {margin_top + plot_height / 2})" text-anchor="middle" font-family="sans-serif">max_abs_error</text>',
    ]
    if not metrics:
        # 空数据不能画成“零误差”；用醒目的 BLOCKED 文本保留失败透明性。
        parts.append(f'<text x="{width / 2:.2f}" y="230" text-anchor="middle" font-family="sans-serif" font-size="20" fill="#a33">BLOCKED: no real CUDA metrics were produced</text>')
    else:
        for index, item in enumerate(metrics):
            x = x_at(index)
            value_y = y_at(float(item["max_abs_error"]))
            threshold_y = y_at(float(item["atol"]))
            label = html.escape(f"{item['dtype']}/n={item['size']}/s={item['seed']}")
            # 绿色/红色来自测试记录的 passed 字段，不在绘图阶段重新判定正确性。
            color = "#167d32" if item.get("passed") else "#c62828"
            parts.append(f'<circle cx="{x:.2f}" cy="{value_y:.2f}" r="4" fill="{color}"/>')
            parts.append(f'<line x1="{x - 5:.2f}" y1="{threshold_y:.2f}" x2="{x + 5:.2f}" y2="{threshold_y:.2f}" stroke="#ff9800" stroke-width="2"/>')
            parts.append(f'<text x="{x:.2f}" y="{margin_top + plot_height + 18}" transform="rotate(55 {x:.2f} {margin_top + plot_height + 18})" font-family="sans-serif" font-size="9">{label}</text>')
    parts.append('</svg>')
    return "\n".join(parts) + "\n"


def main(argv: list[str] | None = None) -> int:
    """验证输入证据存在，并以 UTF-8 写出两个可审查的派生产物。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    environment_path = args.evidence_dir / "environment.json"
    if not environment_path.is_file():
        parser.error(f"Missing {environment_path}")
    report = json.loads(environment_path.read_text(encoding="utf-8"))
    metrics = read_metrics(args.evidence_dir / "metrics.jsonl")
    (args.evidence_dir / "environment-matrix.md").write_text(environment_markdown(report), encoding="utf-8")
    (args.evidence_dir / "vector-add-errors.svg").write_text(error_svg(metrics), encoding="utf-8")
    print(f"wrote {args.evidence_dir / 'environment-matrix.md'}")
    print(f"wrote {args.evidence_dir / 'vector-add-errors.svg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
