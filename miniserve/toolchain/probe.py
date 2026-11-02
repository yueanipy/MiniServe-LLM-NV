"""Collect a machine-readable Phase 0 environment report.

The module imports PyTorch lazily so that its parsing and Gate logic can be
unit-tested on a machine where PyTorch has not been installed yet.

探测与判定刻意分层：``collect_report`` 只记录事实，``evaluate_environment``
再依据 Profile 合同判定 PASS/BLOCKED。这样测试可以构造报告覆盖失败分支，而不必
伪造本机硬件。
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# Profile 是硬件验收合同，不是“尽量兼容”：学习机必须看到 sm_75，正式机预计
# sm_120，任何不同 capability 都应阻断并重新确认机器。
PROFILE_CAPABILITIES = {"L": [7, 5], "D": [12, 0]}

# REQUIRED_COMMANDS 进入强制 Gate；PACKAGE_NAMES 还包含后续 Phase 的观察项，未安装
# 时记录 null，但不会在 Phase 0 自动变成失败。
REQUIRED_COMMANDS = (
    "nvidia-smi",
    "nvcc",
    "gcc",
    "g++",
    "cmake",
    "ninja",
    "cuobjdump",
    "nsys",
)
PACKAGE_NAMES = (
    "torch",
    "scikit-build-core",
    "cmake",
    "ninja",
    "pytest",
    "transformers",
    "tokenizers",
    "safetensors",
    "fastapi",
    "pydantic",
    "uvicorn",
)


def read_os_release(path: Path = Path("/etc/os-release")) -> dict[str, str]:
    """把 /etc/os-release 当纯文本解析，绝不执行其中可能出现的 shell 语法。"""

    if not path.is_file():
        return {}
    result: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        # 只在第一个等号处分割，保留 value 内部合法的等号。
        key, value = line.split("=", 1)
        result[key] = value.strip().strip('"')
    return result


def command_report(command: str, args: Iterable[str] = ()) -> dict[str, Any]:
    """运行有时间上限的版本/探测命令，并完整保留可执行路径与退出状态。"""

    # which 的结果写入证据，便于发现 PATH 选中了错误版本（例如 CUDA 自带旧 nsys）。
    resolved = shutil.which(command)
    if resolved is None:
        return {"found": False, "path": None, "returncode": None, "output": None}
    try:
        completed = subprocess.run(
            [resolved, *args],
            # 探测失败应该进入 JSON，而不是让整个报告在第一条失败命令处中断。
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        output = "\n".join(
            item.strip() for item in (completed.stdout, completed.stderr) if item.strip()
        )
        return {
            "found": True,
            "path": resolved,
            "returncode": completed.returncode,
            "output": output or None,
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        # found=True 表示 PATH 中确有文件，但它无法启动或超时；这与“未安装”不同。
        return {
            "found": True,
            "path": resolved,
            "returncode": None,
            "output": f"{type(error).__name__}: {error}",
        }


def package_versions(names: Iterable[str] = PACKAGE_NAMES) -> dict[str, str | None]:
    """通过 distribution metadata 读取版本，避免导入包产生 GPU/原生库副作用。"""

    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def torch_report() -> dict[str, Any]:
    """读取 PyTorch/CUDA 事实；import 或 device query 失败都保留为结构化数据。"""

    try:
        import torch
    except Exception as error:  # 可能缺少原生依赖，不能只捕获 ModuleNotFoundError。
        return {"import_ok": False, "error": f"{type(error).__name__}: {error}"}

    report: dict[str, Any] = {
        "import_ok": True,
        "version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "compiled_arches": list(torch.cuda.get_arch_list()),
    }
    if not report["cuda_available"]:
        # CUDA 不可用时禁止继续查询 device 0，否则会用第二个异常掩盖根因。
        return report

    try:
        report.update(
            {
                "device_name": torch.cuda.get_device_name(0),
                "device_count": torch.cuda.device_count(),
                "compute_capability": list(torch.cuda.get_device_capability(0)),
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            }
        )
    except Exception as error:
        # CUDA runtime 可用不保证每个查询都成功，故单独保存 device_query_error。
        report["device_query_error"] = f"{type(error).__name__}: {error}"
    return report


def glibc_version() -> str | None:
    """优先用 platform 读取 glibc；必要时再直接查询当前进程已加载的 libc。"""

    libc_name, libc_version = platform.libc_ver()
    if libc_name and libc_version:
        return f"{libc_name} {libc_version}"
    if platform.system() != "Linux":
        return None
    try:
        # CDLL(None) 引用当前进程，不加载任意路径中的新 libc。
        libc = ctypes.CDLL(None)
        function = libc.gnu_get_libc_version
        function.restype = ctypes.c_char_p
        return f"glibc {function().decode('ascii')}"
    except (AttributeError, OSError, TypeError):
        return None


def collect_report(profile: str) -> dict[str, Any]:
    """收集进程、OS、命令、包和 CUDA 事实，最后附加独立 Gate 结果。"""

    normalized_profile = profile.upper()
    if normalized_profile not in PROFILE_CAPABILITIES:
        raise ValueError(f"Unknown profile {profile!r}; expected L or D.")

    # 每条命令使用固定参数，避免探测逻辑受 shell alias 或交互配置影响。
    commands = {
        "nvidia-smi": command_report(
            "nvidia-smi",
            ("--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader"),
        ),
        "nvcc": command_report("nvcc", ("--version",)),
        "gcc": command_report("gcc", ("--version",)),
        "g++": command_report("g++", ("--version",)),
        "cmake": command_report("cmake", ("--version",)),
        "ninja": command_report("ninja", ("--version",)),
        "cuobjdump": command_report("cuobjdump", ("--version",)),
        "nsys": command_report("nsys", ("--version",)),
    }
    # schema_version 与 UTC 时间让不同机器/不同时刻的报告可以明确比较。
    report: dict[str, Any] = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": normalized_profile,
        "expected_compute_capability": PROFILE_CAPABILITIES[normalized_profile],
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "kernel": platform.version(),
            "os_release": read_os_release(),
            "glibc": glibc_version(),
        },
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
            "virtual_env": os.environ.get("VIRTUAL_ENV"),
        },
        "commands": commands,
        "packages": package_versions(),
        "torch": torch_report(),
    }
    # Gate 是输入报告的派生结论，不反过来修改已采集的 observed facts。
    report["environment_gate"] = evaluate_environment(report)
    return report


def _version_pair(text: str | None) -> tuple[int, int] | None:
    """从任意版本输出中提取第一个 major.minor；只用于确认版本已被记录。"""

    if not text:
        return None
    match = re.search(r"(\d+)\.(\d+)", text)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def evaluate_environment(report: dict[str, Any]) -> dict[str, Any]:
    """对只读报告应用 Phase 0 前置不变量，汇总所有失败项而非遇一项即停。"""

    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, observed: Any, expected: Any) -> None:
        # 同时保存 observed/expected，使失败报告能直接回答“哪里不一致”。
        checks.append(
            {"name": name, "passed": bool(passed), "observed": observed, "expected": expected}
        )

    # 平台、libc 与 venv 先保证目标用户态正确，再检查外部工具和 CUDA。
    add("linux_x86_64", report["host"]["system"] == "Linux" and report["host"]["machine"] == "x86_64", {"system": report["host"]["system"], "machine": report["host"]["machine"]}, {"system": "Linux", "machine": "x86_64"})
    add("glibc_recorded", bool(report["host"].get("glibc")), report["host"].get("glibc"), "non-empty")
    add("virtual_environment", bool(report["python"].get("virtual_env")), report["python"].get("virtual_env"), "active venv")

    # 所有 REQUIRED_COMMANDS 必须既可找到又正常退出；只安装文件但无法运行仍失败。
    for command in REQUIRED_COMMANDS:
        command_data = report["commands"][command]
        add(
            f"command_{command}",
            command_data["found"] and command_data["returncode"] == 0,
            {"found": command_data["found"], "returncode": command_data["returncode"]},
            {"found": True, "returncode": 0},
        )

    # PyTorch 层依次验证 import、CUDA availability、精确 capability 和版本记录。
    torch_data = report["torch"]
    add("torch_import", bool(torch_data.get("import_ok")), torch_data.get("error") or torch_data.get("version"), "import succeeds")
    add("torch_cuda_available", bool(torch_data.get("cuda_available")), torch_data.get("cuda_available"), True)
    add("compute_capability", torch_data.get("compute_capability") == report["expected_compute_capability"], torch_data.get("compute_capability"), report["expected_compute_capability"])
    add("torch_cuda_runtime_recorded", _version_pair(torch_data.get("cuda_runtime")) is not None, torch_data.get("cuda_runtime"), "major.minor")
    add("nvcc_version_recorded", _version_pair(report["commands"]["nvcc"].get("output")) is not None, report["commands"]["nvcc"].get("output"), "major.minor")

    # 保留全部 failed_checks，避免用户修复一项后才看到下一项。
    failed = [check["name"] for check in checks if not check["passed"]]
    return {
        "passed": not failed,
        "status": "PASS" if not failed else "BLOCKED",
        "failed_checks": failed,
        "checks": checks,
    }


def write_json(report: dict[str, Any], path: Path) -> None:
    """以 UTF-8 和稳定缩进写证据；先创建明确的目标父目录。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """构造 CLI parser，Profile 只允许 L/D，避免任意字符串进入 Gate。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("L", "D", "l", "d"), required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 入口：PASS 返回 0，前置环境阻断返回 2。"""

    args = build_parser().parse_args(argv)
    report = collect_report(args.profile)
    if args.output is not None:
        write_json(report, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["environment_gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
